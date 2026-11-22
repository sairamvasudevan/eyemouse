#!/usr/bin/env python3
"""EyeMouse Web Server: Interactive Localhost Gaze Tracking, Calibration & Accuracy Validation.

Runs an HTTP server on localhost (port 8080 or available).
Provides:
1. Live camera capture & MediaPipe facial landmark + iris tracking.
2. Self-learning Bayesian Adaptive Gaze Model (RLS online updates).
3. Live Gaze Pointer with real-time distance vector to targets.
4. Interactive keypress calibration (Look at dot -> Press SPACEBAR to capture).
5. Accuracy Validation Test mode: Held-out targets measuring exact pixel & angular error.
6. Interactive Gaze Playground: Look-to-pop target bubbles to test actual eye control.
7. Calibrated Profile Inspector: View/download active_profile.json.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import sys
import time
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision_engine.pipeline import VisionPipeline
from src.vision_engine.types import PixelFormat, VisionOutput
from src.vision_engine.calibration.profile import DEFAULT_PROFILE_DIR, DEFAULT_PROFILE_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("EyeMouseWeb")

# Global Vision Pipeline instance
pipeline: Optional[VisionPipeline] = None
frame_seq: int = 0


def get_pipeline() -> VisionPipeline:
    global pipeline
    if pipeline is None:
        log.info("Initializing VisionPipeline with AdaptiveGazeModel...")
        pipeline = VisionPipeline()
        try:
            profile_path = pipeline.save_active_profile()
            log.info("Active profile initialized at %s", profile_path)
        except Exception as exc:
            log.warning("Could not pre-save profile: %s", exc)
    return pipeline


def find_free_port(preferred: int = 8080) -> int:
    for port in [preferred, 8088, 8090, 8095, 8000, 5000]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EyeMouse - Self-Learning Gaze Tracker & Accuracy Validator</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(20, 28, 47, 0.88);
      --accent-cyan: #00f0ff;
      --accent-green: #00ff88;
      --accent-yellow: #ffea00;
      --accent-orange: #ffaa00;
      --accent-red: #ff4466;
      --text: #f0f6fc;
      --text-dim: #8b949e;
      --border: rgba(255, 255, 255, 0.12);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      position: relative;
    }
    
    /* Fullscreen Drawing Canvas for Error Vectors, Distance Lines, & Gaze Trail */
    #canvas-layer {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 10;
    }
    
    /* High-Visibility Laser Gaze Pointer */
    #gaze-cursor {
      position: absolute;
      width: 36px;
      height: 36px;
      margin-left: -18px;
      margin-top: -18px;
      border: 2px solid var(--accent-cyan);
      border-radius: 50%;
      background: rgba(0, 240, 255, 0.25);
      box-shadow: 0 0 20px var(--accent-cyan);
      pointer-events: none;
      transition: transform 0.04s linear, opacity 0.2s ease;
      z-index: 40;
      display: block;
    }
    #gaze-cursor::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 8px;
      height: 8px;
      margin-left: -4px;
      margin-top: -4px;
      border-radius: 50%;
      background: #fff;
      box-shadow: 0 0 8px #fff;
    }
    .gaze-tag {
      position: absolute;
      top: -24px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 240, 255, 0.9);
      color: #000;
      font-family: monospace;
      font-size: 10px;
      font-weight: 900;
      padding: 2px 6px;
      border-radius: 4px;
      white-space: nowrap;
      pointer-events: none;
      letter-spacing: 0.5px;
    }
    .gaze-cross-h {
      position: absolute;
      top: 50%;
      left: -8px;
      width: 52px;
      height: 1.5px;
      background: rgba(0, 240, 255, 0.6);
      transform: translateY(-50%);
    }
    .gaze-cross-v {
      position: absolute;
      left: 50%;
      top: -8px;
      height: 52px;
      width: 1.5px;
      background: rgba(0, 240, 255, 0.6);
      transform: translateX(-50%);
    }

    /* Calibration / Validation Target Dot */
    #calib-target {
      position: absolute;
      width: 54px;
      height: 54px;
      margin-left: -27px;
      margin-top: -27px;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 30;
      cursor: pointer;
      transition: left 0.3s cubic-bezier(0.16, 1, 0.3, 1), top 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .target-ring {
      position: absolute;
      width: 50px;
      height: 50px;
      border: 3px solid var(--accent-yellow);
      border-radius: 50%;
      animation: pulse-ring 1.8s infinite ease-out;
    }
    .target-center {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--accent-yellow);
      box-shadow: 0 0 16px var(--accent-yellow);
      position: relative;
    }
    .target-center::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 4px;
      height: 4px;
      margin-left: -2px;
      margin-top: -2px;
      background: #000;
      border-radius: 50%;
    }
    .target-cross-h {
      position: absolute;
      width: 38px;
      height: 2px;
      background: rgba(255, 234, 0, 0.5);
    }
    .target-cross-v {
      position: absolute;
      width: 2px;
      height: 38px;
      background: rgba(255, 234, 0, 0.5);
    }
    
    @keyframes pulse-ring {
      0% { transform: scale(0.8); opacity: 1; }
      100% { transform: scale(1.6); opacity: 0; }
    }
    
    .confirmed-pulse {
      animation: confirm-flash 0.35s ease-out !important;
    }
    @keyframes confirm-flash {
      0% { transform: scale(0.9); }
      50% { transform: scale(1.4); }
      100% { transform: scale(1.0); }
    }

    /* Top Header Bar */
    header {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 50;
      background: linear-gradient(180deg, rgba(11, 15, 25, 0.95) 0%, rgba(11, 15, 25, 0) 100%);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 32px;
      height: 32px;
      background: var(--accent-cyan);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #000;
      font-weight: 900;
      font-size: 15px;
    }
    .brand-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    .brand-tag {
      font-size: 11px;
      color: var(--accent-cyan);
      background: rgba(0, 240, 255, 0.12);
      padding: 3px 8px;
      border-radius: 12px;
      font-weight: 600;
      border: 1px solid rgba(0, 240, 255, 0.3);
    }
    
    /* Instruction Banner */
    #banner {
      position: absolute;
      bottom: 90px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      padding: 14px 28px;
      border-radius: 30px;
      display: flex;
      align-items: center;
      gap: 14px;
      z-index: 50;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      font-size: 14px;
    }
    .kbd-key {
      background: #21262d;
      border: 1px solid #30363d;
      border-bottom: 2px solid #30363d;
      color: #fff;
      padding: 3px 8px;
      border-radius: 6px;
      font-family: monospace;
      font-size: 13px;
      font-weight: bold;
    }
    
    /* Bottom Controls */
    #controls {
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 10px;
      z-index: 50;
      flex-wrap: wrap;
      justify-content: center;
      max-width: 90vw;
    }
    .btn {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 9px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
      backdrop-filter: blur(8px);
    }
    .btn:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.3);
    }
    .btn-primary {
      background: var(--accent-cyan);
      color: #000;
      border-color: var(--accent-cyan);
    }
    .btn-primary:hover {
      background: #33f3ff;
    }
    .btn-green {
      background: rgba(0, 255, 136, 0.15);
      color: var(--accent-green);
      border-color: var(--accent-green);
    }
    .btn-green:hover {
      background: rgba(0, 255, 136, 0.28);
    }
    
    /* PIP Video Panel */
    #pip-container {
      position: absolute;
      bottom: 24px;
      right: 24px;
      width: 240px;
      height: 180px;
      background: #000;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      z-index: 50;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
      position: relative;
    }
    #webcam-video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1);
    }
    #pip-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      transform: scaleX(-1);
    }
    #pip-badge {
      position: absolute;
      top: 8px;
      left: 8px;
      font-size: 10px;
      font-weight: 700;
      background: rgba(0, 0, 0, 0.65);
      padding: 3px 6px;
      border-radius: 4px;
      color: var(--accent-green);
      display: flex;
      align-items: center;
      gap: 5px;
      z-index: 5;
    }
    .badge-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent-green);
    }

    /* Diagnostics HUD */
    #hud {
      position: absolute;
      top: 75px;
      left: 24px;
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      padding: 14px 18px;
      border-radius: 10px;
      font-family: monospace;
      font-size: 12px;
      line-height: 1.6;
      color: var(--text-dim);
      z-index: 50;
      min-width: 240px;
    }
    .hud-title {
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: #fff;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }
    .hud-val {
      color: var(--accent-cyan);
      font-weight: bold;
    }

    /* Modals (Profile Inspector, Scorecard) */
    .modal-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
    }
    .modal-box {
      background: #141c2f;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 28px;
      max-width: 650px;
      width: 90%;
      max-height: 85vh;
      overflow-y: auto;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }
    .modal-title {
      font-size: 20px;
      font-weight: bold;
      color: #fff;
    }
    .close-btn {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 20px;
      cursor: pointer;
    }
    pre.code-block {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 14px;
      font-size: 12px;
      color: #c9d1d9;
      overflow-x: auto;
      max-height: 380px;
      font-family: monospace;
    }

    /* Scorecard Table */
    table.metrics-table {
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 13px;
    }
    table.metrics-table th, table.metrics-table td {
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    table.metrics-table th {
      color: var(--text-dim);
      font-size: 11px;
      text-transform: uppercase;
    }

    /* Gaze Target Bubble in Playground Mode */
    .playground-bubble {
      position: absolute;
      width: 74px;
      height: 74px;
      margin-left: -37px;
      margin-top: -37px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(0,240,255,0.4) 0%, rgba(0,240,255,0.1) 70%);
      border: 2px solid var(--accent-cyan);
      box-shadow: 0 0 20px rgba(0,240,255,0.3);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 35;
      font-size: 11px;
      font-weight: bold;
      color: #fff;
      transition: transform 0.15s ease, background 0.15s ease;
      cursor: pointer;
    }
    .playground-bubble.dwell-active {
      background: radial-gradient(circle, rgba(0,255,136,0.6) 0%, rgba(0,255,136,0.2) 70%);
      border-color: var(--accent-green);
      transform: scale(1.15);
      box-shadow: 0 0 25px var(--accent-green);
    }
  </style>
</head>
<body>

  <!-- Fullscreen Drawing Canvas for Error Vectors & Distance Lines -->
  <canvas id="canvas-layer"></canvas>

  <!-- Live Predicted Gaze Pointer -->
  <div id="gaze-cursor" style="left: 50%; top: 50%;">
    <div class="gaze-cross-h"></div>
    <div class="gaze-cross-v"></div>
    <div class="gaze-tag">YOUR GAZE</div>
  </div>

  <!-- Calibration / Validation Target -->
  <div id="calib-target" style="left: 50%; top: 50%;">
    <div class="target-cross-h"></div>
    <div class="target-cross-v"></div>
    <div class="target-ring"></div>
    <div class="target-center"></div>
  </div>

  <!-- Top Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">👁</div>
      <div>
        <span class="brand-title">EyeMouse</span>
        <span class="brand-tag">Online Learning Model</span>
      </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span id="calib-counter" style="font-size: 13px; font-weight: 600; color: var(--accent-cyan);">Target 1 of 5</span>
      <button class="btn" onclick="openProfileModal()">📄 View Profile JSON</button>
      <button class="btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
    </div>
  </header>

  <!-- Diagnostics HUD -->
  <div id="hud">
    <div class="hud-title">
      <span>Diagnostics Telemetry</span>
      <span id="hud-mode-tag" style="color: var(--accent-green); font-size: 10px;">LIVE</span>
    </div>
    <div>Tracking: <span id="hud-face" class="hud-val">Detecting...</span></div>
    <div>Inference: <span id="hud-latency" class="hud-val">0.0 ms</span></div>
    <div>FPS: <span id="hud-fps" class="hud-val">0</span></div>
    <div>Gaze (u, v): <span id="hud-gaze" class="hud-val">0.50, 0.50</span></div>
    <div>Screen (px): <span id="hud-px" class="hud-val">0, 0</span></div>
    <div>Iris UV: <span id="hud-iris" class="hud-val">L: 0.50, 0.00 | R: 0.50, 0.00</span></div>
    <div>Head Pose: <span id="hud-pose" class="hud-val">0°, 0°, 0°</span></div>
    <div>Target Dist: <span id="hud-dist" class="hud-val">0 px</span></div>
    <div>Adapted: <span id="hud-adapted" class="hud-val">0 pts</span></div>
  </div>

  <!-- Instruction Banner -->
  <div id="banner">
    <span>Look directly at the yellow dot, then press</span>
    <span class="kbd-key">SPACEBAR</span>
    <span>to capture eye position. Watch the cyan pointer follow your eyes!</span>
  </div>

  <!-- Bottom Control Toolbar -->
  <div id="controls">
    <button class="btn btn-primary" onclick="capturePoint()">Capture Target (Space)</button>
    <button class="btn" onclick="startQuickOnePoint()">1-Pt Center Calib</button>
    <button class="btn" onclick="startFivePoint()">5-Pt Adaptive Calib</button>
    <button class="btn" onclick="startNinePoint()">9-Pt Calib Grid</button>
    <button class="btn btn-green" onclick="startAccuracyTest()">🎯 Test Calibration Accuracy</button>
    <button class="btn" onclick="startPlayground()">🎮 Gaze Playground</button>
    <button class="btn" onclick="resetAdaptation()">Reset to Prior</button>
  </div>

  <!-- Webcam PIP View -->
  <div id="pip-container">
    <div id="pip-badge">
      <div class="badge-dot"></div>
      <span id="camera-status">WEBCAM LIVE</span>
    </div>
    <video id="webcam-video" autoplay playsinline muted></video>
    <canvas id="pip-canvas" width="240" height="180"></canvas>
  </div>

  <canvas id="hidden-canvas" style="display: none;" width="640" height="480"></canvas>

  <!-- Profile Inspector Modal -->
  <div id="profile-modal" class="modal-overlay" onclick="closeProfileModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div class="modal-title">Active Calibration Profile (JSON)</div>
        <button class="close-btn" onclick="closeProfileModal()">&times;</button>
      </div>
      <p style="font-size: 13px; color: var(--text-dim); margin-bottom: 12px;">
        Loaded from <code style="color: var(--accent-cyan);">~/.eyemouse/profiles/active_profile.json</code>.
        Contains the adapted weights $W$ and prior covariance matrix $P$.
      </p>
      <pre id="profile-json-content" class="code-block">Loading profile...</pre>
      <div style="margin-top: 16px; display: flex; justify-content: flex-end; gap: 10px;">
        <button class="btn" onclick="closeProfileModal()">Close</button>
      </div>
    </div>
  </div>

  <!-- Accuracy Validation Scorecard Modal -->
  <div id="scorecard-modal" class="modal-overlay" onclick="closeScorecardModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div class="modal-title">Objective Accuracy Validation Report</div>
        <button class="close-btn" onclick="closeScorecardModal()">&times;</button>
      </div>
      <div id="scorecard-summary" style="margin-bottom: 16px;"></div>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Target</th>
            <th>Target Pos (px)</th>
            <th>Predicted Gaze (px)</th>
            <th>Pixel Error</th>
            <th>Visual Angle</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="scorecard-table-body"></tbody>
      </table>
      <div style="margin-top: 16px; display: flex; justify-content: flex-end; gap: 10px;">
        <button class="btn btn-primary" onclick="closeScorecardModal()">Done</button>
      </div>
    </div>
  </div>

  <script>
    // State Management
    let mode = "calibration"; // "calibration" | "accuracy_test" | "playground"
    let targets = [
      { name: "Center", x: 0.5, y: 0.5 },
      { name: "Top-Left", x: 0.15, y: 0.15 },
      { name: "Top-Right", x: 0.85, y: 0.15 },
      { name: "Bottom-Left", x: 0.15, y: 0.85 },
      { name: "Bottom-Right", x: 0.85, y: 0.85 }
    ];
    let validationTargets = [
      { name: "Val-Mid-Top", x: 0.50, y: 0.25 },
      { name: "Val-Mid-Left", x: 0.30, y: 0.50 },
      { name: "Val-Mid-Right", x: 0.70, y: 0.50 },
      { name: "Val-Mid-Bottom", x: 0.50, y: 0.75 },
      { name: "Val-Inner-TopLeft", x: 0.35, y: 0.35 },
      { name: "Val-Inner-BottomRight", x: 0.65, y: 0.65 },
      { name: "Val-Center-Offset", x: 0.45, y: 0.55 },
      { name: "Val-Center", x: 0.50, y: 0.50 }
    ];
    let currentTargetIdx = 0;
    let latestFeatures = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0];
    let latestGaze = { u: 0.5, v: 0.5, valid: false };
    let isProcessing = false;
    let fpsFrames = 0;
    let fpsStart = performance.now();
    let adaptedCount = 0;
    let validationResults = [];

    // Playground Targets
    let playgroundScore = 0;
    let currentDwellTarget = null;
    let dwellStartTime = 0;

    const video = document.getElementById("webcam-video");
    const hiddenCanvas = document.getElementById("hidden-canvas");
    const hctx = hiddenCanvas.getContext("2d");
    const drawCanvas = document.getElementById("canvas-layer");
    const dctx = drawCanvas.getContext("2d");
    const pipCanvas = document.getElementById("pip-canvas");
    const pctx = pipCanvas.getContext("2d");
    const targetEl = document.getElementById("calib-target");
    const gazeCursor = document.getElementById("gaze-cursor");
    const counterEl = document.getElementById("calib-counter");

    function resizeCanvas() {
      drawCanvas.width = window.innerWidth;
      drawCanvas.height = window.innerHeight;
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    // Camera Init
    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: { ideal: 30 } }
        });
        video.srcObject = stream;
        video.play().catch(() => {});
        video.onloadedmetadata = () => {
          video.play();
          requestAnimationFrame(processLoop);
        };
        // Fallback trigger in case metadata event fired early
        setTimeout(() => {
          if (video.readyState >= 2) {
            requestAnimationFrame(processLoop);
          }
        }, 500);
      } catch (err) {
        console.error("Camera access error:", err);
        document.getElementById("camera-status").innerText = "CAM PERMISSION NEEDED";
        document.getElementById("camera-status").style.color = "#ff4444";
      }
    }

    // Set Active Calibration / Validation Target
    function setTarget(idx) {
      const activeList = (mode === "accuracy_test") ? validationTargets : targets;
      if (idx >= activeList.length) {
        if (mode === "calibration") {
          document.getElementById("banner").innerHTML = "🎉 <b>Calibration Complete!</b> Gaze model adapted online. Try <b>Test Accuracy</b> or <b>Gaze Playground</b> below.";
          targetEl.style.display = "none";
          counterEl.innerText = "Completed (" + activeList.length + " pts)";
        } else if (mode === "accuracy_test") {
          finishAccuracyTest();
        }
        return;
      }
      targetEl.style.display = "flex";
      currentTargetIdx = idx;
      const t = activeList[idx];
      targetEl.style.left = (t.x * 100) + "%";
      targetEl.style.top = (t.y * 100) + "%";

      const prefix = (mode === "accuracy_test") ? "Validation Point" : "Target";
      counterEl.innerText = prefix + " " + (idx + 1) + " of " + activeList.length + " (" + t.name + ")";
    }

    // Main Vision Processing Loop
    async function processLoop() {
      if (!isProcessing && video.readyState >= 2) {
        isProcessing = true;
        const t0 = performance.now();
        hctx.drawImage(video, 0, 0, 640, 480);
        const dataUrl = hiddenCanvas.toDataURL("image/jpeg", 0.75);

        try {
          const res = await fetch("/api/process_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl })
          });
          const data = await res.json();
          const latency = (performance.now() - t0).toFixed(1);

          // Update HUD
          document.getElementById("hud-latency").innerText = latency + " ms";
          if (data.error) {
            document.getElementById("hud-face").innerText = "Pipeline Error";
            document.getElementById("hud-face").style.color = "var(--accent-red)";
          } else {
            document.getElementById("hud-face").innerText = data.face_detected ? "Tracking (OK)" : "Face Lost";
            document.getElementById("hud-face").style.color = data.face_detected ? "var(--accent-green)" : "var(--accent-red)";
          }

          if (data.head_pose) {
            document.getElementById("hud-pose").innerText =
              data.head_pose.yaw + "°, " + data.head_pose.pitch + "°, " + data.head_pose.roll + "°";
          }

          if (data.eye_features) {
            const lx = data.eye_features.left_x;
            const ly = data.eye_features.left_y;
            const rx = data.eye_features.right_x;
            const ry = data.eye_features.right_y;
            document.getElementById("hud-iris").innerText = `L: ${lx.toFixed(2)}, ${ly.toFixed(2)} | R: ${rx.toFixed(2)}, ${ry.toFixed(2)}`;

            if (data.head_pose) {
              latestFeatures = [
                lx, ly, rx, ry,
                data.head_pose.yaw,
                data.head_pose.pitch,
                data.head_pose.roll
              ];
            }

            // Draw pupil points on PIP canvas
            pctx.clearRect(0, 0, pipCanvas.width, pipCanvas.height);
            if (data.eye_features.left_iris_px && data.eye_features.right_iris_px) {
              const scaleX = pipCanvas.width / 640.0;
              const scaleY = pipCanvas.height / 480.0;
              const lip = data.eye_features.left_iris_px;
              const rip = data.eye_features.right_iris_px;

              pctx.fillStyle = "#00ff88";
              pctx.beginPath();
              pctx.arc(lip[0] * scaleX, lip[1] * scaleY, 4, 0, Math.PI * 2);
              pctx.arc(rip[0] * scaleX, rip[1] * scaleY, 4, 0, Math.PI * 2);
              pctx.fill();
            }
          }

          if (data.gaze && data.gaze.is_valid) {
            latestGaze = { u: data.gaze.filtered_x, v: data.gaze.filtered_y, valid: true };
            const gx = latestGaze.u;
            const gy = latestGaze.v;
            const screenX = Math.round(gx * window.innerWidth);
            const screenY = Math.round(gy * window.innerHeight);

            document.getElementById("hud-gaze").innerText = gx.toFixed(3) + ", " + gy.toFixed(3);
            document.getElementById("hud-px").innerText = screenX + "px, " + screenY + "px";

            // Move the prominent live gaze cursor
            gazeCursor.style.display = "block";
            gazeCursor.style.left = screenX + "px";
            gazeCursor.style.top = screenY + "px";

            // If in calibration or accuracy test, draw real-time distance vector line to active target
            if ((mode === "calibration" || mode === "accuracy_test") && targetEl.style.display !== "none") {
              const activeList = (mode === "accuracy_test") ? validationTargets : targets;
              if (currentTargetIdx < activeList.length) {
                const t = activeList[currentTargetIdx];
                const tx = t.x * window.innerWidth;
                const ty = t.y * window.innerHeight;
                const distPx = Math.round(Math.hypot(screenX - tx, screenY - ty));
                document.getElementById("hud-dist").innerText = distPx + " px";

                dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
                dctx.save();
                dctx.beginPath();
                dctx.setLineDash([4, 4]);
                dctx.moveTo(tx, ty);
                dctx.lineTo(screenX, screenY);
                dctx.strokeStyle = distPx < 70 ? "rgba(0, 255, 136, 0.8)" : "rgba(0, 240, 255, 0.6)";
                dctx.lineWidth = 2;
                dctx.stroke();

                // Draw distance text badge along vector
                dctx.font = "bold 11px monospace";
                dctx.fillStyle = distPx < 70 ? "#00ff88" : "#00f0ff";
                dctx.fillText(distPx + "px", (tx + screenX) / 2 + 8, (ty + screenY) / 2);
                dctx.restore();
              }
            }

            // If in playground mode, check dwell hit
            if (mode === "playground") {
              checkPlaygroundHit(screenX, screenY);
            }
          } else {
            latestGaze.valid = false;
          }

          // FPS calculation
          fpsFrames++;
          const now = performance.now();
          if (now - fpsStart >= 1000) {
            document.getElementById("hud-fps").innerText = fpsFrames;
            fpsFrames = 0;
            fpsStart = now;
          }
        } catch (err) {
          // Display network or fetch error on HUD
          document.getElementById("hud-face").innerText = "Conn Err";
          document.getElementById("hud-face").style.color = "var(--accent-red)";
        } finally {
          isProcessing = false;
        }
      }
      requestAnimationFrame(processLoop);
    }

    // Capture Calibration / Validation Point (Spacebar or Button)
    async function capturePoint() {
      if (mode === "playground") return;
      const activeList = (mode === "accuracy_test") ? validationTargets : targets;
      if (currentTargetIdx >= activeList.length) return;
      const t = activeList[currentTargetIdx];

      // Visual Confirmation Animation
      targetEl.classList.add("confirmed-pulse");
      const center = targetEl.querySelector(".target-center");
      if (center) center.style.background = "var(--accent-green)";

      if (mode === "calibration") {
        try {
          const res = await fetch("/api/adapt_point", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              features: latestFeatures,
              target_norm_x: t.x,
              target_norm_y: t.y,
              weight: 1.0
            })
          });
          const resp = await res.json();
          adaptedCount = resp.adaptation_points_count || (adaptedCount + 1);
          document.getElementById("hud-adapted").innerText = adaptedCount + " pts";
        } catch (err) {
          console.error("Adaptation error:", err);
        }
      } else if (mode === "accuracy_test") {
        // Record validation point prediction
        const targetPxX = t.x * window.innerWidth;
        const targetPxY = t.y * window.innerHeight;
        const predPxX = latestGaze.u * window.innerWidth;
        const predPxY = latestGaze.v * window.innerHeight;
        const errPx = Math.hypot(predPxX - targetPxX, predPxY - targetPxY);
        // Approx visual degrees at ~55 cm on typical ~220 PPI display: ~35 px per degree
        const degErr = (errPx / 35.0).toFixed(2);

        validationResults.push({
          name: t.name,
          targetX: Math.round(targetPxX),
          targetY: Math.round(targetPxY),
          predX: Math.round(predPxX),
          predY: Math.round(predPxY),
          errorPx: Math.round(errPx),
          errorDeg: parseFloat(degErr)
        });

        // Draw persistent error line on canvas
        drawPersistentErrorVector(targetPxX, targetPxY, predPxX, predPxY, errPx);
      }

      setTimeout(() => {
        targetEl.classList.remove("confirmed-pulse");
        if (center) center.style.background = "var(--accent-yellow)";
        setTarget(currentTargetIdx + 1);
      }, 300);
    }

    function drawPersistentErrorVector(tx, ty, px, py, errPx) {
      dctx.save();
      dctx.beginPath();
      dctx.moveTo(tx, ty);
      dctx.lineTo(px, py);
      dctx.strokeStyle = errPx < 70 ? "#00ff88" : (errPx < 130 ? "#ffaa00" : "#ff4466");
      dctx.lineWidth = 3;
      dctx.stroke();

      // Predicted pip
      dctx.beginPath();
      dctx.arc(px, py, 6, 0, Math.PI * 2);
      dctx.fillStyle = dctx.strokeStyle;
      dctx.fill();

      // Error label
      dctx.font = "bold 12px monospace";
      dctx.fillStyle = "#fff";
      dctx.fillText(Math.round(errPx) + "px", (tx + px) / 2 + 8, (ty + py) / 2);
      dctx.restore();
    }

    // Finish Accuracy Test & Show Objective Scorecard
    function finishAccuracyTest() {
      targetEl.style.display = "none";
      const count = validationResults.length;
      if (count === 0) return;

      const errors = validationResults.map(r => r.errorPx).sort((a, b) => a - b);
      const medianErr = errors[Math.floor(count / 2)];
      const p95Idx = Math.min(count - 1, Math.floor(count * 0.95));
      const p95Err = errors[p95Idx];
      const meanDeg = (validationResults.reduce((acc, r) => acc + r.errorDeg, 0) / count).toFixed(2);

      let grade = "EXCELLENT";
      let gradeColor = "var(--accent-green)";
      if (medianErr > 150) {
        grade = "FAIR / UNCALIBRATED";
        gradeColor = "var(--accent-red)";
      } else if (medianErr > 80) {
        grade = "GOOD";
        gradeColor = "var(--accent-orange)";
      }

      const summaryHtml = `
        <div style="background: rgba(255,255,255,0.06); padding: 16px; border-radius: 10px; margin-bottom: 16px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 14px; color: var(--text-dim);">Calibration Accuracy Grade:</span>
            <span style="font-size: 16px; font-weight: bold; color: ${gradeColor};">${grade}</span>
          </div>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-family: monospace;">
            <div><div style="color: var(--text-dim); font-size: 11px;">MEDIAN ERROR</div><div style="font-size: 18px; color: var(--accent-cyan); font-weight: bold;">${medianErr} px</div></div>
            <div><div style="color: var(--text-dim); font-size: 11px;">P95 ERROR</div><div style="font-size: 18px; color: var(--accent-cyan); font-weight: bold;">${p95Err} px</div></div>
            <div><div style="color: var(--text-dim); font-size: 11px;">MEAN ANGLE</div><div style="font-size: 18px; color: var(--accent-cyan); font-weight: bold;">${meanDeg}°</div></div>
          </div>
        </div>
      `;
      document.getElementById("scorecard-summary").innerHTML = summaryHtml;

      const tbody = document.getElementById("scorecard-table-body");
      tbody.innerHTML = "";
      validationResults.forEach(r => {
        const isOk = r.errorPx <= 90;
        const statusHtml = isOk
          ? '<span style="color: var(--accent-green); font-weight: bold;">PASS</span>'
          : '<span style="color: var(--accent-orange); font-weight: bold;">CHECK</span>';
        const row = `
          <tr>
            <td style="font-weight: 600;">${r.name}</td>
            <td style="font-family: monospace;">(${r.targetX}, ${r.targetY})</td>
            <td style="font-family: monospace;">(${r.predX}, ${r.predY})</td>
            <td style="font-family: monospace; font-weight: bold; color: ${isOk ? 'var(--accent-green)' : 'var(--accent-orange)'};">${r.errorPx} px</td>
            <td style="font-family: monospace;">${r.errorDeg}°</td>
            <td>${statusHtml}</td>
          </tr>
        `;
        tbody.innerHTML += row;
      });

      document.getElementById("scorecard-modal").style.display = "flex";
      document.getElementById("banner").innerHTML = "🎯 Accuracy test complete! Scorecard displayed. Lines on screen show error vectors.";
    }

    // Interactive Gaze Playground ("Try It Out" Bubble Test)
    function startPlayground() {
      mode = "playground";
      dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      targetEl.style.display = "none";
      document.getElementById("hud-mode-tag").innerText = "PLAYGROUND";
      document.getElementById("hud-mode-tag").style.color = "var(--accent-cyan)";
      document.getElementById("banner").innerHTML = "🎮 <b>Gaze Playground:</b> Look directly at the target bubbles and dwell for ~0.3s to pop them!";
      counterEl.innerText = "Target Practice: Hits 0 / 5";

      clearPlaygroundBubbles();
      playgroundScore = 0;
      spawnPlaygroundTarget(1);
    }

    function clearPlaygroundBubbles() {
      document.querySelectorAll(".playground-bubble").forEach(el => el.remove());
    }

    function spawnPlaygroundTarget(step) {
      clearPlaygroundBubbles();
      if (step > 5) {
        document.getElementById("banner").innerHTML = "🎉 <b>Target Practice Complete!</b> All 5 bubbles popped by your gaze.";
        counterEl.innerText = "Playground: 5 of 5 Targets Hit!";
        return;
      }
      const coords = [
        { x: 0.50, y: 0.50, name: "Center" },
        { x: 0.20, y: 0.25, name: "Top-Left" },
        { x: 0.80, y: 0.25, name: "Top-Right" },
        { x: 0.25, y: 0.75, name: "Bottom-Left" },
        { x: 0.75, y: 0.75, name: "Bottom-Right" }
      ];
      const cur = coords[step - 1];
      const bubble = document.createElement("div");
      bubble.className = "playground-bubble";
      bubble.style.left = (cur.x * 100) + "%";
      bubble.style.top = (cur.y * 100) + "%";
      bubble.innerHTML = `<span>Look!</span><span style="font-size: 9px; opacity: 0.8;">#${step}</span>`;
      bubble.dataset.step = step;
      bubble.dataset.px = Math.round(cur.x * window.innerWidth);
      bubble.dataset.py = Math.round(cur.y * window.innerHeight);
      document.body.appendChild(bubble);
      counterEl.innerText = `Target Practice: Bubble ${step} of 5 (${cur.name})`;
    }

    function checkPlaygroundHit(gx, gy) {
      const bubble = document.querySelector(".playground-bubble");
      if (!bubble) return;
      const rect = bubble.getBoundingClientRect();
      const bx = rect.left + rect.width / 2;
      const by = rect.top + rect.height / 2;
      const dist = Math.hypot(gx - bx, gy - by);

      // Hit threshold radius ~80 px
      if (dist < 80) {
        bubble.classList.add("dwell-active");
        if (currentDwellTarget !== bubble) {
          currentDwellTarget = bubble;
          dwellStartTime = performance.now();
        } else {
          const dwellTime = performance.now() - dwellStartTime;
          if (dwellTime >= 350) { // 350ms dwell to pop!
            popPlaygroundBubble(bubble);
          }
        }
      } else {
        if (currentDwellTarget === bubble) {
          bubble.classList.remove("dwell-active");
          currentDwellTarget = null;
        }
      }
    }

    function popPlaygroundBubble(bubble) {
      const step = parseInt(bubble.dataset.step, 10);
      bubble.style.transform = "scale(1.8)";
      bubble.style.opacity = "0";
      setTimeout(() => {
        spawnPlaygroundTarget(step + 1);
      }, 250);
    }

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" || e.code === "Enter") {
        e.preventDefault();
        capturePoint();
      } else if (e.key === "1") {
        startQuickOnePoint();
      } else if (e.key === "5") {
        startFivePoint();
      } else if (e.key === "9") {
        startNinePoint();
      } else if (e.key.toLowerCase() === "t") {
        startAccuracyTest();
      } else if (e.key.toLowerCase() === "p") {
        startPlayground();
      } else if (e.code === "Escape") {
        closeScorecardModal();
        closeProfileModal();
        if (document.fullscreenElement) {
          document.exitFullscreen();
        }
      }
    });

    // Launchers
    function startQuickOnePoint() {
      mode = "calibration";
      clearPlaygroundBubbles();
      dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      targets = [{ name: "Screen Center", x: 0.5, y: 0.5 }];
      document.getElementById("hud-mode-tag").innerText = "1-PT CALIB";
      document.getElementById("banner").innerHTML = "Look at the center yellow dot, then press <span class='kbd-key'>SPACEBAR</span>.";
      setTarget(0);
    }

    function startFivePoint() {
      mode = "calibration";
      clearPlaygroundBubbles();
      dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      targets = [
        { name: "Center", x: 0.5, y: 0.5 },
        { name: "Top-Left", x: 0.15, y: 0.15 },
        { name: "Top-Right", x: 0.85, y: 0.15 },
        { name: "Bottom-Left", x: 0.15, y: 0.85 },
        { name: "Bottom-Right", x: 0.85, y: 0.85 }
      ];
      document.getElementById("hud-mode-tag").innerText = "5-PT CALIB";
      document.getElementById("banner").innerHTML = "Look at each yellow dot, then press <span class='kbd-key'>SPACEBAR</span> to capture.";
      setTarget(0);
    }

    function startNinePoint() {
      mode = "calibration";
      clearPlaygroundBubbles();
      dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      targets = [
        { name: "Center", x: 0.5, y: 0.5 },
        { name: "Top-Left", x: 0.15, y: 0.15 },
        { name: "Top-Center", x: 0.5, y: 0.15 },
        { name: "Top-Right", x: 0.85, y: 0.15 },
        { name: "Right-Center", x: 0.85, y: 0.5 },
        { name: "Bottom-Right", x: 0.85, y: 0.85 },
        { name: "Bottom-Center", x: 0.5, y: 0.85 },
        { name: "Bottom-Left", x: 0.15, y: 0.85 },
        { name: "Left-Center", x: 0.15, y: 0.5 }
      ];
      document.getElementById("hud-mode-tag").innerText = "9-PT CALIB";
      document.getElementById("banner").innerHTML = "Look at each yellow dot, then press <span class='kbd-key'>SPACEBAR</span> to capture.";
      setTarget(0);
    }

    function startAccuracyTest() {
      mode = "accuracy_test";
      clearPlaygroundBubbles();
      dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      validationResults = [];
      document.getElementById("hud-mode-tag").innerText = "ACCURACY TEST";
      document.getElementById("hud-mode-tag").style.color = "var(--accent-orange)";
      document.getElementById("banner").innerHTML = "🎯 <b>Accuracy Test:</b> Look at each test dot and press <span class='kbd-key'>SPACEBAR</span> to measure correctness.";
      setTarget(0);
    }

    async function resetAdaptation() {
      try {
        await fetch("/api/reset_adaptation", { method: "POST" });
        adaptedCount = 0;
        document.getElementById("hud-adapted").innerText = "0 pts (Base Prior)";
        alert("Model reset to physiological base prior.");
      } catch (err) {
        console.error(err);
      }
    }

    async function openProfileModal() {
      try {
        const res = await fetch("/api/profile");
        const json = await res.json();
        document.getElementById("profile-json-content").innerText = JSON.stringify(json, null, 2);
      } catch (err) {
        document.getElementById("profile-json-content").innerText = "Failed to load active_profile.json";
      }
      document.getElementById("profile-modal").style.display = "flex";
    }

    function closeProfileModal() {
      document.getElementById("profile-modal").style.display = "none";
    }

    function closeScorecardModal() {
      document.getElementById("scorecard-modal").style.display = "none";
    }

    function toggleFullscreen() {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    }

    // Start
    initCamera();
    setTarget(0);
  </script>
</body>
</html>
"""


class EyeMouseHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for web-based gaze tracking, calibration, and accuracy validation."""

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
        elif self.path == "/api/status":
            p = get_pipeline()
            count = getattr(p.gaze_estimator.model, "adaptation_points_count", 0)
            data = {
                "status": "RUNNING",
                "model": "AdaptiveGazeModel",
                "adaptation_points_count": count,
                "is_calibrated": p.gaze_estimator.is_calibrated,
            }
            self._send_json(data)
        elif self.path == "/api/profile":
            profile_path = DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME
            if profile_path.exists():
                try:
                    with open(profile_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json(data)
                    return
                except Exception as exc:
                    log.error("Failed reading profile: %s", exc)
            # Fallback to current pipeline profile
            p = get_pipeline()
            try:
                saved = p.save_active_profile()
                with open(saved, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json(data)
            except Exception as exc:
                self._send_json({"error": str(exc), "model": "AdaptiveGazeModel"})
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if self.path == "/api/process_frame":
            self._handle_process_frame(payload)
        elif self.path == "/api/adapt_point":
            self._handle_adapt_point(payload)
        elif self.path == "/api/reset_adaptation":
            self._handle_reset_adaptation()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _handle_process_frame(self, payload: Dict[str, Any]) -> None:
        global frame_seq
        p = get_pipeline()

        img_b64 = payload.get("image", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self._send_json({"error": "Failed to decode image", "face_detected": False})
                return

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            frame_seq += 1
            now_ns = time.time_ns()

            output: VisionOutput = p.process_raw_frame(
                pixel_data=rgb,
                width=w,
                height=h,
                stride=w * 3,
                pixel_format=PixelFormat.RGB,
                sequence_num=frame_seq,
                timestamp_ns=now_ns,
            )

            resp: Dict[str, Any] = {
                "face_detected": output.face_detected,
                "confidence": round(output.confidence, 3),
                "processing_time_ms": round(output.processing_time_ms, 2),
            }

            if output.gaze_estimate:
                resp["gaze"] = {
                    "raw_x": round(output.gaze_estimate.raw_x, 4),
                    "raw_y": round(output.gaze_estimate.raw_y, 4),
                    "filtered_x": round(output.gaze_estimate.filtered_x, 4),
                    "filtered_y": round(output.gaze_estimate.filtered_y, 4),
                    "is_valid": output.gaze_estimate.is_valid,
                    "is_calibrated": p.gaze_estimator.is_calibrated,
                }

            if output.eye_metrics:
                l_norm = output.eye_metrics.left_normalized
                r_norm = output.eye_metrics.right_normalized
                resp["eye_features"] = {
                    "left_x": round(l_norm.iris_x, 4),
                    "left_y": round(l_norm.iris_y, 4),
                    "right_x": round(r_norm.iris_x, 4),
                    "right_y": round(r_norm.iris_y, 4),
                    "left_openness": round(l_norm.openness, 4),
                    "right_openness": round(r_norm.openness, 4),
                    "left_iris_px": [round(output.eye_metrics.left_iris_center.x, 1), round(output.eye_metrics.left_iris_center.y, 1)],
                    "right_iris_px": [round(output.eye_metrics.right_iris_center.x, 1), round(output.eye_metrics.right_iris_center.y, 1)],
                }

            if output.head_pose:
                resp["head_pose"] = {
                    "yaw": round(output.head_pose.yaw, 2),
                    "pitch": round(output.head_pose.pitch, 2),
                    "roll": round(output.head_pose.roll, 2),
                }

            self._send_json(resp)
        except Exception as exc:
            log.error("Error processing frame: %s", exc)
            self._send_json({"error": str(exc), "face_detected": False})

    def _handle_adapt_point(self, payload: Dict[str, Any]) -> None:
        p = get_pipeline()
        features = payload.get("features", [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        tx = float(payload.get("target_norm_x", 0.5))
        ty = float(payload.get("target_norm_y", 0.5))
        weight = float(payload.get("weight", 1.0))

        try:
            pred_u, pred_v = p.adapt_calibration_point(
                features=features,
                target_norm_x=tx,
                target_norm_y=ty,
                weight=weight,
            )
            count = getattr(p.gaze_estimator.model, "adaptation_points_count", 0)
            self._send_json({
                "status": "POINT_ADAPTED",
                "pred_x": round(pred_u, 4),
                "pred_y": round(pred_v, 4),
                "adaptation_points_count": count,
                "is_calibrated": p.gaze_estimator.is_calibrated,
            })
        except Exception as exc:
            log.error("Adaptation error: %s", exc)
            self._send_json({"status": "ADAPTATION_ERROR", "error": str(exc)})

    def _handle_reset_adaptation(self) -> None:
        p = get_pipeline()
        try:
            p.reset_adaptation()
            self._send_json({
                "status": "ADAPTATION_RESET",
                "is_calibrated": p.gaze_estimator.is_calibrated,
            })
        except Exception as exc:
            log.error("Reset error: %s", exc)
            self._send_json({"status": "ERROR", "error": str(exc)})

    def _send_json(self, data: Dict[str, Any]) -> None:
        res = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(res)))
        self.end_headers()
        self.wfile.write(res)

    def log_message(self, format: str, *args) -> None:
        # Suppress verbose per-frame log output
        return


def run_server(port: Optional[int] = None) -> None:
    chosen_port = port if port else find_free_port(8080)
    server_address = ("127.0.0.1", chosen_port)
    httpd = HTTPServer(server_address, EyeMouseHTTPRequestHandler)
    print(f"EYEMOUSE_SERVER_RUNNING on http://127.0.0.1:{chosen_port}", flush=True)
    log.info("EyeMouse HTTP server started at http://127.0.0.1:%d", chosen_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_server(p)
