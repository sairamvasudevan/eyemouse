#!/usr/bin/env python3
"""EyeMouse Web Server: Multi-User Gaze Training Ground, Calibration & Accuracy Validation.

Runs an HTTP server on 0.0.0.0 (accessible over LAN / Wi-Fi).
Provides:
1. Non-inverted, calibrated webcam gaze tracking via MediaPipe.
2. Self-learning Bayesian Adaptive Gaze Model (RLS online updates).
3. Crowdsourced Training Ground: Multi-user calibration data collection (JSONL).
4. Global Model Training: Fits Ridge regression over all collected multi-user data.
5. Accuracy Validation Test mode: Held-out targets measuring exact pixel & angular error.
6. Interactive Gaze Playground: Look-to-pop target bubbles.
7. LAN Discovery: Automatically prints and displays shareable IP URL.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone
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
from src.vision_engine.calibration.adaptive_model import train_global_ridge_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("EyeMouseWeb")

# Data directories
DATASET_DIR = PROJECT_ROOT / "data" / "training_ground"
DATASET_FILE = DATASET_DIR / "samples.jsonl"
SESSIONS_DIR = DATASET_DIR / "sessions"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Global Vision Pipeline instance
pipeline: Optional[VisionPipeline] = None
frame_seq: int = 0
SERVER_PORT: int = 8080


def get_lan_ip() -> str:
    """Detect primary LAN IPv4 address for local network sharing."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


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
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def get_dataset_stats() -> Dict[str, Any]:
    """Calculate summary statistics for crowdsourced training ground dataset."""
    if not DATASET_FILE.exists():
        return {"total_samples": 0, "participants": [], "participant_count": 0}
    total = 0
    participants = set()
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
                    try:
                        record = json.loads(line)
                        p = record.get("user_name")
                        if p:
                            participants.add(p)
                    except Exception:
                        pass
    except Exception as exc:
        log.error("Error reading dataset stats: %s", exc)
    return {
        "total_samples": total,
        "participants": sorted(list(participants)),
        "participant_count": len(participants),
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>EyeMouse - Multi-User Gaze Training Ground & Calibrator</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 25, 43, 0.90);
      --accent-cyan: #00f0ff;
      --accent-green: #00ff88;
      --accent-yellow: #ffea00;
      --accent-orange: #ffaa00;
      --accent-red: #ff4466;
      --accent-purple: #b388ff;
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
    
    /* Fullscreen Drawing Canvas */
    #canvas-layer {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 10;
    }
    
    /* Live Predicted Gaze Cursor */
    #gaze-cursor {
      position: absolute;
      width: 34px;
      height: 34px;
      margin-left: -17px;
      margin-top: -17px;
      border: 2px solid var(--accent-cyan);
      border-radius: 50%;
      background: rgba(0, 240, 255, 0.22);
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
      width: 50px;
      height: 1.5px;
      background: rgba(0, 240, 255, 0.6);
      transform: translateY(-50%);
    }
    .gaze-cross-v {
      position: absolute;
      left: 50%;
      top: -8px;
      height: 50px;
      width: 1.5px;
      background: rgba(0, 240, 255, 0.6);
      transform: translateX(-50%);
    }

    /* Target Dot for Calibration & Training Ground */
    #calib-target {
      position: absolute;
      width: 58px;
      height: 58px;
      margin-left: -29px;
      margin-top: -29px;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 30;
      cursor: pointer;
      transition: left 0.35s cubic-bezier(0.16, 1, 0.3, 1), top 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .target-ring {
      position: absolute;
      width: 54px;
      height: 54px;
      border: 3px solid var(--accent-yellow);
      border-radius: 50%;
      animation: pulse-ring 1.6s infinite ease-out;
    }
    .target-center {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--accent-yellow);
      box-shadow: 0 0 18px var(--accent-yellow);
      position: relative;
      transition: background 0.2s, transform 0.2s;
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
      width: 42px;
      height: 2px;
      background: rgba(255, 234, 0, 0.5);
    }
    .target-cross-v {
      position: absolute;
      width: 2px;
      height: 42px;
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
      padding: 14px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 50;
      background: linear-gradient(180deg, rgba(9, 13, 22, 0.96) 0%, rgba(9, 13, 22, 0.2) 100%);
      flex-wrap: wrap;
      gap: 12px;
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
      font-size: 16px;
    }
    .brand-title {
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    
    /* Top Tabs */
    .nav-tabs {
      display: flex;
      background: rgba(255, 255, 255, 0.06);
      padding: 4px;
      border-radius: 10px;
      gap: 4px;
    }
    .nav-tab {
      background: transparent;
      border: none;
      color: var(--text-dim);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .nav-tab:hover {
      color: var(--text);
    }
    .nav-tab.active {
      background: var(--card-bg);
      color: var(--accent-cyan);
      border: 1px solid rgba(0, 240, 255, 0.3);
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    /* LAN Link Share Badge */
    .lan-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(0, 240, 255, 0.1);
      border: 1px solid rgba(0, 240, 255, 0.3);
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-family: monospace;
      color: var(--accent-cyan);
    }
    .lan-copy-btn {
      background: rgba(0, 240, 255, 0.2);
      border: 1px solid var(--accent-cyan);
      color: #fff;
      padding: 2px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 11px;
      font-weight: bold;
    }
    .lan-copy-btn:hover {
      background: var(--accent-cyan);
      color: #000;
    }

    /* User Profile Chip */
    .user-chip {
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 12px;
    }
    .user-input {
      background: transparent;
      border: none;
      color: var(--accent-green);
      font-weight: bold;
      width: 90px;
      outline: none;
      font-size: 12px;
    }

    /* Diagnostics HUD */
    #hud {
      position: absolute;
      top: 75px;
      left: 20px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 11px;
      line-height: 1.6;
      font-family: monospace;
      z-index: 50;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      min-width: 250px;
    }
    .hud-title {
      font-weight: 700;
      color: var(--text-dim);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }
    .hud-val {
      color: var(--accent-cyan);
      font-weight: bold;
    }

    /* Instruction Banner */
    #banner {
      position: absolute;
      bottom: 96px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      padding: 12px 24px;
      border-radius: 30px;
      display: flex;
      align-items: center;
      gap: 12px;
      z-index: 50;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      font-size: 13.5px;
      max-width: 90vw;
      text-align: center;
    }
    .kbd-key {
      background: #21262d;
      border: 1px solid #30363d;
      border-bottom: 2px solid #30363d;
      color: #fff;
      padding: 2px 7px;
      border-radius: 6px;
      font-family: monospace;
      font-size: 12px;
      font-weight: bold;
    }

    /* Bottom Control Bar */
    #controls {
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 8px;
      z-index: 50;
      flex-wrap: wrap;
      justify-content: center;
      max-width: 95vw;
    }
    .btn {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 9px 15px;
      border-radius: 8px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
      backdrop-filter: blur(8px);
    }
    .btn:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.25);
    }
    .btn-primary {
      background: var(--accent-cyan);
      color: #000;
      border-color: var(--accent-cyan);
      font-weight: 700;
    }
    .btn-primary:hover {
      background: #2df4ff;
      box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    }
    .btn-green {
      background: rgba(0, 255, 136, 0.15);
      color: var(--accent-green);
      border-color: rgba(0, 255, 136, 0.4);
    }
    .btn-green:hover {
      background: var(--accent-green);
      color: #000;
    }
    .btn-purple {
      background: rgba(179, 136, 255, 0.15);
      color: var(--accent-purple);
      border-color: rgba(179, 136, 255, 0.4);
    }
    .btn-purple:hover {
      background: var(--accent-purple);
      color: #000;
    }

    /* Webcam PIP Window */
    #pip-container {
      position: absolute;
      bottom: 20px;
      right: 20px;
      width: 220px;
      height: 165px;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #000;
      z-index: 50;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
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
    }
    #pip-badge {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 9px;
      font-weight: bold;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 4px;
      font-family: monospace;
      z-index: 2;
    }
    .badge-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent-green);
    }

    /* Modal dialogs */
    .modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(6px);
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
    }
    .modal-box {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      max-width: 620px;
      width: 90vw;
      max-height: 85vh;
      overflow-y: auto;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .modal-title {
      font-size: 17px;
      font-weight: 700;
      color: var(--accent-cyan);
    }
    .close-btn {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 20px;
      cursor: pointer;
    }
    .close-btn:hover { color: #fff; }

    .stat-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin: 16px 0;
    }
    .stat-card {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 8px;
      text-align: center;
    }
    .stat-val {
      font-size: 20px;
      font-weight: bold;
      color: var(--accent-green);
      margin-top: 4px;
    }
    .stat-lbl {
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
    }

    /* Scorecard Table */
    .metrics-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 12px;
      font-family: monospace;
    }
    .metrics-table th, .metrics-table td {
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .metrics-table th {
      color: var(--text-dim);
      font-size: 11px;
    }
    .status-badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: bold;
    }
    .badge-pass { background: rgba(0, 255, 136, 0.2); color: var(--accent-green); }
    .badge-warn { background: rgba(255, 234, 0, 0.2); color: var(--accent-yellow); }
    .badge-fail { background: rgba(255, 68, 102, 0.2); color: var(--accent-red); }

    /* Playground bubbles */
    .playground-bubble {
      position: absolute;
      width: 60px;
      height: 60px;
      margin-left: -30px;
      margin-top: -30px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(0,240,255,0.4) 0%, rgba(0,240,255,0.1) 70%);
      border: 2px solid var(--accent-cyan);
      box-shadow: 0 0 15px rgba(0,240,255,0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      z-index: 35;
      transition: transform 0.2s, background 0.2s;
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

  <!-- Fullscreen Drawing Canvas for Distance Vectors -->
  <canvas id="canvas-layer"></canvas>

  <!-- Live Predicted Gaze Cursor -->
  <div id="gaze-cursor" style="left: 50%; top: 50%;">
    <div class="gaze-cross-h"></div>
    <div class="gaze-cross-v"></div>
    <div class="gaze-tag">YOUR GAZE</div>
  </div>

  <!-- Calibration / Training Target -->
  <div id="calib-target" style="left: 50%; top: 50%;" onclick="capturePoint()">
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
        <span style="font-size: 11px; color: var(--accent-cyan); margin-left: 6px;">v2.0 Adaptive</span>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
      <button id="tab-tg" class="nav-tab active" onclick="switchMode('training_ground')">🎓 Training Ground</button>
      <button id="tab-5pt" class="nav-tab" onclick="switchMode('calib_5pt')">🎯 5-Pt Calib</button>
      <button id="tab-9pt" class="nav-tab" onclick="switchMode('calib_9pt')">📐 9-Pt Grid</button>
      <button id="tab-test" class="nav-tab" onclick="switchMode('accuracy_test')">📊 Accuracy Test</button>
      <button id="tab-play" class="nav-tab" onclick="switchMode('playground')">🎮 Playground</button>
    </div>

    <div style="display: flex; gap: 10px; align-items: center;">
      <!-- LAN Share Link Badge -->
      <div class="lan-pill" title="Share this URL with anyone on your local Wi-Fi to collect their eye gaze calibration data!">
        <span>🌐 <span id="lan-link-text">Detecting...</span></span>
        <button class="lan-copy-btn" onclick="copyLanLink()">Copy</button>
      </div>

      <!-- User Identification Chip -->
      <div class="user-chip">
        <span>👤 User:</span>
        <input id="user-name" class="user-input" value="User-1" title="Change name to tag your samples" onchange="saveUserName()">
      </div>

      <button class="btn" onclick="toggleFullscreen()" title="Fullscreen">⛶</button>
    </div>
  </header>

  <!-- Diagnostics Telemetry HUD -->
  <div id="hud">
    <div class="hud-title">
      <span>Diagnostics Telemetry</span>
      <span id="hud-mode-tag" style="color: var(--accent-green); font-size: 10px;">TRAINING GROUND</span>
    </div>
    <div>Tracking: <span id="hud-face" class="hud-val">Detecting...</span></div>
    <div>Inference: <span id="hud-latency" class="hud-val">0.0 ms</span></div>
    <div>FPS: <span id="hud-fps" class="hud-val">0</span></div>
    <div>Gaze (u, v): <span id="hud-gaze" class="hud-val">0.50, 0.50</span></div>
    <div>Screen (px): <span id="hud-px" class="hud-val">0, 0</span></div>
    <div>Iris UV: <span id="hud-iris" class="hud-val">L: 0.50, 0.00 | R: 0.50, 0.00</span></div>
    <div>Head Pose: <span id="hud-pose" class="hud-val">0°, 0°, 0°</span></div>
    <div>Target Dist: <span id="hud-dist" class="hud-val">0 px</span></div>
    <div>Session Samples: <span id="hud-session-pts" class="hud-val">0 pts</span></div>
    <div>Global Dataset: <span id="hud-dataset-pts" class="hud-val">0 samples</span></div>
  </div>

  <!-- Instruction Banner -->
  <div id="banner">
    <span id="banner-text">Look directly at the yellow dot, then press <span class="kbd-key">SPACEBAR</span> (or tap dot) to capture. Gaze coordinates adapt online in real-time!</span>
  </div>

  <!-- Bottom Controls -->
  <div id="controls">
    <button class="btn btn-primary" onclick="capturePoint()">Capture Target (Space)</button>
    <button id="tg-pattern-btn" class="btn" onclick="toggleTrainingPattern()">Target Grid: 13-Point</button>
    <button class="btn btn-purple" onclick="trainGlobalModel()">🧠 Train Global Model on All Users</button>
    <button class="btn" onclick="exportDataset()">💾 Export Dataset (JSONL)</button>
    <button class="btn btn-green" onclick="startAccuracyTest()">🎯 Test Accuracy</button>
    <button class="btn" onclick="resetAdaptation()">Reset to Prior</button>
  </div>

  <!-- Webcam PIP View -->
  <div id="pip-container">
    <div id="pip-badge">
      <div class="badge-dot"></div>
      <span id="camera-status">WEBCAM (MIRRORED)</span>
    </div>
    <video id="webcam-video" autoplay playsinline muted></video>
    <canvas id="pip-canvas" width="220" height="165"></canvas>
  </div>

  <canvas id="hidden-canvas" style="display: none;" width="640" height="480"></canvas>

  <!-- Global Model Retraining Report Modal -->
  <div id="train-modal" class="modal-overlay" onclick="closeTrainModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div class="modal-title">🧠 Global Model Retrained on Crowdsourced Dataset</div>
        <button class="close-btn" onclick="closeTrainModal()">&times;</button>
      </div>
      <p style="font-size: 13px; color: var(--text-dim); margin-bottom: 14px;">
        Multi-user Bayesian Ridge regression converged. The physiological base prior was optimized using empirical fixation vectors collected from all participants.
      </p>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-lbl">Total Samples</div>
          <div id="stat-samples" class="stat-val">0</div>
        </div>
        <div class="stat-card">
          <div class="stat-lbl">Participants</div>
          <div id="stat-users" class="stat-val">0</div>
        </div>
        <div class="stat-card">
          <div class="stat-lbl">Prior Error (1080p)</div>
          <div id="stat-prior-err" class="stat-val" style="color: var(--accent-orange);">0 px</div>
        </div>
        <div class="stat-card">
          <div class="stat-lbl">Trained Error (1080p)</div>
          <div id="stat-fitted-err" class="stat-val">0 px</div>
        </div>
      </div>
      <div style="background: rgba(0, 255, 136, 0.1); border: 1px solid var(--accent-green); padding: 10px; border-radius: 8px; font-size: 13px; color: #fff; margin-bottom: 16px;">
        ✅ <b>Active Profile Updated!</b> Your live vision engine is now running with the retrained multi-user weights.
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 10px;">
        <button class="btn btn-primary" onclick="closeTrainModal()">Continue</button>
      </div>
    </div>
  </div>

  <!-- Accuracy Validation Scorecard Modal -->
  <div id="scorecard-modal" class="modal-overlay" onclick="closeScorecardModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div class="modal-title">🎯 Objective Gaze Accuracy Report</div>
        <button class="close-btn" onclick="closeScorecardModal()">&times;</button>
      </div>
      <div id="scorecard-summary" style="margin-bottom: 16px;"></div>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Target</th>
            <th>Target Pos</th>
            <th>Predicted Gaze</th>
            <th>Pixel Error</th>
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
    let currentMode = "training_ground"; // "training_ground" | "calib_5pt" | "calib_9pt" | "accuracy_test" | "playground"
    let trainingPattern = "grid13"; // "grid13" | "random"
    let currentTargetIdx = 0;
    let sessionSamplesCount = 0;
    let totalDatasetCount = 0;
    let sessionId = "tg_" + Date.now().toString(36) + "_" + Math.random().toString(36).substring(2, 6);
    let userName = localStorage.getItem("eyemouse_user") || "User-1";
    document.getElementById("user-name").value = userName;

    // Grid Targets
    const targets13 = [
      { name: "Center", x: 0.50, y: 0.50 },
      { name: "Top-Center", x: 0.50, y: 0.15 },
      { name: "Bottom-Center", x: 0.50, y: 0.85 },
      { name: "Left-Center", x: 0.15, y: 0.50 },
      { name: "Right-Center", x: 0.85, y: 0.50 },
      { name: "Top-Left", x: 0.15, y: 0.15 },
      { name: "Top-Right", x: 0.85, y: 0.15 },
      { name: "Bottom-Left", x: 0.15, y: 0.85 },
      { name: "Bottom-Right", x: 0.85, y: 0.85 },
      { name: "Inner-TopLeft", x: 0.32, y: 0.32 },
      { name: "Inner-TopRight", x: 0.68, y: 0.32 },
      { name: "Inner-BottomLeft", x: 0.32, y: 0.68 },
      { name: "Inner-BottomRight", x: 0.68, y: 0.68 }
    ];
    const targets5 = [
      { name: "Center", x: 0.50, y: 0.50 },
      { name: "Top-Left", x: 0.15, y: 0.15 },
      { name: "Top-Right", x: 0.85, y: 0.15 },
      { name: "Bottom-Left", x: 0.15, y: 0.85 },
      { name: "Bottom-Right", x: 0.85, y: 0.85 }
    ];
    const targets9 = [
      { name: "Center", x: 0.50, y: 0.50 },
      { name: "Top-Left", x: 0.15, y: 0.15 },
      { name: "Top-Center", x: 0.50, y: 0.15 },
      { name: "Top-Right", x: 0.85, y: 0.15 },
      { name: "Right-Center", x: 0.85, y: 0.50 },
      { name: "Bottom-Right", x: 0.85, y: 0.85 },
      { name: "Bottom-Center", x: 0.50, y: 0.85 },
      { name: "Bottom-Left", x: 0.15, y: 0.85 },
      { name: "Left-Center", x: 0.15, y: 0.50 }
    ];
    const validationTargets = [
      { name: "Val-Mid-Top", x: 0.50, y: 0.25 },
      { name: "Val-Mid-Left", x: 0.30, y: 0.50 },
      { name: "Val-Mid-Right", x: 0.70, y: 0.50 },
      { name: "Val-Mid-Bottom", x: 0.50, y: 0.75 },
      { name: "Val-Inner-TopLeft", x: 0.35, y: 0.35 },
      { name: "Val-Inner-BottomRight", x: 0.65, y: 0.65 },
      { name: "Val-Center-Offset", x: 0.45, y: 0.55 },
      { name: "Val-Center", x: 0.50, y: 0.50 }
    ];

    let latestFeatures = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0];
    let latestEyeMetrics = null;
    let latestGaze = { u: 0.5, v: 0.5, valid: false };
    let latestConfidence = 0.0;
    let isProcessing = false;
    let fpsFrames = 0;
    let fpsStart = performance.now();
    let validationResults = [];

    // Playground state
    let playgroundScore = 0;
    let currentDwellTarget = null;
    let dwellStartTime = 0;

    // Elements
    const video = document.getElementById("webcam-video");
    const hiddenCanvas = document.getElementById("hidden-canvas");
    const hctx = hiddenCanvas.getContext("2d");
    const drawCanvas = document.getElementById("canvas-layer");
    const dctx = drawCanvas.getContext("2d");
    const pipCanvas = document.getElementById("pip-canvas");
    const pctx = pipCanvas.getContext("2d");
    const targetEl = document.getElementById("calib-target");
    const gazeCursor = document.getElementById("gaze-cursor");

    // Audio Feedback Generator via Web Audio API
    let audioCtx = null;
    function playBeep(freq = 880, duration = 0.07) {
      try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === "suspended") audioCtx.resume();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
      } catch (e) {}
    }

    function resizeCanvas() {
      drawCanvas.width = window.innerWidth;
      drawCanvas.height = window.innerHeight;
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    function saveUserName() {
      userName = document.getElementById("user-name").value.trim() || "User-1";
      localStorage.setItem("eyemouse_user", userName);
    }

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
        setTimeout(() => {
          if (video.readyState >= 2) requestAnimationFrame(processLoop);
        }, 500);
      } catch (err) {
        console.error("Camera access error:", err);
        document.getElementById("camera-status").innerText = "CAM PERMISSION NEEDED";
        document.getElementById("camera-status").style.color = "#ff4444";
      }
    }

    // Query Network Info & Dataset Stats
    async function loadNetworkInfo() {
      try {
        const res = await fetch("/api/info");
        const data = await res.json();
        const displayUrl = data.network_url || ("http://" + location.host);
        document.getElementById("lan-link-text").innerText = displayUrl;
      } catch (e) {
        document.getElementById("lan-link-text").innerText = "http://" + location.host;
      }
      refreshDatasetStats();
    }

    async function refreshDatasetStats() {
      try {
        const res = await fetch("/api/training_ground/stats");
        const data = await res.json();
        totalDatasetCount = data.total_samples || 0;
        document.getElementById("hud-dataset-pts").innerText =
          `${totalDatasetCount} samples (${data.participant_count || 1} users)`;
      } catch (e) {}
    }

    function copyLanLink() {
      const text = document.getElementById("lan-link-text").innerText;
      navigator.clipboard.writeText(text).then(() => {
        alert("Copied network link to clipboard:\n" + text + "\n\nShare this with anyone on your local Wi-Fi!");
      }).catch(() => {
        prompt("Copy this network link:", text);
      });
    }

    // Target Sequence Resolver
    function getActiveTargetList() {
      if (currentMode === "training_ground") return targets13;
      if (currentMode === "calib_5pt") return targets5;
      if (currentMode === "calib_9pt") return targets9;
      if (currentMode === "accuracy_test") return validationTargets;
      return targets13;
    }

    function setTarget(idx) {
      if (currentMode === "playground") {
        targetEl.style.display = "none";
        return;
      }
      targetEl.style.display = "flex";

      if (currentMode === "training_ground" && trainingPattern === "random") {
        const rx = 0.12 + Math.random() * 0.76;
        const ry = 0.12 + Math.random() * 0.76;
        targetEl.style.left = (rx * 100) + "%";
        targetEl.style.top = (ry * 100) + "%";
        targetEl.dataset.targetX = rx;
        targetEl.dataset.targetY = ry;
        targetEl.dataset.targetName = "Random-" + (sessionSamplesCount + 1);
        document.getElementById("banner-text").innerHTML =
          `Look directly at the yellow dot, then press <span class="kbd-key">SPACEBAR</span>. Gaze features adapt online in real-time!`;
        return;
      }

      const list = getActiveTargetList();
      if (idx >= list.length) {
        if (currentMode === "accuracy_test") {
          finishAccuracyTest();
        } else {
          document.getElementById("banner-text").innerHTML =
            `🎉 <b>Routine Complete!</b> Total: ${sessionSamplesCount} samples. Try <b>Test Accuracy</b> or <b>Train Global Model</b>.`;
          targetEl.style.display = "none";
        }
        return;
      }

      currentTargetIdx = idx;
      const t = list[idx];
      targetEl.style.left = (t.x * 100) + "%";
      targetEl.style.top = (t.y * 100) + "%";
      targetEl.dataset.targetX = t.x;
      targetEl.dataset.targetY = t.y;
      targetEl.dataset.targetName = t.name;

      const prefix = currentMode === "accuracy_test" ? "Validation Point" : "Target";
      document.getElementById("banner-text").innerHTML =
        `Look at <b>${t.name}</b> (${idx + 1}/${list.length}) & press <span class="kbd-key">SPACEBAR</span> to capture eye position.`;
    }

    // MAIN VISION PROCESSING LOOP
    async function processLoop() {
      if (!isProcessing && video.readyState >= 2) {
        isProcessing = true;
        const t0 = performance.now();

        // HORIZONTAL MIRRORING: Draw video flipped horizontally so MediaPipe perspective matches user selfie view
        hctx.save();
        hctx.translate(640, 0);
        hctx.scale(-1, 1);
        hctx.drawImage(video, 0, 0, 640, 480);
        hctx.restore();

        const dataUrl = hiddenCanvas.toDataURL("image/jpeg", 0.75);

        try {
          const res = await fetch("/api/process_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl })
          });
          const data = await res.json();
          const latency = (performance.now() - t0).toFixed(1);

          document.getElementById("hud-latency").innerText = latency + " ms";
          if (data.error) {
            document.getElementById("hud-face").innerText = "Pipeline Error";
            document.getElementById("hud-face").style.color = "var(--accent-red)";
          } else {
            document.getElementById("hud-face").innerText = data.face_detected ? "Tracking (OK)" : "Face Lost";
            document.getElementById("hud-face").style.color = data.face_detected ? "var(--accent-green)" : "var(--accent-red)";
          }

          if (data.confidence !== undefined) latestConfidence = data.confidence;

          if (data.head_pose) {
            document.getElementById("hud-pose").innerText =
              data.head_pose.yaw + "°, " + data.head_pose.pitch + "°, " + data.head_pose.roll + "°";
          }

          if (data.eye_features) {
            latestEyeMetrics = data.eye_features;
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

            // PIP pupil rendering
            pctx.clearRect(0, 0, pipCanvas.width, pipCanvas.height);
            if (data.eye_features.left_iris_px && data.eye_features.right_iris_px) {
              const scaleX = pipCanvas.width / 640.0;
              const scaleY = pipCanvas.height / 480.0;
              // Mirror coordinates on PIP canvas to match selfie preview
              const lip = data.eye_features.left_iris_px;
              const rip = data.eye_features.right_iris_px;
              const lx_pip = (640 - lip[0]) * scaleX;
              const rx_pip = (640 - rip[0]) * scaleX;

              pctx.fillStyle = "#00ff88";
              pctx.beginPath();
              pctx.arc(lx_pip, lip[1] * scaleY, 4, 0, Math.PI * 2);
              pctx.arc(rx_pip, rip[1] * scaleY, 4, 0, Math.PI * 2);
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

            // Move the prominent live laser gaze cursor
            gazeCursor.style.display = "block";
            gazeCursor.style.left = screenX + "px";
            gazeCursor.style.top = screenY + "px";

            // Draw distance vector line to target if active
            if (targetEl.style.display !== "none") {
              const tx = parseFloat(targetEl.dataset.targetX || "0.5") * window.innerWidth;
              const ty = parseFloat(targetEl.dataset.targetY || "0.5") * window.innerHeight;
              const distPx = Math.round(Math.hypot(screenX - tx, screenY - ty));
              document.getElementById("hud-dist").innerText = distPx + " px";

              dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
              dctx.save();
              dctx.beginPath();
              dctx.setLineDash([4, 4]);
              dctx.moveTo(tx, ty);
              dctx.lineTo(screenX, screenY);
              dctx.strokeStyle = distPx < 70 ? "rgba(0, 255, 136, 0.85)" : "rgba(0, 240, 255, 0.65)";
              dctx.lineWidth = 2;
              dctx.stroke();

              dctx.font = "bold 11px monospace";
              dctx.fillStyle = distPx < 70 ? "#00ff88" : "#00f0ff";
              dctx.fillText(distPx + "px", (tx + screenX) / 2 + 8, (ty + screenY) / 2);
              dctx.restore();
            } else {
              dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            }

            if (currentMode === "playground") {
              checkPlaygroundHit(screenX, screenY);
            }
          } else {
            latestGaze.valid = false;
          }

          fpsFrames++;
          const now = performance.now();
          if (now - fpsStart >= 1000) {
            document.getElementById("hud-fps").innerText = fpsFrames;
            fpsFrames = 0;
            fpsStart = now;
          }
        } catch (err) {
          document.getElementById("hud-face").innerText = "Conn Err";
          document.getElementById("hud-face").style.color = "var(--accent-red)";
        } finally {
          isProcessing = false;
        }
      }
      requestAnimationFrame(processLoop);
    }

    // CAPTURE TARGET POINT (Spacebar or Click)
    async function capturePoint() {
      if (currentMode === "playground") return;
      if (targetEl.style.display === "none") return;

      const tx = parseFloat(targetEl.dataset.targetX || "0.5");
      const ty = parseFloat(targetEl.dataset.targetY || "0.5");
      const tName = targetEl.dataset.targetName || "Target";

      playBeep(920, 0.08);

      // Visual flash
      targetEl.classList.add("confirmed-pulse");
      const center = targetEl.querySelector(".target-center");
      if (center) center.style.background = "var(--accent-green)";

      if (currentMode === "accuracy_test") {
        const predX = latestGaze.u * window.innerWidth;
        const predY = latestGaze.v * window.innerHeight;
        const tgtX = tx * window.innerWidth;
        const tgtY = ty * window.innerHeight;
        const errPx = Math.hypot(predX - tgtX, predY - tgtY);

        validationResults.push({
          name: tName,
          targetPx: [Math.round(tgtX), Math.round(tgtY)],
          predPx: [Math.round(predX), Math.round(predY)],
          errPx: Math.round(errPx),
          passed: errPx <= 100
        });

        setTimeout(() => {
          if (center) center.style.background = "var(--accent-yellow)";
          targetEl.classList.remove("confirmed-pulse");
          setTarget(currentTargetIdx + 1);
        }, 400);
        return;
      }

      // Record sample payload for dataset logging
      const samplePayload = {
        session_id: sessionId,
        user_name: userName,
        timestamp_ns: Date.now() * 1000000,
        screen_w: window.innerWidth,
        screen_h: window.innerHeight,
        target: {
          name: tName,
          norm_x: tx,
          norm_y: ty,
          px_x: Math.round(tx * window.innerWidth),
          px_y: Math.round(ty * window.innerHeight)
        },
        features: latestFeatures,
        eye_metrics: latestEyeMetrics,
        prediction_before: [latestGaze.u, latestGaze.v],
        confidence: latestConfidence
      };

      try {
        const res = await fetch("/api/training_ground/record_sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(samplePayload)
        });
        const resp = await res.json();
        sessionSamplesCount++;
        document.getElementById("hud-session-pts").innerText = sessionSamplesCount + " pts";
        if (resp.total_samples) {
          totalDatasetCount = resp.total_samples;
          document.getElementById("hud-dataset-pts").innerText =
            `${totalDatasetCount} samples (${resp.participant_count || 1} users)`;
        }
      } catch (err) {
        console.error("Failed recording sample:", err);
      }

      setTimeout(() => {
        if (center) center.style.background = "var(--accent-yellow)";
        targetEl.classList.remove("confirmed-pulse");
        if (currentMode === "training_ground" && trainingPattern === "random") {
          setTarget(0);
        } else {
          setTarget(currentTargetIdx + 1);
        }
      }, 400);
    }

    // Keyboard capture
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" && e.target.tagName !== "INPUT") {
        e.preventDefault();
        capturePoint();
      }
    });

    // Mode Switching
    function switchMode(newMode) {
      currentMode = newMode;
      document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
      clearPlaygroundBubbles();
      dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);

      if (newMode === "training_ground") {
        document.getElementById("tab-tg").classList.add("active");
        document.getElementById("hud-mode-tag").innerText = "TRAINING GROUND";
        document.getElementById("hud-mode-tag").style.color = "var(--accent-green)";
        setTarget(0);
      } else if (newMode === "calib_5pt") {
        document.getElementById("tab-5pt").classList.add("active");
        document.getElementById("hud-mode-tag").innerText = "5-PT CALIB";
        document.getElementById("hud-mode-tag").style.color = "var(--accent-cyan)";
        setTarget(0);
      } else if (newMode === "calib_9pt") {
        document.getElementById("tab-9pt").classList.add("active");
        document.getElementById("hud-mode-tag").innerText = "9-PT GRID";
        document.getElementById("hud-mode-tag").style.color = "var(--accent-cyan)";
        setTarget(0);
      } else if (newMode === "accuracy_test") {
        startAccuracyTest();
      } else if (newMode === "playground") {
        startPlayground();
      }
    }

    function toggleTrainingPattern() {
      if (trainingPattern === "grid13") {
        trainingPattern = "random";
        document.getElementById("tg-pattern-btn").innerText = "Target Grid: Random Dots";
      } else {
        trainingPattern = "grid13";
        document.getElementById("tg-pattern-btn").innerText = "Target Grid: 13-Point";
      }
      setTarget(0);
    }

    // Global Model Retraining
    async function trainGlobalModel() {
      try {
        const res = await fetch("/api/training_ground/train_global_model", { method: "POST" });
        const data = await res.json();
        if (!data.success) {
          alert("Training Notice: " + (data.error || "Need more samples to train."));
          return;
        }

        document.getElementById("stat-samples").innerText = data.sample_count;
        document.getElementById("stat-users").innerText = data.unique_participants;
        document.getElementById("stat-prior-err").innerText = data.prior_pixel_error_1080p + " px";
        document.getElementById("stat-fitted-err").innerText = data.fitted_pixel_error_1080p + " px";
        document.getElementById("train-modal").style.display = "flex";
      } catch (err) {
        alert("Training request failed: " + err);
      }
    }

    function closeTrainModal() {
      document.getElementById("train-modal").style.display = "none";
    }

    function exportDataset() {
      window.location.href = "/api/training_ground/export";
    }

    // Accuracy Test
    function startAccuracyTest() {
      currentMode = "accuracy_test";
      document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
      document.getElementById("tab-test").classList.add("active");
      clearPlaygroundBubbles();
      validationResults = [];
      document.getElementById("hud-mode-tag").innerText = "ACCURACY TEST";
      document.getElementById("hud-mode-tag").style.color = "var(--accent-orange)";
      setTarget(0);
    }

    function finishAccuracyTest() {
      targetEl.style.display = "none";
      let totalErr = 0;
      let passed = 0;
      const tbody = document.getElementById("scorecard-table-body");
      tbody.innerHTML = "";

      validationResults.forEach(r => {
        totalErr += r.errPx;
        if (r.passed) passed++;
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><b>${r.name}</b></td>
          <td>(${r.targetPx[0]}, ${r.targetPx[1]})</td>
          <td>(${r.predPx[0]}, ${r.predPx[1]})</td>
          <td style="color: ${r.errPx < 80 ? 'var(--accent-green)' : (r.errPx < 140 ? 'var(--accent-yellow)' : 'var(--accent-red)')}">
            ${r.errPx} px
          </td>
          <td>
            <span class="status-badge ${r.errPx < 80 ? 'badge-pass' : (r.errPx < 140 ? 'badge-warn' : 'badge-fail')}">
              ${r.errPx < 80 ? 'ACCURATE' : (r.errPx < 140 ? 'FAIR' : 'POOR')}
            </span>
          </td>
        `;
        tbody.appendChild(row);
      });

      const avgErr = Math.round(totalErr / (validationResults.length || 1));
      document.getElementById("scorecard-summary").innerHTML = `
        <div style="font-size: 14px; margin-bottom: 8px;">
          Mean Error: <b style="color: var(--accent-cyan); font-size: 16px;">${avgErr} px</b>
          (${((avgErr / 1920) * 100).toFixed(1)}% screen width) |
          Passed: <b>${passed} of ${validationResults.length}</b> targets
        </div>
      `;
      document.getElementById("scorecard-modal").style.display = "flex";
    }

    function closeScorecardModal() {
      document.getElementById("scorecard-modal").style.display = "none";
      switchMode("training_ground");
    }

    // Gaze Playground (Look-to-pop bubbles)
    function startPlayground() {
      currentMode = "playground";
      document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
      document.getElementById("tab-play").classList.add("active");
      targetEl.style.display = "none";
      dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      document.getElementById("hud-mode-tag").innerText = "PLAYGROUND";
      document.getElementById("hud-mode-tag").style.color = "var(--accent-green)";
      document.getElementById("banner-text").innerHTML =
        `🎮 <b>Look at bubbles for 1.2s to pop them!</b> Tests actual eye dwell and stability.`;
      spawnPlaygroundBubbles();
    }

    function spawnPlaygroundBubbles() {
      clearPlaygroundBubbles();
      const emojis = ["🎯", "🚀", "🌟", "🎈", "💎", "⭐", "🔥", "🔮"];
      for (let i = 0; i < 6; i++) {
        const bubble = document.createElement("div");
        bubble.className = "playground-bubble";
        bubble.innerText = emojis[Math.floor(Math.random() * emojis.length)];
        const rx = 0.15 + Math.random() * 0.70;
        const ry = 0.15 + Math.random() * 0.70;
        bubble.style.left = (rx * 100) + "%";
        bubble.style.top = (ry * 100) + "%";
        bubble.dataset.bx = rx * window.innerWidth;
        bubble.dataset.by = ry * window.innerHeight;
        document.body.appendChild(bubble);
      }
    }

    function checkPlaygroundHit(gx, gy) {
      const bubbles = document.querySelectorAll(".playground-bubble");
      let hit = null;
      bubbles.forEach(b => {
        const bx = parseFloat(b.dataset.bx);
        const by = parseFloat(b.dataset.by);
        const dist = Math.hypot(gx - bx, gy - by);
        if (dist < 65) hit = b;
      });

      if (hit) {
        if (currentDwellTarget !== hit) {
          currentDwellTarget = hit;
          dwellStartTime = performance.now();
          hit.classList.add("dwell-active");
          playBeep(440, 0.05);
        } else {
          const elapsed = performance.now() - dwellStartTime;
          if (elapsed >= 1200) {
            playBeep(1200, 0.15);
            hit.remove();
            playgroundScore++;
            currentDwellTarget = null;
            if (document.querySelectorAll(".playground-bubble").length === 0) {
              setTimeout(spawnPlaygroundBubbles, 500);
            }
          }
        }
      } else {
        if (currentDwellTarget) {
          currentDwellTarget.classList.remove("dwell-active");
          currentDwellTarget = null;
        }
      }
    }

    function clearPlaygroundBubbles() {
      document.querySelectorAll(".playground-bubble").forEach(b => b.remove());
    }

    async function resetAdaptation() {
      try {
        await fetch("/api/reset_adaptation", { method: "POST" });
        sessionSamplesCount = 0;
        document.getElementById("hud-session-pts").innerText = "0 pts (Base Prior)";
        alert("Gaze model reset to physiological base prior.");
      } catch (e) {
        console.error(e);
      }
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
    loadNetworkInfo();
    setTarget(0);
  </script>
</body>
</html>
"""


class EyeMouseHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for multi-user gaze training, calibration, and validation."""

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))

        elif self.path == "/api/info":
            lan_ip = get_lan_ip()
            data = {
                "status": "RUNNING",
                "lan_ip": lan_ip,
                "port": SERVER_PORT,
                "local_url": f"http://localhost:{SERVER_PORT}",
                "network_url": f"http://{lan_ip}:{SERVER_PORT}",
                "model": "AdaptiveGazeModel",
            }
            self._send_json(data)

        elif self.path == "/api/training_ground/stats":
            stats = get_dataset_stats()
            self._send_json(stats)

        elif self.path == "/api/training_ground/export":
            if not DATASET_FILE.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Dataset is empty.")
                return
            try:
                with open(DATASET_FILE, "rb") as f:
                    content = f.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/jsonl")
                self.send_header("Content-Disposition", 'attachment; filename="eyemouse_crowdsourced_dataset.jsonl"')
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

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
        elif self.path == "/api/training_ground/record_sample":
            self._handle_record_sample(payload)
        elif self.path == "/api/training_ground/train_global_model":
            self._handle_train_global_model()
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

    def _handle_record_sample(self, payload: Dict[str, Any]) -> None:
        """Appends verified multi-user gaze calibration sample to JSONL dataset and adapts online."""
        p = get_pipeline()

        features = payload.get("features", [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        tgt = payload.get("target", {})
        tx = float(tgt.get("norm_x", 0.5))
        ty = float(tgt.get("norm_y", 0.5))

        # 1. Immediate Online RLS Adaptation for this active session
        try:
            pred_u, pred_v = p.adapt_calibration_point(
                features=features,
                target_norm_x=tx,
                target_norm_y=ty,
                weight=1.0,
            )
            payload["prediction_after"] = [round(pred_u, 4), round(pred_v, 4)]
        except Exception as exc:
            log.warning("Online adaptation error during sample logging: %s", exc)

        # 2. Add client network metadata
        client_ip = self.client_address[0]
        payload["client_ip"] = client_ip
        payload["received_at_iso"] = datetime.now(timezone.utc).isoformat()

        # 3. Thread-safe append to dataset JSONL
        try:
            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as exc:
            log.error("Failed writing sample to dataset: %s", exc)

        stats = get_dataset_stats()
        self._send_json({
            "status": "SAMPLE_RECORDED",
            "total_samples": stats["total_samples"],
            "participant_count": stats["participant_count"],
        })

    def _handle_train_global_model(self) -> None:
        """Trains global Ridge model on crowdsourced dataset and updates active pipeline."""
        p = get_pipeline()
        global_model_file = PROJECT_ROOT / "models" / "global_trained_model.json"
        global_model_file.parent.mkdir(parents=True, exist_ok=True)

        res = train_global_ridge_model(
            samples_path=DATASET_FILE,
            alpha=2.0,
            output_path=global_model_file,
        )

        if res.get("success") and res.get("weights"):
            weights = np.array(res["weights"], dtype=np.float64)
            p.load_trained_weights(weights)
            log.info("Global model retrained and activated successfully! %d samples, %.1f px error.",
                     res["sample_count"], res["fitted_pixel_error_1080p"])

        self._send_json(res)

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
        return


def run_server(port: Optional[int] = None) -> None:
    global SERVER_PORT
    chosen_port = port if port else find_free_port(8080)
    SERVER_PORT = chosen_port
    lan_ip = get_lan_ip()

    server_address = ("0.0.0.0", chosen_port)
    httpd = HTTPServer(server_address, EyeMouseHTTPRequestHandler)

    banner = f"""
========================================================================
👁  EyeMouse Multi-User Gaze Training Ground & Calibrator LIVE
========================================================================
📍 Localhost:      http://localhost:{chosen_port}
🌐 Network / Wi-Fi: http://{lan_ip}:{chosen_port}

Anyone on the local network can open the Network URL, look at the target
dots, and press SPACEBAR to contribute to the dataset and train the model!
Dataset Storage:    data/training_ground/samples.jsonl
========================================================================
"""
    print(banner, flush=True)
    log.info("Server started on 0.0.0.0:%d (LAN IP: %s)", chosen_port, lan_ip)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_server(p)
