#!/usr/bin/env python3
"""EyeMouse Web Server: Interactive Localhost Gaze Tracking & Calibration.

Runs a lightweight HTTP server on an available localhost port.
Provides:
1. Live camera capture and MediaPipe facial landmark & iris tracking.
2. Self-learning Bayesian Adaptive Gaze Model.
3. Interactive keypress calibration (Look at dot -> Press SPACEBAR to capture).
4. Real-time gaze prediction reticle and diagnostics HUD.
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
from typing import Optional, Dict, Any

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision_engine.pipeline import VisionPipeline
from src.vision_engine.types import PixelFormat, VisionOutput
from src.vision_engine.calibration.adaptive_model import AdaptiveGazeModel
from src.vision_engine.calibration.estimator import CalibratedGazeEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("EyeMouseWeb")

# Global Vision Pipeline instance
pipeline: Optional[VisionPipeline] = None
frame_seq: int = 0


def get_pipeline() -> VisionPipeline:
    global pipeline
    if pipeline is None:
        log.info("Initializing VisionPipeline with AdaptiveGazeModel...")
        pipeline = VisionPipeline(use_calibrated_model=True)
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
  <title>EyeMouse - Self-Learning Gaze Tracker</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(20, 28, 47, 0.85);
      --accent-cyan: #00f0ff;
      --accent-green: #00ff88;
      --accent-orange: #ffaa00;
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
    
    /* Fullscreen Target & Gaze Layer */
    #canvas-layer {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 10;
    }
    
    /* Live Gaze Reticle */
    #gaze-cursor {
      position: absolute;
      width: 32px;
      height: 32px;
      margin-left: -16px;
      margin-top: -16px;
      border: 2px solid var(--accent-cyan);
      border-radius: 50%;
      background: rgba(0, 240, 255, 0.2);
      box-shadow: 0 0 15px var(--accent-cyan);
      pointer-events: none;
      transition: transform 0.05s linear, opacity 0.2s ease;
      z-index: 20;
      display: none;
    }
    #gaze-cursor::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 6px;
      height: 6px;
      margin-left: -3px;
      margin-top: -3px;
      border-radius: 50%;
      background: #fff;
    }

    /* Calibration Target Dot */
    #calib-target {
      position: absolute;
      width: 48px;
      height: 48px;
      margin-left: -24px;
      margin-top: -24px;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 30;
      cursor: pointer;
      transition: left 0.3s cubic-bezier(0.16, 1, 0.3, 1), top 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .target-ring {
      position: absolute;
      width: 44px;
      height: 44px;
      border: 3px solid var(--accent-cyan);
      border-radius: 50%;
      animation: pulse-ring 1.8s infinite ease-out;
    }
    .target-center {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent-cyan);
      box-shadow: 0 0 16px var(--accent-cyan);
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
      background: #fff;
      border-radius: 50%;
    }
    .target-cross-h {
      position: absolute;
      width: 32px;
      height: 2px;
      background: rgba(0, 240, 255, 0.4);
    }
    .target-cross-v {
      position: absolute;
      width: 2px;
      height: 32px;
      background: rgba(0, 240, 255, 0.4);
    }
    
    @keyframes pulse-ring {
      0% { transform: scale(0.8); opacity: 1; }
      100% { transform: scale(1.6); opacity: 0; }
    }
    
    /* Captured Confirmation Flash */
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
      background: linear-gradient(180deg, rgba(11, 15, 25, 0.9) 0%, rgba(11, 15, 25, 0) 100%);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 28px;
      height: 28px;
      background: var(--accent-cyan);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #000;
      font-weight: 900;
      font-size: 14px;
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
      padding: 2px 8px;
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
    }
    .btn {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 9px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
    }
    .btn:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.25);
    }
    .btn-primary {
      background: var(--accent-cyan);
      color: #000;
      border-color: var(--accent-cyan);
    }
    .btn-primary:hover {
      background: #33f3ff;
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
    }
    #webcam-video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1);
    }
    #pip-badge {
      position: absolute;
      top: 8px;
      left: 8px;
      font-size: 10px;
      font-weight: 700;
      background: rgba(0, 0, 0, 0.6);
      padding: 3px 6px;
      border-radius: 4px;
      color: var(--accent-green);
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .badge-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent-green);
    }

    /* Diagnostics HUD (Top Left) */
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
      min-width: 220px;
    }
    .hud-title {
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: #fff;
      margin-bottom: 6px;
    }
    .hud-val {
      color: var(--accent-cyan);
      font-weight: bold;
    }
  </style>
</head>
<body>

  <!-- Fullscreen Drawing Canvas -->
  <canvas id="canvas-layer"></canvas>

  <!-- Live Predicted Gaze Reticle -->
  <div id="gaze-cursor"></div>

  <!-- Calibration Target -->
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
    <div style="display: flex; gap: 12px; align-items: center;">
      <span id="calib-counter" style="font-size: 13px; font-weight: 600; color: var(--accent-cyan);">Target 1 of 5</span>
      <button class="btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
    </div>
  </header>

  <!-- Diagnostics HUD -->
  <div id="hud">
    <div class="hud-title">Diagnostics Telemetry</div>
    <div>Tracking: <span id="hud-face" class="hud-val">Detecting...</span></div>
    <div>Latency: <span id="hud-latency" class="hud-val">0.0 ms</span></div>
    <div>FPS: <span id="hud-fps" class="hud-val">0</span></div>
    <div>Gaze (u, v): <span id="hud-gaze" class="hud-val">0.50, 0.50</span></div>
    <div>Head Pose: <span id="hud-pose" class="hud-val">0°, 0°, 0°</span></div>
    <div>Model: <span id="hud-model" class="hud-val">Adaptive Online</span></div>
    <div>Adapted: <span id="hud-adapted" class="hud-val">0 pts</span></div>
  </div>

  <!-- Instruction Banner -->
  <div id="banner">
    <span>Look directly at the pulsing dot, then press</span>
    <span class="kbd-key">SPACEBAR</span>
    <span>to capture eye position.</span>
  </div>

  <!-- Bottom Control Toolbar -->
  <div id="controls">
    <button class="btn btn-primary" onclick="capturePoint()">Capture Target (Space)</button>
    <button class="btn" onclick="startQuickOnePoint()">1-Point Center</button>
    <button class="btn" onclick="startFivePoint()">5-Point Calibration</button>
    <button class="btn" onclick="startNinePoint()">9-Point Grid</button>
    <button class="btn" onclick="resetAdaptation()">Reset to Prior</button>
  </div>

  <!-- Webcam PIP View -->
  <div id="pip-container">
    <div id="pip-badge">
      <div class="badge-dot"></div>
      <span id="camera-status">WEBCAM LIVE</span>
    </div>
    <video id="webcam-video" autoplay playsinline muted></video>
  </div>

  <canvas id="hidden-canvas" style="display: none;" width="640" height="480"></canvas>

  <script>
    // State Variables
    let targets = [
      { name: "Center", x: 0.5, y: 0.5 },
      { name: "Top-Left", x: 0.15, y: 0.15 },
      { name: "Top-Right", x: 0.85, y: 0.15 },
      { name: "Bottom-Left", x: 0.15, y: 0.85 },
      { name: "Bottom-Right", x: 0.85, y: 0.85 }
    ];
    let currentTargetIdx = 0;
    let latestFeatures = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0];
    let isProcessing = false;
    let fpsFrames = 0;
    let fpsStart = performance.now();
    let adaptedCount = 0;

    const video = document.getElementById("webcam-video");
    const hiddenCanvas = document.getElementById("hidden-canvas");
    const hctx = hiddenCanvas.getContext("2d");
    const targetEl = document.getElementById("calib-target");
    const gazeCursor = document.getElementById("gaze-cursor");
    const counterEl = document.getElementById("calib-counter");

    // Initialize Camera
    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: { ideal: 30 } }
        });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          video.play();
          requestAnimationFrame(processLoop);
        };
      } catch (err) {
        console.error("Camera access error:", err);
        document.getElementById("camera-status").innerText = "CAM PERMISSION NEEDED";
        document.getElementById("camera-status").style.color = "#ff4444";
      }
    }

    // Set Active Target Position
    function setTarget(idx) {
      if (idx >= targets.length) {
        // Calibration completed!
        document.getElementById("banner").innerHTML = "🎉 <b>Calibration Complete!</b> Model adapted online to your eyes.";
        targetEl.style.display = "none";
        counterEl.innerText = "Completed (" + targets.length + " pts)";
        return;
      }
      targetEl.style.display = "flex";
      currentTargetIdx = idx;
      const t = targets[idx];
      targetEl.style.left = (t.x * 100) + "%";
      targetEl.style.top = (t.y * 100) + "%";
      counterEl.innerText = "Target " + (idx + 1) + " of " + targets.length + " (" + t.name + ")";
    }

    // Main Vision Processing Loop
    async function processLoop() {
      if (!isProcessing && video.readyState === 4) {
        isProcessing = true;
        const t0 = performance.now();
        hctx.drawImage(video, 0, 0, 640, 480);
        const dataUrl = hiddenCanvas.toDataURL("image/jpeg", 0.7);

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
          document.getElementById("hud-face").innerText = data.face_detected ? "Tracking (OK)" : "Face Lost";
          document.getElementById("hud-face").style.color = data.face_detected ? "var(--accent-green)" : "#ff4444";

          if (data.head_pose) {
            document.getElementById("hud-pose").innerText =
              data.head_pose.yaw + "°, " + data.head_pose.pitch + "°, " + data.head_pose.roll + "°";
          }

          if (data.eye_features && data.head_pose) {
            latestFeatures = [
              data.eye_features.left_x,
              data.eye_features.left_y,
              data.eye_features.right_x,
              data.eye_features.right_y,
              data.head_pose.yaw,
              data.head_pose.pitch,
              data.head_pose.roll
            ];
          }

          if (data.gaze && data.gaze.is_valid) {
            const gx = data.gaze.filtered_x;
            const gy = data.gaze.filtered_y;
            document.getElementById("hud-gaze").innerText = gx.toFixed(2) + ", " + gy.toFixed(2);

            // Move gaze cursor
            gazeCursor.style.display = "block";
            gazeCursor.style.left = (gx * window.innerWidth) + "px";
            gazeCursor.style.top = (gy * window.innerHeight) + "px";
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
          // Frame error ignored
        } finally {
          isProcessing = false;
        }
      }
      requestAnimationFrame(processLoop);
    }

    // Capture Calibration Point (Spacebar or Button)
    async function capturePoint() {
      if (currentTargetIdx >= targets.length) return;
      const t = targets[currentTargetIdx];

      // Visual Confirmation Animation
      targetEl.classList.add("confirmed-pulse");
      const center = targetEl.querySelector(".target-center");
      if (center) center.style.background = "var(--accent-green)";

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
        adaptedCount++;
        document.getElementById("hud-adapted").innerText = adaptedCount + " pts";
      } catch (err) {
        console.error("Adaptation error:", err);
      }

      setTimeout(() => {
        targetEl.classList.remove("confirmed-pulse");
        if (center) center.style.background = "var(--accent-cyan)";
        setTarget(currentTargetIdx + 1);
      }, 300);
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
      } else if (e.code === "Escape") {
        if (document.fullscreenElement) {
          document.exitFullscreen();
        }
      }
    });

    // Calibration Launchers
    function startQuickOnePoint() {
      targets = [{ name: "Screen Center", x: 0.5, y: 0.5 }];
      document.getElementById("banner").innerHTML = "Look at the center dot, then press <span class='kbd-key'>SPACEBAR</span>.";
      setTarget(0);
    }

    function startFivePoint() {
      targets = [
        { name: "Center", x: 0.5, y: 0.5 },
        { name: "Top-Left", x: 0.15, y: 0.15 },
        { name: "Top-Right", x: 0.85, y: 0.15 },
        { name: "Bottom-Left", x: 0.15, y: 0.85 },
        { name: "Bottom-Right", x: 0.85, y: 0.85 }
      ];
      document.getElementById("banner").innerHTML = "Look at each dot, then press <span class='kbd-key'>SPACEBAR</span> to capture.";
      setTarget(0);
    }

    function startNinePoint() {
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
      document.getElementById("banner").innerHTML = "Look at each dot, then press <span class='kbd-key'>SPACEBAR</span> to capture.";
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
    """Handles HTTP requests for web-based gaze tracking and online adaptation."""

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
        features = payload.get("features", [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
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
