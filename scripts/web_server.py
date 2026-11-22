#!/usr/bin/env python3
"""EyeMouse Multi-User Gaze Training Ground, Live Monitoring Dashboard & Calibrator.

Features:
1. Dual-server architecture: HTTP (8080) + Secure HTTPS (8443).
   - Solves browser "camera not accessed / not secure" restrictions on local network.
2. Standardized 100-Point Target Curriculum for crowdsourced model training.
3. Live Monitoring Dashboard (/dashboard) with real-time active systems telemetry:
   - Live connected systems count & device info.
   - Progress per participant (X / 100 points with visual progress bar).
   - Total points collected across all systems with auto-refreshing stats.
4. Non-inverted, calibrated webcam gaze tracking via MediaPipe.
5. Online RLS adaptation + Bayesian Ridge global model retraining.
6. Anonymized data collection stored in data/training_ground/samples.jsonl.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import socket
import ssl
import sys
import threading
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
CERTS_DIR = PROJECT_ROOT / "certs"
CERT_FILE = CERTS_DIR / "cert.pem"
KEY_FILE = CERTS_DIR / "key.pem"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Global Vision Pipeline instance
pipeline: Optional[VisionPipeline] = None
frame_seq: int = 0
HTTP_PORT: int = 8080
HTTPS_PORT: int = 8443

# Thread-safe in-memory active systems tracking
sessions_lock = threading.Lock()
active_sessions: Dict[str, Dict[str, Any]] = {}


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


def is_target_in_free_area(x: float, y: float) -> bool:
    """Ensure target dot is strictly positioned in the free, unobstructed display area.
    
    Exclusion zones:
    1. Top Header: y < 0.12
    2. Bottom Banner & Controls: y > 0.85
    3. Outer Screen Margins: x < 0.08 or x > 0.92
    4. Top-Left Diagnostics HUD Box: x < 0.25 and y < 0.42 (The user's reported obstruction)
    5. Bottom-Right Webcam PIP Box: x > 0.78 and y > 0.74
    """
    if y < 0.12 or y > 0.85:
        return False
    if x < 0.08 or x > 0.92:
        return False
    if x < 0.25 and y < 0.42:
        return False
    if x > 0.78 and y > 0.74:
        return False
    return True


def generate_standard_100_targets() -> List[Dict[str, Any]]:
    """Generate standardized, reproducible curriculum of 100 calibration points strictly in the free area."""
    targets: List[Dict[str, Any]] = []

    # 1. Primary Anchor Points (strictly placed in free area)
    anchors = [
        ("Center", 0.50, 0.50),
        ("Top-Left-Free", 0.28, 0.14),       # Safely to the right of the HUD
        ("Mid-Left-Free", 0.10, 0.46),       # Safely below the HUD
        ("Top-Center", 0.50, 0.14),
        ("Top-Right", 0.88, 0.14),
        ("Right-Center", 0.88, 0.50),
        ("Bottom-Right-Free", 0.76, 0.72),  # Safely above/left of PIP
        ("Bottom-Center", 0.50, 0.82),
        ("Bottom-Left", 0.10, 0.82),
    ]
    for name, x, y in anchors:
        if is_target_in_free_area(x, y):
            targets.append({"name": name, "x": round(x, 4), "y": round(y, 4)})

    # 2. Quadrant and Multi-Regional Grid Points
    grid_rows = [0.16, 0.26, 0.36, 0.46, 0.56, 0.66, 0.76, 0.82]
    grid_cols = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    existing_coords = {(p["x"], p["y"]) for p in targets}
    for r in grid_rows:
        for c in grid_cols:
            if is_target_in_free_area(c, r) and (c, r) not in existing_coords:
                targets.append({
                    "name": f"Grid-({int(c*100)},{int(r*100)})",
                    "x": round(c, 4),
                    "y": round(r, 4),
                })
                existing_coords.add((c, r))

    # 3. Remaining Points: Deterministic pseudo-random distribution strictly in free area
    rng = random.Random(42)  # Fixed seed guarantees identical sequence for all users
    while len(targets) < 100:
        rx = round(0.08 + rng.random() * 0.84, 4)
        ry = round(0.12 + rng.random() * 0.72, 4)
        if is_target_in_free_area(rx, ry) and (rx, ry) not in existing_coords:
            targets.append({
                "name": f"Curriculum-{len(targets) + 1}",
                "x": rx,
                "y": ry,
            })
            existing_coords.add((rx, ry))

    return targets[:100]


STANDARD_100_TARGETS = generate_standard_100_targets()


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


def parse_device_info(user_agent: str) -> str:
    """Extract human-readable device and browser summary from User-Agent string."""
    ua = user_agent.lower()
    os_name = "Desktop"
    if "macintosh" in ua or "mac os" in ua: os_name = "macOS"
    elif "iphone" in ua: os_name = "iPhone"
    elif "ipad" in ua: os_name = "iPad"
    elif "android" in ua: os_name = "Android"
    elif "windows" in ua: os_name = "Windows"
    elif "linux" in ua: os_name = "Linux"

    browser = "Browser"
    if "edg" in ua: browser = "Edge"
    elif "chrome" in ua and "safari" in ua: browser = "Chrome"
    elif "safari" in ua: browser = "Safari"
    elif "firefox" in ua: browser = "Firefox"

    return f"{os_name} / {browser}"


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>EyeMouse - 100-Point Gaze Training Ground</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 25, 43, 0.92);
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

    /* Security Notice Banner for Non-Secure Contexts */
    #security-warning {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      background: linear-gradient(90deg, #ff4466 0%, #ff8800 100%);
      color: #fff;
      padding: 10px 16px;
      font-size: 13px;
      font-weight: 600;
      z-index: 1000;
      text-align: center;
      box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }
    #security-warning a {
      color: #fff;
      text-decoration: underline;
      font-weight: bold;
      background: rgba(0,0,0,0.25);
      padding: 3px 10px;
      border-radius: 4px;
      margin-left: 8px;
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

    /* Target Dot for 100-Point Training Ground (Always on top of all UI) */
    #calib-target {
      position: absolute;
      width: 60px;
      height: 60px;
      margin-left: -30px;
      margin-top: -30px;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 80;
      cursor: pointer;
      transition: left 0.35s cubic-bezier(0.16, 1, 0.3, 1), top 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .target-ring {
      position: absolute;
      width: 56px;
      height: 56px;
      border: 3px solid var(--accent-yellow);
      border-radius: 50%;
      animation: pulse-ring 1.6s infinite ease-out;
    }
    .target-center {
      width: 20px;
      height: 20px;
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
      width: 44px;
      height: 2px;
      background: rgba(255, 234, 0, 0.5);
    }
    .target-cross-v {
      position: absolute;
      width: 2px;
      height: 44px;
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
      padding: 12px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 50;
      background: linear-gradient(180deg, rgba(9, 13, 22, 0.96) 0%, rgba(9, 13, 22, 0.2) 100%);
      flex-wrap: wrap;
      gap: 10px;
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
    .nav-tab:hover { color: var(--text); }
    .nav-tab.active {
      background: var(--card-bg);
      color: var(--accent-cyan);
      border: 1px solid rgba(0, 240, 255, 0.3);
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    /* Progress Tracker Bar in Header */
    .curriculum-bar-container {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 170px;
    }
    .curriculum-label {
      font-size: 11px;
      font-family: monospace;
      color: var(--accent-yellow);
      display: flex;
      justify-content: space-between;
      font-weight: bold;
    }
    .progress-track {
      width: 100%;
      height: 7px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent-yellow) 0%, var(--accent-green) 100%);
      transition: width 0.3s ease;
      box-shadow: 0 0 8px var(--accent-green);
    }

    /* Dashboard Button Link */
    .dashboard-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(179, 136, 255, 0.15);
      border: 1px solid rgba(179, 136, 255, 0.4);
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: bold;
      color: var(--accent-purple);
      text-decoration: none;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .dashboard-pill:hover {
      background: var(--accent-purple);
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
      z-index: 45;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      min-width: 250px;
      transition: opacity 0.25s ease, transform 0.2s ease, max-height 0.25s ease;
    }
    #hud.hud-ghost {
      opacity: 0.12 !important;
      pointer-events: none !important;
      filter: blur(1px);
    }
    #hud.hud-minimized {
      min-width: auto;
      width: auto;
      padding: 6px 12px;
      background: rgba(18, 25, 43, 0.7);
    }
    #hud.hud-minimized .hud-content {
      display: none;
    }
    .hud-toggle-btn {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border);
      color: var(--text-dim);
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 10px;
      cursor: pointer;
      font-family: inherit;
    }
    .hud-toggle-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.2);
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
      text-decoration: none;
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
    .btn-green:hover { background: var(--accent-green); color: #000; }
    .btn-purple {
      background: rgba(179, 136, 255, 0.15);
      color: var(--accent-purple);
      border-color: rgba(179, 136, 255, 0.4);
    }
    .btn-purple:hover { background: var(--accent-purple); color: #000; }

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
  </style>
</head>
<body>

  <!-- Insecure Context Warning (when loaded over non-SSL HTTP on local IP) -->
  <div id="security-warning">
    ⚠️ <b>Camera Access Blocked by Browser:</b> You are viewing via insecure HTTP.
    <a id="secure-redirect-link" href="#">Switch to Secure HTTPS (Port 8443)</a>
    <span>to grant camera permission!</span>
  </div>

  <!-- Fullscreen Drawing Canvas -->
  <canvas id="canvas-layer"></canvas>

  <!-- Live Predicted Gaze Cursor -->
  <div id="gaze-cursor" style="left: 50%; top: 50%;">
    <div class="gaze-cross-h"></div>
    <div class="gaze-cross-v"></div>
    <div class="gaze-tag">YOUR GAZE</div>
  </div>

  <!-- 100-Point Calibration / Training Target -->
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
        <span style="font-size: 11px; color: var(--accent-cyan); margin-left: 6px;">100-Point Training Ground</span>
      </div>
    </div>

    <!-- Curriculum Progress Tracker -->
    <div class="curriculum-bar-container">
      <div class="curriculum-label">
        <span id="curriculum-step-text">Point 1 of 100</span>
        <span id="curriculum-pct-text">0%</span>
      </div>
      <div class="progress-track">
        <div id="curriculum-fill" class="progress-fill"></div>
      </div>
    </div>

    <div style="display: flex; gap: 8px; align-items: center;">
      <!-- Toggle Telemetry HUD -->
      <button id="btn-toggle-hud" class="btn" onclick="toggleHud()" title="Show/Hide Telemetry HUD (Hotkey: H)">👁 HUD: Visible</button>

      <!-- Direct Link to Live Monitoring Dashboard -->
      <a href="/dashboard" class="dashboard-pill" title="Open Multi-System Monitoring Dashboard">
        <span>📊 Live Dashboard</span>
      </a>

      <!-- User Identification -->
      <div class="user-chip">
        <span>👤 User:</span>
        <input id="user-name" class="user-input" value="User-1" title="Tag your calibration samples" onchange="saveUserName()">
      </div>

      <button class="btn" onclick="toggleFullscreen()" title="Fullscreen">⛶</button>
    </div>
  </header>

  <!-- Diagnostics Telemetry HUD (Collapsible & Auto-Ghosting) -->
  <div id="hud">
    <div class="hud-title">
      <div style="display: flex; align-items: center; gap: 6px;">
        <span>Diagnostics Telemetry</span>
        <span id="hud-mode-tag" style="color: var(--accent-green); font-size: 10px;">100-PT</span>
      </div>
      <button id="hud-min-btn" class="hud-toggle-btn" onclick="toggleHud(event)">— Minimize</button>
    </div>
    <div id="hud-content" class="hud-content">
      <div>Tracking: <span id="hud-face" class="hud-val">Detecting...</span></div>
      <div>Inference: <span id="hud-latency" class="hud-val">0.0 ms</span></div>
      <div>FPS: <span id="hud-fps" class="hud-val">0</span></div>
      <div>Gaze (u, v): <span id="hud-gaze" class="hud-val">0.50, 0.50</span></div>
      <div>Screen (px): <span id="hud-px" class="hud-val">0, 0</span></div>
      <div>Iris UV: <span id="hud-iris" class="hud-val">L: 0.50, 0.00 | R: 0.50, 0.00</span></div>
      <div>Head Pose: <span id="hud-pose" class="hud-val">0°, 0°, 0°</span></div>
      <div>Target Dist: <span id="hud-dist" class="hud-val">0 px</span></div>
      <div>Your Progress: <span id="hud-user-pts" class="hud-val">0 / 100</span></div>
      <div>Global Dataset: <span id="hud-dataset-pts" class="hud-val">0 samples</span></div>
      <div>Active Systems: <span id="hud-systems-online" class="hud-val">1 online</span></div>
    </div>
  </div>

  <!-- Instruction Banner -->
  <div id="banner">
    <span id="banner-text">Look directly at the yellow dot, then press <span class="kbd-key">SPACEBAR</span> (or tap dot) to capture. Gaze coordinates adapt online in real-time!</span>
  </div>

  <!-- Bottom Controls -->
  <div id="controls">
    <button class="btn btn-primary" onclick="capturePoint()">Capture Target (Space)</button>
    <a href="/dashboard" class="btn btn-purple">📊 Open Monitoring Dashboard</a>
    <button class="btn" onclick="exportDataset()">💾 Export Dataset (JSONL)</button>
    <button class="btn" onclick="resetCurriculum()">Restart 100 Points</button>
    <button class="btn" onclick="resetAdaptation()">Reset to Base Prior</button>
  </div>

  <!-- Webcam PIP View -->
  <div id="pip-container">
    <div id="pip-badge">
      <div class="badge-dot"></div>
      <span id="camera-status">WEBCAM LIVE</span>
    </div>
    <video id="webcam-video" autoplay playsinline muted></video>
    <canvas id="pip-canvas" width="220" height="165"></canvas>
  </div>

  <canvas id="hidden-canvas" style="display: none;" width="640" height="480"></canvas>

  <script>
    // State
    const targets100 = """ + json.dumps(STANDARD_100_TARGETS) + """;
    let currentTargetIdx = 0;
    let completedPointsCount = 0;
    let sessionId = "tg_" + Date.now().toString(36) + "_" + Math.random().toString(36).substring(2, 6);
    let userName = localStorage.getItem("eyemouse_user") || ("User-" + Math.floor(100 + Math.random()*900));
    document.getElementById("user-name").value = userName;

    let latestFeatures = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0];
    let latestEyeMetrics = null;
    let latestGaze = { u: 0.5, v: 0.5, valid: false };
    let latestConfidence = 0.0;
    let isProcessing = false;
    let fpsFrames = 0;
    let fpsStart = performance.now();

    const video = document.getElementById("webcam-video");
    const hiddenCanvas = document.getElementById("hidden-canvas");
    const hctx = hiddenCanvas.getContext("2d");
    const drawCanvas = document.getElementById("canvas-layer");
    const dctx = drawCanvas.getContext("2d");
    const pipCanvas = document.getElementById("pip-canvas");
    const pctx = pipCanvas.getContext("2d");
    const targetEl = document.getElementById("calib-target");
    const gazeCursor = document.getElementById("gaze-cursor");

    // Audio Feedback Generator
    let audioCtx = null;
    function playBeep(freq = 920, duration = 0.07) {
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

    // Check Security Context for Camera
    function checkSecurityContext() {
      const isSecure = window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1";
      if (!isSecure) {
        const warn = document.getElementById("security-warning");
        const link = document.getElementById("secure-redirect-link");
        const secureUrl = "https://" + location.hostname + ":""" + str(HTTPS_PORT) + """" + location.pathname;
        link.href = secureUrl;
        warn.style.display = "block";
      }
    }
    checkSecurityContext();

    // Camera Init
    async function initCamera() {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("getUserMedia not supported on insecure HTTP. Please use HTTPS.");
        }
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
        const statusEl = document.getElementById("camera-status");
        statusEl.innerText = "CAMERA BLOCKED / INSECURE";
        statusEl.style.color = "#ff4444";
        document.getElementById("banner-text").innerHTML =
          `⚠️ <b>Camera Permission Blocked:</b> Please access via <b><a href="https://${location.hostname}:""" + str(HTTPS_PORT) + """" style="color: #00f0ff; text-decoration: underline;">Secure HTTPS</a></b> to allow camera permissions.`;
      }
    }

    function setTarget(idx) {
      if (idx >= targets100.length) {
        targetEl.style.display = "none";
        document.getElementById("banner-text").innerHTML =
          `🎉 <b>Incredible Job! All 100 Points Completed!</b> Your data is logged and the model is adapted. View the <a href="/dashboard" style="color: #00f0ff;">Live Dashboard</a> to see all systems.`;
        updateProgressUI(100);
        return;
      }
      targetEl.style.display = "flex";
      currentTargetIdx = idx;
      const t = targets100[idx];
      targetEl.style.left = (t.x * 100) + "%";
      targetEl.style.top = (t.y * 100) + "%";
      targetEl.dataset.targetX = t.x;
      targetEl.dataset.targetY = t.y;
      targetEl.dataset.targetName = t.name;

      // Auto-ghost HUD if target dot is near the top-left area so it NEVER obstructs view
      const hudEl = document.getElementById("hud");
      if (hudEl) {
        if (t.x < 0.36 && t.y < 0.48) {
          hudEl.classList.add("hud-ghost");
        } else {
          hudEl.classList.remove("hud-ghost");
        }
      }

      document.getElementById("banner-text").innerHTML =
        `Point <b>${idx + 1} of 100</b> (${t.name}): Look at the yellow dot and press <span class="kbd-key">SPACEBAR</span>.`;

      updateProgressUI(idx);
    }

    function updateProgressUI(idx) {
      const pct = Math.round((idx / 100) * 100);
      document.getElementById("curriculum-step-text").innerText = `Point ${idx} of 100`;
      document.getElementById("curriculum-pct-text").innerText = `${pct}%`;
      document.getElementById("curriculum-fill").style.width = `${pct}%`;
      document.getElementById("hud-user-pts").innerText = `${idx} / 100`;
    }

    // MAIN VISION PROCESSING LOOP
    async function processLoop() {
      if (!isProcessing && video.readyState >= 2) {
        isProcessing = true;
        const t0 = performance.now();

        // Horizontal Mirroring: Match user selfie display
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

            // Live laser gaze cursor
            gazeCursor.style.display = "block";
            gazeCursor.style.left = screenX + "px";
            gazeCursor.style.top = screenY + "px";

            // Real-time distance vector
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

    // CAPTURE TARGET POINT
    async function capturePoint() {
      if (targetEl.style.display === "none") return;

      const tx = parseFloat(targetEl.dataset.targetX || "0.5");
      const ty = parseFloat(targetEl.dataset.targetY || "0.5");
      const tName = targetEl.dataset.targetName || `Point-${currentTargetIdx + 1}`;

      playBeep(940, 0.08);

      targetEl.classList.add("confirmed-pulse");
      const center = targetEl.querySelector(".target-center");
      if (center) center.style.background = "var(--accent-green)";

      const samplePayload = {
        session_id: sessionId,
        user_name: userName,
        timestamp_ns: Date.now() * 1000000,
        screen_w: window.innerWidth,
        screen_h: window.innerHeight,
        target: {
          name: tName,
          index: currentTargetIdx + 1,
          total_curriculum: 100,
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
        completedPointsCount++;
        if (resp.total_samples) {
          document.getElementById("hud-dataset-pts").innerText = `${resp.total_samples} samples`;
        }
      } catch (err) {
        console.error("Failed recording sample:", err);
      }

      setTimeout(() => {
        if (center) center.style.background = "var(--accent-yellow)";
        targetEl.classList.remove("confirmed-pulse");
        setTarget(currentTargetIdx + 1);
      }, 350);
    }

    // Spacebar capture & H key to toggle HUD
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" && e.target.tagName !== "INPUT") {
        e.preventDefault();
        capturePoint();
      } else if ((e.key === "h" || e.key === "H") && e.target.tagName !== "INPUT") {
        toggleHud();
      }
    });

    let isHudMinimized = false;
    function toggleHud(e) {
      if (e) e.stopPropagation();
      isHudMinimized = !isHudMinimized;
      const hud = document.getElementById("hud");
      const minBtn = document.getElementById("hud-min-btn");
      const topBtn = document.getElementById("btn-toggle-hud");
      if (isHudMinimized) {
        hud.classList.add("hud-minimized");
        if (minBtn) minBtn.innerText = "+ Expand";
        if (topBtn) topBtn.innerText = "👁 HUD: Hidden";
      } else {
        hud.classList.remove("hud-minimized");
        if (minBtn) minBtn.innerText = "— Minimize";
        if (topBtn) topBtn.innerText = "👁 HUD: Visible";
      }
    }

    // Client Heartbeat: Sends live progress & system status every 3 seconds
    async function sendHeartbeat() {
      try {
        const res = await fetch("/api/heartbeat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            user_name: userName,
            points_completed: completedPointsCount,
            screen_width: window.innerWidth,
            screen_height: window.innerHeight,
            user_agent: navigator.userAgent
          })
        });
        const data = await res.json();
        if (data.online_count !== undefined) {
          document.getElementById("hud-systems-online").innerText = `${data.online_count} online`;
        }
        if (data.total_samples !== undefined) {
          document.getElementById("hud-dataset-pts").innerText = `${data.total_samples} samples`;
        }
      } catch (e) {}
    }
    setInterval(sendHeartbeat, 3000);
    sendHeartbeat();

    function resetCurriculum() {
      if (confirm("Restart 100-point curriculum from Point 1?")) {
        completedPointsCount = 0;
        setTarget(0);
      }
    }

    async function resetAdaptation() {
      try {
        await fetch("/api/reset_adaptation", { method: "POST" });
        alert("Gaze model reset to physiological base prior.");
      } catch (e) {}
    }

    function exportDataset() {
      window.location.href = "/api/training_ground/export";
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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EyeMouse - Live Multi-System Monitoring Dashboard</title>
  <style>
    :root {
      --bg: #070a12;
      --card-bg: rgba(16, 22, 38, 0.94);
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
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 24px;
      min-height: 100vh;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-icon {
      width: 38px;
      height: 38px;
      background: var(--accent-purple);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #000;
      font-weight: 900;
      font-size: 20px;
    }
    .brand-title {
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }

    /* Top Stats Metric Grid */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      position: relative;
      overflow: hidden;
    }
    .metric-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background: var(--accent-purple);
    }
    .card-green::before { background: var(--accent-green); }
    .card-cyan::before { background: var(--accent-cyan); }
    .card-yellow::before { background: var(--accent-yellow); }
    
    .metric-val {
      font-size: 32px;
      font-weight: 800;
      font-family: monospace;
      color: #fff;
      margin-top: 6px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .metric-lbl {
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 1px;
      font-weight: 700;
    }

    /* Live Pulse Dot */
    .pulse-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent-green);
      box-shadow: 0 0 10px var(--accent-green);
      animation: live-pulse 1.8s infinite;
    }
    @keyframes live-pulse {
      0% { transform: scale(0.9); opacity: 1; }
      50% { transform: scale(1.3); opacity: 0.5; }
      100% { transform: scale(0.9); opacity: 1; }
    }

    /* Network URL Share Box */
    .share-box {
      background: rgba(0, 240, 255, 0.06);
      border: 1px solid rgba(0, 240, 255, 0.3);
      border-radius: 14px;
      padding: 16px 20px;
      margin-bottom: 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 14px;
    }
    .share-info {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .share-title {
      font-size: 14px;
      font-weight: 700;
      color: var(--accent-cyan);
    }
    .share-desc {
      font-size: 12px;
      color: var(--text-dim);
    }
    .share-links-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .url-chip {
      background: #101626;
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 12px;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .copy-btn {
      background: rgba(0, 240, 255, 0.15);
      border: 1px solid var(--accent-cyan);
      color: var(--accent-cyan);
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: bold;
      cursor: pointer;
    }
    .copy-btn:hover { background: var(--accent-cyan); color: #000; }

    /* Live Systems Table */
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .section-title {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }

    .table-container {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      margin-bottom: 28px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 14px 18px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    th {
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-dim);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 700;
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }

    /* Progress Bar */
    .table-progress-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 180px;
    }
    .table-track {
      flex: 1;
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      overflow: hidden;
    }
    .table-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-yellow) 0%, var(--accent-green) 100%);
      border-radius: 4px;
      box-shadow: 0 0 6px var(--accent-green);
    }
    .table-pct {
      font-family: monospace;
      font-size: 12px;
      font-weight: bold;
      width: 45px;
      color: var(--accent-yellow);
    }

    /* Status Pills */
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: bold;
      font-family: monospace;
    }
    .status-online {
      background: rgba(0, 255, 136, 0.15);
      color: var(--accent-green);
      border: 1px solid rgba(0, 255, 136, 0.4);
    }
    .status-idle {
      background: rgba(255, 234, 0, 0.15);
      color: var(--accent-yellow);
      border: 1px solid rgba(255, 234, 0, 0.4);
    }
    .status-offline {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-dim);
      border: 1px solid var(--border);
    }

    /* Buttons */
    .btn {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 9px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
      text-decoration: none;
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
    .btn-purple {
      background: rgba(179, 136, 255, 0.2);
      color: var(--accent-purple);
      border-color: rgba(179, 136, 255, 0.5);
    }
    .btn-purple:hover {
      background: var(--accent-purple);
      color: #000;
    }

    /* Modal dialog */
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
      max-width: 580px;
      width: 90vw;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">📊</div>
      <div>
        <div class="brand-title">EyeMouse Live Telemetry Dashboard</div>
        <div style="font-size: 12px; color: var(--text-dim);">Multi-system gaze training & crowdsourced dataset monitor</div>
      </div>
    </div>
    <div style="display: flex; gap: 10px;">
      <a href="/" class="btn btn-primary">👁 Open Gaze Trainer</a>
      <button class="btn btn-purple" onclick="trainGlobalModel()">🧠 Train Global Model</button>
      <a href="/api/training_ground/export" class="btn">💾 Download Dataset</a>
    </div>
  </header>

  <!-- Metric Counters Grid -->
  <div class="metric-grid">
    <div class="metric-card card-green">
      <div class="metric-lbl">Active Systems Online</div>
      <div class="metric-val">
        <span class="pulse-dot"></span>
        <span id="stat-online-systems">0</span>
      </div>
    </div>
    <div class="metric-card card-yellow">
      <div class="metric-lbl">Total Points Collected</div>
      <div class="metric-val" id="stat-total-points">0</div>
    </div>
    <div class="metric-card card-cyan">
      <div class="metric-lbl">Total Participants</div>
      <div class="metric-val" id="stat-unique-users">0</div>
    </div>
    <div class="metric-card">
      <div class="metric-lbl">Curriculum Progress (Avg)</div>
      <div class="metric-val" id="stat-avg-progress">0%</div>
    </div>
  </div>

  <!-- Share & Join Links Card -->
  <div class="share-box">
    <div class="share-info">
      <div class="share-title">🌐 Shareable Training Ground Links for Other Devices</div>
      <div class="share-desc">Share the Secure HTTPS link with anyone on your local Wi-Fi to collect their gaze data:</div>
    </div>
    <div class="share-links-row">
      <div class="url-chip">
        <span>🔒 Secure Training Link: <b id="share-https-url">Loading...</b></span>
        <button class="copy-btn" onclick="copyText('share-https-url')">Copy</button>
      </div>
      <div class="url-chip">
        <span>📊 Dashboard Link: <b id="share-dash-url">Loading...</b></span>
        <button class="copy-btn" onclick="copyText('share-dash-url')">Copy</button>
      </div>
    </div>
  </div>

  <!-- Live Connected Systems Table -->
  <div class="section-header">
    <div class="section-title">Connected Systems & Calibration Progress (out of 100)</div>
    <div style="font-size: 12px; color: var(--text-dim);">Auto-refreshing every 2s</div>
  </div>

  <div class="table-container">
    <table>
      <thead>
        <tr>
          <th>Participant / System</th>
          <th>Client IP</th>
          <th>Device / Browser</th>
          <th>Screen Res</th>
          <th>Status</th>
          <th>100-Point Progress</th>
          <th>Last Active</th>
        </tr>
      </thead>
      <tbody id="systems-table-body">
        <tr>
          <td colspan="7" style="text-align: center; color: var(--text-dim); padding: 30px;">
            Connecting to active systems telemetry...
          </td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Retraining Report Modal -->
  <div id="train-modal" class="modal-overlay" onclick="closeTrainModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <h3 style="color: var(--accent-cyan);">🧠 Global Model Retraining Report</h3>
        <button class="btn" onclick="closeTrainModal()">&times;</button>
      </div>
      <div id="train-modal-body" style="font-size: 13.5px; line-height: 1.6; color: var(--text);">
        Retraining in progress...
      </div>
      <div style="margin-top: 18px; display: flex; justify-content: flex-end;">
        <button class="btn btn-primary" onclick="closeTrainModal()">Done</button>
      </div>
    </div>
  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch("/api/dashboard/stats");
        const data = await res.json();

        // Update top metrics
        document.getElementById("stat-online-systems").innerText = data.online_count || 0;
        document.getElementById("stat-total-points").innerText = data.total_samples || 0;
        document.getElementById("stat-unique-users").innerText = data.unique_participants || 0;

        // Share URLs
        const httpsUrl = data.share_url_https || window.location.origin;
        document.getElementById("share-https-url").innerText = httpsUrl;
        document.getElementById("share-dash-url").innerText = httpsUrl + "/dashboard";

        // Render Systems Table
        const tbody = document.getElementById("systems-table-body");
        const systems = data.systems || [];

        if (systems.length === 0) {
          tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 30px;">No connected systems detected yet. Share the training link above to begin!</td></tr>`;
          document.getElementById("stat-avg-progress").innerText = "0%";
          return;
        }

        let totalProgressPct = 0;
        let html = "";
        systems.forEach(s => {
          totalProgressPct += s.progress_pct;
          const statusClass = s.status === "ONLINE" ? "status-online" : (s.status === "IDLE" ? "status-idle" : "status-offline");
          const statusIcon = s.status === "ONLINE" ? "●" : "○";

          html += `
            <tr>
              <td><b>${s.user_name}</b></td>
              <td style="font-family: monospace;">${s.client_ip}</td>
              <td>${s.device}</td>
              <td style="font-family: monospace;">${s.screen}</td>
              <td><span class="status-pill ${statusClass}">${statusIcon} ${s.status}</span></td>
              <td>
                <div class="table-progress-wrap">
                  <div class="table-track">
                    <div class="table-fill" style="width: ${s.progress_pct}%"></div>
                  </div>
                  <span class="table-pct">${s.points_completed}/100</span>
                </div>
              </td>
              <td style="color: var(--text-dim); font-size: 12px;">${s.last_active_str}</td>
            </tr>
          `;
        });
        tbody.innerHTML = html;

        const avg = Math.round(totalProgressPct / systems.length);
        document.getElementById("stat-avg-progress").innerText = avg + "%";
      } catch (err) {
        console.error("Dashboard update error:", err);
      }
    }

    function copyText(elemId) {
      const text = document.getElementById(elemId).innerText;
      navigator.clipboard.writeText(text).then(() => {
        alert("Copied link to clipboard:\n" + text);
      }).catch(() => {
        prompt("Copy this link:", text);
      });
    }

    async function trainGlobalModel() {
      document.getElementById("train-modal").style.display = "flex";
      document.getElementById("train-modal-body").innerHTML = "Running Bayesian Ridge regression across all collected samples...";
      try {
        const res = await fetch("/api/training_ground/train_global_model", { method: "POST" });
        const data = await res.json();
        if (!data.success) {
          document.getElementById("train-modal-body").innerHTML = `<span style="color: var(--accent-red); font-weight: bold;">Training Notice:</span> ${data.error}`;
          return;
        }
        document.getElementById("train-modal-body").innerHTML = `
          <div style="background: rgba(0, 255, 136, 0.1); border: 1px solid var(--accent-green); padding: 12px; border-radius: 8px; margin-bottom: 12px;">
            ✅ <b>Model Retrained Successfully!</b> Live vision engine updated with new weights.
          </div>
          <p><b>Total Samples Used:</b> ${data.sample_count} points across ${data.unique_participants} participants</p>
          <p><b>Prior Pixel Error:</b> ${data.prior_pixel_error_1080p} px</p>
          <p><b>Trained Pixel Error:</b> <span style="color: var(--accent-green); font-weight: bold;">${data.fitted_pixel_error_1080p} px</span></p>
          <p><b>Improvement:</b> ${data.improvement_percent}% error reduction</p>
        `;
      } catch (err) {
        document.getElementById("train-modal-body").innerText = "Request failed: " + err;
      }
    }

    function closeTrainModal(e) {
      document.getElementById("train-modal").style.display = "none";
    }

    setInterval(updateDashboard, 2000);
    updateDashboard();
  </script>
</body>
</html>
"""


class EyeMouseHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP & HTTPS requests for gaze calibration, 100-point curriculum, and live dashboard."""

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))

        elif self.path == "/dashboard" or self.path.startswith("/dashboard?"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif self.path == "/api/info":
            lan_ip = get_lan_ip()
            data = {
                "status": "RUNNING",
                "lan_ip": lan_ip,
                "http_port": HTTP_PORT,
                "https_port": HTTPS_PORT,
                "http_url": f"http://{lan_ip}:{HTTP_PORT}",
                "https_url": f"https://{lan_ip}:{HTTPS_PORT}",
                "model": "AdaptiveGazeModel",
                "curriculum_total_points": 100,
            }
            self._send_json(data)

        elif self.path == "/api/dashboard/stats":
            self._handle_dashboard_stats()

        elif self.path == "/api/training_ground/export":
            if not DATASET_FILE.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Dataset is empty.")
                return
            try:
                with open(DATASET_FILE, "rb") as f:
                    content = f.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/jsonl")
                self.send_header("Content-Disposition", 'attachment; filename="eyemouse_100pt_dataset.jsonl"')
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
        elif self.path == "/api/heartbeat":
            self._handle_heartbeat(payload)
        elif self.path == "/api/training_ground/record_sample":
            self._handle_record_sample(payload)
        elif self.path == "/api/training_ground/train_global_model":
            self._handle_train_global_model()
        elif self.path == "/api/reset_adaptation":
            self._handle_reset_adaptation()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _handle_heartbeat(self, payload: Dict[str, Any]) -> None:
        """Records active client heartbeat and returns live telemetry."""
        session_id = payload.get("session_id", "anon")
        user_name = payload.get("user_name", "Anonymous")
        points = int(payload.get("points_completed", 0))
        sw = payload.get("screen_width", 1920)
        sh = payload.get("screen_height", 1080)
        ua = payload.get("user_agent", self.headers.get("User-Agent", ""))
        client_ip = self.client_address[0]

        now = time.time()
        with sessions_lock:
            existing = active_sessions.get(session_id, {})
            first_seen = existing.get("first_seen", now)
            active_sessions[session_id] = {
                "session_id": session_id,
                "user_name": user_name,
                "client_ip": client_ip,
                "points_completed": min(100, max(0, points)),
                "screen": f"{sw}x{sh}",
                "device": parse_device_info(ua),
                "first_seen": first_seen,
                "last_seen": now,
            }

        # Count online systems
        online_count = sum(
            1 for s in active_sessions.values() if (now - s["last_seen"]) < 12
        )
        ds = get_dataset_stats()

        self._send_json({
            "status": "HEARTBEAT_ACK",
            "online_count": online_count,
            "total_samples": ds["total_samples"],
        })

    def _handle_dashboard_stats(self) -> None:
        """Returns structured JSON for the live monitoring dashboard."""
        now = time.time()
        lan_ip = get_lan_ip()
        systems_list = []

        with sessions_lock:
            for s_id, s in active_sessions.items():
                elapsed = now - s["last_seen"]
                if elapsed < 12:
                    status = "ONLINE"
                    last_str = "Just now"
                elif elapsed < 60:
                    status = "IDLE"
                    last_str = f"{int(elapsed)}s ago"
                else:
                    status = "OFFLINE"
                    last_str = f"{int(elapsed // 60)}m ago"

                pct = int(round((s["points_completed"] / 100.0) * 100))
                systems_list.append({
                    "session_id": s_id,
                    "user_name": s["user_name"],
                    "client_ip": s["client_ip"],
                    "device": s["device"],
                    "screen": s["screen"],
                    "points_completed": s["points_completed"],
                    "progress_pct": pct,
                    "status": status,
                    "last_active_str": last_str,
                    "last_seen_epoch": s["last_seen"],
                })

        # Sort: ONLINE first, then most recently active
        status_rank = {"ONLINE": 0, "IDLE": 1, "OFFLINE": 2}
        systems_list.sort(key=lambda x: (status_rank.get(x["status"], 3), -x["last_seen_epoch"]))

        online_count = sum(1 for s in systems_list if s["status"] == "ONLINE")
        ds = get_dataset_stats()

        data = {
            "online_count": online_count,
            "total_samples": ds["total_samples"],
            "unique_participants": ds["participant_count"],
            "share_url_https": f"https://{lan_ip}:{HTTPS_PORT}",
            "share_url_http": f"http://{lan_ip}:{HTTP_PORT}",
            "systems": systems_list,
        }
        self._send_json(data)

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
        """Appends verified gaze calibration sample to JSONL dataset and adapts online."""
        p = get_pipeline()

        features = payload.get("features", [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        tgt = payload.get("target", {})
        tx = float(tgt.get("norm_x", 0.5))
        ty = float(tgt.get("norm_y", 0.5))

        # 1. Immediate Online RLS Adaptation
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

        # Update session points count
        session_id = payload.get("session_id")
        if session_id:
            with sessions_lock:
                if session_id in active_sessions:
                    active_sessions[session_id]["points_completed"] += 1
                    active_sessions[session_id]["last_seen"] = time.time()

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
            log.info("Global model retrained successfully! %d samples, %.1f px error.",
                     res["sample_count"], res["fitted_pixel_error_1080p"])

        self._send_json(res)

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


def run_dual_server(http_port: int = 8080, https_port: int = 8443) -> None:
    global HTTP_PORT, HTTPS_PORT
    HTTP_PORT = http_port
    HTTPS_PORT = https_port
    lan_ip = get_lan_ip()

    # 1. Setup HTTP Server
    httpd = HTTPServer(("0.0.0.0", http_port), EyeMouseHTTPRequestHandler)

    # 2. Setup HTTPS Server (using local SAN SSL certificate)
    httpsd = None
    if CERT_FILE.exists() and KEY_FILE.exists():
        try:
            httpsd = HTTPServer(("0.0.0.0", https_port), EyeMouseHTTPRequestHandler)
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))
            httpsd.socket = ssl_ctx.wrap_socket(httpsd.socket, server_side=True)
            log.info("Secure HTTPS listener initialized on port %d with SSL", https_port)
        except Exception as exc:
            log.warning("Could not start HTTPS listener: %s", exc)
            httpsd = None

    banner = f"""
========================================================================================
👁  EyeMouse 100-Point Gaze Training Ground & Live Dashboard LIVE
========================================================================================
📍 Localhost (HTTP):         http://localhost:{http_port}
🔒 Secure Network (HTTPS):    https://{lan_ip}:{https_port} (Recommended for all network devices)
🌐 Standard Network (HTTP):   http://{lan_ip}:{http_port}
📊 Live Telemetry Dashboard:  https://{lan_ip}:{https_port}/dashboard

Notice for other Wi-Fi devices (iPhone, Android, other laptops):
1. Open the Secure Link: https://{lan_ip}:{https_port}
2. Click 'Advanced' -> 'Proceed to {lan_ip}' once to allow the camera certificate.
3. The browser will grant camera permissions and start the 100-point calibration!
========================================================================================
"""
    print(banner, flush=True)

    # Launch HTTPS server in background thread if configured
    if httpsd is not None:
        https_thread = threading.Thread(target=httpsd.serve_forever, daemon=True)
        https_thread.start()

    # Run HTTP server on main thread
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down servers.")
        httpd.server_close()
        if httpsd:
            httpsd.server_close()


if __name__ == "__main__":
    hp = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    sp = int(sys.argv[2]) if len(sys.argv) > 2 else 8443
    run_dual_server(hp, sp)
