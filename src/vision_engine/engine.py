"""Headless vision engine daemon integrating shared-memory frame processing and IPC."""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional, Dict, Any

from .pipeline import VisionPipeline
from .ipc.shm_consumer import ShmConsumer
from .ipc.control_server import ControlServer
from .types import VisionOutput

log = logging.getLogger("vision_engine")


class VisionEngine:
    """Headless vision processing server daemon."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self.control_server = ControlServer(socket_path)
        self.pipeline = VisionPipeline()
        self.shm_consumer: Optional[ShmConsumer] = None
        self._running = False

    def start(self) -> None:
        """Start control server and enter the event loop."""
        self.control_server.start()
        self._running = True

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        log.info("VisionEngine ready. Waiting for host connection...")
        if not self.control_server.accept_client(timeout=None):
            log.warning("Control server stopped without client connection.")
            return

        self._run_loop()

    def _handle_signal(self, signum: int, frame: Any) -> None:
        log.info("Received termination signal %d. Shutting down...", signum)
        self._running = False

    def _run_loop(self) -> None:
        """Process incoming control commands and frames."""
        # 1. Wait for HANDSHAKE
        handshake_msg = self.control_server.read_message(timeout=10.0)
        if not handshake_msg or handshake_msg.get("cmd") != "HANDSHAKE":
            log.error("Missing or invalid HANDSHAKE message: %s", handshake_msg)
            self.control_server.send_message({"status": "ERROR", "message": "Expected HANDSHAKE"})
            self.stop()
            return

        shm_name = handshake_msg.get("shm_name")
        if not shm_name:
            log.error("HANDSHAKE missing 'shm_name'")
            self.control_server.send_message({"status": "ERROR", "message": "Missing shm_name"})
            self.stop()
            return

        # 2. Attach to shared memory
        try:
            self.shm_consumer = ShmConsumer(shm_name)
            log.info("Attached to shared memory: %s", shm_name)
        except Exception as exc:
            log.error("Failed to attach to shared memory %s: %s", shm_name, exc)
            self.control_server.send_message({"status": "ERROR", "message": f"SHM attach failed: {exc}"})
            self.stop()
            return

        # 3. Send READY acknowledgement to host
        self.control_server.send_message({"status": "READY", "version": 1})

        # 4. Main message pump
        while self._running:
            msg = self.control_server.read_message(timeout=1.0)
            if msg is None:
                if not self._running:
                    break
                # Read timeout or client disconnect check
                continue

            cmd = msg.get("cmd")
            if cmd == "FRAME":
                self._handle_frame(msg)
            elif cmd == "LOAD_CALIBRATION":
                profile_path = msg.get("profile_path")
                success = self.pipeline.load_calibration_profile(profile_path)
                self.control_server.send_message({
                    "status": "CALIBRATION_LOADED" if success else "CALIBRATION_ERROR",
                    "is_calibrated": self.pipeline.gaze_estimator.is_calibrated,
                })
            elif cmd == "SHUTDOWN":
                log.info("Received SHUTDOWN request from host.")
                self.control_server.send_message({"status": "SHUTDOWN_ACK"})
                break
            else:
                log.warning("Unknown command received: %s", cmd)

        self.stop()

    def _handle_frame(self, msg: Dict[str, Any]) -> None:
        """Process a published frame notification from shared memory."""
        slot = msg.get("slot")
        seq = msg.get("seq", 0)

        if slot is None or self.shm_consumer is None:
            self.control_server.send_message({"status": "ERROR", "message": "Invalid slot or SHM not attached"})
            return

        try:
            # Zero-copy access from shared memory
            meta, raw_frame = self.shm_consumer.acquire_slot_for_reading(slot)

            # Execute vision pipeline
            output: VisionOutput = self.pipeline.process_raw_frame(
                pixel_data=raw_frame,
                width=meta.width,
                height=meta.height,
                stride=meta.stride,
                pixel_format=meta.pixel_format,
                sequence_num=meta.sequence_num,
                timestamp_ns=meta.timestamp_ns,
            )

            # Release slot immediately so producer can reuse it
            self.shm_consumer.release_slot(slot)

            # Format lightweight response
            response: Dict[str, Any] = {
                "status": "ACK",
                "slot": slot,
                "seq": meta.sequence_num,
                "face_detected": output.face_detected,
                "confidence": round(output.confidence, 3),
                "processing_time_ms": round(output.processing_time_ms, 2),
            }

            if output.gaze_estimate:
                response["gaze"] = {
                    "raw_x": round(output.gaze_estimate.raw_x, 4),
                    "raw_y": round(output.gaze_estimate.raw_y, 4),
                    "filtered_x": round(output.gaze_estimate.filtered_x, 4),
                    "filtered_y": round(output.gaze_estimate.filtered_y, 4),
                    "is_valid": output.gaze_estimate.is_valid,
                    "is_calibrated": self.pipeline.gaze_estimator.is_calibrated,
                }

            if output.eye_metrics:
                l_norm = output.eye_metrics.left_normalized
                r_norm = output.eye_metrics.right_normalized
                response["eye_features"] = {
                    "left_x": round(l_norm.iris_x, 4),
                    "left_y": round(l_norm.iris_y, 4),
                    "right_x": round(r_norm.iris_x, 4),
                    "right_y": round(r_norm.iris_y, 4),
                    "left_openness": round(l_norm.openness, 4),
                    "right_openness": round(r_norm.openness, 4),
                }

            if output.blink_event:
                response["blink"] = {
                    "phase": output.blink_event.phase.value,
                    "left": round(output.blink_event.left_blink_score, 3),
                    "right": round(output.blink_event.right_blink_score, 3),
                    "duration_ms": round(output.blink_event.closure_duration_ms, 1),
                    "is_sustained": output.blink_event.is_intentional_sustained,
                }

            if output.head_pose:
                response["head_pose"] = {
                    "yaw": round(output.head_pose.yaw, 2),
                    "pitch": round(output.head_pose.pitch, 2),
                    "roll": round(output.head_pose.roll, 2),
                }

            self.control_server.send_message(response)

        except Exception as exc:
            log.exception("Error processing frame in slot %s: %s", slot, exc)
            if self.shm_consumer is not None:
                try:
                    self.shm_consumer.release_slot(slot)
                except Exception:
                    pass
            self.control_server.send_message({
                "status": "ERROR",
                "slot": slot,
                "seq": seq,
                "message": str(exc),
            })

    def stop(self) -> None:
        """Release resources, unmap memory, and stop server."""
        self._running = False
        if self.shm_consumer is not None:
            self.shm_consumer.close()
            self.shm_consumer = None
        self.pipeline.close()
        self.control_server.close()
        log.info("VisionEngine stopped cleanly.")

