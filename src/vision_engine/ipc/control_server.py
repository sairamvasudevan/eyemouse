"""Unix Domain Socket control-plane server handling command and notification messages."""

from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path
from typing import Optional, Dict, Any, Callable

log = logging.getLogger("vision_engine.ipc")


class ControlServer:
    """Unix Domain Socket server for lightweight JSON control and frame notifications."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self._server_sock: Optional[socket.socket] = None
        self._client_sock: Optional[socket.socket] = None
        self._is_running = False

    def start(self) -> None:
        """Bind and listen on the Unix domain socket."""
        # Unlink preexisting socket if present
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        # Ensure owner-only permissions (0600)
        os.chmod(self.socket_path, 0o600)
        self._server_sock.listen(1)
        self._is_running = True
        log.info("ControlServer listening on %s", self.socket_path)

    def accept_client(self, timeout: Optional[float] = None) -> bool:
        """Wait for host/producer client to connect."""
        if self._server_sock is None:
            raise RuntimeError("Server not started")
        self._server_sock.settimeout(timeout)
        try:
            client, _ = self._server_sock.accept()
            self._client_sock = client
            # Line-buffered communication
            self._client_file = client.makefile("r", encoding="utf-8")
            log.info("Control client connected.")
            return True
        except (socket.timeout, TimeoutError):
            return False

    def read_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Read a single newline-delimited JSON message from the connected client."""
        if self._client_sock is None or not hasattr(self, "_client_file"):
            return None
        self._client_sock.settimeout(timeout)
        try:
            line = self._client_file.readline()
            if not line:
                # Client disconnected (EOF)
                return None
            return json.loads(line.strip())
        except (socket.timeout, TimeoutError):
            return None
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read control message: %s", exc)
            return None

    def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a single newline-delimited JSON message to the connected client."""
        if self._client_sock is None:
            return False
        try:
            payload = json.dumps(message) + "\n"
            self._client_sock.sendall(payload.encode("utf-8"))
            return True
        except OSError as exc:
            log.warning("Failed to send control message: %s", exc)
            return False

    def close(self) -> None:
        """Close client and server sockets and clean up socket file."""
        self._is_running = False
        if self._client_sock is not None:
            try:
                self._client_sock.close()
            except OSError:
                pass
            self._client_sock = None

        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

