"""Health and readiness endpoints for Docker orchestration."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class BotHealthState:
    """Shared bot health state used by HTTP probes."""

    ready: bool = False
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)

    def mark_alive(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()


def start_health_server(state: BotHealthState) -> threading.Thread:
    """Start background HTTP server exposing `/health` and `/ready`.

    Applies SO_REUSEADDR + SO_REUSEPORT at socket level (before bind)
    so a fast bot restart never blocks on 'Address already in use'.
    If the port is truly occupied by another live process, the health
    server is skipped and a warning is logged  – the bot keeps running.
    """

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status_code: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                lag = time.time() - state.last_heartbeat
                self._send(200, {"status": "ok", "heartbeat_lag_sec": round(lag, 3)})
                return

            if self.path == "/ready":
                lag = time.time() - state.last_heartbeat
                is_ready = state.ready and lag <= (2 * Config.SCAN_INTERVAL + 10)
                status = 200 if is_ready else 503
                self._send(status, {
                    "ready": is_ready,
                    "heartbeat_lag_sec": round(lag, 3),
                    "metadata": state.metadata,
                })
                return

            self._send(404, {"error": "not_found"})

        def log_message(self, *args) -> None:  # silence access logs
            return

    class RobustHealthServer(ThreadingHTTPServer):
        """ThreadingHTTPServer with SO_REUSEADDR + SO_REUSEPORT before bind."""

        # stdlib flag – set here for belt-and-suspenders
        allow_reuse_address = True

        def server_bind(self) -> None:
            # SO_REUSEADDR must be set on the socket BEFORE bind()
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # SO_REUSEPORT (Linux ≥ 3.9) – silently ignored if unavailable
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass
            super().server_bind()

    host = Config.HEALTH_HOST
    port = Config.HEALTH_PORT

    try:
        server = RobustHealthServer((host, port), Handler)
        thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="health-server",
        )
        thread.start()
        logger.info("Health server listening on %s:%d", host, port)
        return thread

    except OSError as exc:
        # Port is held by another live process – skip health endpoint.
        # The bot itself continues trading normally.
        logger.warning(
            "Health server could NOT bind to %s:%d (%s). "
            "A previous instance may still hold the port. "
            "Run:  fuser -k %d/tcp   to free it, then restart the bot. "
            "The bot will continue WITHOUT a health endpoint for now.",
            host, port, exc, port,
        )
        dummy = threading.Thread(
            target=lambda: None,
            daemon=True,
            name="health-server-noop",
        )
        dummy.start()
        return dummy
