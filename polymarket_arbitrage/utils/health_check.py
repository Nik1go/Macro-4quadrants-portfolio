"""Health and readiness endpoints for Docker orchestration."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict

from .config import Config


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
    """Start background HTTP server exposing `/health` and `/ready`."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status_code: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib signature
            if self.path == "/health":
                lag = time.time() - state.last_heartbeat
                payload = {"status": "ok", "heartbeat_lag_sec": round(lag, 3)}
                self._send(200, payload)
                return

            if self.path == "/ready":
                lag = time.time() - state.last_heartbeat
                is_ready = state.ready and lag <= (2 * Config.SCAN_INTERVAL + 10)
                status = 200 if is_ready else 503
                payload = {
                    "ready": is_ready,
                    "heartbeat_lag_sec": round(lag, 3),
                    "metadata": state.metadata,
                }
                self._send(status, payload)
                return

            self._send(404, {"error": "not_found"})

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    class HealthCheckServer(ThreadingHTTPServer):
        allow_reuse_address = True

    server = HealthCheckServer((Config.HEALTH_HOST, Config.HEALTH_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="health-server")
    thread.start()
    return thread
