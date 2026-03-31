"""Notification helpers for operational alerts (Telegram / Email)."""

from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from typing import Optional

import httpx

from .config import Config
from .logger import Logger

logger = Logger().logger


class Notifier:
    """Send alert messages to configured channels."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    async def send(self, message: str) -> None:
        """Send message to all configured channels."""
        if not self.enabled:
            return

        await self._send_telegram(message)
        self._send_email(message)

    async def _send_telegram(self, message: str) -> None:
        if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
            return

        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": message}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)

    def _send_email(self, message: str) -> None:
        if not all([Config.EMAIL_HOST, Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD, Config.EMAIL_TO]):
            return

        try:
            mime = MIMEText(message, _charset="utf-8")
            mime["Subject"] = "Polymarket Arbitrage Alert"
            mime["From"] = Config.EMAIL_USERNAME
            mime["To"] = Config.EMAIL_TO

            with smtplib.SMTP(Config.EMAIL_HOST, Config.EMAIL_PORT, timeout=10) as server:
                server.starttls()
                server.login(Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD)
                server.sendmail(Config.EMAIL_USERNAME, [Config.EMAIL_TO], mime.as_string())
        except Exception as exc:
            logger.warning("Email notification failed: %s", exc)
