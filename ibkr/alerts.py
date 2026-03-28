"""
IBKR Alerting & Notifications
=============================
Handles notifications via Telegram and local logging.
"""

import os
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ERROR_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "execution_error.log")

def send_alert(message: str, severity: str = "error"):
    """
    Send an alert via Telegram and log it locally.
    
    Args:
        message: The alert message
        severity: 'info', 'warning', or 'error'
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "❌ ERROR" if severity == "error" else "⚠️ WARNING" if severity == "warning" else "ℹ️ INFO"
    full_message = f"{prefix} [{timestamp}]\n{message}"
    
    # 1. Log to console
    if severity == "error":
        logger.error(message)
    elif severity == "warning":
        logger.warning(message)
    else:
        logger.info(message)
        
    # 2. Log to file
    try:
        os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {severity.upper()}: {message}\n")
    except Exception as e:
        logger.error(f"Failed to write to alert log: {e}")

    # 3. Send Telegram message
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": full_message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Telegram alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    else:
        logger.debug("Telegram alerting not configured (missing token or chat_id)")

if __name__ == "__main__":
    # Test alert
    print("Sending test alert...")
    send_alert("Test alert from IBKR Macro Pipeline", severity="info")
