import requests
from celery.utils.log import get_task_logger
from app.config import settings
from app.core.celery_app import celery_app

logger = get_task_logger(__name__)

@celery_app.task(autoretry_for=(Exception,), retry_kwargs={"max_retries": 3}, retry_backoff=True)
def send_notification_to_user(user_id: int, message_text: str):
    """Отправляет уведомление в Telegram."""
    try:
        token = settings.TELEGRAM_BOT_TOKEN
        if not token:
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": user_id, "text": message_text, "parse_mode": "HTML"}

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send notification to {user_id}: {e}")
        raise e