from contextlib import contextmanager
from app.db.database import SessionLocal
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@contextmanager
def session_scope():
    """
    Контекстный менеджер для безопасной работы с БД.
    Гарантирует commit при успехе, rollback при ошибке и close всегда.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"DB Session Error: {e}")
        session.rollback()
        raise
    finally:
        session.close()