import logging
import sys
import structlog
from app.config import settings

def setup_logging():
    """
    Настраивает структурированное логирование (JSON для продакшена, Console для разработки).
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,  # Вливаем контекст (request_id и т.д.)
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]

    if settings.DEBUG:
        # Для разработки: цветной красивый вывод
        processors = shared_processors + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        # Для продакшена: JSON формат (для ELK/Loki)
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Перехватываем стандартные логи (uvicorn, sqlalchemy и т.д.)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    
    # Настраиваем уровни для шумных библиотек
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)