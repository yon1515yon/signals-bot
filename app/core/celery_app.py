from celery import Celery
from celery.schedules import crontab
from app.config import settings

celery_app = Celery("neurovision", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

# Настройки
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_default_queue="default",
)

# Импортируем задачи, чтобы воркер их видел при запуске
celery_app.conf.imports = [
    "app.worker.tasks.ml",
    "app.worker.tasks.scanners",
    "app.worker.tasks.maintenance",
    "app.worker.tasks.notifications",
    "app.worker.tasks.nlp",
]

# Маршрутизация (Heavy vs Light)
celery_app.conf.task_routes = {
    "app.worker.tasks.ml.*": {"queue": "ml_training"},
    "app.worker.tasks.scanners.run_signal_scanner": {"queue": "default"},
    "app.worker.tasks.scanners.*": {"queue": "default"},
    "app.worker.tasks.maintenance.*": {"queue": "default"},
    "app.worker.tasks.notifications.*": {"queue": "default"},
    "app.worker.tasks.nlp.*": {"queue": "nlp_queue"}, 
}

# Расписание (Beat)
celery_app.conf.beat_schedule = {
    # --- SCANNERS ---
    "scan-order-book-breakouts": {
        "task": "app.worker.tasks.scanners.schedule_order_book_scan",
        "schedule": crontab(minute="*"),
    },
    "run-intraday-scanner-frequently": {
        "task": "app.worker.tasks.scanners.schedule_intraday_scan",
        "schedule": crontab(minute="*/5", hour="7-18", day_of_week="mon-fri"),
    },
    "run-medium-term-scanner-hourly": {
        "task": "app.worker.tasks.scanners.run_signal_scanner",
        "schedule": crontab(minute="0"),
    },
    "scan-ghost-activity-15min": {
        "task": "app.worker.tasks.scanners.schedule_ghost_scan",
        "schedule": crontab(minute="*/15", hour="7-18", day_of_week="mon-fri"),
    },
    
    # --- ML & TRAINING ---
    "train-global-model-weekly": {
        "task": "app.worker.tasks.ml.run_global_model_training",
        "schedule": crontab(day_of_week="sunday", hour=1, minute=0),
    },
    "retrain-price-models-daily": {
        "task": "app.worker.tasks.ml.schedule_full_retrains",
        "schedule": crontab(hour=3, minute=0),
    },
    "train-meta-model-daily": {
        "task": "app.worker.tasks.ml.run_meta_model_training",
        "schedule": crontab(hour=6, minute=0),
    },
    "train-intraday-models-market-hours": {
        "task": "app.worker.tasks.ml.schedule_intraday_train_batch",
        "schedule": crontab(hour="7-16", minute=15, day_of_week="mon-fri"),
    },
    "retrain-drawdown-models-weekly": {
        "task": "app.worker.tasks.ml.schedule_drawdown_model_retrains",
        "schedule": crontab(day_of_week="sunday", hour=5, minute=0),
    },
    
    # --- MAINTENANCE & SYSTEM ---
    "discover-stocks-daily": {
        "task": "app.worker.tasks.maintenance.discover_and_track_stocks",
        "schedule": crontab(hour=0, minute=1),
    },
    "recalculate-key-levels-daily": {
        "task": "app.worker.tasks.maintenance.schedule_key_level_recalculation",
        "schedule": crontab(hour=2, minute=0),
    },
    "update-global-bias-daily": {
        "task": "app.worker.tasks.maintenance.schedule_global_bias_update",
        "schedule": crontab(hour=6, minute=30, day_of_week="mon-fri"),
    },
    "recalculate-strategy-stats-daily": {
        "task": "app.worker.tasks.maintenance.recalculate_strategy_performance",
        "schedule": crontab(hour=4, minute=0),
    },
    "update-portfolio-prices": {
        "task": "app.worker.tasks.maintenance.update_portfolio_prices", 
        "schedule": crontab(minute="*/5")
    },
}