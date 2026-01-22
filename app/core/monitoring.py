import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

SIGNALS_GENERATED = Counter(
    "neurovision_signals_total", 
    "Total trading signals generated",
    ["ticker", "strategy", "direction"] 
)

TRADES_EXECUTED = Counter(
    "neurovision_trades_total",
    "Total trades executed via API",
    ["ticker", "direction", "status"]
)

# 2. ML Метрики
INFERENCE_LATENCY = Histogram(
    "neurovision_ml_inference_seconds",
    "Time spent in ML model inference",
    ["model_name"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0] 
)

MODEL_LOAD_TIME = Gauge(
    "neurovision_model_load_seconds",
    "Time taken to load model from disk/cache",
    ["ticker"]
)

# 3. Инфраструктура
CELERY_TASK_ERRORS = Counter(
    "neurovision_celery_errors_total",
    "Total errors in Celery tasks",
    ["task_name"]
)



def track_inference_time(model_name: str):
    """Декоратор для замера времени инференса."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                INFERENCE_LATENCY.labels(model_name=model_name).observe(duration)
        return wrapper
    return decorator

def track_task_errors(task_name: str):
    """Декоратор для подсчета ошибок в Celery (если не используем встроенный мониторинг)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                CELERY_TASK_ERRORS.labels(task_name=task_name).inc()
                raise e
        return wrapper
    return decorator