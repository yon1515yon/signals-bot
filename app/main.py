import logging

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# Импортируем роутер, куда мы перенесли все эндпоинты
from app.api.routes import router

# Импортируем настройки (если понадобится доступ к ним здесь, например для версионирования)
from app.config import settings

# Настройка базового логгера
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Создаем приложение FastAPI
app = FastAPI(
    title="NeuroVision API for Russian Stocks",
    description="AI-powered trading bot backend with ML forecasting and market analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Подключаем основной роутер с префиксом (опционально, можно без prefix)
# Это делает API доступным по путям /api/v1/tickers и т.д., или просто /tickers
app.include_router(router)

# Подключаем метрики Prometheus для мониторинга
Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def on_startup():
    """Действия при запуске приложения."""
    logger.info("NeuroVision API is starting up...")
    # Здесь можно добавить проверку соединения с Redis или БД,
    # но SQLAlchemy и Celery обычно делают это лениво.


@app.get("/health", tags=["System"], summary="Проверка работоспособности")
async def health_check():
    """
    Простой эндпоинт для проверки, что сервис жив.
    Используется Docker healthcheck или Kubernetes liveness probe.
    """
    return {"status": "ok", "environment": "production" if not settings.DEBUG else "dev"}


@app.get("/", tags=["System"])
async def root():
    """Корневой эндпоинт с приветствием."""
    return {"message": "Welcome to NeuroVision API!", "documentation": "/docs", "version": "1.0.0"}
