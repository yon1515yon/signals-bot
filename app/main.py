import logging
from asgi_correlation_id import CorrelationIdMiddleware
from app.core.logging import setup_logging
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router

from app.config import settings

setup_logging()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="signals-bot API for Russian Stocks",
    description="AI-powered trading bot backend with ML forecasting and market analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(CorrelationIdMiddleware)

app.include_router(router)

Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def on_startup():
    """Действия при запуске приложения."""
    logger.info("signals-bot API is starting up...")



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
    return {"message": "Welcome to signals-bot API!", "documentation": "/docs", "version": "1.0.0"}
