# Файл: app/schemas.py

from datetime import date, datetime
from typing import Any, List

from pydantic import BaseModel

# --- Схемы для базовой аналитики ---


class TickerResponse(BaseModel):
    ticker: str
    name: str
    sentiment_score: float | None
    risk_score: int | None
    global_bias: str | None
    last_updated_at: datetime | None

    class Config:
        from_attributes = True


class ForecastResponse(BaseModel):
    ticker: str
    forecast_date: date
    forecast_value: float
    upper_bound: float | None
    lower_bound: float | None

    class Config:
        from_attributes = True


class KeyLevelZoneResponse(BaseModel):
    ticker: str
    start_price: float
    end_price: float
    level_type: str
    intensity: int

    class Config:
        from_attributes = True


class SignalResponse(BaseModel):
    id: int
    ticker: str
    signal_type: str
    potential_profit_pct: float
    details: dict[str, Any]
    generated_at: datetime
    forecast_horizon_days: int
    target_date: date
    description: str

    class Config:
        from_attributes = True


class DarkPoolSignalResponse(BaseModel):
    ticker: str
    signal_time: datetime
    price: float
    activity_type: str
    description: str

    class Config:
        from_attributes = True


class IntradaySignalResponse(BaseModel):
    ticker: str
    signal_type: str
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    strategy: str
    global_bias: str | None
    generated_at: datetime
    context_score: int | None

    class Config:
        from_attributes = True


# --- Схемы для калькуляторов ---


class StopLossSuggestion(BaseModel):
    method: str
    stop_loss_price: float
    risk_percent: float
    description: str


class PositionSizeRequest(BaseModel):
    ticker: str
    entry_price: float
    stop_loss_price: float
    account_size: float
    risk_percent: float = 1.0


# --- Схемы для фидбэка и статистики ---


class FeedbackIn(BaseModel):
    signal_id: int
    user_id: int
    reaction: str


class SignalStats(BaseModel):
    total_signals: int
    success_count: int
    fail_count: int
    breakeven_count: int
    success_rate: float


# --- Схемы для портфолио ---


class PositionBase(BaseModel):
    ticker: str
    direction: str
    entry_price: float
    size_shares: int


class PositionCreate(PositionBase):
    portfolio_id: int


class PositionClose(BaseModel):
    exit_price: float


class PositionResponse(BaseModel):
    id: int
    ticker: str
    entry_price: float
    size_shares: int
    status: str
    entry_date: datetime
    current_price: float | None = None
    unrealized_pnl: float | None = None

    class Config:
        from_attributes = True


class PortfolioCreate(BaseModel):
    user_id: int
    name: str = "My Portfolio"
    initial_capital: float


class PortfolioResponse(BaseModel):
    id: int
    user_id: int
    initial_capital: float
    current_capital: float
    open_positions: List[PositionResponse]
    total_pnl: float
    total_pnl_percent: float

    class Config:
        from_attributes = True


class SubscriberCreate(BaseModel):
    user_id: int
    username: str | None = None


class ForecastMetricsResponse(BaseModel):
    ticker: str
    backtest_metrics: dict | None
    multi_horizon_forecast: dict | None
    last_calculated_at: datetime

    class Config:
        from_attributes = True

class LocalTradeRequest(schemas.PositionBase):
    ticker: str
    price: float
    direction: str = "LONG"
    amount: float = 50000