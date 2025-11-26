
from app.db.database import Base
from sqlalchemy import BIGINT, JSON, Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import relationship


class TrackedTicker(Base):
    __tablename__ = "tracked_tickers"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    figi = Column(String(20), unique=True, nullable=False)
    name = Column(String(255))
    sentiment_score = Column(Float)
    risk_score = Column(Integer)
    global_bias = Column(String(20))
    last_updated_at = Column(DateTime(timezone=True))


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    forecast_date = Column(DateTime, nullable=False)
    forecast_value = Column(Float, nullable=False)
    upper_bound = Column(Float)  
    lower_bound = Column(Float)  
    __table_args__ = (UniqueConstraint("ticker", "forecast_date", name="_ticker_date_uc"),)


class ForecastMetrics(Base):
    """Хранит метрики качества и бэктеста для каждого тикера."""

    __tablename__ = "forecast_metrics"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    backtest_metrics = Column(JSON)
    multi_horizon_forecast = Column(JSON)  
    bear_scenario_forecast = Column(JSON) 
    last_calculated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KeyLevel(Base):
    __tablename__ = "key_levels"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    start_price = Column(Float, nullable=False)
    end_price = Column(Float, nullable=False)
    level_type = Column(String(20), nullable=False)
    strength = Column(BIGINT)
    intensity = Column(Integer)
    last_calculated_at = Column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("ticker", "start_price", "end_price", name="_ticker_price_uc"),)


class TradingSignal(Base):
    __tablename__ = "trading_signals"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    signal_type = Column(String(50), nullable=False)
    potential_profit_pct = Column(Float)
    forecast_days = Column(Integer)
    details = Column(JSON)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())


class DarkPoolSignal(Base):
    __tablename__ = "dark_pool_signals"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    signal_time = Column(DateTime(timezone=True), nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(BIGINT)
    avg_volume = Column(BIGINT)
    activity_type = Column(String(20))
    description = Column(String)
    __table_args__ = (UniqueConstraint("ticker", "signal_time", name="_ticker_time_uc"),)


class IntradaySignal(Base):
    __tablename__ = "intraday_signals"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    signal_type = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    strategy = Column(String(50))
    global_bias = Column(String(20))
    context_score = Column(Integer)
    related_zone_id = Column(Integer)
    generated_at = Column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("ticker", "generated_at", name="_ticker_generated_uc"),)


class OrderBookWall(Base):
    __tablename__ = "order_book_walls"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(BIGINT, nullable=False)
    wall_type = Column(String(10), nullable=False)
    detected_at = Column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("ticker", "wall_type", name="_ticker_wall_uc"),)


class SignalFeedback(Base):
    __tablename__ = "signal_feedback"

    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("trading_signals.id", ondelete="CASCADE"))
    user_id = Column(BIGINT)
    reaction = Column(String(20), nullable=False)
    comment = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("signal_id", "user_id", name="_signal_user_uc"),)


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True)
    user_id = Column(BIGINT, unique=True, nullable=False, index=True)
    initial_balance = Column(Float, default=1000000.0)
    current_balance = Column(Float, default=1000000.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Связь с позициями
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    ticker = Column(String(10), index=True)
    figi = Column(String(20))
    direction = Column(String(10))

    entry_price = Column(Float)
    quantity = Column(Integer)
    current_price = Column(Float)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)

    status = Column(String(20), default="OPEN")
    entry_date = Column(DateTime(timezone=True), server_default=func.now())
    exit_date = Column(DateTime(timezone=True), nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0)

    portfolio = relationship("Portfolio", back_populates="positions")


class Subscriber(Base):
    __tablename__ = "subscribers"

    id = Column(Integer, primary_key=True)
    user_id = Column(BIGINT, unique=True, nullable=False, index=True)
    username = Column(String(100))
    subscribed_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Subscriber(user_id={self.user_id})>"


class StrategyPerformance(Base):
    """Хранит статистику эффективности для каждой стратегии/типа сигнала."""

    __tablename__ = "strategy_performance"

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), unique=True, nullable=False, index=True)
    win_rate = Column(Float, default=0.5)
    avg_profit_pct = Column(Float, default=5.0)  
    avg_loss_pct = Column(Float, default=-3.0)  
    trades_count = Column(Integer, default=0)
    last_calculated_at = Column(DateTime(timezone=True), onupdate=func.now())
