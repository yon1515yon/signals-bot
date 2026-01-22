import pandas as pd
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert  
import structlog
import numpy as np

from tinkoff.invest import Client, CandleInterval

from app.config import settings
from app.core.celery_app import celery_app
from app.db import models
from app.db.session import session_scope
from app.services.services import get_all_russian_stocks, get_historical_data, get_order_book
from app.core.levels import find_key_level_zones
from app.core.intraday import determine_global_bias
from app.constants import LIQUID_TICKERS

logger = structlog.get_logger()

@celery_app.task
def discover_and_track_stocks():
    logger.info("MAINTENANCE: Обнаружение акций...")
    stocks = get_all_russian_stocks()
    with session_scope() as db:
        stmt = insert(models.TrackedTicker).values(stocks)
        stmt = stmt.on_conflict_do_nothing(index_elements=['ticker'])
        
        db.execute(stmt)

@celery_app.task(name="app.worker.tasks.maintenance.update_portfolio_prices")
def update_portfolio_prices():
    with session_scope() as db:
        positions = db.query(models.Position).filter(models.Position.status == "OPEN").all()
        if not positions:
            return
        unique_figis = {p.figi for p in positions if p.figi}
        prices = {}
        try:
            with Client(settings.TINKOFF_API_TOKEN) as client:
                for figi in unique_figis:
                    df = get_historical_data(figi, days=1, interval=CandleInterval.CANDLE_INTERVAL_1_MIN, client=client)
                    if not df.empty:
                        prices[figi] = df.iloc[-1]["close"]
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
            return
        for p in positions:
            if p.figi in prices:
                p.current_price = float(prices[p.figi])

@celery_app.task
def schedule_key_level_recalculation():
    with session_scope() as db:
        stmt = select(models.TrackedTicker.ticker, models.TrackedTicker.figi)
        tickers = db.execute(stmt).fetchall()
        for ticker, figi in tickers:
            recalculate_levels_for_ticker.delay(ticker, figi)

@celery_app.task(bind=True, rate_limit="10/s")
def recalculate_levels_for_ticker(self, ticker: str, figi: str):
    try:
        hist = get_historical_data(figi, days=365)
        if hist.empty: return
        current_price = hist.iloc[-1]["close"]
        order_book = get_order_book(figi)
        zones = find_key_level_zones(hist, order_book, current_price)
        if not zones: return
        with session_scope() as db:
            db.execute(delete(models.KeyLevel).where(models.KeyLevel.ticker == ticker))
        if zones:
                data_to_insert = [
                    {
                        "ticker": ticker,
                        "start_price": z["start_price"],
                        "end_price": z["end_price"],
                        "level_type": z["level_type"],
                        "strength": z["strength"],
                        "intensity": z["intensity"],
                        "last_calculated_at": pd.Timestamp.utcnow() # Важно: Python datetime, не SQL NOW()
                    }
                    for z in zones
                ]
                db.execute(insert(models.KeyLevel), data_to_insert)
    except Exception as e:
        logger.error(f"Level calc error {ticker}: {e}")

@celery_app.task
def recalculate_strategy_performance():
    logger.info("MAINTENANCE: Пересчет статистики (WinRate/PnL)...")
    
    with session_scope() as db:
        stmt = (
            select(
                models.TradingSignal.signal_type,
                models.TradingSignal.potential_profit_pct,
                models.SignalFeedback.reaction
            )
            .join(models.SignalFeedback, models.TradingSignal.id == models.SignalFeedback.signal_id)
        )
        
        rows = db.execute(stmt).fetchall()

        if not rows:
            return

        df = pd.DataFrame(rows, columns=["strategy_name", "profit_pct", "reaction"])
        
        # Заполняем пропуски нулями, чтобы не упало при расчетах
        df["profit_pct"] = df["profit_pct"].fillna(0.0)
        
        updates = []

        for name, group in df.groupby("strategy_name"):
            total_trades = len(group)
            
            # Игнорируем стратегии, где слишком мало статистики
            if total_trades < 5: 
                continue

            # Фильтруем успешные и неудачные сделки
            wins = group[group["reaction"] == "SUCCESS"]
            losses = group[group["reaction"] == "FAIL"]

            # --- РАСЧЕТ МЕТРИК ---
            
            win_rate = len(wins) / total_trades

            avg_profit_pct = wins["profit_pct"].mean() if not wins.empty else 0.0

            avg_loss_potential = losses["profit_pct"].mean() if not losses.empty else 0.0
            avg_loss_pct = -1.0 * abs(avg_loss_potential / 2.0) 

            updates.append({
                "strategy_name": name,
                "win_rate": float(win_rate),
                "avg_profit_pct": float(round(avg_profit_pct, 2)),
                "avg_loss_pct": float(round(avg_loss_pct, 2)),
                "trades_count": total_trades,
                "last_calculated_at": pd.Timestamp.utcnow()
            })

        # 3. Bulk Upsert (Вставка или Обновление)
        if updates:
            stmt = insert(models.StrategyPerformance).values(updates)
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['strategy_name'],
                set_={
                    "win_rate": stmt.excluded.win_rate,
                    "avg_profit_pct": stmt.excluded.avg_profit_pct,
                    "avg_loss_pct": stmt.excluded.avg_loss_pct,
                    "trades_count": stmt.excluded.trades_count,
                    "last_calculated_at": stmt.excluded.last_calculated_at
                }
            )
            
            db.execute(stmt)
            logger.info(f"MAINTENANCE: Обновлена статистика для {len(updates)} стратегий.")

@celery_app.task
def schedule_global_bias_update():
    with session_scope() as db:
        stmt = select(models.TrackedTicker.ticker, models.TrackedTicker.figi).where(models.TrackedTicker.ticker.in_(LIQUID_TICKERS))
        tickers = db.execute(stmt).fetchall()
        for t, f in tickers:
             update_bias_for_ticker.delay(t, f)

@celery_app.task(rate_limit="2/s")
def update_bias_for_ticker(ticker: str, figi: str):
    with session_scope() as db:
        stmt = select(models.Forecast.forecast_value).where(models.Forecast.ticker == ticker).limit(7)
        fc = db.execute(stmt).fetchall()
        if not fc: return
        hist = get_historical_data(figi, days=1)
        if hist.empty: return
        bias = determine_global_bias(pd.DataFrame(fc, columns=["forecast_value"]), hist.iloc[-1]["close"], 0.0)
        db.query(models.TrackedTicker).filter(models.TrackedTicker.ticker == ticker).update({"global_bias": bias})