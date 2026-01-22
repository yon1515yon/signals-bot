import json
import structlog
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text, select, delete, and_
from sqlalchemy.dialects.postgresql import insert

from app.core.celery_app import celery_app
from app.db import models
from app.db.session import session_scope
from app.core.scanner import run_scanner
from app.ml.market_analysis import get_market_regime_cached
from app.worker.tasks.notifications import send_notification_to_user

# Core Logic Imports
from app.services.services import get_historical_data, get_order_book, CandleInterval
from app.core.darkpool import detect_dark_pool_footprints
from app.core.ghost import detect_ghost_accumulation
from app.core.intraday import find_rsi_bounce_long, find_ema_cross_short
from app.core.orderbook_analysis import find_walls, validate_breakout
from app.constants import LIQUID_TICKERS
from app.core.monitoring import SIGNALS_GENERATED

logger = structlog.get_logger()

@celery_app.task
def run_signal_scanner():
    log = logger.bind(scanner="global_scanner")
    log.info("Scanner started")
    with session_scope() as db:
        market_regime = get_market_regime_cached() # Redis
        existing = {r[0] for r in db.query(models.TradingSignal.ticker).all()}
        
        found = run_scanner(db, market_regime)
        log.info("Scanner finished", signals_found=found)
        if found > 0:
            fresh = db.query(models.TradingSignal).filter(models.TradingSignal.generated_at > datetime.utcnow() - timedelta(hours=1)).all()
            new_sigs = [s for s in fresh if s.ticker not in existing]
            
            subs = [r[0] for r in db.query(models.Subscriber.user_id).all()]
            for sig in new_sigs:
                det = sig.details if isinstance(sig.details, dict) else json.loads(sig.details)
                msg = f"🔔 <b>New Signal: {sig.ticker}</b>\n{sig.signal_type}\nWinRate: {det.get('model_win_rate')}%"
                log.info("New trading signal", 
                         ticker=sig.ticker, 
                         type=sig.signal_type, 
                         profit_potential=sig.potential_profit_pct,
                         win_rate=json.loads(sig.details).get('model_win_rate'))

                SIGNALS_GENERATED.labels(
                    ticker=sig.ticker, 
                    strategy=sig.signal_type, 
                    direction="LONG"
                ).inc()
                for uid in subs:
                    send_notification_to_user.delay(uid, msg)

@celery_app.task
def schedule_order_book_scan():
    with session_scope() as db:
        stmt = select(models.TrackedTicker.ticker, models.TrackedTicker.figi)\
               .where(models.TrackedTicker.global_bias != 'Neutral')
        tickers = db.execute(stmt).fetchall()
        for t, f in tickers: scan_order_book_for_ticker.delay(t, f)

def scan_order_book_for_ticker(ticker: str, figi: str):
    with session_scope() as db:
        order_book = get_order_book(figi)
        if not order_book.get("bids") or not order_book.get("asks"):
            return

        # Core Select
        stmt = select(models.OrderBookWall.price, models.OrderBookWall.wall_type)\
               .where(models.OrderBookWall.ticker == ticker)
        old_walls = db.execute(stmt).fetchall()
        
        old_walls_map = {row.wall_type: row.price for row in old_walls}
        best_bid = order_book["bids"][0]["price"]

        # 1. Обработка пробоя
        if "ASK" in old_walls_map and best_bid > old_walls_map["ASK"]:
            val = validate_breakout(ticker, old_walls_map["ASK"], "UP", db)
            if val["score"] >= 70:
                # Вставка сигнала
                new_signal = {
                    "ticker": ticker,
                    "signal_type": "LONG",
                    "entry_price": old_walls_map["ASK"],
                    "strategy": "OB_Breakout",
                    "context_score": val["score"],
                    "generated_at": pd.Timestamp.utcnow()
                }
                # Тут обычный insert, конфликтов быть не должно (уникальность по времени + тикеру)
                db.execute(insert(models.IntradaySignal), new_signal)

                # Удаление стенки
                db.execute(
                    delete(models.OrderBookWall)
                    .where(
                        and_(models.OrderBookWall.ticker == ticker, 
                             models.OrderBookWall.wall_type == 'ASK')
                    )
                )

        # 2. Поиск и обновление стенок (UPSERT)
        hist_data = get_historical_data(figi, days=30)
        avg_vol = hist_data["volume"].mean() if not hist_data.empty else 1
        new_walls = find_walls(order_book, avg_vol)

        for wall_type, wall_data in new_walls.items():
            if wall_data:
                w_type = "BID" if "bid" in wall_type else "ASK"
                
                wall_values = {
                    "ticker": ticker,
                    "price": wall_data["price"],
                    "volume": wall_data["quantity"],
                    "wall_type": w_type,
                    "detected_at": pd.Timestamp.utcnow()
                }

                stmt = insert(models.OrderBookWall).values(wall_values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['ticker', 'wall_type'], 
                    set_={
                        "price": stmt.excluded.price,
                        "volume": stmt.excluded.volume,
                        "detected_at": stmt.excluded.detected_at
                    }
                )
                db.execute(stmt)

@celery_app.task
def schedule_intraday_scan():
    with session_scope() as db:
        tickers = db.execute(text("SELECT ticker, figi FROM tracked_tickers WHERE global_bias != 'Neutral'")).fetchall()
        for t, f in tickers: scan_intraday_for_ticker.delay(t, f)

@celery_app.task(rate_limit="3/s")
def scan_intraday_for_ticker(ticker: str, figi: str):
    with session_scope() as db:
        bias = db.execute(select(models.TrackedTicker.global_bias).where(models.TrackedTicker.ticker == ticker)).scalar()
        metrics_json = db.execute(
            text("SELECT multi_horizon_forecast FROM forecast_metrics WHERE ticker = :t"), {"t": ticker}
        ).scalar()

        neural_trend = "Neutral"
        if metrics_json:
            meta = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json)
            if "intraday" in meta:
                upd = datetime.fromisoformat(meta["intraday"]["updated_at"])
                if (datetime.utcnow() - upd).total_seconds() < 3600 * 3:
                    neural_trend = meta["intraday"]["trend"]

        final_trend = "Neutral"
        if bias == "Bullish" and neural_trend == "UP":
            final_trend = "UP"
        elif bias == "Bearish" and neural_trend == "DOWN":
            final_trend = "DOWN"

        if final_trend == "Neutral":
            return

        data = get_historical_data(figi, days=1, interval=CandleInterval.CANDLE_INTERVAL_5_MIN)
        if data.empty or len(data) < 55:
            return

        signal = None
        if final_trend == "UP":
            signal = find_rsi_bounce_long(data, neural_trend="UP")
        elif final_trend == "DOWN":
            signal = find_ema_cross_short(data, neural_trend="DOWN")

        if signal:
             signal_data = {
                "ticker": ticker,
                "signal_type": signal["signal_type"],
                "entry_price": signal["entry_price"],
                "stop_loss": signal.get("stop_loss"),
                "take_profit": signal.get("take_profit"),
                "strategy": signal["strategy"],
                "global_bias": bias,
                "generated_at": pd.Timestamp.utcnow()
             }
             
             stmt = insert(models.IntradaySignal).values(signal_data)
             stmt = stmt.on_conflict_do_nothing(index_elements=['ticker', 'generated_at'])
             db.execute(stmt)

@celery_app.task
def schedule_ghost_scan():
     with session_scope() as db:
        tickers = db.execute(text("SELECT ticker, figi FROM tracked_tickers WHERE ticker IN :l"), {"l": tuple(LIQUID_TICKERS)}).fetchall()
        for t, f in tickers: scan_ghost_activity.delay(t, f)

@celery_app.task(name="app.worker.tasks.scan_ghost_activity", rate_limit="2/s")
def scan_ghost_activity(ticker: str, figi: str):
    with session_scope() as db:
        data = get_historical_data(figi, days=2, interval=CandleInterval.CANDLE_INTERVAL_5_MIN)
        signals = detect_ghost_accumulation(data)
        if not signals: return

        fresh_signals = []
        for s in signals:
            sig_time = s["signal_time"]

            if sig_time.tzinfo is None:
                sig_time = sig_time.replace(tzinfo=None) 
            
            if (datetime.utcnow() - sig_time.replace(tzinfo=None)).total_seconds() < 3600 * 4:
                fresh_signals.append({
                    "ticker": ticker,
                    "signal_time": s["signal_time"],
                    "price": s["price"],
                    "volume": s["volume"],
                    "avg_volume": s["avg_volume"],
                    "activity_type": s["activity_type"],
                    "description": s["description"]
                })

        if fresh_signals:
            # BULK UPSERT (ON CONFLICT DO NOTHING)
            stmt = insert(models.DarkPoolSignal).values(fresh_signals)
            stmt = stmt.on_conflict_do_nothing(index_elements=['ticker', 'signal_time'])
            db.execute(stmt)

            # Уведомления
            subs = db.execute(select(models.Subscriber.user_id)).scalars().all()
            for uid in subs:
                desc = fresh_signals[0]['description'] 
                send_notification_to_user.delay(uid, f"👻 Ghost Activity {ticker}: {desc}")