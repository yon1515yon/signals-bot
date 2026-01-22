import json
import torch
import joblib
import structlog
import pandas as pd 
from sqlalchemy import text, delete
from sqlalchemy.dialects.postgresql import insert 

from app.core.monitoring import track_inference_time, MODEL_LOAD_TIME
from app.config import settings
from app.core.celery_app import celery_app
from app.db.session import session_scope
from app.constants import LIQUID_TICKERS, ML_CONFIG
from app.services.services import get_historical_data, CandleInterval
from app.services.sentiment import get_sentiment_score
from app.db import models # Импортируем модели

# ML Imports (без изменений)
from app.ml.loader import load_model, load_scaler, clear_model_cache
from app.ml.training import (
    train_and_predict_lstm, 
    train_global_base_model, 
    train_drawdown_model
)
from app.ml.prediction import predict_with_uncertainty
from app.ml.meta_model import train_meta_model

logger = structlog.get_logger()

@celery_app.task(bind=True, time_limit=7200)
def run_global_model_training(self):
    ticker_map = {}
    with session_scope() as db:
        rows = db.execute(text("SELECT ticker, figi FROM tracked_tickers WHERE ticker IN :tickers"), {"tickers": tuple(LIQUID_TICKERS)}).fetchall()
        for r in rows: ticker_map[r.ticker] = r.figi
    if ticker_map:
        train_global_base_model(list(ticker_map.keys()), ticker_map)

@celery_app.task
def schedule_full_retrains():
    with session_scope() as db:
        tickers = db.execute(text("SELECT ticker, figi, name FROM tracked_tickers")).fetchall()
        for t, f, n in tickers:
            full_train_model.delay(t, f, n)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 3}, rate_limit="2/s")
def full_train_model(self, ticker: str, figi: str, name: str):
    log = logger.bind(task_id=self.request.id, ticker=ticker)
    log.info("Starting full training task")
    try:
        data = get_historical_data(figi, days=365*4)
        if data.empty or len(data) < 300: return
        
        model, scaler, preds, upper, lower, _, bear, multi = train_and_predict_lstm(data, ticker=ticker, future_predictions=30)
        log.info("Model saved successfully", path=f"{settings.MODEL_STORAGE_PATH}/{ticker}.pth")
        if model:
            torch.save(model.state_dict(), f"{settings.MODEL_STORAGE_PATH}/{ticker}.pth")
            joblib.dump(scaler, f"{settings.MODEL_STORAGE_PATH}/{ticker}.pkl")
            clear_model_cache() 
            
            sent, _ = get_sentiment_score(ticker, name)
            _save_forecast_to_db(ticker, preds, upper, lower, {}, multi, bear, sent)
            
    except Exception as e:
        log.error("Training task failed", error=str(e), exc_info=True)
        raise e

@celery_app.task(bind=True, rate_limit="5/s")
def update_forecast(self, ticker: str, figi: str, name: str):
    log = logger.bind(task_id=self.request.id, ticker=ticker)
    try:
        scaler = load_scaler(ticker)
        if not scaler:
            full_train_model.delay(ticker, figi, name)
            return
        
        data = get_historical_data(figi, days=90)
        if data.empty: return

        processed, data_norm, _, feats, outs = prepare_features_and_scale(data)
        if data_norm is None: return

        model = load_model(ticker, len(feats), len(outs))
        if not model:
            full_train_model.delay(ticker, figi, name)
            return

        mean, upper, lower = predict_with_uncertainty(model, scaler, data_norm[-ML_CONFIG["TRAIN_WINDOW"]:], processed, feats, outs, 30)
        sent, _ = get_sentiment_score(ticker, name)
        _save_forecast_to_db(ticker, mean, upper, lower, None, None, None, sent)
        log.info("Forecast updated", status="success")
    except Exception as e:
        log.error("Inference failed", error=str(e))

def _save_forecast_to_db(ticker, preds, upper, lower, metrics, multi, bear, sent):
    """
    Helper для записи прогноза в БД.
    Использует SQLAlchemy Core для массовой вставки (Bulk Insert) и Upsert.
    """
    with session_scope() as db:
        # 1. Очистка старых прогнозов (Core Delete)
        db.execute(delete(models.Forecast).where(models.Forecast.ticker == ticker))

        # 2. Подготовка данных для Bulk Insert
        forecast_objects = []
        today = pd.Timestamp.now(tz="UTC").normalize()
        
        for i, val in enumerate(preds):
            forecast_objects.append({
                "ticker": ticker,
                "forecast_date": today + pd.Timedelta(days=i + 1),
                "forecast_value": float(val),
                "upper_bound": float(upper[i]),
                "lower_bound": float(lower[i])
            })
        
        # 3. Массовая вставка (один запрос в БД вместо 30)
        if forecast_objects:
            db.execute(insert(models.Forecast), forecast_objects)
        
        # 4. Upsert для метрик
        metric_values = {
            "ticker": ticker,
            "backtest_metrics": metrics, 
            "multi_horizon_forecast": multi,
            "bear_scenario_forecast": bear,
            "last_calculated_at": pd.Timestamp.utcnow()
        }
        
        stmt_metrics = insert(models.ForecastMetrics).values(metric_values)
        stmt_metrics = stmt_metrics.on_conflict_do_update(
            index_elements=['ticker'],
            set_={
                "backtest_metrics": stmt_metrics.excluded.backtest_metrics,
                "multi_horizon_forecast": stmt_metrics.excluded.multi_horizon_forecast,
                "bear_scenario_forecast": stmt_metrics.excluded.bear_scenario_forecast,
                "last_calculated_at": stmt_metrics.excluded.last_calculated_at
            }
        )
        db.execute(stmt_metrics)
        
        # 5. Обновление сентимента
        if sent is not None:
            # Используем update через ORM Query (проще для одного поля)
            db.query(models.TrackedTicker).\
                filter(models.TrackedTicker.ticker == ticker).\
                update({"sentiment_score": sent, "last_updated_at": pd.Timestamp.utcnow()})

@celery_app.task(name="app.worker.tasks.ml.run_meta_model_training")
def run_meta_model_training():
    with session_scope() as db:
        res = db.execute(text("SELECT backtest_metrics FROM forecast_metrics")).fetchall()
        trades = []
        for r in res:
            if r[0]:
                js = r[0] if isinstance(r[0], dict) else json.loads(r[0])
                if "meta_data" in js: trades.extend(js["meta_data"])
        if len(trades) > 50:
            train_meta_model(trades)

@celery_app.task
def schedule_intraday_train_batch():
    with session_scope() as db:
        tickers = db.execute(text("SELECT ticker, figi FROM tracked_tickers WHERE ticker IN :l"), {"l": tuple(LIQUID_TICKERS)}).fetchall()
        for t, f in tickers: train_intraday_model.delay(t, f)

@celery_app.task(bind=True, rate_limit="5/s")
def train_intraday_model(self, ticker: str, figi: str):
    hour = datetime.utcnow().hour
    if hour < 6 or hour > 16:
        return

    try:
        hourly_data = get_historical_data(figi, days=60, interval=CandleInterval.CANDLE_INTERVAL_HOUR)
        if hourly_data.empty or len(hourly_data) < 100:
            return

        _, _, preds, _, _, _, _, _ = train_and_predict_lstm(hourly_data.copy(), ticker=ticker, future_predictions=8)

        if preds is None:
            return

        intraday_json = {
            "forecast_8h": preds.tolist(),
            "current_price": float(hourly_data.iloc[-1]["close"]),
            "updated_at": datetime.utcnow().isoformat(),
            "trend": "UP" if preds[-1] > preds[0] else "DOWN",
        }

        with session_scope() as db:
            curr = db.execute(
                text("SELECT multi_horizon_forecast FROM forecast_metrics WHERE ticker = :t"), {"t": ticker}
            ).scalar()
            meta = curr if curr else {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta["intraday"] = intraday_json
            db.execute(
                text("UPDATE forecast_metrics SET multi_horizon_forecast = :m WHERE ticker = :t"),
                {"m": json.dumps(meta), "t": ticker},
            )
    except Exception as e:
        logger.error(f"Intraday train error {ticker}: {e}")

@celery_app.task
def schedule_drawdown_model_retrains():
    with session_scope() as db:
        tickers = db.execute(text("SELECT ticker, figi FROM tracked_tickers")).fetchall()
        for t, f in tickers: train_and_save_drawdown_model.delay(t, f)

@celery_app.task(bind=True, rate_limit="1/s")
def train_and_save_drawdown_model(self, ticker: str, figi: str):
    try:
        data = get_historical_data(figi, days=365 * 5)
        if data.empty or len(data) < 100:
            return
        model, scaler = train_drawdown_model(data, ticker)
        if model:
            torch.save(model.state_dict(), f"{settings.MODEL_STORAGE_PATH}/{ticker}_drawdown.pth")
            joblib.dump(scaler, f"{settings.MODEL_STORAGE_PATH}/{ticker}_drawdown.pkl")
    except Exception as e:
        logger.error(f"Drawdown error {ticker}: {e}")
