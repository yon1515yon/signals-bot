# Файл: app/worker/tasks.py

import json
import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import requests
import torch
from app.ml.market_analysis import get_market_regime_cached

# === Сервисы ===
from app.services.services import CandleInterval, get_all_russian_stocks, get_historical_data, get_order_book
from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger
from sqlalchemy import create_engine, text

# === Конфигурация и Константы ===
from app.config import settings
from app.constants import LIQUID_TICKERS
from app.core.darkpool import detect_dark_pool_footprints
from app.core.ghost import detect_ghost_accumulation
from app.core.intraday import determine_global_bias, find_ema_cross_short, find_rsi_bounce_long
from app.core.levels import find_key_level_zones
from app.core.orderbook_analysis import find_walls, validate_breakout
from app.core.scanner import run_scanner
from app.db import models

# === База Данных ===
from app.db.database import SessionLocal
from app.ml.meta_model import train_meta_model

# === ML & Core Logic ===
from app.ml.training import (
    run_walk_forward_validation,
    train_and_predict_lstm,
    train_drawdown_model,
    train_global_base_model,
)
from app.ml.tuning import run_tuning_session  
from app.services.sentiment import get_sentiment_score

logger = get_task_logger(__name__)


os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)

engine = create_engine(settings.DATABASE_URL)

celery_app = Celery("neurovision", broker=settings.REDIS_URL, backend=settings.REDIS_URL, include=["app.worker.tasks"])

# Настройки Celery
celery_app.conf.task_default_queue = "default"
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,  # Повторить задачу, если воркер упал во время выполнения
)

# === Маршрутизация (Heavy vs Light) ===
celery_app.conf.task_routes = {
    # Тяжелые ML задачи -> очередь ml_training
    "app.worker.tasks.run_global_model_training": {"queue": "ml_training"},
    "app.worker.tasks.full_train_model": {"queue": "ml_training"},
    "app.worker.tasks.run_meta_model_training": {"queue": "ml_training"},
    "app.worker.tasks.train_intraday_model": {"queue": "ml_training"},
    "app.worker.tasks.schedule_full_retrains": {"queue": "ml_training"},
    "app.worker.tasks.tune_hyperparameters": {"queue": "ml_training"},  # <--- Optuna
    # Легкие задачи -> default
    "app.worker.tasks.schedule_order_book_scan": {"queue": "default"},
    "app.worker.tasks.scan_order_book_for_ticker": {"queue": "default"},
    "app.worker.tasks.run_signal_scanner": {"queue": "default"},
    "app.worker.tasks.scan_intraday_for_ticker": {"queue": "default"},
    "app.worker.tasks.send_notification_to_user": {"queue": "default"},
    "app.worker.tasks.schedule_intraday_scan": {"queue": "default"},
    "app.worker.tasks.update_portfolio_prices": {"queue": "default"},
}

# === Расписание (Beat) ===
celery_app.conf.beat_schedule = {
    # 1. СКАНИРОВАНИЕ СТАКАНА (Каждую минуту)
    "scan-order-book-breakouts": {
        "task": "app.worker.tasks.schedule_order_book_scan",
        "schedule": crontab(minute="*"),
    },
    # 2. ИНТРАДЕЙ СКАНЕР (Каждые 5 мин в рабочее время)
    "run-intraday-scanner-frequently": {
        "task": "app.worker.tasks.schedule_intraday_scan",
        "schedule": crontab(minute="*/5", hour="7-18", day_of_week="mon-fri"),
    },
    # 3. СРЕДНЕСРОЧНЫЙ СКАНЕР (Каждый час)
    "run-medium-term-scanner-hourly": {"task": "app.worker.tasks.run_signal_scanner", "schedule": crontab(minute="0")},
    # 4. ОБНОВЛЕНИЕ СПИСКА АКЦИЙ (Раз в сутки)
    "discover-stocks-daily": {
        "task": "app.worker.tasks.discover_and_track_stocks",
        "schedule": crontab(hour=0, minute=1),
    },
    # 5. ГЛОБАЛЬНОЕ ОБУЧЕНИЕ (Раз в неделю, воскресенье)
    "train-global-model-weekly": {
        "task": "app.worker.tasks.run_global_model_training",
        "schedule": crontab(day_of_week="sunday", hour=1, minute=0),
    },
    # 6. МАССОВОЕ ПЕРЕОБУЧЕНИЕ (Каждую ночь)
    "retrain-price-models-daily": {
        "task": "app.worker.tasks.schedule_full_retrains",
        "schedule": crontab(hour=3, minute=0),
    },
    # 7. МЕТА-МОДЕЛЬ (Утром)
    "train-meta-model-daily": {
        "task": "app.worker.tasks.run_meta_model_training",
        "schedule": crontab(hour=6, minute=0),
    },
    # 8. ИНТРАДЕЙ МОДЕЛИ (В течение дня)
    "train-intraday-models-market-hours": {
        "task": "app.worker.tasks.schedule_intraday_train_batch",
        "schedule": crontab(hour="7-16", minute=15, day_of_week="mon-fri"),
    },
    # 9. DRAWDOWN MODELS (Еженедельно)
    "retrain-drawdown-models-weekly": {
        "task": "app.worker.tasks.schedule_drawdown_model_retrains",
        "schedule": crontab(day_of_week="sunday", hour=5, minute=0),
    },
    # 10. КЛЮЧЕВЫЕ УРОВНИ (Ночью)
    "recalculate-key-levels-daily": {
        "task": "app.worker.tasks.schedule_key_level_recalculation",
        "schedule": crontab(hour=2, minute=0),
    },
    # 11. GLOBAL BIAS (Утром перед торгами)
    "update-global-bias-daily": {
        "task": "app.worker.tasks.schedule_global_bias_update",
        "schedule": crontab(hour=6, minute=30, day_of_week="mon-fri"),
    },
    # 12. СТАТИСТИКА СТРАТЕГИЙ (Ночью)
    "recalculate-strategy-stats-daily": {
        "task": "app.worker.tasks.recalculate_strategy_performance",
        "schedule": crontab(hour=4, minute=0),
    },
    # 13. GHOST ACTIVITY (Каждые 15 мин)
    "scan-ghost-activity-15min": {
        "task": "app.worker.tasks.schedule_ghost_scan",
        "schedule": crontab(minute="*/15", hour="7-18", day_of_week="mon-fri"),
    },
    # 14. ОБНОВЛЕНИЕ ЦЕН ПОРТФЕЛЯ (Часто)
    "update-portfolio-prices": {"task": "app.worker.tasks.update_portfolio_prices", "schedule": crontab(minute="*/5")},
}

# ===========================================
# ЗАДАЧИ (TASKS)
# ===========================================


@celery_app.task(bind=True, time_limit=7200)
def run_global_model_training(self):
    """Задача на обучение материнской модели."""
    logger.warning("BEAT: Запуск обучения GLOBAL BASE MODEL...")

    ticker_map = {}
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT ticker, figi FROM tracked_tickers WHERE ticker IN :tickers"),
            {"tickers": tuple(LIQUID_TICKERS)},
        ).fetchall()
        for r in rows:
            ticker_map[r.ticker] = r.figi

    if not ticker_map:
        logger.error("GLOBAL TASK: Не найдены FIGI для ликвидных акций!")
        return

    train_global_base_model(list(ticker_map.keys()), ticker_map)
    logger.warning("BEAT: Обучение GLOBAL BASE MODEL завершено.")


@celery_app.task(name="app.worker.tasks.tune_hyperparameters")
def tune_hyperparameters(ticker: str, figi: str):
    """AutoML: Подбор гиперпараметров через Optuna."""
    logger.info(f"[{ticker}] Запуск AutoML тюнинга...")
    data = get_historical_data(figi, days=365 * 2)  # Берем 2 года
    if data.empty:
        return

    best_params = run_tuning_session(ticker, data, n_trials=15)

    # Сохраняем параметры в JSON рядом с моделью
    params_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}_params.json"
    try:
        with open(params_path, "w") as f:
            json.dump(best_params, f)
        logger.info(f"[{ticker}] Параметры сохранены в {params_path}")
    except Exception as e:
        logger.error(f"[{ticker}] Ошибка сохранения параметров: {e}")


@celery_app.task
def recalculate_strategy_performance():
    logger.info("PERFORMANCE: Пересчет статистики...")
    db = SessionLocal()
    try:
        feedback_query = (
            db.query(
                models.TradingSignal.signal_type,
                models.TradingSignal.potential_profit_pct,
                models.SignalFeedback.reaction,
            )
            .join(models.SignalFeedback, models.TradingSignal.id == models.SignalFeedback.signal_id)
            .all()
        )

        if not feedback_query:
            return

        df = pd.DataFrame(feedback_query, columns=["strategy_name", "profit_pct", "reaction"])
        grouped = df.groupby("strategy_name")

        for name, group in grouped:
            total_trades = len(group)
            if total_trades < 5:
                continue
            success_trades = group[group["reaction"] == "SUCCESS"]
            win_rate = len(success_trades) / total_trades
            avg_profit = success_trades["profit_pct"].mean()
            # Упрощенная оценка убытка (т.к. реальный убыток не всегда известен точно)
            avg_loss = -abs(group["profit_pct"].mean() / 2)

            stmt = text(
                """
                INSERT INTO strategy_performance (strategy_name, win_rate, avg_profit_pct, avg_loss_pct, trades_count)
                VALUES (:name, :wr, :ap, :al, :tc)
                ON CONFLICT (strategy_name) DO UPDATE SET
                    win_rate = EXCLUDED.win_rate,
                    avg_profit_pct = EXCLUDED.avg_profit_pct,
                    avg_loss_pct = EXCLUDED.avg_loss_pct,
                    trades_count = EXCLUDED.trades_count,
                    last_calculated_at = NOW();
            """
            )
            db.execute(stmt, {"name": name, "wr": win_rate, "ap": avg_profit, "al": avg_loss, "tc": total_trades})
        db.commit()
    except Exception as e:
        logger.error(f"PERFORMANCE Error: {e}")
        db.rollback()
    finally:
        db.close()


@celery_app.task(autoretry_for=(Exception,), retry_kwargs={"max_retries": 3}, retry_backoff=True)
def send_notification_to_user(user_id: int, message_text: str):
    """Отправляет уведомление в Telegram."""
    try:
        token = settings.TELEGRAM_BOT_TOKEN
        if not token:
            logger.warning("Telegram Token не установлен.")
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": user_id, "text": message_text, "parse_mode": "HTML"}

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info(f"Уведомление отправлено пользователю {user_id}")
    except Exception as e:
        logger.error(f"Не удалось отправить уведомление {user_id}: {e}")
        raise e


@celery_app.task
def discover_and_track_stocks():
    logger.info("BEAT: Обнаружение акций...")
    stocks = get_all_russian_stocks()
    with engine.connect() as connection:
        with connection.begin():
            for stock in stocks:
                connection.execute(
                    text(
                        """
                    INSERT INTO tracked_tickers (ticker, figi, name) VALUES (:ticker, :figi, :name) ON CONFLICT (ticker) DO NOTHING;
                """
                    ),
                    stock,
                )
    logger.warning(f"BEAT: Обнаружение завершено. Акций: {len(stocks)}.")


@celery_app.task
def schedule_full_retrains():
    with engine.connect() as connection:
        tickers = connection.execute(text("SELECT ticker, figi, name FROM tracked_tickers")).fetchall()
        for ticker, figi, name in tickers:
            full_train_model.delay(ticker, figi, name)


@celery_app.task(
    bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 3}, retry_backoff=True, rate_limit="2/s"
)
def full_train_model(self, ticker: str, figi: str, name: str):
    """Полный цикл обучения модели для конкретного тикера."""
    logger.info(f"[{ticker}] Запуск Fine-Tuning...")
    try:
        stock_data = get_historical_data(figi, days=365 * 4)
        if stock_data.empty or len(stock_data) < 300:
            logger.info(f"[{ticker}] Мало данных. Пропуск.")
            return

        (
            model,
            scaler,
            final_predictions,
            final_upper_bound,
            final_lower_bound,
            _,
            bear_scenario_preds,
            multi_horizon_forecasts,
        ) = train_and_predict_lstm(stock_data, ticker=ticker, future_predictions=30)

        if model is None or final_predictions is None:
            return

        # Сохраняем веса
        torch.save(model.state_dict(), f"{settings.MODEL_STORAGE_PATH}/{ticker}.pth")
        joblib.dump(scaler, f"{settings.MODEL_STORAGE_PATH}/{ticker}.pkl")

        sentiment_score, _ = get_sentiment_score(ticker, name)

        # WFA (Backtesting)
        wfa_metrics = {}
        try:
            wfa_metrics = run_walk_forward_validation(stock_data.copy(), ticker, window_size=252, step_size=30)
        except Exception as e:
            logger.error(f"[{ticker}] WFA Ошибка: {e}")

        update_db_with_full_forecast(
            ticker,
            final_predictions,
            final_upper_bound,
            final_lower_bound,
            wfa_metrics,
            multi_horizon_forecasts,
            bear_scenario_preds,
            sentiment_score,
        )
        logger.info(f"[{ticker}] Обучение завершено.")

        # После успешного обучения запускаем Optuna тюнинг на будущее (в отдельной задаче)
        # tune_hyperparameters.delay(ticker, figi)

    except Exception as e:
        logger.error(f"[{ticker}] Ошибка обучения: {e}")
        raise e


@celery_app.task(
    bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 3}, retry_backoff=True, rate_limit="5/s"
)
def update_forecast(self, ticker: str, figi: str, name: str):
    """
    Быстрое обновление прогноза (Inference Only).
    Загружает сохраненную модель и делает прогноз на свежих данных.
    """
    logger.info(f"[{ticker}] Обновление прогноза (Inference)...")

    try:
        model_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}.pth"
        scaler_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.info(f"[{ticker}] Модель не найдена. Запуск полного обучения.")
            full_train_model.delay(ticker, figi, name)
            return

        stock_data = get_historical_data(figi, days=90)

        # Проверка достаточности данных
        from app.constants import ML_CONFIG

        if stock_data.empty or len(stock_data) < ML_CONFIG["TRAIN_WINDOW"] + 10:
            logger.warning(f"[{ticker}] Недостаточно свежих данных для обновления.")
            return

        from app.ml.training import prepare_features_and_scale

        processed_data, data_normalized, scaler, feature_columns, output_columns = prepare_features_and_scale(
            stock_data
        )

        if data_normalized is None:
            return

        from app.ml.architecture import LSTMTransformerModel

        input_size = len(feature_columns)
        output_size = len(output_columns)

        # Пытаемся загрузить тюнингованные параметры, если есть
        params_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}_params.json"
        # Дефолтные
        current_params = ML_CONFIG.copy()

        if os.path.exists(params_path):
            try:
                with open(params_path, "r") as f:
                    tuned = json.load(f)
                    # Маппинг ключей
                    if "hidden_size" in tuned:
                        current_params["HIDDEN_SIZE"] = tuned["hidden_size"]
                    if "num_layers" in tuned:
                        current_params["NUM_LAYERS"] = tuned["num_layers"]
                    if "dropout" in tuned:
                        current_params["DROPOUT"] = tuned["dropout"]
            except Exception:
                pass

        model = LSTMTransformerModel(
            input_size=input_size,
            hidden_size=current_params["HIDDEN_SIZE"],
            num_layers=current_params["NUM_LAYERS"],
            n_head=ML_CONFIG["N_HEAD"],
            dropout=current_params["DROPOUT"],
            output_size=output_size,
        )

        model.load_state_dict(torch.load(model_path))
        model.eval()

        train_window = ML_CONFIG["TRAIN_WINDOW"]
        last_sequence = data_normalized[-train_window:]

        from app.ml.prediction import predict_with_uncertainty

        mean_preds, upper, lower = predict_with_uncertainty(
            model, scaler, last_sequence, processed_data, feature_columns, output_columns, future_predictions=30
        )

        sentiment_score, _ = get_sentiment_score(ticker, name)

        update_db_with_full_forecast(ticker, mean_preds, upper, lower, None, None, None, sentiment_score)

        logger.info(f"[{ticker}] Прогноз успешно обновлен.")

    except Exception as e:
        logger.error(f"[{ticker}] Ошибка update_forecast: {e}")
        raise e


def update_db_with_full_forecast(
    ticker: str,
    forecast_values: list[float],
    upper_bounds: list[float],
    lower_bounds: list[float],
    profitability_metrics: dict,
    multi_horizon: dict,
    bear_scenario: dict | None,
    sentiment_score: float | None,
) -> None:
    """Вспомогательная функция записи в БД."""
    with engine.connect() as connection:
        with connection.begin():
            connection.execute(text("DELETE FROM forecasts WHERE ticker = :ticker"), {"ticker": ticker})
            for i, value in enumerate(forecast_values):
                connection.execute(
                    text(
                        """
                    INSERT INTO forecasts (ticker, forecast_date, forecast_value, upper_bound, lower_bound) 
                    VALUES (:ticker, current_date + interval '1 day' * :day_offset, :value, :upper, :lower)
                """
                    ),
                    {
                        "ticker": ticker,
                        "day_offset": i + 1,
                        "value": float(value),
                        "upper": float(upper_bounds[i]),
                        "lower": float(lower_bounds[i]),
                    },
                )

            connection.execute(
                text(
                    """
                INSERT INTO forecast_metrics (ticker, backtest_metrics, multi_horizon_forecast, bear_scenario_forecast)
                VALUES (:ticker, :backtest, :multi_horizon, :bear_scenario)
                ON CONFLICT (ticker) DO UPDATE SET backtest_metrics=EXCLUDED.backtest_metrics, last_calculated_at=NOW()
            """
                ),
                {
                    "ticker": ticker,
                    "backtest": json.dumps(profitability_metrics),
                    "multi_horizon": json.dumps(multi_horizon),
                    "bear_scenario": json.dumps(bear_scenario) if bear_scenario else None,
                },
            )

            if sentiment_score is not None:
                connection.execute(
                    text(
                        "UPDATE tracked_tickers SET sentiment_score = :sentiment, last_updated_at = NOW() WHERE ticker = :ticker"
                    ),
                    {"ticker": ticker, "sentiment": sentiment_score},
                )


@celery_app.task
def run_signal_scanner():
    logger.info("SCANNER: Запуск...")
    db = SessionLocal()
    try:
        market_regime = get_market_regime_cached(db, None)
        existing_signals_query = db.query(models.TradingSignal.ticker).all()
        existing_tickers = {ticker for (ticker,) in existing_signals_query}

        num_signals_found = run_scanner(db, market_regime)

        if num_signals_found > 0:
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            fresh_signals = (
                db.query(models.TradingSignal).filter(models.TradingSignal.generated_at > one_hour_ago).all()
            )

            new_signals = [s for s in fresh_signals if s.ticker not in existing_tickers]

            if new_signals:
                logger.info(f"SCANNER: Найдено {len(new_signals)} новых сигналов.")
                subscribers_ids = [row[0] for row in db.query(models.Subscriber.user_id).all()]

                for sig in new_signals:
                    details = sig.details if isinstance(sig.details, dict) else json.loads(sig.details)
                    msg = (
                        f"🔔 <b>Новый Сигнал!</b>\n\n"
                        f"<b>Тикер:</b> {sig.ticker}\n"
                        f"<b>Тип:</b> {sig.signal_type}\n"
                        f"<b>WinRate:</b> {details.get('model_win_rate', 'N/A')}%\n"
                    )
                    for user_id in subscribers_ids:
                        send_notification_to_user.delay(user_id, msg)
    finally:
        db.close()


@celery_app.task
def schedule_key_level_recalculation():
    with engine.connect() as connection:
        tickers = connection.execute(text("SELECT ticker, figi FROM tracked_tickers")).fetchall()
        for ticker, figi in tickers:
            recalculate_levels_for_ticker.delay(ticker, figi)


@celery_app.task(bind=True, rate_limit="10/s")
def recalculate_levels_for_ticker(self, ticker: str, figi: str):
    try:
        hist = get_historical_data(figi, days=365)
        if hist.empty:
            return
        current_price = hist.iloc[-1]["close"]
        order_book = get_order_book(figi)
        zones = find_key_level_zones(hist, order_book, current_price)

        if not zones:
            return

        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text("DELETE FROM key_levels WHERE ticker = :ticker"), {"ticker": ticker})
                for z in zones:
                    connection.execute(
                        text(
                            "INSERT INTO key_levels (ticker, start_price, end_price, level_type, strength, intensity, last_calculated_at) VALUES (:ticker, :start_price, :end_price, :level_type, :strength, :intensity, NOW())"
                        ),
                        {"ticker": ticker, **z},
                    )
    except Exception as e:
        logger.error(f"Level calc error {ticker}: {e}")


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


@celery_app.task
def schedule_drawdown_model_retrains():
    with engine.connect() as connection:
        tickers = connection.execute(text("SELECT ticker, figi FROM tracked_tickers")).fetchall()
        for ticker, figi in tickers:
            train_and_save_drawdown_model.delay(ticker, figi)


@celery_app.task
def schedule_dark_pool_scans():
    with engine.connect() as connection:
        tickers = connection.execute(text("SELECT ticker, figi FROM tracked_tickers")).fetchall()
        for ticker, figi in tickers:
            scan_dark_pool_for_ticker.delay(ticker, figi)


@celery_app.task(rate_limit="1/s")
def scan_dark_pool_for_ticker(ticker: str, figi: str):
    try:
        hourly_data = get_historical_data(figi, days=14, interval=CandleInterval.CANDLE_INTERVAL_HOUR)
        signals = detect_dark_pool_footprints(hourly_data)
        if not signals:
            return

        with engine.connect() as connection:
            with connection.begin():
                for signal in signals:
                    stmt = text(
                        """
                        INSERT INTO dark_pool_signals (ticker, signal_time, price, volume, avg_volume, activity_type, description)
                        VALUES (:ticker, :timestamp, :price, :volume, :avg_volume, :activity_type, :description)
                        ON CONFLICT (ticker, signal_time) DO NOTHING;
                    """
                    )
                    connection.execute(stmt, {"ticker": ticker, **signal})
    except Exception as e:
        logger.error(f"Darkpool error {ticker}: {e}")


@celery_app.task(rate_limit="2/s")
def update_bias_for_ticker(ticker: str, figi: str):
    with SessionLocal() as db:
        try:
            forecast = db.execute(
                text("SELECT forecast_value FROM forecasts WHERE ticker = :ticker ORDER BY forecast_date LIMIT 7"),
                {"ticker": ticker},
            ).fetchall()
            if not forecast:
                return
            hist = get_historical_data(figi, days=1)
            if hist.empty:
                return

            f_df = pd.DataFrame(forecast, columns=["forecast_value"])
            # Получаем сентимент
            sentiment = db.execute(
                text("SELECT sentiment_score FROM tracked_tickers WHERE ticker = :t"), {"t": ticker}
            ).scalar()

            bias = determine_global_bias(f_df, hist.iloc[-1]["close"], sentiment or 0.0)
            db.execute(
                text("UPDATE tracked_tickers SET global_bias = :bias WHERE ticker = :ticker"),
                {"bias": bias, "ticker": ticker},
            )
            db.commit()
        except Exception:
            db.rollback()


@celery_app.task
def schedule_intraday_scan():
    with SessionLocal() as db:
        # Сканируем только те, где есть направленный тренд
        tickers = db.execute(text("SELECT ticker, figi FROM tracked_tickers WHERE global_bias != 'Neutral'")).fetchall()
        for ticker, figi in tickers:
            scan_intraday_for_ticker.delay(ticker, figi)


@celery_app.task(rate_limit="3/s")
def scan_intraday_for_ticker(ticker: str, figi: str):
    with SessionLocal() as db:
        try:
            bias = db.execute(
                text("SELECT global_bias FROM tracked_tickers WHERE ticker = :ticker"), {"ticker": ticker}
            ).scalar()
            metrics_json = db.execute(
                text("SELECT multi_horizon_forecast FROM forecast_metrics WHERE ticker = :t"), {"t": ticker}
            ).scalar()

            neural_trend = "Neutral"
            if metrics_json:
                meta = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json)
                if "intraday" in meta:
                    # Проверяем свежесть (3 часа)
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
                db.execute(
                    text(
                        "INSERT INTO intraday_signals (ticker, signal_type, entry_price, stop_loss, take_profit, strategy, global_bias, generated_at) VALUES (:ticker, :st, :ep, :sl, :tp, :strat, :gb, NOW())"
                    ),
                    {
                        "ticker": ticker,
                        "st": signal["signal_type"],
                        "ep": signal["entry_price"],
                        "sl": signal["stop_loss"],
                        "tp": signal["take_profit"],
                        "strat": signal["strategy"],
                        "gb": bias,
                    },
                )
                db.commit()
        except Exception as e:
            logger.error(f"Intraday error {ticker}: {e}")
            db.rollback()


@celery_app.task
def schedule_order_book_scan():
    with SessionLocal() as db:
        tickers = db.execute(text("SELECT ticker, figi FROM tracked_tickers WHERE global_bias != 'Neutral'")).fetchall()
        for ticker, figi in tickers:
            scan_order_book_for_ticker.delay(ticker, figi)


@celery_app.task(rate_limit="5/s")
def scan_order_book_for_ticker(ticker: str, figi: str):
    with SessionLocal() as db:
        try:
            order_book = get_order_book(figi)
            if not order_book.get("bids") or not order_book.get("asks"):
                return

            old_walls = db.execute(
                text("SELECT price, wall_type FROM order_book_walls WHERE ticker = :ticker"), {"ticker": ticker}
            ).fetchall()
            old_walls_map = {row.wall_type: row.price for row in old_walls}

            best_bid = order_book["bids"][0]["price"]
            # best_ask = order_book['asks'][0]['price']

            # Пробой вверх
            if "ASK" in old_walls_map and best_bid > old_walls_map["ASK"]:
                val = validate_breakout(ticker, old_walls_map["ASK"], "UP", db)
                if val["score"] >= 70:
                    db.execute(
                        text(
                            "INSERT INTO intraday_signals (ticker, signal_type, entry_price, strategy, context_score, generated_at) VALUES (:t, 'LONG', :p, 'OB_Breakout', :s, NOW())"
                        ),
                        {"t": ticker, "p": old_walls_map["ASK"], "s": val["score"]},
                    )
                    db.execute(
                        text("DELETE FROM order_book_walls WHERE ticker = :t AND wall_type = 'ASK'"), {"t": ticker}
                    )

            # Поиск новых стенок
            hist_data = get_historical_data(figi, days=30)
            avg_vol = hist_data["volume"].mean() if not hist_data.empty else 1
            new_walls = find_walls(order_book, avg_vol)

            for wall_type, wall_data in new_walls.items():
                if wall_data:
                    w_type = "BID" if "bid" in wall_type else "ASK"
                    db.execute(
                        text(
                            "INSERT INTO order_book_walls (ticker, price, volume, wall_type, detected_at) VALUES (:t, :p, :v, :wt, NOW()) ON CONFLICT (ticker, wall_type) DO UPDATE SET price=EXCLUDED.price, volume=EXCLUDED.volume, detected_at=NOW()"
                        ),
                        {"t": ticker, "p": wall_data["price"], "v": wall_data["quantity"], "wt": w_type},
                    )
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"OB Scan error {ticker}: {e}")


@celery_app.task(name="app.worker.tasks.run_meta_model_training")
def run_meta_model_training():
    logger.info("META: Обучение...")
    db = SessionLocal()
    try:
        results = db.execute(text("SELECT backtest_metrics FROM forecast_metrics")).fetchall()
        all_trades = []
        for row in results:
            if row[0]:
                metrics = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                if "meta_data" in metrics:
                    all_trades.extend(metrics["meta_data"])

        if len(all_trades) > 50:
            train_meta_model(all_trades)
    finally:
        db.close()


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

        with engine.connect() as connection:
            with connection.begin():
                curr = connection.execute(
                    text("SELECT multi_horizon_forecast FROM forecast_metrics WHERE ticker = :t"), {"t": ticker}
                ).scalar()
                meta = curr if curr else {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
                meta["intraday"] = intraday_json
                connection.execute(
                    text("UPDATE forecast_metrics SET multi_horizon_forecast = :m WHERE ticker = :t"),
                    {"m": json.dumps(meta), "t": ticker},
                )
    except Exception as e:
        logger.error(f"Intraday train error {ticker}: {e}")


@celery_app.task
def schedule_intraday_train_batch():
    with engine.connect() as connection:
        tickers = connection.execute(
            text("SELECT ticker, figi FROM tracked_tickers WHERE ticker IN :liquids"),
            {"liquids": tuple(LIQUID_TICKERS)},
        ).fetchall()
        for t, f in tickers:
            train_intraday_model.delay(t, f)


@celery_app.task(name="app.tasks.schedule_ghost_scan")
def schedule_ghost_scan():
    with SessionLocal() as db:
        tickers = db.execute(
            text("SELECT ticker, figi FROM tracked_tickers WHERE ticker IN :liquids"),
            {"liquids": tuple(LIQUID_TICKERS)},
        ).fetchall()
        for t, f in tickers:
            scan_ghost_activity.delay(t, f)


@celery_app.task(name="app.worker.tasks.scan_ghost_activity", rate_limit="2/s")
def scan_ghost_activity(ticker: str, figi: str):
    with SessionLocal() as db:
        try:
            data = get_historical_data(figi, days=2, interval=CandleInterval.CANDLE_INTERVAL_5_MIN)
            signals = detect_ghost_accumulation(data)
            if not signals:
                return

            fresh = []
            for s in signals:
                sig_time = s["signal_time"].replace(tzinfo=None)
                if (datetime.utcnow() - sig_time).total_seconds() < 3600 * 4:
                    fresh.append(s)

            for s in fresh:
                stmt = text(
                    """
                    INSERT INTO dark_pool_signals (ticker, signal_time, price, volume, avg_volume, activity_type, description)
                    VALUES (:t, :ts, :p, :v, :av, :at, :d)
                    ON CONFLICT (ticker, signal_time) DO NOTHING
                """
                )
                db.execute(
                    stmt,
                    {
                        "t": ticker,
                        "ts": s["signal_time"],
                        "p": s["price"],
                        "v": s["volume"],
                        "av": s["avg_volume"],
                        "at": s["activity_type"],
                        "d": s["description"],
                    },
                )

                subs = [r[0] for r in db.query(models.Subscriber.user_id).all()]
                for uid in subs:
                    send_notification_to_user.delay(uid, f"👻 Ghost Activity {ticker}: {s['description']}")
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Ghost scan error {ticker}: {e}")


@celery_app.task(name="app.worker.tasks.update_portfolio_prices")
def update_portfolio_prices():
    db = SessionLocal()
    try:
        positions = db.query(models.Position).filter(models.Position.status == "OPEN").all()
        if not positions:
            return

        unique_figis = {p.figi for p in positions if p.figi}
        prices = {}
        for figi in unique_figis:
            df = get_historical_data(figi, days=1, interval=CandleInterval.CANDLE_INTERVAL_1_MIN)
            if not df.empty:
                prices[figi] = df.iloc[-1]["close"]

        for p in positions:
            if p.figi in prices:
                p.current_price = float(prices[p.figi])
        db.commit()
    finally:
        db.close()
