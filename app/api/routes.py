# Файл: app/api/routes.py

import logging
from datetime import date, datetime, timedelta
from typing import List

import pandas as pd

# Сервисы
from app.services.services import get_historical_data
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

# Схемы Pydantic
from app import schemas

# Конфигурация
from app.core import plotting
from app.core.sizing import calculate_fixed_risk_position_size, calculate_kelly_criterion_position_size

# Бизнес-логика (Core)
from app.core.stoploss import get_stop_loss_suggestions
from app.db import models

#   Импорты из модулей приложения
# База данных
from app.db.database import SessionLocal

# ML
from app.ml.training import train_and_predict_lstm
from app.services.portfolio_service import PortfolioService

# Настройка логгера для этого модуля
logger = logging.getLogger(__name__)

# Создаем роутер вместо app
router = APIRouter()


# Dependency: Получение сессии БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Эндпоинты: Аналитика и Данные


@router.get("/tickers", response_model=List[schemas.TickerResponse], summary="Получить список отслеживаемых акций")
def get_tracked_tickers(db: Session = Depends(get_db)):
    tickers = db.query(models.TrackedTicker).order_by(models.TrackedTicker.ticker).all()
    return tickers


@router.get("/forecast/{ticker}", response_model=List[schemas.ForecastResponse], summary="Получить прогноз для тикера")
def get_forecast(ticker: str, db: Session = Depends(get_db)):
    forecasts = (
        db.query(models.Forecast)
        .filter(models.Forecast.ticker == ticker.upper())
        .order_by(models.Forecast.forecast_date)
        .all()
    )
    if not forecasts:
        raise HTTPException(status_code=404, detail="Прогноз не найден.")
    return forecasts


@router.get("/levels/{ticker}", response_model=List[schemas.KeyLevelZoneResponse], summary="Получить ключевые уровни")
def get_key_levels(ticker: str, db: Session = Depends(get_db)):
    levels = (
        db.query(models.KeyLevel)
        .filter(models.KeyLevel.ticker == ticker.upper())
        .order_by(models.KeyLevel.start_price.desc())
        .all()
    )
    if not levels:
        raise HTTPException(status_code=404, detail="Ключевые зоны не рассчитаны.")
    return levels


@router.get("/signals", response_model=List[schemas.SignalResponse], summary="Получить торговые сигналы")
def get_trading_signals(show_all: bool = False, db: Session = Depends(get_db)):
    query = db.query(models.TradingSignal)
    if not show_all:
        query = query.filter(models.TradingSignal.generated_at >= datetime.utcnow() - timedelta(hours=24))

    db_signals = query.order_by(models.TradingSignal.potential_profit_pct.desc()).all()

    response_signals = []
    today = date.today()
    for signal in db_signals:
        details = signal.details if signal.details else {}
        target_price = details.get("target_price") or details.get("price_in_14_days") or 0.0
        profit_pct = signal.potential_profit_pct or 0.0
        forecast_days = signal.forecast_days or 0

        target_date = today + timedelta(days=forecast_days)

        description = "Описание не сформировано."
        if signal.signal_type == "bullish_momentum":
            description = f"Прогнозируется рост. Цель ~{target_price:.2f} руб. (+{profit_pct:.2f}%) к {target_date.strftime('%d.%m.%Y')}."
        elif signal.signal_type == "support_bounce":
            support_start = details.get("support_zone_start") or 0.0
            description = (
                f"Отскок от поддержки ~{support_start:.2f}. Цель ~{target_price:.2f} руб. (+{profit_pct:.2f}%)."
            )
        elif signal.signal_type == "resistance_breakout":
            res_start = details.get("resistance_zone_start") or 0.0
            description = f"Пробой сопротивления ~{res_start:.2f}. Цель ~{target_price:.2f} руб. (+{profit_pct:.2f}%)."

        response_signals.append(
            {
                "id": signal.id,
                "ticker": signal.ticker,
                "signal_type": signal.signal_type,
                "potential_profit_pct": signal.potential_profit_pct,
                "details": details,
                "generated_at": signal.generated_at,
                "forecast_horizon_days": forecast_days,
                "target_date": target_date,
                "description": description,
            }
        )

    return response_signals


@router.get("/stoploss/{ticker}", response_model=List[schemas.StopLossSuggestion], summary="Рассчитать стоп-лосс")
def calculate_stop_loss(ticker: str, entry_price: float, db: Session = Depends(get_db)):
    ticker_upper = ticker.upper()
    figi = db.query(models.TrackedTicker.figi).filter(models.TrackedTicker.ticker == ticker_upper).scalar()
    if not figi:
        raise HTTPException(status_code=404, detail="Тикер не найден.")

    suggestions = get_stop_loss_suggestions(db, ticker_upper, figi, entry_price)
    if not suggestions:
        raise HTTPException(status_code=404, detail="Не удалось рассчитать стоп-лосс (мало данных).")
    return suggestions


@router.get("/darkpool", response_model=List[schemas.DarkPoolSignalResponse], summary="Сигналы Dark Pool")
def get_dark_pool_signals(db: Session = Depends(get_db)):
    signals = (
        db.query(models.DarkPoolSignal)
        .filter(models.DarkPoolSignal.signal_time >= datetime.utcnow() - timedelta(days=3))
        .order_by(models.DarkPoolSignal.signal_time.desc())
        .all()
    )
    return signals


@router.get("/intraday-signals", response_model=List[schemas.IntradaySignalResponse], summary="Интрадей сигналы")
def get_intraday_signals(db: Session = Depends(get_db)):
    signals = (
        db.query(models.IntradaySignal)
        .filter(models.IntradaySignal.generated_at >= datetime.utcnow() - timedelta(hours=4))
        .order_by(models.IntradaySignal.generated_at.desc())
        .all()
    )
    return signals


@router.post("/feedback", status_code=201, summary="Отправить отзыв на сигнал")
def post_feedback(feedback: schemas.FeedbackIn, db: Session = Depends(get_db)):
    db_signal = db.query(models.TradingSignal).filter(models.TradingSignal.id == feedback.signal_id).first()
    if not db_signal:
        raise HTTPException(status_code=404, detail="Сигнал не найден")

    db_feedback = (
        db.query(models.SignalFeedback).filter_by(signal_id=feedback.signal_id, user_id=feedback.user_id).first()
    )
    if db_feedback:
        db_feedback.reaction = feedback.reaction
    else:
        db_feedback = models.SignalFeedback(**feedback.dict())
        db.add(db_feedback)

    db.commit()
    return {"status": "success", "message": "Отзыв сохранен."}


@router.get("/stats", response_model=schemas.SignalStats, summary="Статистика сигналов")
def get_signal_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func as sql_func

    results = (
        db.query(models.SignalFeedback.reaction, sql_func.count(models.SignalFeedback.id))
        .group_by(models.SignalFeedback.reaction)
        .all()
    )
    stats = {"SUCCESS": 0, "FAIL": 0, "BREAKEVEN": 0}
    for reaction, count in results:
        if reaction in stats:
            stats[reaction] = count

    total_rated = sum(stats.values())
    success_rate = (stats["SUCCESS"] / total_rated * 100) if total_rated > 0 else 0

    return {
        "total_signals": total_rated,
        "success_count": stats["SUCCESS"],
        "fail_count": stats["FAIL"],
        "breakeven_count": stats["BREAKEVEN"],
        "success_rate": round(success_rate, 2),
    }


#   Эндпоинты: Калькуляторы


@router.post("/position-size", summary="Калькулятор размера позиции (Фикс. риск)")
def get_position_size(request: schemas.PositionSizeRequest, db: Session = Depends(get_db)):
    # TODO:
    lot_size = 1

    result = calculate_fixed_risk_position_size(
        request.account_size, request.risk_percent, request.entry_price, request.stop_loss_price, lot_size
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/position-size/kelly", summary="Калькулятор Келли")
def get_kelly_position_size(request: schemas.PositionSizeRequest, db: Session = Depends(get_db)):
    last_signal = (
        db.query(models.TradingSignal)
        .filter(models.TradingSignal.ticker == request.ticker.upper())
        .order_by(models.TradingSignal.generated_at.desc())
        .first()
    )
    if not last_signal:
        raise HTTPException(
            status_code=404, detail=f"Нет активных сигналов для {request.ticker}, невозможно определить стратегию."
        )

    strategy_name = last_signal.signal_type
    lot_size = 1

    result = calculate_kelly_criterion_position_size(
        db, request.account_size, request.entry_price, request.stop_loss_price, strategy_name, lot_size
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/forecast-metrics/{ticker}", response_model=schemas.ForecastMetricsResponse, summary="Метрики прогноза")
def get_forecast_metrics(ticker: str, db: Session = Depends(get_db)):
    metrics = db.query(models.ForecastMetrics).filter(models.ForecastMetrics.ticker == ticker.upper()).first()
    if not metrics:
        raise HTTPException(status_code=404, detail="Метрики не найдены.")
    return metrics


#   Эндпоинты: Портфолио


@router.post("/portfolio/create/{user_id}", summary="Создать портфель пользователя")
def create_user_portfolio(user_id: int, db: Session = Depends(get_db)):
    svc = PortfolioService(db)
    svc.create_portfolio(user_id)
    return {"status": "ok"}


@router.get("/portfolio/{user_id}", summary="Получить портфель")
def get_my_portfolio(user_id: int, db: Session = Depends(get_db)):
    svc = PortfolioService(db)
    portfolio = svc.get_portfolio_summary(user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Портфель не найден")
    return portfolio


class TradeRequest(schemas.PositionBase):
    ticker: str
    price: float
    direction: str = "LONG"
    amount: float = 50000


@router.post("/portfolio/trade/{user_id}", summary="Совершить сделку")
def execute_trade(user_id: int, req: TradeRequest, db: Session = Depends(get_db)):
    svc = PortfolioService(db)
    res = svc.open_position(user_id, req.ticker, req.price, req.direction, req.amount)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.post("/portfolio/close/{user_id}/{position_id}", summary="Закрыть позицию")
def close_trade(user_id: int, position_id: int, price: float, db: Session = Depends(get_db)):
    svc = PortfolioService(db)
    res = svc.close_position(user_id, position_id, price)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.post("/portfolio", response_model=schemas.PortfolioResponse, status_code=201, summary="Создать портфель (CRUD)")
def create_portfolio_crud(portfolio_data: schemas.PortfolioCreate, db: Session = Depends(get_db)):
    db_portfolio = db.query(models.Portfolio).filter(models.Portfolio.user_id == portfolio_data.user_id).first()
    if db_portfolio:
        raise HTTPException(status_code=409, detail="Портфель уже существует.")

    new_portfolio = models.Portfolio(
        user_id=portfolio_data.user_id,
        name=portfolio_data.name,
        initial_capital=portfolio_data.initial_capital,
        current_capital=portfolio_data.initial_capital,
    )
    db.add(new_portfolio)
    db.commit()
    db.refresh(new_portfolio)

    return {
        "id": new_portfolio.id,
        "user_id": new_portfolio.user_id,
        "initial_capital": new_portfolio.initial_capital,
        "current_capital": new_portfolio.current_capital,
        "open_positions": [],
        "total_pnl": 0.0,
        "total_pnl_percent": 0.0,
    }


#   Эндпоинты: Графики и Подписка


@router.get("/plots/forecast/{ticker}", summary="График прогноза")
def get_forecast_plot(ticker: str, db: Session = Depends(get_db)):
    ticker_upper = ticker.upper()
    figi = db.query(models.TrackedTicker.figi).filter(models.TrackedTicker.ticker == ticker_upper).scalar()
    if not figi:
        raise HTTPException(status_code=404, detail="Тикер не найден.")

    logger.info(f"[{ticker_upper}] Запрос графика. Загрузка истории...")
    historical_data = get_historical_data(figi, days=365 * 5)

    if historical_data.empty or len(historical_data) < 252:
        raise HTTPException(status_code=404, detail="Недостаточно данных.")

    try:
        (
            model,
            scaler,
            final_predictions,
            final_upper_bound,
            final_lower_bound,
            profitability_metrics,
            bear_scenario_preds,
            multi_horizon_forecasts,
        ) = train_and_predict_lstm(stock_data=historical_data.copy(), ticker=ticker_upper, future_predictions=30)

        if final_predictions is None or len(final_predictions) == 0:
            raise ValueError("ML модель вернула пустой прогноз.")

    except Exception as e:
        logger.error(f"Ошибка ML пайплайна для графика {ticker_upper}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка расчета: {e}")

    forecast_dates = pd.to_datetime(
        pd.date_range(start=historical_data["time"].iloc[-1].date() + timedelta(days=1), periods=30)
    )
    forecast_df = pd.DataFrame({"forecast_date": forecast_dates, "forecast_value": final_predictions})

    try:
        image_buffer = plotting.plot_forecast(
            historical_data=historical_data,
            forecast_data=forecast_df,
            upper_bound=final_upper_bound,
            lower_bound=final_lower_bound,
            ticker=ticker_upper,
            bear_scenario=bear_scenario_preds,
        )
    except Exception as e:
        logger.error(f"Ошибка генерации картинки: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка отрисовки.")

    return StreamingResponse(image_buffer, media_type="image/png")


@router.post("/subscribe", status_code=201, summary="Подписаться на уведомления")
def subscribe_user(subscriber_data: schemas.SubscriberCreate, db: Session = Depends(get_db)):
    existing = db.query(models.Subscriber).filter(models.Subscriber.user_id == subscriber_data.user_id).first()
    if existing:
        return {"status": "ok", "message": "Уже подписан."}

    new_sub = models.Subscriber(user_id=subscriber_data.user_id, username=subscriber_data.username)
    db.add(new_sub)
    try:
        db.commit()
    except Exception:
        db.rollback()
        return {"status": "ok", "message": "Ошибка или уже подписан."}

    return {"status": "success", "message": "Подписка оформлена."}


@router.post("/unsubscribe", status_code=200, summary="Отписаться")
def unsubscribe_user(subscriber_data: schemas.SubscriberCreate, db: Session = Depends(get_db)):
    sub = db.query(models.Subscriber).filter(models.Subscriber.user_id == subscriber_data.user_id).first()
    if not sub:
        return {"status": "ok", "message": "Не был подписан."}

    db.delete(sub)
    db.commit()
    return {"status": "success", "message": "Подписка отменена."}
