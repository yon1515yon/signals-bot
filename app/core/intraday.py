# Файл: app/intraday.py
import pandas as pd


def determine_global_bias(daily_forecast_df: pd.DataFrame, current_price: float, sentiment_score: float) -> str:
    """Определяет глобальное настроение для акции (только Long, только Short, или ничего)."""
    if daily_forecast_df.empty:
        return "Neutral"

    sentiment_score = sentiment_score or 0.0

    # Тренд прогноза на 7 дней
    price_in_7_days = daily_forecast_df.iloc[6]["forecast_value"]
    trend_strength = (price_in_7_days / current_price - 1) * 100

    if trend_strength > 1.5 and sentiment_score > 0.1:
        return "Bullish"  # Разрешаем Long
    elif trend_strength < -1.5 and sentiment_score < -0.1:
        return "Bearish"  # Разрешаем Short
    else:
        return "Neutral"  # Торговля запрещена


def find_rsi_bounce_long(intraday_data: pd.DataFrame, neural_trend: str = "Neutral") -> dict | None:
    """
    Long по RSI (5 min), только если Нейросеть (1 hour) тоже смотрит вверх.
    """
    if len(intraday_data) < 20:
        return None

    # --- ФИЛЬТР НЕЙРОСЕТИ ---
    # Если нейросеть говорит "ВНИЗ" или молчит, мы не лонгуем (даже если RSI красивый)
    if neural_trend != "UP":
        return None
    # ------------------------

    intraday_data.ta.rsi(length=14, append=True)
    last = intraday_data.iloc[-1]
    prev = intraday_data.iloc[-2]

    # RSI пересекает 30 снизу вверх
    if prev["RSI_14"] < 30 and last["RSI_14"] >= 30:
        entry = last["close"]
        stop = intraday_data["low"].tail(10).min() * 0.998
        tp = entry + (entry - stop) * 2

        return {
            "signal_type": "SCALP_LONG",
            "entry_price": float(entry),
            "stop_loss": float(stop),
            "take_profit": float(tp),
            "strategy": "RSI_Bounce + AI_Confirm",
        }
    return None


def find_ema_cross_short(intraday_data: pd.DataFrame, neural_trend: str = "Neutral") -> dict | None:
    """
    Short по EMA, только если Нейросеть смотрит вниз.
    """
    if len(intraday_data) < 55:
        return None

    # --- ФИЛЬТР НЕЙРОСЕТИ ---
    if neural_trend != "DOWN":
        return None
    # ------------------------

    intraday_data.ta.ema(length=20, append=True)
    intraday_data.ta.ema(length=50, append=True)
    last = intraday_data.iloc[-1]
    prev = intraday_data.iloc[-2]

    # Dead Cross (20 пересекает 50 вниз)
    if prev["EMA_20"] > prev["EMA_50"] and last["EMA_20"] <= last["EMA_50"]:
        entry = last["close"]
        stop = intraday_data["high"].tail(10).max() * 1.002
        tp = entry - (stop - entry) * 2

        return {
            "signal_type": "SCALP_SHORT",
            "entry_price": float(entry),
            "stop_loss": float(stop),
            "take_profit": float(tp),
            "strategy": "EMA_Cross + AI_Confirm",
        }
    return None
