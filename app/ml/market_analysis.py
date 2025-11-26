from __future__ import annotations

import pandas as pd
from celery.utils.log import get_task_logger

from app.services.services import find_imoex_future_figi, get_historical_data

logger = get_task_logger(__name__)

_market_regime_cache = {"regime": "Neutral", "timestamp": pd.Timestamp(0, tz="UTC")}


def get_global_market_regime() -> str:
    """
    Determines global market regime (Risk-On / Risk-Off) using IMOEX index.
    Returns: 'Bullish', 'Bearish', 'Neutral', 'Extreme_Volatility'
    """
    try:
        figi = find_imoex_future_figi()
        if not figi:
            return "Neutral"

        df = get_historical_data(figi, days=200)
        if df.empty:
            return "Neutral"

        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)

        last = df.iloc[-1]

        price = last["close"]
        ema50 = last["EMA_50"]
        ema200 = last["EMA_200"]
        adx = last["ADX_14"]

        atr_pct = (last["ATRr_14"] / price) * 100
        if atr_pct > 3.0:
            return "Extreme_Volatility"

        if price > ema50 and price > ema200:
            if adx > 25:
                return "Bullish"
            return "Neutral_Bullish"

        if price < ema50 and price < ema200:
            if adx > 25:
                return "Bearish"
            return "Neutral_Bearish"

        return "Neutral"

    except Exception as e:
        print(f"Market Regime Error: {e}")
        return "Neutral"


def get_market_regime_cached(db_session, services, cache_duration_minutes=60):
    """
    Returns market state using in-memory cache.
    """
    global _market_regime_cache
    now = pd.Timestamp.now(tz="UTC")

    if (now - _market_regime_cache["timestamp"]).total_seconds() > cache_duration_minutes * 60:
        logger.info("MARKET REGIME: Кэш устарел, обновляю состояние рынка...")

        try:
            market_index_figi = services.find_imoex_future_figi()
            if not market_index_figi:
                raise ValueError("Не удалось найти FIGI для фьючерса на IMOEX.")
            index_data = services.get_historical_data(figi=market_index_figi, days=90)

            if index_data.empty:
                logger.warning("MARKET REGIME WARNING: Не удалось получить данные по фьючерсу IMOEX. Режим 'Neutral'.")
                regime = "Neutral"
            else:
                regime = get_global_market_regime(index_data)

        except Exception as e:
            logger.error(f"MARKET REGIME ERROR: Ошибка при получении данных по индексу: {e}. Режим 'Neutral'.")
            regime = "Neutral"

        _market_regime_cache["regime"] = regime
        _market_regime_cache["timestamp"] = now
        logger.warning(f"MARKET REGIME: Новый режим рынка: {regime}")

    return _market_regime_cache["regime"]
