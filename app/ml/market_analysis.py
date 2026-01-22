from __future__ import annotations

import pandas as pd
from celery.utils.log import get_task_logger

from app.core.cache import redis_client
from app.services.services import find_imoex_future_figi, get_historical_data

logger = get_task_logger(__name__)

CACHE_KEY_REGIME = "global_market_regime"
CACHE_TTL_SECONDS = 3600 


def get_global_market_regime() -> str:
    """
    Determines global market regime (Risk-On / Risk-Off) using IMOEX index.
    Returns: 'Bullish', 'Bearish', 'Neutral', 'Extreme_Volatility'
    Calculates purely from fresh data (Heavy operation).
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
        logger.error(f"Market Regime Calculation Error: {e}")
        return "Neutral"


def get_market_regime_cached(db_session=None, services=None) -> str:
    """
    Returns market state using REDIS cache.
    Arguments db_session and services are kept for backward compatibility but unused.
    """
    try:
        # 1. Пытаемся получить значение из Redis
        cached_regime = redis_client.get(CACHE_KEY_REGIME)
        
        if cached_regime:
            logger.info(f"MARKET REGIME: Взято из кэша Redis ({cached_regime})")
            return cached_regime

        # 2. Если в кэше нет, считаем заново (Тяжелая операция)
        logger.info("MARKET REGIME: Кэш пуст или устарел, обновляю состояние рынка...")
        regime = get_global_market_regime()

        # 3. Сохраняем в Redis с TTL
        redis_client.set(CACHE_KEY_REGIME, regime, ex=CACHE_TTL_SECONDS)
        logger.warning(f"MARKET REGIME: Новый режим рынка: {regime} (сохранен в Redis)")
        
        return regime

    except Exception as e:
        logger.error(f"Redis Error in get_market_regime_cached: {e}")
        # Fallback: если Redis упал, просто считаем напрямую, не кэшируя
        return get_global_market_regime()