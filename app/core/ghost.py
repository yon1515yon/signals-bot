from __future__ import annotations
import numpy as np
import pandas as pd
from app.constants import GHOST_PRICE_CHANGE_LIMIT, GHOST_VOL_RATIO, GHOST_WINDOW


def detect_ghost_accumulation(intraday_data: pd.DataFrame) -> list[dict]:
    """
    Detects 'Ghost Accumulation' pattern.
    
    Conditions:
    1. Volume over the window > 300% of the average.
    2. Price change over the window < 0.4% (price stays flat).
    """
    if intraday_data.empty or len(intraday_data) < 100:
        return []

    WINDOW = GHOST_WINDOW

    signals = []

    intraday_data["vol_ma_long"] = intraday_data["volume"].rolling(window=200).mean()


    for i in range(len(intraday_data) - WINDOW):
        window = intraday_data.iloc[i : i + WINDOW]

        total_vol = window["volume"].sum()

        normal_vol = window.iloc[-1]["vol_ma_long"] * WINDOW

        if np.isnan(normal_vol) or normal_vol == 0:
            continue

        vol_ratio = total_vol / normal_vol

        max_price = window["high"].max()
        min_price = window["low"].min()
        price_change_pct = (max_price - min_price) / min_price * 100


        if vol_ratio > GHOST_VOL_RATIO and price_change_pct < GHOST_PRICE_CHANGE_LIMIT:

            last_candle = window.iloc[-1]

            signals.append(
                {
                    "signal_time": last_candle["time"],
                    "price": float(last_candle["close"]),
                    "volume": int(total_vol),
                    "avg_volume": int(normal_vol),
                    "activity_type": "GHOST_ACCUMULATION", 
                    "description": (
                        f"Обнаружено скрытое накопление! За 30 мин прошел объем x{vol_ratio:.1f} "
                        f"от нормы, а цена изменилась всего на {price_change_pct:.2f}%. "
                        f"Кит набирает позицию."
                    ),
                }
            )


    if signals:
        last_sig = signals[-1]

        return [last_sig]

    return []
