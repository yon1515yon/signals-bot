from __future__ import annotations

import pandas as pd

from app.constants import DARKPOOL_VOL_SPIKE, DARKPOOL_VOLATILITY_QUANTILE


def detect_dark_pool_footprints(hourly_data: pd.DataFrame) -> list[dict]:
    """
    Analyzes hourly data to find Dark Pool activity traces
    using the "Price-Volume" divergence method.
    """
    if hourly_data.empty or len(hourly_data) < 50:
        return []

    hourly_data["volume_ma"] = hourly_data["volume"].rolling(window=20).mean()
    hourly_data["price_volatility"] = hourly_data["close"].pct_change().rolling(window=10).std()

    hourly_data.dropna(inplace=True)
    if hourly_data.empty:
        return []

    high_volume_condition = hourly_data["volume"] > (hourly_data["volume_ma"] * DARKPOOL_VOL_SPIKE)

    low_volatility_threshold = hourly_data["price_volatility"].quantile(DARKPOOL_VOLATILITY_QUANTILE)
    low_volatility_condition = hourly_data["price_volatility"] < low_volatility_threshold

    footprint_candles = hourly_data[high_volume_condition & low_volatility_condition]

    signals = []
    for __, row in footprint_candles.iterrows():
        activity_type = "accumulation" if row["close"] > row["open"] else "distribution"

        signals.append(
            {
                "timestamp": row["time"].isoformat(),
                "price": float(row["close"]),
                "volume": int(row["volume"]),
                "avg_volume": int(row["volume_ma"]),
                "activity_type": activity_type,
                "description": (
                    f"Обнаружена аномальная активность: объем ({int(row['volume'])}) "
                    f"в {row['volume']/row['volume_ma']:.1f} раз выше среднего при очень низкой волатильности. "
                    f"Вероятно, крупный игрок проводит {activity_type}."
                ),
            }
        )

    return signals
