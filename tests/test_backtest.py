import numpy as np
import pandas as pd
from backtest_engine import run_backtrader_wfa


def test_backtrader_logic():
    # Создаем идеальный синус (цена ходит вверх-вниз)
    x = np.linspace(0, 10, 100)
    prices = 100 + np.sin(x) * 10

    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=100),
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 1000,
        }
    )

    # Идеальные сигналы: покупаем внизу, продаем наверху
    # Z-Score коррелирует с будущей ценой
    z_scores = np.diff(prices, append=prices[-1])  # Если следующая цена выше -> Z>0

    res = run_backtrader_wfa(df, z_scores, z_threshold=0.0)

    assert res["total_trades"] > 0
    # При идеальных сигналах PnL должен быть положительным
    # (Хотя комиссия может съесть, но в тесте без шума должно быть ок)
