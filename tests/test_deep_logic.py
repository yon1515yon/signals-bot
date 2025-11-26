# Файл: app/tests/test_deep_logic.py


import numpy as np
import pandas as pd
import pytest
from ghost import detect_ghost_accumulation
from model import prepare_features_and_scale
from scanner import find_patterns

# --- ТЕСТЫ MODEL.PY ---


def test_prepare_features_basic():
    # Создаем фейковый OHLCV
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "time": dates,
            "open": np.random.rand(100) * 100,
            "high": np.random.rand(100) * 100,
            "low": np.random.rand(100) * 100,
            "close": np.random.rand(100) * 100,
            "volume": np.random.randint(100, 1000, 100),
        }
    )

    # Вызываем функцию
    stock_data, norm, scaler, feats, outs = prepare_features_and_scale(df.copy())

    # Проверки
    assert stock_data is not None
    assert "log_return" in stock_data.columns  # Критично для сканера!
    assert "RSI_14" in stock_data.columns  # Критично для XGBoost!
    assert len(feats) == norm.shape[1]
    assert scaler is not None


def test_prepare_features_empty():
    df = pd.DataFrame({"time": pd.to_datetime([]), "close": [], "volume": []})  # <--- Явно datetime

    res = prepare_features_and_scale(df)
    assert res == (None, None, None, None, None)


# --- ТЕСТЫ SCANNER.PY (ОБНОВЛЕННЫЕ) ---


def test_find_patterns_support_bounce():
    curr_price = 100.0
    # Прогноз: цена вырастет до 110 (+10%)
    forecast_df = pd.DataFrame({"forecast_value": [105] * 9 + [110]})
    # Уровни: Поддержка прямо под ценой
    levels = pd.DataFrame([{"start_price": 99.0, "end_price": 100.0, "level_type": "support", "intensity": 50}])
    ob = {}  # Пустой стакан

    # Ищем паттерны (MTF Neutral)
    signals = find_patterns("TEST", curr_price, forecast_df, levels, ob)

    # Должен найти Momentum и Support Bounce
    assert len(signals) > 0

    types = [s["signal_type"] for s in signals]
    assert "support_bounce" in types
    assert "bullish_momentum" in types

    bounce_sig = next(s for s in signals if s["signal_type"] == "support_bounce")
    assert bounce_sig["potential_profit_pct"] == pytest.approx(10.0)


def test_find_patterns_short():
    curr_price = 100.0
    # Прогноз: падение до 90 (-10%)
    forecast_df = pd.DataFrame({"forecast_value": [95] * 9 + [90]})
    levels = pd.DataFrame([{"start_price": 100.0, "end_price": 101.0, "level_type": "resistance", "intensity": 50}])
    ob = {}

    # Ищем Short
    signals = find_patterns("TEST", curr_price, forecast_df, levels, ob)

    types = [s["signal_type"] for s in signals]
    assert "bearish_momentum" in types
    assert "resistance_bounce_short" in types

    short_sig = next(s for s in signals if s["signal_type"] == "bearish_momentum")
    assert short_sig["details"]["direction"] == "SHORT"


# --- ТЕСТЫ GHOST.PY ---


def test_ghost_accumulation():
    # Создаем паттерн: цена стоит, объем огромный
    dates = pd.date_range("2024-01-01", periods=300, freq="5min")
    df = pd.DataFrame(
        {
            "time": dates,  # длина 300
            "close": [100.0] * 300,
            "high": [100.1] * 300,
            "low": [99.9] * 300,
            "volume": [100] * 290 + [10000] * 10,  # 300
        }
    )

    signals = detect_ghost_accumulation(df)

    # Должен найтись хотя бы один сигнал
    assert len(signals) > 0
    assert signals[-1]["activity_type"] == "GHOST_ACCUMULATION"
    assert signals[-1]["volume"] >= 60000  # Суммарный объем окна
