# Файл: app/tests/test_logic.py

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from app.core.scanner import find_patterns, get_dynamic_threshold
from app.worker.tasks import full_train_model


# --- ТЕСТ СКАНЕРА ---
def test_find_patterns_bullish():
    # Тестируем универсальную функцию
    current_price = 100.0
    forecast_df = pd.DataFrame({"forecast_value": [105] * 9 + [110]})  # Рост +10%
    levels = pd.DataFrame()  # Без уровней
    ob = {}  # Пустой стакан

    # Должен найти momentum
    signals = find_patterns("TEST", current_price, forecast_df, levels, ob)

    assert len(signals) > 0
    assert signals[0]["signal_type"] == "bullish_momentum"
    assert signals[0]["potential_profit_pct"] == pytest.approx(10.0)


def test_dynamic_threshold():
    # Спокойный рынок
    df_calm = pd.DataFrame({"close": [100, 100.1, 100.2, 100.1] * 10})
    thresh_calm = get_dynamic_threshold(df_calm)
    assert thresh_calm == 2.4  # Минимальный

    # Бешеный рынок
    df_vol = pd.DataFrame({"close": [100, 200, 50, 300] * 25})
    # Добавляем high/low/open, так как ATR их требует!
    df_vol["high"] = df_vol["close"] + 10
    df_vol["low"] = df_vol["close"] - 10
    df_vol["open"] = df_vol["close"]
    thresh_vol = get_dynamic_threshold(df_vol)
    assert thresh_vol > 3.0  # Должен быть высоким


# --- ТЕСТ ЗАДАЧИ (MOCKING) ---
@patch("tasks.get_historical_data")  # Мокаем запрос к Тинькофф
@patch("tasks.train_and_predict_lstm")  # Мокаем долгое обучение
@patch("tasks.update_db_with_full_forecast")  # Мокаем запись в БД
@patch("torch.save")  # <--- Добавить
@patch("joblib.dump")  # <--- Добавить
def test_full_train_task(mock_joblib, mock_torch, mock_db, mock_train, mock_data):
    # Настраиваем моки
    mock_data.return_value = pd.DataFrame(
        {"close": [100] * 300, "volume": [1000] * 300, "time": pd.date_range("2024-01-01", periods=300)}
    )

    # train_and_predict возвращает кортеж (model, scaler, preds...)
    mock_train.return_value = (MagicMock(), MagicMock(), np.array([100] * 30), [110] * 30, [90] * 30, {}, None, {})

    # Запускаем задачу (синхронно, без Celery)
    full_train_model("TEST", "FIGI", "TestName")

    # Проверяем, что функции вызывались
    mock_data.assert_called_once()
    mock_train.assert_called_once()
    mock_db.assert_called_once()
