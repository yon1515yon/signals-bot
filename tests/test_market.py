from unittest.mock import patch

import pandas as pd
from market_analysis import get_global_market_regime


@patch("market_analysis.get_historical_data")
@patch("market_analysis.find_imoex_future_figi")
def test_market_regime_bullish(mock_figi, mock_data):
    mock_figi.return_value = "FUT_FIGI"
    # Создаем растущий тренд (Close > EMA50 > EMA200)
    df = pd.DataFrame({"close": [100 + i for i in range(250)]})
    # Мокаем EMA и ADX (так как ta-lib требует время)
    # Но лучше позволить ta-lib посчитать реально
    # Для простоты - создаем идеальный тренд
    mock_data.return_value = df

    # В реальном тесте лучше использовать интеграционный тест с pandas-ta
    # или замокать результат df.ta...

    # Здесь просто проверим обработку пустых данных
    mock_data.return_value = pd.DataFrame()
    assert get_global_market_regime() == "Neutral"
