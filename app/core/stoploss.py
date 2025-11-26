import os
import joblib
import numpy as np
import pandas as pd
import torch
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.constants import (
    AI_SL_SAFETY_MARGIN, 
    ATR_SL_MULTIPLIER, 
    LEVEL_SL_MARGIN, 
    FIXED_SL_PERCENT
)
from app.ml.architecture import DrawdownLSTMModel
from app.services.services import get_historical_data

def calculate_ai_drawdown_stop_loss(historical_data: pd.DataFrame, ticker: str, entry_price: float) -> dict | None:
    """Рассчитывает стоп-лосс на основе AI-прогноза максимального отката."""
    model_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}_drawdown.pth"
    scaler_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}_drawdown.pkl"

    if not os.path.exists(model_path):
        return None 

    try:
        model = DrawdownLSTMModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        scaler = joblib.load(scaler_path)

        if "volume" not in historical_data.columns:
            historical_data["volume"] = 0

        last_30_days_features = historical_data[["close", "volume"]].tail(30).astype(float).values
        if len(last_30_days_features) < 30:
            return None 

        temp_data_for_scaling = np.zeros((len(last_30_days_features), 3))
        temp_data_for_scaling[:, :2] = last_30_days_features
        features_normalized = scaler.transform(temp_data_for_scaling)[:, :2]

        input_tensor = torch.FloatTensor(features_normalized).unsqueeze(0)
        with torch.no_grad():
            predicted_drawdown_normalized = model(input_tensor)[0][0].item()

        value_to_inverse = np.zeros((1, 3))
        value_to_inverse[0, 2] = predicted_drawdown_normalized
        predicted_drawdown_percent = scaler.inverse_transform(value_to_inverse)[0, 2]

        predicted_drawdown_percent *= AI_SL_SAFETY_MARGIN

        stop_loss_price = entry_price * (1 - predicted_drawdown_percent)

        return {
            "method": "AI Drawdown Forecast",
            "stop_loss_price": float(round(stop_loss_price, 4)),
            "description": (
                f"AI-модель прогнозирует максимальную просадку в ~{predicted_drawdown_percent*100:.2f}% "
                f"в течение следующих 10 дней. Стоп установлен на этом уровне."
            ),
        }
    except Exception as e:
        print(f"Ошибка при расчете AI стоп-лосса для {ticker}: {e}")
        return None


def calculate_atr_stop_loss(historical_data: pd.DataFrame, entry_price: float) -> dict | None:
    """Рассчитывает стоп-лосс на основе ATR."""
    if historical_data.empty or len(historical_data) < 20:
        return None

    historical_data.ta.atr(length=14, append=True)
    last_atr = historical_data["ATRr_14"].iloc[-1]

    if pd.isna(last_atr) or last_atr == 0:
        return None

    stop_loss_price = entry_price - (ATR_SL_MULTIPLIER * last_atr)

    return {
        "method": "Volatility (ATR)",
        "stop_loss_price": float(round(stop_loss_price, 4)),
        "description": (
            f"Основан на средней волатильности за 14 дней (ATR = {last_atr:.2f}). "
            f"Стоп размещен на расстоянии {ATR_SL_MULTIPLIER}x ATR от цены входа."
        ),
    }


def calculate_level_stop_loss(key_levels: pd.DataFrame, entry_price: float) -> dict | None:
    """Рассчитывает стоп-лосс на основе ближайшей зоны поддержки."""
    if key_levels.empty:
        return None

    support_zones = key_levels[key_levels["level_type"] == "support"].copy()
    if support_zones.empty:
        return None

    nearby_supports = support_zones[support_zones["end_price"] < entry_price]
    if nearby_supports.empty:
        return None

    strongest_support = nearby_supports.loc[nearby_supports["intensity"].idxmax()]

    stop_loss_price = strongest_support["start_price"] * LEVEL_SL_MARGIN

    return {
        "method": "Market Structure (AI Key Level)",
        "stop_loss_price": float(round(stop_loss_price, 4)),
        "description": (
            f"Основан на сильной зоне поддержки ({strongest_support['start_price']:.2f} - {strongest_support['end_price']:.2f}) "
            f"с интенсивностью {strongest_support['intensity']:.0f}%. Стоп размещен под этой зоной."
        ),
    }


def get_stop_loss_suggestions(db: Session, ticker: str, figi: str, entry_price: float) -> list:
    suggestions = []
    historical_data = get_historical_data(figi, days=90)

    levels_data = db.execute(
        text("SELECT start_price, end_price, level_type, intensity FROM key_levels WHERE ticker = :ticker"),
        {"ticker": ticker},
    ).fetchall()

    if levels_data:
        key_levels_df = pd.DataFrame(levels_data, columns=["start_price", "end_price", "level_type", "intensity"])
    else:
        key_levels_df = pd.DataFrame()

    atr_sl = calculate_atr_stop_loss(historical_data, entry_price)
    if atr_sl:
        suggestions.append(atr_sl)

    level_sl = calculate_level_stop_loss(key_levels_df, entry_price)
    if level_sl:
        suggestions.append(level_sl)

    ai_sl = calculate_ai_drawdown_stop_loss(historical_data, ticker, entry_price)
    if ai_sl:
        suggestions.append(ai_sl)
    percent_sl_price = entry_price * FIXED_SL_PERCENT
    suggestions.append(
        {
            "method": "Fixed Percentage",
            "stop_loss_price": float(round(percent_sl_price, 4)),
            "description": f"Простой стоп-лосс, основанный на фиксированном риске в {int((1-FIXED_SL_PERCENT)*100)}% от цены входа.",
        }
    )

    for s in suggestions:
        risk_percent = ((entry_price - s["stop_loss_price"]) / entry_price) * 100
        s["risk_percent"] = float(round(risk_percent, 2))

    return suggestions