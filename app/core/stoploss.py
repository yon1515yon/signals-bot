import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
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

def calculate_ai_drawdown_stop_loss(historical_data: pd.DataFrame, ticker: str, entry_price: float, direction: str = "LONG") -> dict | None:
    """AI-based drawdown stop-loss (LONG only)."""
    direction = (direction or "LONG").upper()
    if direction not in {"LONG", "SHORT"}:
        return None
    if direction == "SHORT":
        return None
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
                f"AI model estimates max drawdown ~{predicted_drawdown_percent*100:.2f}% in next 10 days. "
                f"Stop is placed at that level."
            ),
        }
    except Exception as e:
        print(f"Error calculating AI stop-loss for {ticker}: {e}")
        return None

def calculate_atr_stop_loss(historical_data: pd.DataFrame, entry_price: float, direction: str = "LONG") -> dict | None:
    """Calculates stop-loss based on ATR."""
    direction = (direction or "LONG").upper()
    if direction not in {"LONG", "SHORT"}:
        return None
    if historical_data.empty or len(historical_data) < 20:
        return None

    historical_data.ta.atr(length=14, append=True)
    last_atr = historical_data["ATRr_14"].iloc[-1]

    if pd.isna(last_atr) or last_atr == 0:
        return None

    if direction == "SHORT":
        stop_loss_price = entry_price + (ATR_SL_MULTIPLIER * last_atr)
    else:
        stop_loss_price = entry_price - (ATR_SL_MULTIPLIER * last_atr)

    return {
        "method": "Volatility (ATR)",
        "stop_loss_price": float(round(stop_loss_price, 4)),
        "description": (
            f"ATR-based stop (ATR={last_atr:.2f}, x{ATR_SL_MULTIPLIER})."
        ),
    }

def calculate_level_stop_loss(key_levels: pd.DataFrame, entry_price: float, direction: str = "LONG") -> dict | None:
    """Calculates stop-loss based on nearest key level."""
    direction = (direction or "LONG").upper()
    if direction not in {"LONG", "SHORT"}:
        return None
    if key_levels.empty:
        return None

    if direction == "SHORT":
        resistance_zones = key_levels[key_levels["level_type"] == "resistance"].copy()
        if resistance_zones.empty:
            return None
        nearby_resists = resistance_zones[resistance_zones["start_price"] > entry_price]
        if nearby_resists.empty:
            return None
        strongest_resist = nearby_resists.loc[nearby_resists["intensity"].idxmax()]
        stop_loss_price = strongest_resist["end_price"] / LEVEL_SL_MARGIN
        return {
            "method": "Market Structure (Key Level)",
            "stop_loss_price": float(round(stop_loss_price, 4)),
            "description": (
                f"Stop above resistance zone ({strongest_resist['start_price']:.2f} - {strongest_resist['end_price']:.2f})."
            ),
        }

    support_zones = key_levels[key_levels["level_type"] == "support"].copy()
    if support_zones.empty:
        return None

    nearby_supports = support_zones[support_zones["end_price"] < entry_price]
    if nearby_supports.empty:
        return None

    strongest_support = nearby_supports.loc[nearby_supports["intensity"].idxmax()]

    stop_loss_price = strongest_support["start_price"] * LEVEL_SL_MARGIN

    return {
        "method": "Market Structure (Key Level)",
        "stop_loss_price": float(round(stop_loss_price, 4)),
        "description": (
            f"Stop below support zone ({strongest_support['start_price']:.2f} - {strongest_support['end_price']:.2f})."
        ),
    }

def get_stop_loss_suggestions(db: Session, ticker: str, figi: str, entry_price: float, direction: str = "LONG") -> list:
    suggestions = []
    direction = (direction or "LONG").upper()
    if direction not in {"LONG", "SHORT"}:
        direction = "LONG"

    historical_data = get_historical_data(figi, days=90)

    levels_data = db.execute(
        text("SELECT start_price, end_price, level_type, intensity FROM key_levels WHERE ticker = :ticker"),
        {"ticker": ticker},
    ).fetchall()

    if levels_data:
        key_levels_df = pd.DataFrame(levels_data, columns=["start_price", "end_price", "level_type", "intensity"])
    else:
        key_levels_df = pd.DataFrame()

    atr_sl = calculate_atr_stop_loss(historical_data, entry_price, direction)
    if atr_sl:
        suggestions.append(atr_sl)

    level_sl = calculate_level_stop_loss(key_levels_df, entry_price, direction)
    if level_sl:
        suggestions.append(level_sl)

    ai_sl = calculate_ai_drawdown_stop_loss(historical_data, ticker, entry_price, direction)
    if ai_sl:
        suggestions.append(ai_sl)

    if direction == "SHORT":
        percent_sl_price = entry_price * (2 - FIXED_SL_PERCENT)
    else:
        percent_sl_price = entry_price * FIXED_SL_PERCENT

    suggestions.append(
        {
            "method": "Fixed Percentage",
            "stop_loss_price": float(round(percent_sl_price, 4)),
            "description": f"Fixed stop based on {int((1-FIXED_SL_PERCENT)*100)}% risk from entry."
        }
    )

    for s in suggestions:
        if direction == "SHORT":
            risk_percent = ((s["stop_loss_price"] - entry_price) / entry_price) * 100
        else:
            risk_percent = ((entry_price - s["stop_loss_price"]) / entry_price) * 100
        s["risk_percent"] = float(round(risk_percent, 2))

    return suggestions
