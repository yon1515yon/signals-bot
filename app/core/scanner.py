import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.constants import MAX_SPREAD_PCT, ROBUST_FACTOR
from app.ml.market_analysis import get_global_market_regime
from app.ml.meta_model import get_meta_prediction_with_shap
from app.services.services import get_historical_data, get_order_book

_regime_cache = {"val": "Neutral", "time": datetime.min}
MODEL_STORAGE_PATH = settings.MODEL_STORAGE_PATH


def get_global_regime_cached():
    now = datetime.utcnow()
    if (now - _regime_cache["time"]).total_seconds() > 3600 * 4:  # 4 часа
        try:
            _regime_cache["val"] = get_global_market_regime()
            _regime_cache["time"] = now
            print(f"SCANNER: Обновлен режим рынка: {_regime_cache['val']}")
        except Exception:
            pass
    return _regime_cache["val"]


def check_liquidity_and_spread(order_book: dict, MIN_LIQUIDITY_RUB) -> bool:
    """Filter by liquidity and spread."""
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    if not bids or not asks:
        return False

    best_bid = bids[0]["price"]
    best_ask = asks[0]["price"]

    # 1. Спред
    spread_pct = (best_ask - best_bid) / best_bid * 100
    if spread_pct > MAX_SPREAD_PCT:  # Если спред > 1.5%, бумага неликвид
        return False

    # 2. Объем в стакане (глубина)
    total_bid_vol = sum(b["quantity"] * b["price"] for b in bids)
    if total_bid_vol < MIN_LIQUIDITY_RUB:  # Если в стакане меньше миллиона, опасно
        return False

    return True


def get_level_weight(level_row) -> float:
    """Вес уровня в зависимости от свежести (чем старше, тем слабее)."""
    # Предполагаем, что в DataFrame есть колонка 'last_calculated_at'
    # Если нет, вернем 1.0
    # Для простоты вернем 1.0 пока, т.к. в run_scanner мы передаем DataFrame без дат
    return 1.0


def get_dynamic_threshold(historical_data: pd.DataFrame) -> float:
    """
    Считает минимально необходимый % прибыли на основе ATR.
    Порог = 1.5 * ATR (в процентах).
    """
    if historical_data.empty:
        return 2.0

    last = historical_data.iloc[-1]
    # Если ATR еще не посчитан, считаем быстро
    if "ATRr_14" not in historical_data.columns:
        historical_data.ta.atr(length=14, append=True)
        last = historical_data.iloc[-1]

    atr_val = last.get("ATRr_14", last["close"] * 0.02)
    atr_pct = (atr_val / last["close"]) * 100

    return max(1.5, atr_pct * 1.2)  # Порог = 1.2 дневных ATR


def calculate_risk_score(forecast_df: pd.DataFrame, sentiment_score: float) -> int:
    vol = forecast_df["forecast_value"].std() / forecast_df["forecast_value"].mean()
    score = 5.0
    if vol > 0.05:
        score += 2.0
    if sentiment_score < -0.2:
        score += 2.0
    if sentiment_score > 0.5:
        score -= 1.0
    return max(1, min(10, int(round(score))))


def prepare_meta_features(
    ticker: str, historical_data: pd.DataFrame, forecast_df: pd.DataFrame, global_regime: str
) -> dict | None:
    scaler_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}.pkl"
    if not os.path.exists(scaler_path):
        return None

    try:
        scaler = joblib.load(scaler_path)
        df = historical_data.copy()

        try:
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.adx(length=14, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.ema(length=50, append=True)
        except Exception:
            pass

        df["vol_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["vol_ma_20"]

        df["day_of_week"] = df["time"].dt.dayofweek
        df = pd.get_dummies(df, columns=["day_of_week"], prefix="day")
        df["log_return"] = np.log1p(df["close"].pct_change()).fillna(0)

        for d in range(7):
            col = f"day_{d}"
            if col not in df.columns:
                df[col] = 0

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True).fillna(0, inplace=True)

        if hasattr(scaler, "feature_names_in_"):
            expected_cols = scaler.feature_names_in_
        else:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            expected_cols = [c for c in num_cols if c not in ["open", "high", "low"]]

        last_row_df = df.iloc[[-1]][expected_cols]
        norm_features = scaler.transform(last_row_df)[0]

        last = df.iloc[-1]
        atr_rub = last.get("ATRr_14", last["close"] * 0.02)

        pred_mean = forecast_df.iloc[0]["forecast_value"]
        pred_lower = forecast_df.iloc[0]["lower_bound"]
        pred_upper = forecast_df.iloc[0]["upper_bound"]

        uncertainty_rub = (pred_upper - pred_lower) / 4
        raw_diff = pred_mean - last["close"]

        # Robust logic
        if raw_diff > 0:
            robust_diff = (pred_mean - uncertainty_rub) - last["close"]
        else:
            robust_diff = (pred_mean + uncertainty_rub) - last["close"]

        z_score_robust = robust_diff / atr_rub if atr_rub > 0 else 0

        regime_map = {"Bullish": 1, "Bearish": -1, "Neutral": 0, "Extreme_Volatility": -2}
        regime_val = regime_map.get(global_regime, 0)
        adx = last.get("ADX_14", 20)
        z_threshold = 1.0 if adx < 25 else 0.5

        return {
            "z_score": float(z_score_robust),
            "z_threshold": float(z_threshold),
            "global_regime": int(regime_val),
            "rsi": float(last.get("RSI_14", 50)),
            "adx": float(adx),
            "volume_ratio": float(last.get("volume_ratio", 1.0)),
            "atr_pct": (atr_rub / last["close"]) * 100,
            "macd": float(last.get("MACD_12_26_9", 0)),
            "ema_bias": 1 if last["close"] > last.get("EMA_50", last["close"]) else -1,
            "volatility_norm": 0.0,
            "lstm_confidence": float(abs(raw_diff / last["close"])),
            "day_of_week": pd.Timestamp.now().dayofweek,
            "is_month_end": 1 if pd.Timestamp.now().is_month_end else 0,
        }
    except Exception:
        return None


def find_patterns(ticker, current_price, forecast_df, key_levels, order_book):
    """Универсальный искатель паттернов."""
    signals = []

    try:
        p10 = forecast_df.iloc[9]["forecast_value"]
        profit_pct = (p10 / current_price - 1) * 100
    except:
        return []

    if profit_pct > MAX_SPREAD_PCT:
        # Momentum
        signals.append(
            {
                "signal_type": "bullish_momentum",
                "potential_profit_pct": profit_pct,
                "forecast_days": 10,
                "details": {"target_price": p10},
            }
        )

        # Support Bounce
        if not key_levels.empty:
            supports = key_levels[key_levels["level_type"] == "support"]
            if not supports.empty:
                closest = supports.iloc[(supports["start_price"] - current_price).abs().argsort()[:1]]
                s_start, s_end = closest["start_price"].iloc[0], closest["end_price"].iloc[0]
                # Если мы в зоне поддержки
                if s_start <= current_price < s_end * 1.02:
                    signals.append(
                        {
                            "signal_type": "support_bounce",
                            "potential_profit_pct": profit_pct,
                            "forecast_days": 10,
                            "details": {"support_zone_start": s_start, "target_price": p10},
                        }
                    )

    elif profit_pct < -1.5:
        signals.append(
            {
                "signal_type": "bearish_momentum",  
                "potential_profit_pct": abs(profit_pct),  
                "forecast_days": 10,
                "details": {"target_price": p10, "direction": "SHORT"},
            }
        )

        if not key_levels.empty:
            resists = key_levels[key_levels["level_type"] == "resistance"]
            if not resists.empty:
                closest = resists.iloc[(resists["start_price"] - current_price).abs().argsort()[:1]]
                r_start, r_end = closest["start_price"].iloc[0], closest["end_price"].iloc[0]
                # Если мы под сопротивлением
                if r_start * 0.98 < current_price <= r_end:
                    signals.append(
                        {
                            "signal_type": "resistance_bounce_short",
                            "potential_profit_pct": abs(profit_pct),
                            "forecast_days": 10,
                            "details": {"resistance_zone_end": r_end, "target_price": p10, "direction": "SHORT"},
                        }
                    )

    return signals


def analyze_multitimeframe(ticker, db):
    """
    Проверяет совпадение Дневного и Часового тренда.
    """
    # 1. Daily
    daily_fc = db.execute(
        text("SELECT forecast_value FROM forecasts WHERE ticker = :t ORDER BY forecast_date LIMIT 5"), {"t": ticker}
    ).fetchall()
    if not daily_fc:
        return "Neutral"
    daily_trend = (
        "Bullish"
        if daily_fc[-1][0] > daily_fc[0][0] * 1.01
        else "Bearish" if daily_fc[-1][0] < daily_fc[0][0] * 0.99 else "Neutral"
    )

    # 2. Hourly
    metrics_json = db.execute(
        text("SELECT multi_horizon_forecast FROM forecast_metrics WHERE ticker = :t"), {"t": ticker}
    ).scalar()
    if not metrics_json:
        return "Neutral"

    metrics = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json)
    intra = metrics.get("intraday")
    if not intra:
        return "Neutral"

    # Свежесть < 4 часов
    upd = datetime.fromisoformat(intra["updated_at"])
    if (datetime.utcnow() - upd).total_seconds() > 3600 * 4:
        return "Neutral"

    hourly_trend = (
        "Bullish"
        if intra["forecast_8h"][-1] > intra["current_price"] * 1.005
        else "Bearish" if intra["forecast_8h"][-1] < intra["current_price"] * 0.995 else "Neutral"
    )

    if daily_trend == "Bullish" and hourly_trend == "Bullish":
        return "Strong_Buy"
    elif daily_trend == "Bearish" and hourly_trend == "Bearish":
        return "Strong_Sell"
    return "Mixed"



def run_scanner(db: Session, market_regime_unused: str):
    global_regime = get_global_regime_cached()
    if global_regime == "Extreme_Volatility":
        return 0

    tickers_rows = db.execute(text("SELECT DISTINCT ticker FROM forecasts")).fetchall()
    metrics = db.execute(text("SELECT ticker, backtest_metrics FROM forecast_metrics")).fetchall()
    wr_map = {}
    for t, m_json in metrics:
        if m_json:
            m = m_json if isinstance(m_json, dict) else json.loads(m_json)
            wr_map[t] = m.get("win_rate", 0.0)

    raw_signals = {}

    for row in tickers_rows:
        ticker = row[0]
        wr = wr_map.get(ticker, 0.0)
        if wr < 40.0:
            continue

        figi = db.execute(text("SELECT figi FROM tracked_tickers WHERE ticker = :ticker"), {"ticker": ticker}).scalar()
        if not figi:
            continue

        fc_data = db.execute(
            text(
                "SELECT forecast_date, forecast_value, upper_bound, lower_bound FROM forecasts WHERE ticker = :ticker ORDER BY forecast_date LIMIT 15"
            ),
            {"ticker": ticker},
        ).fetchall()
        if len(fc_data) < 14:
            continue
        fc_df = pd.DataFrame(fc_data, columns=["forecast_date", "forecast_value", "upper_bound", "lower_bound"])

        hist = get_historical_data(figi, days=60)
        if hist.empty:
            continue
        curr_price = hist.iloc[-1]["close"]

        ob = get_order_book(figi)
        if not check_liquidity_and_spread(ob):
            continue

        lvl_data = db.execute(
            text("SELECT start_price, end_price, level_type, intensity FROM key_levels WHERE ticker = :ticker"),
            {"ticker": ticker},
        ).fetchall()
        lvl_df = (
            pd.DataFrame(lvl_data, columns=["start_price", "end_price", "level_type", "intensity"])
            if lvl_data
            else pd.DataFrame()
        )

        sigs = find_patterns(ticker, curr_price, fc_df, lvl_df, ob)

        if sigs:
            raw_signals[ticker] = {"signals": sigs, "fc": fc_df, "hist": hist, "wr": wr}

    final = []

    for ticker, item in raw_signals.items():
        allowed_direction = "BOTH"
        if global_regime == "Bullish":
            allowed_direction = "LONG"
        elif global_regime == "Bearish":
            allowed_direction = "SHORT"

        valid_sigs = []
        for s in item["signals"]:
            direction = s["details"].get("direction", "LONG")
            if allowed_direction == "BOTH" or direction == allowed_direction:
                valid_sigs.append(s)

        if not valid_sigs:
            continue

        best = sorted(valid_sigs, key=lambda s: s["potential_profit_pct"], reverse=True)[0]

        meta_feats = prepare_meta_features(ticker, item["hist"], item["fc"], global_regime)
        meta_result = {"score": 0.5, "explanation": []}

        if meta_feats:
            meta_result = get_meta_prediction_with_shap(meta_feats)
            meta_score = meta_result["score"]
            ai_reasoning = meta_result["explanation"]

        meta_score = meta_result["score"]

        if meta_score < ROBUST_FACTOR:
            continue

        size_mult = 1.0
        if meta_score > 0.8:
            size_mult = 1.5
        elif meta_score > 0.9:
            size_mult = 2.0
        elif meta_score < 0.6:
            size_mult = 0.5

        if best["details"].get("confidence") == "High (MTF)":
            size_mult *= 1.2

        best["details"]["meta_score"] = round(meta_score * 100, 1)
        best["details"]["model_win_rate"] = round(item["wr"], 1)
        best["details"]["position_size_mult"] = round(size_mult, 2)
        best["details"]["ai_reasoning"] = ai_reasoning  

        priority_score = meta_score * best["potential_profit_pct"]
        best["_priority"] = priority_score

        final.append({**best, "ticker": ticker})

    final.sort(key=lambda x: x["_priority"], reverse=True)

    final_top = final[:20]

    try:
        db.execute(text("DELETE FROM trading_signals WHERE generated_at < NOW() - INTERVAL '2 days'"))
        for s in final_top:
            del s["_priority"]
            db.execute(
                text(
                    """
                INSERT INTO trading_signals (ticker, signal_type, potential_profit_pct, forecast_days, details, generated_at)
                VALUES (:ticker, :st, :pp, :fd, :det, NOW())
                ON CONFLICT (ticker) DO UPDATE SET
                    signal_type = EXCLUDED.signal_type,
                    potential_profit_pct = EXCLUDED.potential_profit_pct,
                    forecast_days = EXCLUDED.forecast_days,
                    details = EXCLUDED.details,
                    generated_at = NOW()
            """
                ),
                {
                    "ticker": s["ticker"],
                    "st": s["signal_type"],
                    "pp": s["potential_profit_pct"],
                    "fd": s["forecast_days"],
                    "det": json.dumps(s["details"]),
                },
            )
        db.commit()
    except Exception:
        db.rollback()

    return len(final_top)
