from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from celery.utils.log import get_task_logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from app.config import settings
from app.constants import META_MODEL_MIN_TRADES, WINRATE_CONFIDENCE_THRESHOLD

logger = get_task_logger(__name__)
MODEL_STORAGE_PATH = settings.MODEL_STORAGE_PATH
META_MODEL_PATH = f"{MODEL_STORAGE_PATH}/meta_xgboost_calibrated.pkl"


def _coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    for col in clean.columns:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return clean


def _build_tscv(n_samples: int) -> TimeSeriesSplit:
    n_splits = min(5, max(2, n_samples // 40))
    return TimeSeriesSplit(n_splits=n_splits)


def _extract_base_estimator(calibrated_model: Any):
    calibrated = getattr(calibrated_model, "calibrated_classifiers_", None)
    if not calibrated:
        return None
    for cls in calibrated:
        est = getattr(cls, "estimator", None)
        if est is not None:
            return est
    return None


def _build_explanation(feature_row: np.ndarray, feature_columns: list[str], calibrated_model: Any) -> list[str]:
    base_model = _extract_base_estimator(calibrated_model)
    if base_model is None or not hasattr(base_model, "feature_importances_"):
        return []

    importances = np.asarray(base_model.feature_importances_, dtype=float)
    if len(importances) != len(feature_columns):
        return []

    contributions = importances * np.abs(feature_row)
    top_idx = np.argsort(contributions)[::-1][:3]

    explanation = []
    for idx in top_idx:
        if contributions[idx] <= 0:
            continue
        explanation.append(f"{feature_columns[idx]} contribution={contributions[idx]:.4f}")
    return explanation


def fit_meta_model(all_trades_data: list, confidence_threshold: float = WINRATE_CONFIDENCE_THRESHOLD):
    """
    Trains calibrated meta model with TimeSeriesSplit.
    """
    if not all_trades_data:
        logger.warning("META MODEL: no trade data for training")
        return

    df = pd.DataFrame(all_trades_data)
    if "target" not in df.columns:
        logger.warning("META MODEL: target column is missing")
        return

    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
        df = df.sort_values("event_time")

    df = df.dropna(subset=["target"]).copy()
    if len(df) < META_MODEL_MIN_TRADES:
        logger.warning(f"META MODEL: not enough trades ({len(df)}), need >= {META_MODEL_MIN_TRADES}")
        return

    y = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=["target"], errors="ignore")
    X = X.drop(columns=["event_time"], errors="ignore")
    X = _coerce_numeric_frame(X)

    if y.nunique() < 2:
        logger.warning("META MODEL: target has <2 classes, training skipped")
        return

    tscv = _build_tscv(len(X))
    base_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    try:
        calibrator = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=tscv)
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=tscv)

    try:
        calibrator.fit(X.values, y.values)
        payload = {
            "model": calibrator,
            "feature_columns": X.columns.tolist(),
            "confidence_threshold": float(confidence_threshold),
        }
        os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
        joblib.dump(payload, META_MODEL_PATH)
        logger.warning(f"META MODEL: trained and saved ({len(X)} samples) -> {META_MODEL_PATH}")
    except Exception as e:
        logger.error(f"META MODEL training failed: {e}")


def train_meta_model(all_trades_data: list):
    """
    Backward-compatible wrapper.
    """
    fit_meta_model(all_trades_data, confidence_threshold=WINRATE_CONFIDENCE_THRESHOLD)


def predict_meta_probability(features: dict) -> dict:
    """
    Returns calibrated success probability and high-confidence flag.
    """
    if not os.path.exists(META_MODEL_PATH):
        return {
            "score": 0.5,
            "threshold": WINRATE_CONFIDENCE_THRESHOLD,
            "is_high_confidence": False,
            "explanation": [],
        }

    try:
        payload = joblib.load(META_MODEL_PATH)
        model = payload["model"]
        feature_columns = payload.get("feature_columns", [])
        threshold = float(payload.get("confidence_threshold", WINRATE_CONFIDENCE_THRESHOLD))

        if not feature_columns:
            return {
                "score": 0.5,
                "threshold": threshold,
                "is_high_confidence": False,
                "explanation": [],
            }

        row = pd.DataFrame([features])
        for col in feature_columns:
            if col not in row.columns:
                row[col] = 0.0
        row = row[feature_columns]
        row = _coerce_numeric_frame(row)

        score = float(model.predict_proba(row.values)[0][1])
        explanation = _build_explanation(row.values[0], feature_columns, model)
        return {
            "score": score,
            "threshold": threshold,
            "is_high_confidence": bool(score >= threshold),
            "explanation": explanation,
        }
    except Exception as e:
        logger.error(f"META MODEL prediction error: {e}")
        return {
            "score": 0.5,
            "threshold": WINRATE_CONFIDENCE_THRESHOLD,
            "is_high_confidence": False,
            "explanation": [],
        }


def get_meta_prediction_with_shap(features: dict) -> dict:
    """
    Backward-compatible wrapper for scanner.
    """
    return predict_meta_probability(features)
