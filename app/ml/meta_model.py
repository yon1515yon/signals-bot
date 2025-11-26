import os
import pandas as pd
import shap
import xgboost as xgb
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)
MODEL_STORAGE_PATH = "/app/models_storage"
META_MODEL_PATH = f"{MODEL_STORAGE_PATH}/meta_xgboost.json"


def train_meta_model(all_trades_data: list):
    """
    Обучает XGBoost на собранных сделках со всех тикеров.
    """
    if not all_trades_data:
        logger.warning("META MODEL: Нет данных для обучения.")
        return

    df = pd.DataFrame(all_trades_data)

    X = df.drop(columns=["target"])
    y = df["target"]

    logger.info(f"META MODEL: Обучение на {len(df)} сделках...")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
    )

    try:
        model.fit(X, y)
        model.save_model(META_MODEL_PATH)
        logger.info(f"META MODEL: Успешно обучена и сохранена в {META_MODEL_PATH}")
    except Exception as e:
        logger.error(f"META MODEL: Ошибка обучения: {e}")


def get_meta_prediction_with_shap(features: dict) -> dict:
    """
    Возвращает вероятность успеха И объяснение (SHAP values).
    """
    if not os.path.exists(META_MODEL_PATH):
        return {"score": 0.5, "explanation": []}

    try:
        model = xgb.XGBClassifier()
        model.load_model(META_MODEL_PATH)

        df = pd.DataFrame([features])

        prob_success = float(model.predict_proba(df)[0][1])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        feature_names = df.columns.tolist()
        contributions = zip(feature_names, shap_values[0], strict=False) 

        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:3]

        explanation = []
        for name, val in top_features:
            impact = "повышает" if val > 0 else "понижает"
            explanation.append(f"{name} {impact} вероятность ({val:.2f})")

        return {"score": prob_success, "explanation": explanation}

    except Exception as e:
        logger.error(f"SHAP Error: {e}")
        return {"score": 0.5, "explanation": []}
