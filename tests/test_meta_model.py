import os
from unittest.mock import patch

from meta_model import get_meta_prediction_with_shap, train_meta_model  # <--- SHAP версия

TEST_MODEL_PATH = "/tmp/test_meta.json"


@patch("meta_model.META_MODEL_PATH", TEST_MODEL_PATH)
def test_train_and_predict():
    # 1. Обучаем
    data = [
        {"rsi": 30, "volume": 0.5, "z_score": 1.5, "target": 1},  # Добавили z_score
        {"rsi": 80, "volume": 0.5, "z_score": 0.2, "target": 0},
    ] * 50

    train_meta_model(data)
    assert os.path.exists(TEST_MODEL_PATH)

    # 2. Предсказываем (с SHAP)
    res = get_meta_prediction_with_shap({"rsi": 30, "volume": 0.5, "z_score": 1.5})
    assert res["score"] > 0.5
    assert isinstance(res["explanation"], list)

    os.remove(TEST_MODEL_PATH)
