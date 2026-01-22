import os
import json
import joblib
import torch
import structlog
from functools import lru_cache
from typing import Optional, Tuple, Any

from app.config import settings
from app.constants import ML_CONFIG
from app.ml.architecture import LSTMTransformerModel

logger = structlog.get_logger()


CACHE_SIZE_MODELS = 20
CACHE_SIZE_SCALERS = 50

@lru_cache(maxsize=CACHE_SIZE_SCALERS)
def load_scaler(ticker: str) -> Any:
    """
    Загружает Scaler из кэша или с диска.
    """
    path = f"{settings.MODEL_STORAGE_PATH}/{ticker}.pkl"
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        logger.error(f"[{ticker}] Ошибка загрузки Scaler: {e}")
        return None

@lru_cache(maxsize=CACHE_SIZE_MODELS)
def load_model(ticker: str, input_size: int, output_size: int = 4) -> Optional[LSTMTransformerModel]:
    """
    Загружает веса PyTorch модели. Использует LRU-кэширование.
    """
    logger.info("Model cache miss - loading from disk", ticker=ticker)
    path = f"{settings.MODEL_STORAGE_PATH}/{ticker}.pth"
    if not os.path.exists(path):
        logger.warning("Model file not found", ticker=ticker, path=path)
        return None

    params_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}_params.json"
    
    current_params = {
        "hidden_size": ML_CONFIG["HIDDEN_SIZE"],
        "num_layers": ML_CONFIG["NUM_LAYERS"],
        "dropout": ML_CONFIG["DROPOUT"],
        "n_head": ML_CONFIG["N_HEAD"]
    }

    if os.path.exists(params_path):
        try:
            with open(params_path, "r") as f:
                tuned = json.load(f)
                if "hidden_size" in tuned: current_params["hidden_size"] = tuned["hidden_size"]
                if "num_layers" in tuned: current_params["num_layers"] = tuned["num_layers"]
                if "dropout" in tuned: current_params["dropout"] = tuned["dropout"]
        except Exception:
            pass

    try:
        # Инициализация архитектуры
        model = LSTMTransformerModel(
            input_size=input_size,
            hidden_size=current_params["hidden_size"],
            num_layers=current_params["num_layers"],
            n_head=current_params["n_head"],
            dropout=current_params["dropout"],
            output_size=output_size,
        )
        
        # Загрузка весов
        # map_location='cpu' важно, если обучали на GPU, а инференс на CPU
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval() # Переводим в режим инференса (выключаем dropout слои)
        
        logger.info("Model loaded successfully", ticker=ticker)
        return model
    except Exception as e:
        logger.error("Failed to load model architecture", ticker=ticker, error=str(e))
        return None

def clear_model_cache():
    """Очистка кэша (например, после переобучения модели)."""
    load_model.cache_clear()
    load_scaler.cache_clear()