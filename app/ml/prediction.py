import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
from app.ml.architecture import LSTMTransformerModel

class FeatureUpdater:
    """
    Отвечает за инкрементальное обновление технических индикаторов
    на основе нового предсказанного значения цены (Close).
    """
    def __init__(self, last_row: dict):
        self.last_price = last_row.get("close", 0.0)
        self.ema_12 = last_row.get("EMA_12", self.last_price)
        self.ema_26 = last_row.get("EMA_26", self.last_price)
        self.macd_signal = last_row.get("MACDs_12_26_9", 0.0)
        self.atr = last_row.get("ATRr_14", self.last_price * 0.01)

    def _update_ema(self, prev: float, new_val: float, period: int) -> float:
        k = 2 / (period + 1)
        return (new_val * k) + (prev * (1 - k))

    def update(self, new_price: float) -> dict:
        """
        Возвращает словарь с обновленными индикаторами.
        """
        updates = {}
        
        # 1. Log Return
        updates["log_return"] = np.log1p((new_price - self.last_price) / self.last_price) if self.last_price > 0 else 0.0

        # 2. EMA
        self.ema_12 = self._update_ema(self.ema_12, new_price, 12)
        self.ema_26 = self._update_ema(self.ema_26, new_price, 26)
        updates["EMA_12"] = self.ema_12
        updates["EMA_26"] = self.ema_26

        # 3. MACD
        macd_line = self.ema_12 - self.ema_26
        self.macd_signal = self._update_ema(self.macd_signal, macd_line, 9)
        updates["MACD_12_26_9"] = macd_line
        updates["MACDs_12_26_9"] = self.macd_signal
        updates["MACDh_12_26_9"] = macd_line - self.macd_signal

        # 4. ATR (Приближенно, т.к. High/Low тоже предсказываются, но тут берем Close-Close)
        true_range = abs(new_price - self.last_price)
        self.atr = self._update_ema(self.atr, true_range, 14)
        updates["ATRr_14"] = self.atr

        self.last_price = new_price
        return updates


class AutoregressivePredictor:
    """
    Класс для выполнения авторегрессионного прогноза с Monte Carlo Dropout.
    """
    def __init__(
        self,
        model: LSTMTransformerModel,
        scaler,
        feature_columns: List[str],
        output_columns: List[str]
    ):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.output_columns = output_columns
        self.output_indices = [feature_columns.index(col) for col in output_columns]
        self.close_idx = feature_columns.index("close")

    def _predict_single_step(self, input_seq: torch.Tensor) -> np.ndarray:
        """Делает один шаг прогноза и возвращает денормализованный вектор."""
        with torch.no_grad():
            # (1, seq_len, features) -> (1, features)
            pred_norm = self.model(input_seq)[0].numpy()
        
        # Денормализация (создаем пустышку нужной формы)
        dummy = np.zeros((1, len(self.feature_columns)))
        dummy[0, self.output_indices] = pred_norm
        pred_denorm = self.scaler.inverse_transform(dummy)[0]
        
        return pred_denorm

    def predict_horizon(
        self, 
        initial_sequence_norm: np.ndarray, 
        initial_last_row_denorm: dict, 
        steps: int
    ) -> List[float]:
        """
        Генерирует одну траекторию цены на steps шагов вперед.
        """
        current_seq_norm = torch.FloatTensor(initial_sequence_norm).unsqueeze(0) # (1, win, feat)
        
        updater = FeatureUpdater(initial_last_row_denorm)
        
        # Текущий вектор признаков (денормализованный)
        current_features_denorm = initial_last_row_denorm.copy()
        
        trajectory_prices = []

        for _ in range(steps):
            pred_vals = self._predict_single_step(current_seq_norm)
            pred_price = pred_vals[self.close_idx]
            trajectory_prices.append(pred_price)

            new_indicators = updater.update(pred_price)
            
            next_row = current_features_denorm.copy()
            # Обновляем предсказанными значениями (Close, Volume и т.д.)
            for i, col in enumerate(self.output_columns):
                next_row[col] = pred_vals[self.output_indices[i]]
            # Обновляем расчетными индикаторами
            next_row.update(new_indicators)
            
            next_row_arr = np.array([[next_row.get(c, 0.0) for c in self.feature_columns]])
            next_row_norm = self.scaler.transform(next_row_arr) # (1, features)
            
            # 5. Добавление в последовательность (FIFO)
            next_step_tensor = torch.FloatTensor(next_row_norm).unsqueeze(0) # (1, 1, feat)
            current_seq_norm = torch.cat((current_seq_norm[:, 1:, :], next_step_tensor), dim=1)
            
            current_features_denorm = next_row

        return trajectory_prices


def predict_with_uncertainty(
    model: LSTMTransformerModel,
    scaler,
    last_sequence_normalized: np.ndarray,
    stock_data: pd.DataFrame,
    feature_columns: List[str],
    output_columns: List[str],
    future_predictions: int,
    n_iter: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    model.train() 
    
    predictor = AutoregressivePredictor(model, scaler, feature_columns, output_columns)
    last_row_dict = stock_data.iloc[-1].to_dict()

    all_paths = []

    for i in range(n_iter):
        path = predictor.predict_horizon(
            last_sequence_normalized, 
            last_row_dict, 
            future_predictions
        )
        all_paths.append(path)

    all_paths = np.array(all_paths)
    
    mean_preds = np.mean(all_paths, axis=0)
    std_preds = np.std(all_paths, axis=0)
    
    upper_bound = mean_preds + (1.96 * std_preds)
    lower_bound = mean_preds - (1.96 * std_preds)

    return mean_preds, upper_bound, lower_bound