import numpy as np
import pandas as pd
import torch


def predict_with_uncertainty(
    model, scaler, last_sequence_normalized, stock_data, feature_columns, output_columns, future_predictions, n_iter=30
):
    """
    Генерация прогноза с использованием Monte Carlo Dropout для оценки неопределенности.
    """
    model.train()

    output_indices = [feature_columns.index(col) for col in output_columns]
    all_future_predictions = []

    def update_ema(prev_ema, new_value, period):
        return (new_value * (2 / (period + 1))) + (prev_ema * (1 - (2 / (period + 1))))

    for _ in range(n_iter):
        iter_predictions = []
        test_inputs = torch.FloatTensor(last_sequence_normalized).clone().unsqueeze(0)

        last_values = stock_data[feature_columns].iloc[-1].to_dict()
        last_price = stock_data["close"].iloc[-1]

        last_atr = last_values.get("ATRr_14", 0.01)
        last_ema_fast = last_values.get("EMA_12", last_price)
        last_ema_slow = last_values.get("EMA_26", last_price)
        last_macd_signal = last_values.get("MACDs_12_26_9", 0)

        with torch.no_grad():
            for _ in range(future_predictions):
                pred_vec_norm = model(test_inputs)[0].numpy()

                temp_denorm = np.zeros((1, len(feature_columns)))
                temp_denorm[0, output_indices] = pred_vec_norm
                pred_vals_denorm = scaler.inverse_transform(temp_denorm)[0]

                pred_price = pred_vals_denorm[feature_columns.index("close")]
                iter_predictions.append(pred_price)

                new_vec_denorm = last_values.copy()

                # Обновляем выходы
                for i_col, col_name in enumerate(output_columns):
                    new_vec_denorm[col_name] = pred_vals_denorm[output_indices[i_col]]

                # Ручной пересчет индикаторов (авторегрессия)
                calculated_log_return = np.log1p((pred_price - last_price) / last_price)
                new_vec_denorm["log_return"] = calculated_log_return

                true_range = abs(pred_price - last_price)
                new_atr = update_ema(last_atr, true_range, 14)
                new_vec_denorm["ATRr_14"] = new_atr

                new_ema_fast = update_ema(last_ema_fast, pred_price, 12)
                new_ema_slow = update_ema(last_ema_slow, pred_price, 26)
                macd_line = new_ema_fast - new_ema_slow
                new_macd_signal = update_ema(last_macd_signal, macd_line, 9)
                new_vec_denorm["MACD_12_26_9"] = macd_line
                new_vec_denorm["MACDs_12_26_9"] = new_macd_signal
                new_vec_denorm["MACDh_12_26_9"] = macd_line - new_macd_signal

                day_cols = [f"day_{d}" for d in range(7)]
                last_day_idx = np.argmax([last_values.get(c, 0) for c in day_cols])
                next_day_idx = (last_day_idx + 1) % 5
                for idx, col in enumerate(day_cols):
                    if col in new_vec_denorm:
                        new_vec_denorm[col] = 1 if idx == next_day_idx else 0

                df_new = pd.DataFrame([new_vec_denorm])
                for col in feature_columns:
                    if col not in df_new.columns:
                        df_new[col] = last_values.get(col, 0)

                new_vec_norm = scaler.transform(df_new[feature_columns].values)[0]
                new_point = torch.FloatTensor(new_vec_norm).unsqueeze(0).unsqueeze(0)

                test_inputs = torch.cat((test_inputs[:, 1:, :], new_point), dim=1)

                last_values = new_vec_denorm
                last_price = pred_price
                last_atr, last_ema_fast, last_ema_slow, last_macd_signal = (
                    new_atr,
                    new_ema_fast,
                    new_ema_slow,
                    new_macd_signal,
                )

        all_future_predictions.append(iter_predictions)

    all_future_predictions = np.array(all_future_predictions)
    mean_preds = np.mean(all_future_predictions, axis=0)
    std_preds = np.std(all_future_predictions, axis=0)

    upper = mean_preds + (1.96 * std_preds)
    lower = mean_preds - (1.96 * std_preds)

    return mean_preds, upper, lower
