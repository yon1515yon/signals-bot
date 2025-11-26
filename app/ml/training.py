from __future__ import annotations
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from app.ml.backtest_engine import run_backtrader_wfa 
from app.services.services import get_combined_macro_data, get_cross_asset_data, get_historical_data
from celery.utils.log import get_task_logger
from prophet import Prophet
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, TensorDataset


from app.config import settings
from app.constants import (
    LSTM_PARAMS,
    ML_CONFIG,
)  
from app.ml.architecture import DrawdownLSTMModel, FocalLoss, LSTMTransformerModel
from app.ml.prediction import predict_with_uncertainty

warnings.filterwarnings("ignore")
logger = get_task_logger(__name__)

GLOBAL_MODEL_PATH = f"{settings.MODEL_STORAGE_PATH}/GLOBAL_BASE_MODEL.pth"


def prepare_features_and_scale(stock_data: pd.DataFrame):
    """Подготовка фичей и нормализация данных."""
    try:
        stock_data.ta.rsi(length=14, append=True)
        stock_data.ta.macd(fast=12, slow=26, signal=9, append=True)
        stock_data.ta.cci(length=20, append=True)
        stock_data.ta.ema(length=50, append=True)
        stock_data.ta.ema(length=200, append=True)
        stock_data.ta.adx(length=14, append=True)
        stock_data.ta.atr(length=14, append=True)
        stock_data.ta.obv(append=True)

        stock_data["vol_ma_20"] = stock_data["volume"].rolling(window=20).mean()
        stock_data["volume_ratio"] = stock_data["volume"] / stock_data["vol_ma_20"]
    except Exception:
        pass

    stock_data["day_of_week"] = stock_data["time"].dt.dayofweek
    stock_data = pd.get_dummies(stock_data, columns=["day_of_week"], prefix="day")
    stock_data["log_return"] = np.log1p(stock_data["close"].pct_change())

    stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    stock_data.dropna(inplace=True)

    if stock_data.empty:
        return None, None, None, None, None

    if "volume" in stock_data.columns:
        stock_data["volume"] = winsorize(stock_data["volume"], limits=[0.01, 0.01])

    numeric_cols = stock_data.select_dtypes(include=np.number).columns.tolist()
    feature_columns = [col for col in numeric_cols if col not in ["open", "high", "low"]]
    output_columns = ["close", "volume", "RSI_14", "log_return"]

    for col in output_columns:
        if col not in feature_columns:
            return None, None, None, None, None

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(stock_data[feature_columns])

    return stock_data, data_normalized, scaler, feature_columns, output_columns


def train_global_base_model(tickers_list: list, figi_map: dict):
    """Обучение общей модели на данных нескольких тикеров."""
    logger.warning("GLOBAL MODEL: Запуск обучения материнской модели...")

    macro_data = get_combined_macro_data()
    cross_asset_data = get_cross_asset_data(days=365 * 2)

    train_window = ML_CONFIG["TRAIN_WINDOW"]
    all_X, all_y = [], []
    input_size = 0
    output_size = 0

    for ticker in tickers_list:
        figi = figi_map.get(ticker)
        if not figi:
            continue

        try:
            raw_df = get_historical_data(figi, days=365 * 2)
            if raw_df.empty:
                continue

            raw_df["time_date"] = raw_df["time"].dt.normalize()
            if not cross_asset_data.empty:
                raw_df = pd.merge_asof(
                    raw_df.sort_values("time_date"),
                    cross_asset_data,
                    left_on="time_date",
                    right_on="time",
                    direction="backward",
                )
            if not macro_data.empty:
                raw_df = pd.merge_asof(
                    raw_df.sort_values("time_date"),
                    macro_data,
                    left_on="time_date",
                    right_on="time",
                    direction="backward",
                )

            raw_df.drop(
                columns=[col for col in raw_df.columns if col.startswith("time_y") or col == "time_date"],
                inplace=True,
                errors="ignore",
            )
            raw_df.rename(columns={"time_x": "time"}, inplace=True, errors="ignore")

            raw_df = raw_df.ffill().bfill()

            _, data_norm, _, feats, outs = prepare_features_and_scale(raw_df)

            if data_norm is None:
                continue

            input_size = len(feats)
            output_size = len(outs)
            output_indices = [feats.index(col) for col in outs]

            for i in range(len(data_norm) - train_window):
                all_X.append(data_norm[i : i + train_window])
                all_y.append(data_norm[i + train_window, output_indices])

        except Exception as e:
            logger.error(f"GLOBAL MODEL: Ошибка с {ticker}: {e}")
            continue

    if not all_X:
        logger.error("GLOBAL MODEL: Не удалось собрать данные!")
        return

    X_tensor = torch.FloatTensor(np.array(all_X))
    y_tensor = torch.FloatTensor(np.array(all_y))

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=ML_CONFIG["BATCH_SIZE_GLOBAL"], shuffle=True)

    model = LSTMTransformerModel(input_size=input_size, output_size=output_size, **LSTM_PARAMS)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = FocalLoss(1.0, [2.0, 1.0, 0.5, 1.5])

    epochs = ML_CONFIG["EPOCHS_GLOBAL"]
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            logger.info(f"GLOBAL EPOCH {epoch+1}: Loss {total_loss/len(loader):.5f}")

    torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
    logger.warning(f"GLOBAL MODEL: Сохранена в {GLOBAL_MODEL_PATH}")


def train_and_predict_lstm(stock_data: pd.DataFrame, ticker: str, future_predictions: int = 30):
    """Fine-tuning модели для конкретного тикера и прогноз."""
    current_config = ML_CONFIG.copy()
    params_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}_params.json"
    if os.path.exists(params_path):
        try:
            with open(params_path, "r") as f:
                tuned_params = json.load(f)
            if "hidden_size" in tuned_params:
                current_config["HIDDEN_SIZE"] = tuned_params["hidden_size"]
            if "num_layers" in tuned_params:
                current_config["NUM_LAYERS"] = tuned_params["num_layers"]
            if "dropout" in tuned_params:
                current_config["DROPOUT"] = tuned_params["dropout"]

            logger.info(f"[{ticker}] Используются оптимизированные параметры Optuna")
        except Exception as e:
            logger.error(f"Ошибка загрузки params.json: {e}")
    days_history = (stock_data["time"].max() - stock_data["time"].min()).days + 90
    macro_data = get_combined_macro_data()
    cross_asset_data = get_cross_asset_data(days=days_history)

    stock_data["time_date"] = stock_data["time"].dt.normalize()
    if not cross_asset_data.empty:
        stock_data = pd.merge_asof(
            stock_data.sort_values("time_date"),
            cross_asset_data,
            left_on="time_date",
            right_on="time",
            direction="backward",
        )
    if not macro_data.empty:
        stock_data = pd.merge_asof(
            stock_data.sort_values("time_date"), macro_data, left_on="time_date", right_on="time", direction="backward"
        )

    stock_data.drop(
        columns=[col for col in stock_data.columns if col.startswith("time_y") or col == "time_date"],
        inplace=True,
        errors="ignore",
    )
    stock_data.rename(columns={"time_x": "time"}, inplace=True, errors="ignore")
    stock_data = stock_data.ffill().bfill()

    processed_data, data_normalized, scaler, feature_columns, output_columns = prepare_features_and_scale(stock_data)

    if data_normalized is None:
        return None, None, [], [], [], {}, None, {}

    train_window = ML_CONFIG["TRAIN_WINDOW"]
    output_indices = [feature_columns.index(col) for col in output_columns]
    X_all, y_all = [], []
    for i in range(len(data_normalized) - train_window):
        X_all.append(data_normalized[i : i + train_window])
        y_all.append(data_normalized[i + train_window, output_indices])

    full_dataset = TensorDataset(torch.FloatTensor(np.array(X_all)), torch.FloatTensor(np.array(y_all)))
    loader = DataLoader(full_dataset, batch_size=ML_CONFIG["BATCH_SIZE_FINE_TUNE"], shuffle=False)

    model = LSTMTransformerModel(
        input_size=len(feature_columns),
        hidden_size=current_config["HIDDEN_SIZE"],
        num_layers=current_config["NUM_LAYERS"],
        n_head=ML_CONFIG["N_HEAD"],
        dropout=current_config["DROPOUT"],
        output_size=len(output_columns),
    )
    params_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}_params.json"

    if os.path.exists(GLOBAL_MODEL_PATH):
        logger.info(f"[{ticker}] Transfer Learning: Загрузка весов...")
        try:
            model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
            learning_rate = ML_CONFIG["LEARNING_RATE_TRANSFER"]
            epochs = ML_CONFIG["EPOCHS_TRANSFER"]
        except Exception as e:
            logger.error(f"[{ticker}] Ошибка загрузки глобальной модели: {e}")
            learning_rate = ML_CONFIG["LEARNING_RATE_FINE_TUNE"]
            epochs = ML_CONFIG["EPOCHS_FINE_TUNE"]
    else:
        learning_rate = ML_CONFIG["LEARNING_RATE_FINE_TUNE"]
        epochs = ML_CONFIG["EPOCHS_FINE_TUNE"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = FocalLoss(gamma=ML_CONFIG["FOCAL_LOSS_GAMMA"], weights=ML_CONFIG["FOCAL_LOSS_WEIGHTS"])

    model.train()
    for _ in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
    try:
        train_size_simple = int(len(processed_data) * 0.9)
        train_df = processed_data.iloc[:train_size_simple]
        prophet_train_df = train_df.reset_index().rename(columns={"time": "ds", "close": "y"})
        prophet_train_df["ds"] = prophet_train_df["ds"].dt.tz_localize(None)
        m = Prophet(daily_seasonality=True)
        m.fit(prophet_train_df)
        future = m.make_future_dataframe(periods=future_predictions)
        prophet_preds = m.predict(future)["yhat"].tail(future_predictions).values
    except Exception:
        prophet_preds = np.full(future_predictions, processed_data["close"].iloc[-1])

    try:
        arima_model = ARIMA(processed_data["close"], order=(5, 1, 0)).fit()
        arima_preds = arima_model.forecast(steps=future_predictions).values
    except Exception:
        arima_preds = np.full(future_predictions, processed_data["close"].iloc[-1])

    mean_lstm, upper_raw, lower_raw = predict_with_uncertainty(
        model,
        scaler,
        data_normalized[-train_window:],
        processed_data,
        feature_columns,
        output_columns,
        future_predictions,
    )

    w_lstm, w_prophet, w_arima = 0.6, 0.2, 0.2
    ensemble_preds = (mean_lstm * w_lstm) + (prophet_preds * w_prophet) + (arima_preds * w_arima)

    uncertainty = (upper_raw - lower_raw) / 2
    final_upper = ensemble_preds + uncertainty
    final_lower = ensemble_preds - uncertainty

    final_preds = pd.Series(ensemble_preds).ewm(alpha=0.5).mean().values
    final_upper = pd.Series(final_upper).ewm(alpha=0.5).mean().values
    final_lower = pd.Series(final_lower).ewm(alpha=0.5).mean().values

    return model, scaler, final_preds, final_upper, final_lower, {}, None, {}


def train_drawdown_model(stock_data: pd.DataFrame, ticker: str):
    """Training the drawdown prediction model."""
    if "volume" not in stock_data.columns:
        stock_data["volume"] = 0
    features = stock_data[["close", "volume"]].astype(float)
    lookahead_period = 10
    min_price_in_future = stock_data["low"].rolling(window=lookahead_period).min().shift(-lookahead_period)
    drawdown = (stock_data["close"] - min_price_in_future) / stock_data["close"]
    data = features.copy()
    data["target_drawdown"] = drawdown
    data.dropna(inplace=True)
    if data.empty:
        return None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)

    train_window = 30

    def create_sequences(input_data, tw):
        X, y = [], []
        for i in range(len(input_data) - tw):
            X.append(input_data[i : i + tw, 0:2])
            y.append(input_data[i + tw, 2])
        return np.array(X), np.array(y)

    X_data, y_data = create_sequences(data_normalized, train_window)
    X_train = torch.FloatTensor(X_data)
    y_train = torch.FloatTensor(y_data).view(-1, 1)

    model = DrawdownLSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = ML_CONFIG["EPOCHS_FINE_TUNE"]
    batch_size = ML_CONFIG["BATCH_SIZE_FINE_TUNE"]
    for _ in range(epochs):
        for j in range(0, len(X_train), batch_size):
            X_batch, y_batch = X_train[j : j + batch_size], y_train[j : j + batch_size]
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    return model, scaler


def run_walk_forward_validation(stock_data: pd.DataFrame, ticker: str, window_size=252, step_size=60):

    wfa_metrics = {"total_pnl": 0.0, "total_trades": 0, "winning_trades": 0, "meta_data": []}

    if len(stock_data) < window_size + step_size:
        return None

    processed_data, data_normalized, scaler, feature_columns, output_columns = prepare_features_and_scale(stock_data)
    if data_normalized is None:
        return None

    total_samples = len(data_normalized)
    train_window_lstm = ML_CONFIG["TRAIN_WINDOW"]

    meta_data = []

    bt_rows = []
    bt_z_scores = []

    model = LSTMTransformerModel(input_size=len(feature_columns), output_size=len(output_columns), **GLOBAL_PARAMS)
    loss_fn = nn.MSELoss()

    start_index = 0
    end_index = window_size
    close_idx = feature_columns.index("close")

    while end_index + step_size <= total_samples:
        if os.path.exists(GLOBAL_MODEL_PATH):
            try:
                model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
            except Exception:
                pass

        train_data_norm = data_normalized[start_index:end_index]
        test_data_norm = data_normalized[end_index : end_index + step_size]

        X_train, y_train = [], []
        for i in range(len(train_data_norm) - train_window_lstm):
            X_train.append(train_data_norm[i : i + train_window_lstm])
            y_train.append(train_data_norm[i + train_window_lstm, close_idx])

        if not X_train:
            break
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(np.array(X_train)), torch.FloatTensor(np.array(y_train)).view(-1, 1)),
            batch_size=64,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=ML_CONFIG["LEARNING_RATE_FINE_TUNE"])
        model.train()
        for _ in range(5):
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(X)[:, 0].view(-1, 1), y)
                loss.backward()
                optimizer.step()

        model.train()
        context_data = data_normalized[end_index - train_window_lstm : end_index + step_size]
        test_slice_start = end_index

        with torch.no_grad():
            for i in range(len(test_data_norm)):
                input_seq = context_data[i : i + train_window_lstm]
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)

                current_norm = input_seq[-1, close_idx]

                mc_preds = [model(input_tensor)[0, 0].item() for _ in range(5)]
                pred_norm_mean = np.mean(mc_preds)
                uncertainty = np.std(mc_preds)

                dummy = np.zeros((1, len(feature_columns)))
                dummy[0, close_idx] = current_norm
                price_c = scaler.inverse_transform(dummy)[0, close_idx]

                current_idx_raw = test_slice_start + i
                if current_idx_raw >= len(processed_data):
                    break
                raw_row = processed_data.iloc[current_idx_raw]

                atr_rub = raw_row.get("ATRr_14", price_c * 0.02)
                if atr_rub == 0:
                    atr_rub = price_c * 0.01

                dummy[0, close_idx] = pred_norm_mean
                pred_rub = scaler.inverse_transform(dummy)[0, close_idx]

                if pred_rub > price_c:
                    dummy[0, close_idx] = pred_norm_mean - uncertainty
                    robust_rub = scaler.inverse_transform(dummy)[0, close_idx]
                    z_score_robust = (robust_rub - price_c) / atr_rub
                else:
                    dummy[0, close_idx] = pred_norm_mean + uncertainty
                    robust_rub = scaler.inverse_transform(dummy)[0, close_idx]
                    z_score_robust = (robust_rub - price_c) / atr_rub
                bt_rows.append(raw_row)
                bt_z_scores.append(z_score_robust)
                adx = raw_row.get("ADX_14", 20)
                volume_ratio = raw_row.get("volume_ratio", 1.0)
                z_thresh = 0.5 if adx < 25 else 0.3

                signal = 0
                if z_score_robust > z_thresh and volume_ratio > 0.5:
                    signal = 1
                elif z_score_robust < -z_thresh and volume_ratio > 0.5:
                    signal = -1

                if signal != 0:
                    exit_idx = min(i + 5, len(test_data_norm) - 1)
                    true_norm = test_data_norm[exit_idx, close_idx]
                    dummy[0, close_idx] = true_norm
                    price_t = scaler.inverse_transform(dummy)[0, close_idx]
                    pnl = price_t - price_c if signal == 1 else price_c - price_t

                    meta_data.append(
                        {
                            "z_score": float(z_score_robust),
                            "z_threshold": float(z_thresh),
                            "rsi": float(raw_row.get("RSI_14", 50)),
                            "adx": float(adx),
                            "volume_ratio": float(volume_ratio),
                            "atr_pct": (atr_rub / price_c) * 100,
                            "macd": float(raw_row.get("MACD_12_26_9", 0)),
                            "ema_bias": 1 if price_c > raw_row.get("EMA_50", price_c) else -1,
                            "volatility_norm": float(uncertainty),
                            "lstm_confidence": float(abs(pred_norm_mean - current_norm)),
                            "day_of_week": raw_row["time"].dayofweek,
                            "is_month_end": 1 if raw_row["time"].is_month_end else 0,
                            "target": 1 if pnl > 0 else 0,
                        }
                    )

        start_index += step_size
        end_index += step_size

    if not bt_rows:
        return None

    bt_df = pd.DataFrame(bt_rows)
    bt_df["time"] = pd.to_datetime(bt_df["time"])

    bt_df["z_score"] = bt_z_scores

    bt_df = bt_df.drop_duplicates(subset=["time"], keep="last").sort_values("time")

    clean_z_scores = bt_df["z_score"].values

    bt_metrics = run_backtrader_wfa(bt_df, clean_z_scores, z_threshold=0.7)

    logger.info(
        f"[{ticker}] BT WFA: Sharpe={bt_metrics['sharpe']:.2f}, WR={bt_metrics['win_rate']:.1f}%, DD={bt_metrics['drawdown']:.1f}%"
    )

    return {
        "total_pnl": bt_metrics["total_pnl"],
        "win_rate": bt_metrics["win_rate"],
        "trades": bt_metrics["total_trades"],
        "sharpe": bt_metrics["sharpe"],
        "drawdown": bt_metrics["drawdown"],
        "meta_data": wfa_metrics["meta_data"],
    }
