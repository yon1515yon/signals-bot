import numpy as np
import optuna
import pandas as pd
import torch
from celery.utils.log import get_task_logger
from torch.utils.data import DataLoader, TensorDataset

from app.ml.architecture import FocalLoss, LSTMTransformerModel
from app.ml.training import prepare_features_and_scale

logger = get_task_logger(__name__)


def objective(trial, stock_data):
    """
    Целевая функция для Optuna.
    Она строит модель с параметрами, которые предлагает trial,
    и возвращает ошибку (Loss) на валидации.
    """
    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }

    processed, data_norm, scaler, feats, outs = prepare_features_and_scale(stock_data.copy())
    if data_norm is None:
        raise optuna.TrialPruned()

    input_size = len(feats)
    output_size = len(outs)
    output_indices = [feats.index(col) for col in outs]
    train_window = 60

    X, y = [], []
    for i in range(len(data_norm) - train_window):
        X.append(data_norm[i : i + train_window])
        y.append(data_norm[i + train_window, output_indices])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_val = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
    y_train, y_val = torch.FloatTensor(y[:split]), torch.FloatTensor(y[split:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=params["batch_size"], shuffle=False)

    model = LSTMTransformerModel(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        n_head=4, 
        dropout=params["dropout"],
        output_size=output_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = FocalLoss(gamma=1.0)
    for epoch in range(5):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                pred = model(X_v)
                val_loss += loss_fn(pred, y_v).item()

        avg_val_loss = val_loss / len(val_loader)

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return avg_val_loss


def run_tuning_session(ticker: str, stock_data: pd.DataFrame, n_trials=20):
    """Запускает сессию подбора параметров."""
    logger.info(f"[{ticker}] Запуск подбора гиперпараметров (Optuna)...")

    study = optuna.create_study(direction="minimize")

    func = lambda trial: objective(trial, stock_data)

    study.optimize(func, n_trials=n_trials)

    logger.info(f"[{ticker}] Лучшие параметры: {study.best_params}")
    return study.best_params
