# Файл: app/plotting.py

from io import BytesIO

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def plot_forecast(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ticker: str,
) -> BytesIO:
    """
    Создает график с историей, прогнозом и доверительным интервалом.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    historical_data["time"] = pd.to_datetime(historical_data["time"]).dt.tz_localize(None)
    forecast_data["forecast_date"] = pd.to_datetime(forecast_data["forecast_date"]).dt.tz_localize(None)

    history_to_plot = historical_data.tail(60)

    # Чтобы линии соединялись, добавляем последнюю историческую точку в начало прогноза
    last_hist_point = history_to_plot.iloc[-1:]

    plot_dates = pd.concat([pd.Series(last_hist_point["time"].iloc[0]), forecast_data["forecast_date"]]).reset_index(
        drop=True
    )

    plot_values = np.concatenate(([last_hist_point["close"].iloc[0]], forecast_data["forecast_value"]))
    plot_upper = np.concatenate(([last_hist_point["close"].iloc[0]], upper_bound))
    plot_lower = np.concatenate(([last_hist_point["close"].iloc[0]], lower_bound))

    # Отрисовка линий
    ax.plot(history_to_plot["time"], history_to_plot["close"], label="Цена (история)", color="royalblue", linewidth=2)

    ax.plot(
        plot_dates,
        plot_values,
        label="Прогноз (Ансамбль)",
        color="darkorange",
        linestyle="--",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    # Добавляем доверительный интервал как закрашенную область
    ax.fill_between(plot_dates, plot_lower, plot_upper, color="darkorange", alpha=0.2, label="95% Дов. интервал")

    ax.set_title(f"Прогноз цены для {ticker} на 30 дней", fontsize=16, fontweight="bold")
    ax.set_ylabel("Цена (руб.)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=45)

    fig.text(0.9, 0.03, "NeuroVision Bot", fontsize=12, color="gray", ha="right", va="bottom", alpha=0.7)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    return buf
