# Файл: app/levels.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def find_key_level_zones(historical_data: pd.DataFrame, order_book: dict, current_price: float) -> list[dict]:
    """
    Находит ценовые зоны поддержки и сопротивления, анализируя историю цен и стакан ордеров.
    """
    if historical_data.empty or len(historical_data) < 50:
        return []

    # --- 1. Анализ исторических данных (Гистограмма) ---
    prices = pd.concat([historical_data["close"], historical_data["high"], historical_data["low"]]).dropna().to_numpy()

    price_range = prices.max() - prices.min()
    if price_range == 0:
        return []

    num_bins = int(price_range / (price_range * 0.005))
    if num_bins < 10:
        num_bins = 10  # Минимальное количество корзин

    hist, bin_edges = np.histogram(prices, bins=num_bins)

    # Отсеиваем пики, которые составляют менее 10% от самого сильного
    peaks, _ = find_peaks(hist, prominence=hist.max() * 0.10)

    zones_from_history = []
    for peak_index in peaks:
        zones_from_history.append(
            {
                "start_price": bin_edges[peak_index],
                "end_price": bin_edges[peak_index + 1],
                "strength_history": int(hist[peak_index]),
            }
        )

    # --- 2. Анализ стакана ордеров (Order Book) ---
    bids = order_book.get("bids", [])  # Заявки на покупку (формируют поддержку)
    asks = order_book.get("asks", [])  # Заявки на продажу (формируют сопротивление)

    all_orders = bids + asks
    zones_from_orders = []

    if all_orders:
        order_prices = np.array([float(order["price"]) for order in all_orders])
        order_volumes = np.array([int(order["quantity"]) for order in all_orders])

        # Создаем гистограмму для объемов в стакане, используя те же границы
        order_hist, _ = np.histogram(order_prices, bins=bin_edges, weights=order_volumes)

        # Ищем пики в стакане. `prominence` здесь меньше, т.к. объемы могут быть распределены
        order_peaks, _ = find_peaks(order_hist, prominence=order_hist.max() * 0.05 if order_hist.max() > 0 else 1)

        for peak_index in order_peaks:
            zones_from_orders.append(
                {
                    "start_price": bin_edges[peak_index],
                    "end_price": bin_edges[peak_index + 1],
                    "strength_order_book": int(order_hist[peak_index]),
                }
            )

    # --- 3. Объединение и финальный расчет ---

    # Объединяем зоны из истории и стакана в один DataFrame
    df_hist = pd.DataFrame(zones_from_history)
    df_orders = pd.DataFrame(zones_from_orders)

    if df_hist.empty and df_orders.empty:
        return []
    elif df_hist.empty:
        merged_df = df_orders
    elif df_orders.empty:
        merged_df = df_hist
    else:
        merged_df = pd.merge(df_hist, df_orders, on=["start_price", "end_price"], how="outer")

    merged_df = merged_df.fillna(0)  # Заполняем пропуски нулями

    # --- ВОТ ГДЕ ИСПОЛЬЗУЕТСЯ СТАКАН ---
    # Комбинируем "силу" из истории и из стакана.
    # Даем стакану ордеров ВЕС в 5 раз больше, так как это "живые" деньги.
    merged_df["total_strength"] = merged_df["strength_history"] + merged_df.get("strength_order_book", 0) * 5

    final_zones = []
    for _, row in merged_df.iterrows():
        avg_price = (row["start_price"] + row["end_price"]) / 2
        level_type = "support" if avg_price < current_price else "resistance"

        final_zones.append(
            {
                "start_price": float(row["start_price"]),
                "end_price": float(row["end_price"]),
                "level_type": level_type,
                "strength": int(row["total_strength"]),
            }
        )

    if not final_zones:
        return []

    # Нормализуем силу от 0 до 100 для удобства фронтенда
    max_strength = max((z["strength"] for z in final_zones), default=1)
    if max_strength == 0:
        max_strength = 1

    for zone in final_zones:
        zone["intensity"] = int((zone["strength"] / max_strength) * 100)

    # Возвращаем только значимые зоны (с интенсивностью > 5)
    return [z for z in final_zones if z["intensity"] > 5]
