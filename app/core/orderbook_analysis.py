from celery.utils.log import get_task_logger
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = get_task_logger(__name__)


def find_walls(order_book: dict, avg_daily_volume: float) -> dict:
    """
    Ищет аномально крупные заявки ('стенки') в стакане.
    Возвращает словарь с информацией о самой большой заявке на покупку и продажу.
    """
    walls = {"bid_wall": None, "ask_wall": None}

    # Критерий "стенки": заявка, объем которой составляет > 2% от среднесуточного объема.
    # Этот порог можно настраивать.
    if avg_daily_volume == 0:
        avg_daily_volume = 1  # Избегаем деления на ноль
    wall_threshold = avg_daily_volume * 0.02

    if order_book.get("bids"):
        try:
            # Находим самую крупную заявку (по количеству лотов)
            largest_bid = max(order_book["bids"], key=lambda x: x["quantity"])
            if largest_bid["quantity"] > wall_threshold:
                walls["bid_wall"] = largest_bid
        except (ValueError, KeyError):
            pass  # Если стакан пустой или имеет неверный формат

    if order_book.get("asks"):
        try:
            largest_ask = max(order_book["asks"], key=lambda x: x["quantity"])
            if largest_ask["quantity"] > wall_threshold:
                walls["ask_wall"] = largest_ask
        except (ValueError, KeyError):
            pass

    return walls


def validate_breakout(ticker: str, breakout_price: float, direction: str, db: Session) -> dict:
    """
    Проверяет пробой на соответствие глобальному контексту (зоны, тренд, сентимент).
    Возвращает словарь с оценкой качества сигнала и деталями.
    """
    score = 50  # Базовая оценка
    reasons = []

    # 1. Проверка №1: Связь с Ключевыми Зонами (Вес: до +40 баллов)
    level_type_to_check = "resistance" if direction == "UP" else "support"
    zones_query = text(
        "SELECT start_price, end_price, intensity FROM key_levels WHERE ticker = :ticker AND level_type = :type"
    )
    zones = db.execute(zones_query, {"ticker": ticker, "type": level_type_to_check}).fetchall()

    zone_hit = None
    for z in zones:
        if z.start_price * 0.99 <= breakout_price <= z.end_price * 1.01:
            score += 30
            zone_hit = z
            reasons.append(f"Пробой совпадает с зоной {level_type_to_check} (Интенс.: {z.intensity}%)")
            if z.intensity > 80:
                score += 10
                reasons.append("Пробиваемая зона очень сильная (>80%)")
            break

    # 2. Проверка №2: Соответствие Глобальному Тренду (Вес: +20 / -40 баллов)
    bias = db.execute(
        text("SELECT global_bias FROM tracked_tickers WHERE ticker = :ticker"), {"ticker": ticker}
    ).scalar()

    if (direction == "UP" and bias == "Bullish") or (direction == "DOWN" and bias == "Bearish"):
        score += 20
        reasons.append(f"Соответствует глобальному тренду ({bias})")
    elif (direction == "UP" and bias == "Bearish") or (direction == "DOWN" and bias == "Bullish"):
        score -= 40
        reasons.append(f"Движение ПРОТИВ глобального тренда ({bias})")

    # 3. Проверка №3: Соответствие Сентименту (Вес: +/- 15 баллов)
    sentiment = (
        db.execute(
            text("SELECT sentiment_score FROM tracked_tickers WHERE ticker = :ticker"), {"ticker": ticker}
        ).scalar()
        or 0.0
    )

    if sentiment > 0.3:
        score += 15
        reasons.append(f"Поддерживается позитивным сентиментом ({sentiment:.2f})")
    elif sentiment < -0.3:
        score -= 15
        reasons.append(f"Происходит на фоне негативного сентимента ({sentiment:.2f})")

    # 4. Финальный результат
    final_score = max(0, min(100, int(score)))

    return {"score": final_score, "reasons": reasons, "zone_hit": zone_hit}
