import app.db.models as models
from sqlalchemy.orm import Session


def calculate_fixed_risk_position_size(
    account_size: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float,
    lot_size: int = 1, 
    direction: str = "LONG",
) -> dict:
    """Рассчитывает размер позиции на основе фиксированного процента риска."""

    direction = (direction or "LONG").upper()
    if direction not in {"LONG", "SHORT"}:
        return {"error": "Invalid direction. Use LONG or SHORT."}
    if direction == "LONG" and entry_price <= stop_loss_price:
        return {"error": "For LONG, entry price must be above stop-loss price."}
    if direction == "SHORT" and entry_price >= stop_loss_price:
        return {"error": "For SHORT, stop-loss price must be above entry price."}

    # 1. Рассчитываем сумму риска в деньгах
    risk_amount = account_size * (risk_percent / 100.0)

    # 2. Рассчитываем риск на одну акцию
    risk_per_share = abs(entry_price - stop_loss_price)

    # 3. Рассчитываем, сколько акций мы можем себе позволить купить
    num_shares = risk_amount / risk_per_share

    # 4. Округляем до количества лотов
    num_lots = int(num_shares / lot_size)

    if num_lots == 0:
        return {"error": f"Risk too high for account size {account_size:,.2f} and risk {risk_percent}%."}

    position_value = num_lots * lot_size * entry_price

    return {
        "method": "Fixed Percentage Risk",
        "position_size_lots": num_lots,
        "position_size_shares": num_lots * lot_size,
        "position_value": round(position_value, 2),
        "risk_on_trade_percent": risk_percent,
        "risk_on_trade_value": round(risk_amount, 2),
        "description": (
            f"Для риска в {risk_percent}% ({risk_amount:,.2f} руб.) от депозита {account_size:,.2f} руб. "
            f"рекомендуется купить {num_lots} лот(ов)."
        ),
    }


def calculate_kelly_criterion_position_size(
    db: Session, account_size: float, entry_price: float, stop_loss_price: float, strategy_name: str, lot_size: int = 1, direction: str = "LONG"
) -> dict:
    """
    Рассчитывает размер позиции, используя Критерий Келли, на основе
    исторической эффективности конкретной торговой стратегии.
    """
    direction = (direction or "LONG").upper()
    if direction not in {"LONG", "SHORT"}:
        return {"error": "Invalid direction. Use LONG or SHORT."}
    if direction == "LONG" and entry_price <= stop_loss_price:
        return {"error": "For LONG, entry price must be above stop-loss price."}
    if direction == "SHORT" and entry_price >= stop_loss_price:
        return {"error": "For SHORT, stop-loss price must be above entry price."}

    stats = (
        db.query(models.StrategyPerformance).filter(models.StrategyPerformance.strategy_name == strategy_name).first()
    )

    if not stats or stats.trades_count < 10:
        return {
            "error": f"Недостаточно данных для стратегии '{strategy_name}' (нужно мин. 10 сделок). Используйте метод фиксированного риска."
        }

    win_rate = stats.win_rate
    if stats.avg_loss_pct == 0:
        return {"error": "Средний убыток по стратегии равен нулю, невозможно рассчитать R/R."}
    reward_to_risk_ratio = abs(stats.avg_profit_pct / stats.avg_loss_pct)

    kelly_fraction = win_rate - ((1 - win_rate) / reward_to_risk_ratio)

    if kelly_fraction <= 0:
        return {
            "error": f"Стратегия '{strategy_name}' имеет отрицательное мат. ожидание (Kelly={kelly_fraction:.2f}). Торговать не рекомендуется."
        }

    conservative_kelly = kelly_fraction * 0.5

    position_value = account_size * conservative_kelly
    num_shares = position_value / entry_price
    num_lots = int(num_shares / lot_size)

    if num_lots == 0:
        return {"error": "Deposit too small for at least one lot (Kelly)."}

    final_position_value = num_lots * lot_size * entry_price
    risk_on_trade_value = abs(entry_price - stop_loss_price) * num_lots * lot_size
    risk_on_trade_percent = (risk_on_trade_value / account_size) * 100

    return {
        "method": "Kelly Criterion (Conservative)",
        "position_size_lots": num_lots,
        "position_size_shares": num_lots * lot_size,
        "position_value": round(final_position_value, 2),
        "risk_on_trade_percent": round(risk_on_trade_percent, 2),
        "risk_on_trade_value": round(risk_on_trade_value, 2),
        "description": (
            f"На основе {stats.trades_count} сделок (Winrate: {win_rate*100:.1f}%), критерий Келли "
            f"рекомендует инвестировать {conservative_kelly*100:.1f}% вашего капитала в эту сделку."
        ),
    }
