from datetime import datetime

from app.db.models import Portfolio, Position, TrackedTicker
from sqlalchemy.orm import Session


class PortfolioService:
    def __init__(self, db: Session):
        self.db = db

    def create_portfolio(self, user_id: int):
        """Создает новый портфель, если его нет."""
        port = self.db.query(Portfolio).filter_by(user_id=user_id).first()
        if not port:
            port = Portfolio(user_id=user_id, initial_balance=1000000.0, current_balance=1000000.0)
            self.db.add(port)
            self.db.commit()
        return port

    def get_portfolio_summary(self, user_id: int):
        """Возвращает полную статистику портфеля."""
        port = self.db.query(Portfolio).filter_by(user_id=user_id).first()
        if not port:
            return None

        positions = self.db.query(Position).filter_by(portfolio_id=port.id, status="OPEN").all()

        total_invested = 0.0
        total_current_value = 0.0

        pos_data = []
        for p in positions:
            # Текущая цена (обновляется воркером) или цена входа, если нет данных
            curr_price = p.current_price if p.current_price else p.entry_price

            # Стоимость позиции
            cost_basis = p.entry_price * p.quantity
            market_value = curr_price * p.quantity

            unrealized_pnl = market_value - cost_basis
            pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0

            total_invested += cost_basis
            total_current_value += market_value

            pos_data.append(
                {
                    "id": p.id,
                    "ticker": p.ticker,
                    "direction": p.direction,
                    "entry": p.entry_price,
                    "current": curr_price,
                    "qty": p.quantity,
                    "pnl": round(unrealized_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                }
            )

        equity = port.current_balance + total_current_value
        total_pnl = equity - port.initial_balance
        total_pnl_pct = (total_pnl / port.initial_balance) * 100

        return {
            "cash": round(port.current_balance, 2),
            "equity": round(equity, 2),
            "pnl": round(total_pnl, 2),
            "pnl_pct": round(total_pnl_pct, 2),
            "positions": pos_data,
        }

    def open_position(
        self, user_id: int, ticker: str, price: float, direction: str = "LONG", amount_rub: float = 50000
    ):
        """Открывает позицию на фиксированную сумму."""
        port = self.get_portfolio(user_id)
        if not port:
            return {"error": "Портфель не найден"}

        ticker_info = self.db.query(TrackedTicker).filter_by(ticker=ticker).first()
        if not ticker_info:
            return {"error": "Тикер не найден"}

        lot_size = 1 

        # Расчет количества
        if amount_rub > port.current_balance:
            return {"error": f"Недостаточно средств. Баланс: {port.current_balance}"}

        qty = int(amount_rub / (price * lot_size))
        if qty < 1:
            return {"error": "Сумма слишком мала для покупки 1 лота"}

        total_cost = qty * price * lot_size

        commission = total_cost * 0.0005
        final_cost = total_cost + commission

        if final_cost > port.current_balance:
            qty -= 1  
            total_cost = qty * price * lot_size
            final_cost = total_cost * 1.0005

        port.current_balance -= final_cost

        new_pos = Position(
            portfolio_id=port.id,
            ticker=ticker,
            figi=ticker_info.figi,
            direction=direction,
            entry_price=price,
            current_price=price,
            quantity=qty * lot_size,
            status="OPEN",
        )
        self.db.add(new_pos)
        self.db.commit()

        return {"status": "ok", "msg": f"Куплено {qty} лотов {ticker} по {price}"}

    def close_position(self, user_id: int, position_id: int, current_price: float):
        """Закрывает позицию."""
        pos = self.db.query(Position).filter_by(id=position_id, status="OPEN").first()
        if not pos:
            return {"error": "Позиция не найдена"}

        port = pos.portfolio
        if port.user_id != user_id:
            return {"error": "Доступ запрещен"}

        proceeds = pos.quantity * current_price
        commission = proceeds * 0.0005
        final_proceeds = proceeds - commission

        entry_cost = pos.quantity * pos.entry_price
        pnl = final_proceeds - (entry_cost + (entry_cost * 0.0005))  

        port.current_balance += final_proceeds
        pos.status = "CLOSED"
        pos.exit_price = current_price
        pos.exit_date = datetime.utcnow()
        pos.pnl = pnl

        self.db.commit()
        return {"status": "ok", "msg": f"Закрыто {pos.ticker}. PnL: {pnl:.2f}"}

    def get_portfolio(self, user_id):
        return self.db.query(Portfolio).filter_by(user_id=user_id).first()
