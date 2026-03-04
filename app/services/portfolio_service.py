from datetime import datetime

from app.db.models import Portfolio, Position, TrackedTicker
from sqlalchemy.orm import Session

COMMISSION_RATE = 0.0005


class PortfolioService:
    def __init__(self, db: Session):
        self.db = db

    def create_portfolio(self, user_id: int):
        """Создает новый портфель, если его нет."""
        port = self.db.query(Portfolio).filter_by(user_id=user_id).first()
        if not port:
            port = Portfolio(user_id=user_id, initial_capital=1000000.0, current_capital=1000000.0)
            self.db.add(port)
            self.db.commit()
        return port

    def get_portfolio_summary(self, user_id: int):
        """Returns full portfolio summary."""
        port = self.db.query(Portfolio).filter_by(user_id=user_id).first()
        if not port:
            return None

        positions = self.db.query(Position).filter_by(portfolio_id=port.id, status="OPEN").all()

        total_invested = 0.0
        total_current_value = 0.0

        pos_data = []
        for p in positions:
            curr_price = p.current_price if p.current_price else p.entry_price
            direction = (p.direction or "LONG").upper()

            cost_basis = p.entry_price * p.quantity
            if direction == "SHORT":
                market_value = -curr_price * p.quantity
                unrealized_pnl = (p.entry_price - curr_price) * p.quantity
            else:
                market_value = curr_price * p.quantity
                unrealized_pnl = market_value - cost_basis

            pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0

            total_invested += cost_basis
            total_current_value += market_value

            pos_data.append(
                {
                    "id": p.id,
                    "ticker": p.ticker,
                    "direction": direction,
                    "entry": p.entry_price,
                    "current": curr_price,
                    "qty": p.quantity,
                    "pnl": round(unrealized_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                }
            )

        equity = port.current_capital + total_current_value
        total_pnl = equity - port.initial_capital
        total_pnl_pct = (total_pnl / port.initial_capital) * 100

        return {
            "cash": round(port.current_capital, 2),
            "equity": round(equity, 2),
            "pnl": round(total_pnl, 2),
            "pnl_pct": round(total_pnl_pct, 2),
            "positions": pos_data,
        }

    def open_position(
        self, user_id: int, ticker: str, price: float, direction: str = "LONG", amount_rub: float = 50000
    ):
        """Opens a position for a fixed RUB amount."""
        port = self.get_portfolio(user_id)
        if not port:
            return {"error": "Portfolio not found"}

        ticker_info = self.db.query(TrackedTicker).filter_by(ticker=ticker).first()
        if not ticker_info:
            return {"error": "Ticker not found"}

        direction = (direction or "LONG").upper()
        if direction not in {"LONG", "SHORT"}:
            return {"error": "Invalid direction. Use LONG or SHORT."}

        lot_size = 1

        if amount_rub > port.current_capital:
            return {"error": f"Insufficient funds. Balance: {port.current_capital}"}

        qty = int(amount_rub / (price * lot_size))
        if qty < 1:
            return {"error": "Amount too small to buy 1 lot"}

        notional = qty * price * lot_size
        commission = notional * COMMISSION_RATE

        if direction == "LONG":
            total_cost = notional + commission
            if total_cost > port.current_capital:
                qty -= 1
                if qty < 1:
                    return {"error": "Amount too small to buy 1 lot"}
                notional = qty * price * lot_size
                commission = notional * COMMISSION_RATE
                total_cost = notional + commission
            port.current_capital -= total_cost
            msg = f"Bought {qty} lots {ticker} at {price}"
        else:
            proceeds = notional - commission
            port.current_capital += proceeds
            msg = f"Opened short {qty} lots {ticker} at {price}"

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

        return {"status": "ok", "msg": msg}

    def close_position(self, user_id: int, position_id: int, current_price: float):
        """Closes a position."""
        pos = self.db.query(Position).filter_by(id=position_id, status="OPEN").first()
        if not pos:
            return {"error": "Position not found"}

        port = pos.portfolio
        if port.user_id != user_id:
            return {"error": "Access denied"}

        direction = (pos.direction or "LONG").upper()
        entry_cost = pos.quantity * pos.entry_price
        entry_commission = entry_cost * COMMISSION_RATE
        exit_value = pos.quantity * current_price
        exit_commission = exit_value * COMMISSION_RATE

        if direction == "SHORT":
            final_cost = exit_value + exit_commission
            port.current_capital -= final_cost
            pnl = (entry_cost - exit_value) - (entry_commission + exit_commission)
        else:
            final_proceeds = exit_value - exit_commission
            port.current_capital += final_proceeds
            pnl = (exit_value - entry_cost) - (entry_commission + exit_commission)

        pos.status = "CLOSED"
        pos.exit_price = current_price
        pos.exit_date = datetime.utcnow()
        pos.pnl = pnl

        self.db.commit()
        return {"status": "ok", "msg": f"Closed {pos.ticker}. PnL: {pnl:.2f}"}

    def get_portfolio(self, user_id):
        return self.db.query(Portfolio).filter_by(user_id=user_id).first()
