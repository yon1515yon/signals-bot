# Файл: app/backtest_engine.py

import backtrader as bt
import pandas as pd


class NeuralStrategy(bt.Strategy):
    params = (
        ("z_scores", []),
        ("z_threshold", 1.0),
        ("sl_mult", 3.0),
        ("tp_mult", 3.0),
        ("max_hold", 14),
        ("risk_per_trade", 0.95),  
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.z_scores = self.p.z_scores
        self.order = None
        self.entry_bar = 0

    def next(self):
        if self.order:
            return

        idx = len(self) - 1
        if idx >= len(self.z_scores):
            return

        z_score = self.z_scores[idx]


        if self.position:
            if len(self) - self.entry_bar >= self.p.max_hold:
                self.close()
                return

        if not self.position:
            cash = self.broker.get_cash()
            price = self.data.close[0]

            size = (cash * self.p.risk_per_trade) / price

            size = int(size)
            if size < 1:
                return  

            if z_score > self.p.z_threshold:
                sl_price = price - (self.atr[0] * self.p.sl_mult)
                tp_price = price + (self.atr[0] * self.p.sl_mult * self.p.tp_mult)

                self.entry_bar = len(self)
                self.order = self.buy(size=size)
                self.sell(exectype=bt.Order.Stop, price=sl_price, size=size)
                self.sell(exectype=bt.Order.Limit, price=tp_price, size=size)

            elif z_score < -self.p.z_threshold:
                sl_price = price + (self.atr[0] * self.p.sl_mult)
                tp_price = price - (self.atr[0] * self.p.sl_mult * self.p.tp_mult)

                self.entry_bar = len(self)
                self.order = self.sell(size=size)
                self.buy(exectype=bt.Order.Stop, price=sl_price, size=size)
                self.buy(exectype=bt.Order.Limit, price=tp_price, size=size)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                pass
            else:
                pass
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass
        self.order = None


def run_backtrader_wfa(df: pd.DataFrame, z_scores: list, z_threshold=1.0):
    """
    Запускает симуляцию.
    df: Pandas DataFrame с колонками 'open', 'high', 'low', 'close', 'volume', 'time'
    z_scores: Список значений Z-Score для каждой свечи в df.
    """
    if df.empty or len(df) != len(z_scores):
        return {"sharpe": 0, "drawdown": 0, "total_pnl": 0, "win_rate": 0}

    cerebro = bt.Cerebro()

    # Подготовка данных
    data_df = df.copy()
    data_df["time"] = pd.to_datetime(data_df["time"])
    data_df.set_index("time", inplace=True)

    data = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data)

    cerebro.addstrategy(NeuralStrategy, z_scores=z_scores, z_threshold=z_threshold)

    start_cash = 100000.0
    cerebro.broker.setcash(start_cash)

    cerebro.broker.setcommission(commission=0.0005)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1, riskfreerate=0.0
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    try:
        results = cerebro.run()
        strat = results[0]

        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", 0.0)
        if sharpe is None:
            sharpe = 0.0

        dd = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0)

        trade_analysis = strat.analyzers.trades.get_analysis()
        total_closed = trade_analysis.get("total", {}).get("closed", 0)
        won = trade_analysis.get("won", {}).get("total", 0)
        win_rate = (won / total_closed * 100) if total_closed > 0 else 0.0

        final_value = cerebro.broker.getvalue()
        total_pnl = final_value - start_cash

        return {
            "total_pnl": total_pnl,
            "sharpe": sharpe,
            "drawdown": dd,
            "win_rate": win_rate,
            "total_trades": total_closed,
        }
    except Exception as e:
        print(f"Backtrader Error: {e}")
        return {"sharpe": 0, "drawdown": 0, "total_pnl": 0, "win_rate": 0}
