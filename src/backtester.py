"""
Simple backtesting helpers for strategy trade results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Tuple

import pandas as pd

from .strategy import Trade


@dataclass
class BacktestResult:
    """Summary metrics for a collection of trades."""

    trades: List[Trade]
    total_trades: int
    winners: int
    losers: int
    total_pnl: float
    win_rate: float
    average_win: float
    average_loss: float
    expectancy: float
    long_win_rate: float
    short_win_rate: float
    long_average_win: float
    long_average_loss: float
    short_average_win: float
    short_average_loss: float
    equity_curve: pd.Series
    sharpe_ratio: float
    portfolio_max_drawdown: float
    asset_drawdowns: Dict[str, float]
    average_holding_minutes: Dict[str, float]
    total_fees_paid: float


class Backtester:
    """Compute performance statistics given a list of trades."""

    def __init__(self, *, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital

    def run(self, trades: Iterable[Trade]) -> BacktestResult:
        trade_list = [t for t in trades if t.pnl() is not None]

        total_trades = len(trade_list)
        winners = [t for t in trade_list if t.exit_reason == "take_profit"]
        losers = [t for t in trade_list if t.exit_reason == "stop_loss"]

        long_trades = [t for t in trade_list if t.direction == "long"]
        short_trades = [t for t in trade_list if t.direction == "short"]

        long_winners = [t for t in long_trades if t.exit_reason == "take_profit"]
        long_losers = [t for t in long_trades if t.exit_reason == "stop_loss"]
        short_winners = [t for t in short_trades if t.exit_reason == "take_profit"]
        short_losers = [t for t in short_trades if t.exit_reason == "stop_loss"]

        pnl_values = [t.pnl() for t in trade_list if t.pnl() is not None]
        total_pnl = float(sum(pnl_values)) if pnl_values else 0.0

        win_rate = len(winners) / total_trades if total_trades else 0.0

        avg_win = (
            sum(pnl for pnl in pnl_values if pnl is not None and pnl > 0) / len(winners)
            if winners
            else 0.0
        )
        avg_loss = (
            sum(pnl for pnl in pnl_values if pnl is not None and pnl < 0) / len(losers)
            if losers
            else 0.0
        )

        long_pnls = [t.pnl() for t in long_trades if t.pnl() is not None]
        short_pnls = [t.pnl() for t in short_trades if t.pnl() is not None]

        long_avg_win = (
            sum(p for p in long_pnls if p is not None and p > 0) / len(long_winners)
            if long_winners
            else 0.0
        )
        long_avg_loss = (
            sum(p for p in long_pnls if p is not None and p < 0) / len(long_losers)
            if long_losers
            else 0.0
        )
        short_avg_win = (
            sum(p for p in short_pnls if p is not None and p > 0) / len(short_winners)
            if short_winners
            else 0.0
        )
        short_avg_loss = (
            sum(p for p in short_pnls if p is not None and p < 0) / len(short_losers)
            if short_losers
            else 0.0
        )

        long_win_rate = len(long_winners) / len(long_trades) if long_trades else 0.0
        short_win_rate = len(short_winners) / len(short_trades) if short_trades else 0.0

        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss if total_trades else 0.0

        equity_values = [self.initial_capital]
        for pnl in pnl_values:
            equity_values.append(equity_values[-1] + pnl)
        equity_curve = pd.Series(equity_values, name="equity")

        returns = pd.Series(pnl_values) / self.initial_capital if pnl_values else pd.Series(dtype=float)
        sharpe_ratio = (
            (returns.mean() / returns.std()) * (len(returns) ** 0.5)
            if len(returns) > 1 and returns.std() != 0
            else 0.0
        )

        running_max = equity_curve.cummax()
        drawdowns = equity_curve - running_max
        portfolio_max_drawdown = drawdowns.min() if not drawdowns.empty else 0.0

        asset_drawdowns = self._asset_drawdowns(trade_list)
        average_holding_minutes = self._average_holding_minutes(trade_list)

        total_fees_paid = sum(t.fees_paid for t in trade_list)

        return BacktestResult(
            trades=trade_list,
            total_trades=total_trades,
            winners=len(winners),
            losers=len(losers),
            total_pnl=total_pnl,
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss,
            expectancy=expectancy,
            long_win_rate=long_win_rate,
            short_win_rate=short_win_rate,
            long_average_win=long_avg_win,
            long_average_loss=long_avg_loss,
            short_average_win=short_avg_win,
            short_average_loss=short_avg_loss,
            equity_curve=equity_curve,
            sharpe_ratio=float(sharpe_ratio),
            portfolio_max_drawdown=float(portfolio_max_drawdown),
            asset_drawdowns=asset_drawdowns,
            average_holding_minutes=average_holding_minutes,
            total_fees_paid=float(total_fees_paid),
        )

    def _asset_drawdowns(self, trades: List[Trade]) -> Dict[str, float]:
        asset_to_drawdown: Dict[str, float] = {}
        for symbol in {trade.symbol for trade in trades}:
            symbol_trades = [t for t in trades if t.symbol == symbol]
            pnl_values = [t.pnl() for t in symbol_trades if t.pnl() is not None]
            equity_values = [0.0]
            for pnl in pnl_values:
                equity_values.append(equity_values[-1] + pnl)
            equity_curve = pd.Series(equity_values)
            drawdown = (equity_curve - equity_curve.cummax()).min() if not equity_curve.empty else 0.0
            asset_to_drawdown[symbol] = float(drawdown)
        return asset_to_drawdown

    def _average_holding_minutes(self, trades: List[Trade]) -> Dict[str, float]:
        asset_to_minutes: Dict[str, float] = {}
        for symbol in {trade.symbol for trade in trades}:
            durations = [
                t.holding_period().total_seconds() / 60
                for t in trades
                if t.symbol == symbol and t.holding_period() is not None
            ]
            asset_to_minutes[symbol] = float(sum(durations) / len(durations)) if durations else 0.0
        return asset_to_minutes

