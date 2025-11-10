"""
CLI entry point for running the strategy sandbox on historical Binance data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from typing import Optional, cast

from .binance_client import BinanceConfig
from .data_fetcher import fetch_ohlcv_dataframe
from .backtester import Backtester
from .strategy import (
    IntradayRangeStrategy,
    MomentumBreakoutStrategy,
    StopMode,
    StrategySettings,
    Trade,
    TradePreference,
)
def apply_trading_cost(trade: Trade, trading_cost: float) -> Trade:
    if trading_cost <= 0:
        return trade

    if trading_cost >= 1:
        raise SystemExit("--trading-cost must be less than 1 (100%).")

    cost_multiplier = 1 - 2 * trading_cost

    adjusted_trade = replace(trade)
    adjusted_trade.cost_multiplier = cost_multiplier
    gross_notional = trade.entry_price * trade.quantity
    fees_paid = gross_notional * trading_cost * 2
    adjusted_trade.fees_paid = fees_paid
    note_suffix = f"(fees {trading_cost:.4%} per side, total paid {fees_paid:.2f})"
    adjusted_trade.notes = f"{trade.notes} {note_suffix}".strip()

    return adjusted_trade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selectable trading strategies on Binance OHLCV data.",
    )
    parser.add_argument(
        "--strategy",
        choices=["intraday_range", "momentum"],
        default="intraday_range",
        help="Choose which strategy to evaluate.",
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair symbol or comma-separated list for multi-asset runs.",
    )
    parser.add_argument(
        "--timeframe",
        default="5m",
        help="Candle timeframe (Binance-compatible, e.g. 1m, 5m, 15m).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of recent days to backtest.",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR lookback period.",
    )
    parser.add_argument(
        "--reward-multiple",
        type=float,
        default=2.0,
        help="Take-profit multiple of ATR.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Starting balance used for backtest equity curve.",
    )
    parser.add_argument(
        "--session-offset",
        type=int,
        default=0,
        help="Hour offset from UTC for the session start (e.g. -3 for UTC-3).",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory to store cached OHLCV data (set empty string to disable).",
    )
    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=6.0,
        help="Reuse cached OHLCV data younger than this many hours (use 0 to force refresh).",
    )
    parser.add_argument(
        "--plot-equity",
        action="store_true",
        help="Display a matplotlib chart of the equity curve after running the backtest.",
    )
    parser.add_argument(
        "--position-fraction",
        type=float,
        default=1.0,
        help="Fraction of capital to allocate per trade (1.0 = 100%, >1.0 uses leverage).",
    )
    parser.add_argument(
        "--trading-cost",
        type=float,
        default=0.0,
        help="Percentage trading cost applied to each trade (e.g. 0.001 = 0.1%).",
    )
    parser.add_argument(
        "--sma-period",
        type=int,
        default=None,
        help="Optional SMA period to filter trades (only long above SMA, short below).",
    )
    parser.add_argument(
        "--stop-mode",
        choices=["atr", "swing"],
        default="atr",
        help="Choose risk basis: 'atr' uses ATR stop, 'swing' uses distance to recent swing high/low.",
    )
    parser.add_argument(
        "--swing-period",
        type=int,
        default=20,
        help="Lookback length for swing-based stops (only if --stop-mode swing).",
    )
    parser.add_argument(
        "--trade-directions",
        choices=["long", "short", "both"],
        default="both",
        help="Limit trades to long-only, short-only, or both directions.",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Binance testnet (public data still comes from production).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Binance API key (only needed for authenticated endpoints).",
    )
    parser.add_argument(
        "--api-secret",
        default=None,
        help="Optional Binance API secret.",
    )

    return parser.parse_args()


def print_trade_summary(trades: list[Trade]) -> None:
    if not trades:
        print("No trades triggered with the given parameters.")
        return

    winners = [t for t in trades if t.exit_reason == "take_profit"]
    losers = [t for t in trades if t.exit_reason == "stop_loss"]
    pnl_values = [t.pnl() for t in trades if t.pnl() is not None]
    cumulative_pnl = sum(pnl_values) if pnl_values else 0.0

    print(f"Total trades: {len(trades)}")
    print(f"Winners: {len(winners)}")
    print(f"Losers: {len(losers)}")
    print(f"Cumulative PnL (quote currency): {cumulative_pnl:.2f}")
    print()

    for trade in trades:
        pnl = trade.pnl()
        pnl_str = f"{pnl:.2f}" if pnl is not None else "N/A"
        print(
            f"[{trade.entry_time:%Y-%m-%d %H:%M}] {trade.symbol} {trade.direction.upper()} "
            f"entry={trade.entry_price:.2f} stop={trade.stop_loss:.2f} "
            f"target={trade.take_profit:.2f} exit={trade.exit_reason} pnl={pnl_str}"
            f" qty={trade.quantity:.4f}"
        )


def main() -> None:
    args = parse_args()

    config = BinanceConfig(
        api_key=args.api_key,
        api_secret=args.api_secret,
        use_testnet=args.testnet,
    )

    symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No valid symbols provided via --symbol.")

    if args.position_fraction <= 0:
        raise SystemExit("--position-fraction must be greater than zero.")

    end_time = datetime.now(tz=timezone.utc)
    base_settings = StrategySettings(
        atr_period=args.atr_period,
        reward_multiple=args.reward_multiple,
        session_timezone=timezone(timedelta(hours=args.session_offset)),
        sma_period=args.sma_period if args.sma_period and args.sma_period > 0 else None,
        stop_mode=cast(StopMode, args.stop_mode),
        swing_stop_period=args.swing_period if args.swing_period and args.swing_period > 0 else 20,
        trade_directions=cast(TradePreference, args.trade_directions),
        capital_base=args.initial_capital,
        position_fraction=args.position_fraction,
    )
    settings_kwargs = asdict(base_settings)
    strategy_cls = IntradayRangeStrategy if args.strategy == "intraday_range" else MomentumBreakoutStrategy

    all_trades: list[Trade] = []
    for symbol in symbols:
        symbol_settings = StrategySettings(**settings_kwargs)
        frame = fetch_ohlcv_dataframe(
            symbol,
            args.timeframe,
            days=args.days,
            end_time=end_time,
            config=config,
            cache_dir=args.cache_dir or None,
            cache_ttl=None if args.cache_ttl_hours <= 0 else timedelta(hours=args.cache_ttl_hours),
        )
        strategy = strategy_cls(symbol=symbol, settings=symbol_settings)
        all_trades.extend(strategy.evaluate(frame))

    max_positions = 1 if args.position_fraction >= 1 else max(1, int(1 / args.position_fraction))
    accepted_trades: list[Trade] = []
    active_trades: list[Trade] = []

    for trade in sorted(all_trades, key=lambda t: t.entry_time):
        active_trades = [
            t for t in active_trades if t.exit_time is None or t.exit_time > trade.entry_time
        ]

        if len(active_trades) >= max_positions:
            continue

        accepted_trades.append(trade)
        active_trades.append(trade)

    trading_cost = args.trading_cost
    if trading_cost < 0:
        raise SystemExit("--trading-cost must be non-negative.")

    adjusted_trades: list[Trade] = []
    for trade in accepted_trades:
        adjusted_trade = apply_trading_cost(trade, trading_cost)
        adjusted_trades.append(adjusted_trade)

    print_trade_summary(adjusted_trades)

    backtester = Backtester(initial_capital=args.initial_capital)
    result = backtester.run(adjusted_trades)

    print()
    print("Backtest metrics")
    print("----------------")
    print(f"Total trades: {result.total_trades}")
    print(f"Win rate: {result.win_rate*100:.1f}%")
    print(f"Average win: {result.average_win:.2f}")
    print(f"Average loss: {result.average_loss:.2f}")
    print(f"Expectancy (per trade): {result.expectancy:.2f}")
    print(f"Final equity: {result.equity_curve.iloc[-1]:.2f}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"Portfolio max drawdown: {result.portfolio_max_drawdown:.2f}")
    print(f"Total fees paid: {result.total_fees_paid:.2f}")
    print()
    print("Long trades")
    print(f"  Count: {len([t for t in result.trades if t.direction == 'long'])}")
    print(f"  Win rate: {result.long_win_rate*100:.1f}%")
    print(f"  Average win: {result.long_average_win:.2f}")
    print(f"  Average loss: {result.long_average_loss:.2f}")
    print("Short trades")
    print(f"  Count: {len([t for t in result.trades if t.direction == 'short'])}")
    print(f"  Win rate: {result.short_win_rate*100:.1f}%")
    print(f"  Average win: {result.short_average_win:.2f}")
    print(f"  Average loss: {result.short_average_loss:.2f}")
    print()
    print("Average holding time (minutes) per asset:")
    for asset, minutes in result.average_holding_minutes.items():
        print(f"  {asset}: {minutes:.1f}")
    print("Max drawdown per asset:")
    for asset, dd in result.asset_drawdowns.items():
        print(f"  {asset}: {dd:.2f}")

    if args.plot_equity:
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SystemExit(
                "matplotlib is required for --plot-equity. Install it via pip."
            ) from exc

        plt.figure(figsize=(10, 5))  # type: ignore[attr-defined]
        plt.plot(result.equity_curve.index, result.equity_curve.values, marker="o")
        plt.title(f"Equity Curve – {', '.join(symbols)} ({args.timeframe})")
        plt.xlabel("Trade #")
        plt.ylabel("Equity")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

