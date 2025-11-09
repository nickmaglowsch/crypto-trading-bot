# Algo Trade – Strategy Sandbox

This project provides a small framework for experimenting with Binance spot strategies. It currently includes:

- **Intraday Range Fade** – watch the first 4 hours of the session, fade breaks that return into the range.
- **Momentum Breakout** – go with momentum when price pushes above yesterday's high or below yesterday's low.

The code uses [`ccxt`](https://github.com/ccxt/ccxt) to pull historical candles and pandas to backtest the logic.

## Getting Started

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the strategy on recent data

```bash
python -m src.run_strategy --symbol BTC/USDT --timeframe 5m --days 7
```

To try the momentum breakout with swing-based risk:

```bash
python -m src.run_strategy --strategy momentum --symbol BTC/USDT --timeframe 15m --days 10 --stop-mode swing --swing-period 20
```

Command line options:

- `--strategy`: Choose `intraday_range` (default) or `momentum`.
- `--symbol`: Binance market symbol.
- `--timeframe`: Candle timeframe (1m, 5m, 15m, etc).
- `--days`: Number of trailing days to evaluate.
- `--atr-period`: Lookback for ATR (default 14).
- `--reward-multiple`: Take-profit multiplier vs ATR (default 2.0).
- `--initial-capital`: Starting balance for the backtest equity curve (default 10,000).
- `--session-offset`: Hour offset from UTC that defines the start of the trading day (e.g. `-3` for UTC-3).
- `--cache-dir`: Location to store cached OHLCV data (default `.cache`; set empty to disable caching).
- `--cache-ttl-hours`: Reuse cached data younger than this many hours (default 6.0, set 0 to force refresh).
- `--plot-equity`: Show a matplotlib equity curve after the run (requires `matplotlib`).
- `--position-fraction`: Fraction of capital used per trade (1.0 = 100%, >1.0 allows leverage).
- `--sma-period`: Apply a simple moving average filter (longs only above SMA, shorts only below).
- `--stop-mode`: Choose `atr` (default) or `swing` to use distance from the latest swing high/low as the stop.
- `--swing-period`: Number of candles to look back for swing-based stops (defaults to 20 when `--stop-mode swing`).
- `--trade-directions`: Limit trades to `long`, `short`, or `both` (default).
- `--testnet`: Toggle Binance sandbox mode for future live order routing.
- `--api-key`/`--api-secret`: Optional credentials (not needed for historical data).

### 3. Output

The script prints each trade, its entry/exit details and cumulative PnL (in quote currency units). Use this to gauge performance before considering a live deployment.

The summary now includes basic backtest metrics such as win rate, average win/loss, expectancy, an equity curve based on the chosen starting capital, plus separate long/short win rates and average outcomes.

## Next Steps

- Connect the signals to live order execution via ccxt.
- Add position sizing and risk controls.
- Expand with unit tests and a more robust back-testing engine.


