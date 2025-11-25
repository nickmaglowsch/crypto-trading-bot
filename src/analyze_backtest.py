"""
Analyze backtest results from test.py
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path

# Read the results
results_path = Path(__file__).parent / 'backtest_results.csv'
df = pd.read_csv(results_path, index_col=0, parse_dates=True)

print("=" * 80)
print("BACKTEST RESULTS ANALYSIS")
print("=" * 80)
print(f"\nDate Range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total Days: {len(df)}")
print(f"Years: {len(df) / 365.25:.2f}")

# Basic metrics
initial_value = df['portfolio_value'].iloc[0]
final_value = df['portfolio_value'].iloc[-1]
total_return = (final_value / initial_value) - 1

print(f"\n{'='*80}")
print("PERFORMANCE METRICS")
print(f"{'='*80}")
print(f"Initial Capital: ${initial_value:,.2f}")
print(f"Final Capital: ${final_value:,.2f}")
print(f"Total Return: {total_return:.2%}")
print(f"Total Return (multiple): {final_value / initial_value:.2f}x")

# Annualized metrics
years = len(df) / 365.25
ann_return = (1 + total_return) ** (1 / years) - 1
print(f"\nAnnualized Return: {ann_return:.2%}")

# Volatility
df['daily_ret'] = df['portfolio_value'].pct_change().fillna(0)
ann_vol = df['daily_ret'].std() * math.sqrt(365.25)
print(f"Annualized Volatility: {ann_vol:.2%}")

# Sharpe ratio (assuming risk-free rate = 0)
sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan
print(f"Sharpe Ratio (rf=0): {sharpe:.2f}")

# Drawdown analysis
df['cummax'] = df['portfolio_value'].cummax()
df['drawdown'] = (df['portfolio_value'] - df['cummax']) / df['cummax']
max_drawdown = df['drawdown'].min()
print(f"\nMaximum Drawdown: {max_drawdown:.2%}")

# Find worst drawdown period
worst_dd_idx = df['drawdown'].idxmin()
worst_dd_date = worst_dd_idx
print(f"Worst Drawdown Date: {worst_dd_date.date()}")

# Win rate (positive days)
positive_days = (df['daily_ret'] > 0).sum()
total_days = len(df[df['daily_ret'] != 0])
win_rate = positive_days / total_days if total_days > 0 else 0
print(f"Win Rate (positive days): {win_rate:.2%} ({positive_days}/{total_days})")

# Best and worst days
best_day = df['daily_ret'].max()
worst_day = df['daily_ret'].min()
best_day_date = df['daily_ret'].idxmax()
worst_day_date = df['daily_ret'].idxmin()

print(f"\n{'='*80}")
print("EXTREME DAYS")
print(f"{'='*80}")
print(f"Best Day: {best_day_date.date()} - {best_day:.2%}")
print(f"Worst Day: {worst_day_date.date()} - {worst_day:.2%}")

# Monthly returns
df['year_month'] = df.index.to_period('M')
monthly = df.groupby('year_month')['portfolio_value'].last().pct_change().dropna()
print(f"\n{'='*80}")
print("MONTHLY STATISTICS")
print(f"{'='*80}")
print(f"Average Monthly Return: {monthly.mean():.2%}")
print(f"Best Month: {monthly.max():.2%}")
print(f"Worst Month: {monthly.min():.2%}")
print(f"Monthly Volatility: {monthly.std():.2%}")

# Yearly returns
df['year'] = df.index.year
yearly = df.groupby('year')['portfolio_value'].last().pct_change().dropna()
print(f"\n{'='*80}")
print("YEARLY RETURNS")
print(f"{'='*80}")
for year, ret in yearly.items():
    print(f"{year}: {ret:.2%}")

# Recent performance (last 30, 90, 365 days)
if len(df) >= 30:
    ret_30d = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[-30] - 1)
    print(f"\n{'='*80}")
    print("RECENT PERFORMANCE")
    print(f"{'='*80}")
    print(f"Last 30 days: {ret_30d:.2%}")

if len(df) >= 90:
    ret_90d = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[-90] - 1)
    print(f"Last 90 days: {ret_90d:.2%}")

if len(df) >= 365:
    ret_1y = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[-365] - 1)
    print(f"Last 365 days: {ret_1y:.2%}")

# Growth trajectory
print(f"\n{'='*80}")
print("MILESTONE DATES")
print(f"{'='*80}")
milestones = [2, 5, 10, 20, 50]
for mult in milestones:
    target = initial_value * mult
    milestone_dates = df[df['portfolio_value'] >= target]
    if len(milestone_dates) > 0:
        milestone_date = milestone_dates.index[0]
        days_to_milestone = (milestone_date - df.index[0]).days
        print(f"{mult}x return ({target:,.0f}): {milestone_date.date()} (Day {days_to_milestone})")

print(f"\n{'='*80}")

