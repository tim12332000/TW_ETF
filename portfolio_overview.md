# Portfolio Overview

This project reads TW and US transaction exports, normalizes cashflows and holdings, then generates portfolio reports, benchmark comparisons, and charts.

## Main entry points

- `portfolio_report.py`: thin script entry point
- `portfolio/app.py`: orchestration for loading data, building reports, and rendering charts

## Core modules

- `portfolio/transactions.py`: transaction normalization, cash ledgers, share-sign handling
- `portfolio/market_data.py`: price history, FX history, option pricing helpers, live price fallback
- `portfolio/tw_portfolio.py`: TW position processing
- `portfolio/us_portfolio.py`: US position processing
- `portfolio/performance.py`: XIRR, TWR, and risk metrics
- `portfolio/reporting.py`: tables, stock-performance chart, rebalance suggestion, put analysis
- `portfolio/benchmarking.py`: benchmark simulation using the same external cashflows
- `portfolio/cache.py`: local cache for downloaded market data
- `portfolio/logging_utils.py`: console plus report file logging

## Outputs

Generated artifacts are written under `output/`, including:

- `report.txt`
- `asset_trend.png`
- `funding_ratio.png`
- `cumulative_return_comparison.png`
- `portfolio_vs_benchmark_usd.png`
- `drawdown_underwater.png`
- `asset_pie_chart.png`
- `monthly_investment.png`
- `stock_performance.png`
- `total_asset_protection.png`
