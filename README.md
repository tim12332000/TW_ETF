# TW_ETF

A local portfolio analysis project for combined TW and US holdings.

## Entry Points

- `python portfolio_report.py`
- `python -m portfolio.app`

## Project Structure

- `portfolio/`: core portfolio logic and report orchestration
- `research/`: exploratory analysis and strategy scripts
- `tools/`: verification, debugging, and maintenance utilities
- `tests/`: focused regression and behavior checks
- `docs/`: project notes and design references
- `cache/`: cached market data snapshots
- `output/`: generated reports and charts

## Core Modules

- `portfolio/app.py`: application entry and orchestration
- `portfolio/tw_portfolio.py`: TW portfolio pipeline
- `portfolio/us_portfolio.py`: US portfolio pipeline
- `portfolio/market_data.py`: prices, FX, option valuation, live price helpers
- `portfolio/performance.py`: TWR, XIRR, risk metrics
- `portfolio/benchmarking.py`: benchmark simulation helpers
- `portfolio/reporting.py`: report rendering and chart output
- `portfolio/positions.py`: position and closed-PnL helpers
- `portfolio/transactions.py`: transaction normalization and cashflow helpers
- `portfolio/cache.py`: local cache layer

## Outputs

Generated files are written to `output/`.
Cached market data is written to `cache/`.

## Current Design Goal

Keep `portfolio/app.py` thin and move reusable logic into focused modules with clear names.
