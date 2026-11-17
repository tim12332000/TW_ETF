# Portfolio Overview

This project reads TW and US transaction exports, normalizes cashflows and holdings, then generates portfolio reports, benchmark comparisons, and charts.

## Main entry points

- `startup.ipynb`: daily notebook entry point
- `portfolio_report.py`: thin script entry point
- `portfolio/app.py`: orchestration for loading data, building reports, and rendering charts

Active transactions are `data/transactions/tw_train.csv` and
`data/transactions/us_train.csv`. Historical imports and price snapshots live in
`data/imports/` and `data/reference/` and are not selected by the report pipeline.

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

The current reports are `output/report.md` and `output/report_details.md`.
Their 19 freshly rendered SVG figures are in `output/charts/`; the shared daily cash ledger is
`output/combined_cash_ledger.csv`.

The core chart renderer is `portfolio/svg_charts.py`. It writes SVG directly
without Matplotlib, PNG encoding, downsampling or completed-chart reuse.

Research artifacts live in `output/research/` by topic. Old validation files
live in `output/diagnostics/`, spreadsheet exports in `output/exports/`, and
the historical text report in `output/archive/report.txt`.

See [the complete folder map](project_layout.md) for research, tools, historical
inputs and cache locations.
