from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from portfolio.performance import calculate_twr_series, twr_to_daily_returns
from portfolio.transactions import build_cash_ledgers


def test_zero_cash_portfolio_uses_every_transaction_as_external_cashflow():
    dates = pd.bdate_range("2026-01-05", periods=3)
    transactions = pd.DataFrame(
        {
            "Date": dates,
            "Amount": [-100.0, 50.0, -50.0],
        }
    )
    portfolio_value = pd.Series([100.0, 50.0, 100.0], index=dates)

    transaction_cashflows, inferred_cashflows, _ = build_cash_ledgers(transactions)

    zero_cash_twr = calculate_twr_series(portfolio_value, transaction_cashflows)
    inferred_cash_twr = calculate_twr_series(portfolio_value, inferred_cashflows)

    assert twr_to_daily_returns(zero_cash_twr).tolist() == [0.0, 0.0]
    assert twr_to_daily_returns(inferred_cash_twr).tolist() == [-0.5, 1.0]
