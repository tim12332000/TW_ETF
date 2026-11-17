from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from portfolio.performance import calculate_twr_series, twr_to_daily_returns
from portfolio.transactions import build_cash_ledgers


@pytest.mark.parametrize(
    "values, amounts, expected",
    [
        ([100.0], [-999.0], [0.0]),
        ([0.0, 0.0, 0.0], [-100.0, 50.0, 0.0], [0.0, 0.0, 0.0]),
        ([-10.0, 0.0, -5.0], [10.0, -20.0, 30.0], [0.0, 0.0, 0.0]),
        (
            [-10.0, 0.0, -5.0, 100.0, 110.0],
            [0.0, 0.0, 0.0, -500.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10.0],
        ),
        (
            [100.0, 50.0, 0.0, 50.0, 60.0],
            [0.0, 0.0, 25.0, 0.0, 0.0],
            [0.0, -50.0, -75.0, -75.0, -70.0],
        ),
        (
            [float("nan"), 100.0, float("nan"), 50.0, 55.0],
            [0.0, float("nan"), 25.0, float("nan"), 0.0],
            [0.0, 0.0, -75.0, -75.0, -72.5],
        ),
    ],
    ids=[
        "single-positive-start",
        "all-zero",
        "never-positive",
        "delayed-positive-start",
        "zero-after-start",
        "missing-values-and-cashflows",
    ],
)
def test_twr_start_and_zero_value_rules(values, amounts, expected):
    dates = pd.date_range("2026-01-01", periods=len(values), name="valuation_date")
    portfolio_value = pd.Series(values, index=dates, name="Portfolio value")

    result = calculate_twr_series(portfolio_value, list(zip(dates, amounts)))

    pd.testing.assert_series_equal(result, pd.Series(expected, index=dates.normalize()))


def test_twr_empty_preserves_result_metadata():
    portfolio_value = pd.Series(
        dtype=float,
        index=pd.DatetimeIndex([], name="valuation_date"),
        name="Portfolio value",
    )

    result = calculate_twr_series(portfolio_value, [])

    pd.testing.assert_series_equal(
        result,
        pd.Series(dtype=object, index=pd.DatetimeIndex([], name="valuation_date")),
    )


def test_twr_combined_external_funding_keeps_cashflow_signs():
    dates = pd.bdate_range("2026-01-05", periods=4)
    combined_value = pd.Series([1000.0, 1550.0, 1275.0, 1402.5], index=dates)
    cashflows = [
        (dates[0], -1000.0),
        (dates[1], -600.0),
        (dates[1], 100.0),
        (dates[2], 300.0),
        (dates[2], -25.0),
    ]

    result = calculate_twr_series(combined_value, cashflows)

    pd.testing.assert_series_equal(
        result, pd.Series([0.0, 5.0, 5.0, 15.5], index=dates.normalize())
    )


def test_twr_normalizes_duplicate_dates_and_retains_join_order():
    portfolio_value = pd.Series(
        [150.0, 100.0, 120.0, 110.0],
        index=pd.to_datetime(
            ["2026-01-03 15:00", "2026-01-01 12:00", "2026-01-02 14:00", "2026-01-01 09:00"]
        ),
    )
    cashflows = [
        (pd.Timestamp("2026-01-02 10:00"), -20.0),
        (pd.Timestamp("2026-01-02 16:00"), 10.0),
        (pd.Timestamp("2026-01-03 09:00"), 10.0),
        (pd.Timestamp("2026-01-04"), 999.0),
    ]

    result = calculate_twr_series(portfolio_value, cashflows)

    expected_index = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"])
    pd.testing.assert_series_equal(
        result, pd.Series([0.0, 10.0, 10.0, 46.66666666666667], index=expected_index)
    )


def test_twr_keeps_shuffled_unique_dates_in_left_join_order():
    dates = pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"])
    portfolio_value = pd.Series([120.0, 100.0, 110.0], index=dates)

    result = calculate_twr_series(portfolio_value, [])

    pd.testing.assert_series_equal(
        result, pd.Series([0.0, -16.66666666666667, -8.33333333333333], index=dates)
    )


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
