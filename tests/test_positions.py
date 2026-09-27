from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from portfolio.positions import calculate_dividends_for_position, calculate_total_buy_for_position
from portfolio.transactions import clean_currency, fix_share_sign
from portfolio.us_portfolio import process_us_data


def test_us_journal_rows_do_not_poison_position_costs():
    def fake_daily_price(symbols, start_date, end_date, is_tw=False):
        index = pd.date_range(start=start_date, end=end_date, freq="B")
        return pd.DataFrame({"SPCX": 200.0}, index=index)

    result = process_us_data(
        clean_currency=clean_currency,
        build_cash_ledgers=lambda df: ([], [], 0.0),
        fix_share_sign=fix_share_sign,
        get_daily_price=fake_daily_price,
        build_option_history_series=lambda symbol, date_range: pd.Series(index=date_range, dtype=float),
        resolve_market_price=lambda symbol, history_series=None, is_tw=False: 200.0,
        get_latest_available_price=lambda series: 200.0,
        calculate_total_pnl_for_closed_position=lambda symbol, df: (0.0, 0.0, 0.0),
    )

    spcx = result["portfolio_df"].loc[result["portfolio_df"]["Symbol"] == "SPCX"].iloc[0]
    assert spcx["Quantity_now"] == 3
    assert round(spcx["Cost"], 2) == 521.68
    assert round(spcx["Price_Total"], 2) == 600.00
    # Cash-bearing journal rows must survive even if removed from share history.
    cash = result['cash_transactions']
    raw = pd.read_csv(ROOT / 'us_train.csv', encoding='utf-8-sig')
    raw['Date'] = pd.to_datetime(raw['Date'])
    expected = raw['Amount'].apply(clean_currency).groupby(raw['Date']).sum()
    actual = cash.groupby('Date')['Amount'].sum()
    pd.testing.assert_series_equal(actual, expected)


def test_open_position_pnl_percent_uses_total_buy_denominator():
    df = pd.read_csv(ROOT / "us_train.csv", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    # This regression covers the remaining share before the July liquidation.
    df = df.loc[df['Date'] < '2026-07-01'].copy()
    df = df.apply(fix_share_sign, axis=1)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Amount"] = df["Amount"].apply(clean_currency)

    edv = df[(df["Symbol"] == "EDV") & df["Quantity"].notna()]

    assert edv["Quantity"].sum() == 1
    assert round(-edv["Amount"].sum(), 2) == 851.05
    assert round(calculate_total_buy_for_position("EDV", df), 2) == 7383.06

    current_value = 62.59
    total_pnl = current_value - 851.05
    assert round(total_pnl / calculate_total_buy_for_position("EDV", df) * 100, 2) == -10.68


def test_us_dividends_are_grouped_by_symbol():
    df = pd.read_csv(ROOT / "us_train.csv", encoding="utf-8-sig")
    df["Amount"] = df["Amount"].apply(clean_currency)

    assert round(calculate_dividends_for_position("EDV", df), 2) == 393.48
    assert round(calculate_dividends_for_position("TQQQ", df), 2) == 138.50


def test_tw_cash_dividends_are_grouped_by_symbol():
    df = pd.read_csv(ROOT / "tw_train.csv", encoding="utf-8-sig")
    # Freeze the original fixture period so subsequent dividends do not alter it.
    df = df.loc[pd.to_datetime(df.iloc[:, 0]) < '2026-08-10'].copy()
    df = df.rename(
        columns={
            df.columns[1]: "Action",
            df.columns[2]: "Symbol",
            df.columns[6]: "Amount",
        }
    )
    df["Amount"] = df["Amount"].apply(clean_currency)

    assert round(calculate_dividends_for_position("006208", df), 2) == 42916.00
    assert round(calculate_dividends_for_position("2376", df), 2) == 2245.00
    assert round(calculate_dividends_for_position("0056", df), 2) == 486.00
    assert round(calculate_dividends_for_position("2330", df), 2) == 611.00
