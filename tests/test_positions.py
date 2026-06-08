from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from portfolio.positions import calculate_dividends_for_position, calculate_total_buy_for_position
from portfolio.transactions import clean_currency, fix_share_sign


def test_open_position_pnl_percent_uses_total_buy_denominator():
    df = pd.read_csv(ROOT / "us_train.csv", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
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

    assert round(calculate_dividends_for_position("EDV", df), 2) == 392.72
    assert round(calculate_dividends_for_position("TQQQ", df), 2) == 138.50


def test_tw_cash_dividends_are_grouped_by_symbol():
    df = pd.read_csv(ROOT / "tw_train.csv", encoding="utf-8-sig")
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
