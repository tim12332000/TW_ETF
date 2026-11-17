from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from portfolio import us_portfolio
from portfolio.market_data import align_fx_series, get_latest_available_price
from portfolio.positions import calculate_total_pnl_for_closed_position
from portfolio.transactions import build_cash_ledgers, clean_currency, fix_share_sign
from portfolio.tw_portfolio import process_tw_data


OPTION = "QQQ270115P00350000"
COMMON_KEYS = {
    "df", "date_range", "portfolio_value", "transaction_cashflows",
    "external_cashflows", "total_investment", "invested_capital",
    "final_portfolio_value", "total_profit", "total_profit_pct",
    "portfolio_df", "portfolio_snapshot", "price_data", "symbols",
}


def _sample_rows(market):
    rows = [
        ("2026-01-05", "buy", "ALPHA", 2.0, 100.0, -200.0),
        ("2026-01-06", "buy", "BETA", 3.0, 200.0, -600.0),
        ("2026-01-07", "buy", "CLOSED", 1.0, 50.0, -50.0),
        ("2026-01-09", "sell", "CLOSED", -1.0, 50.0, 60.0),
        ("2026-01-12", "dividend", "ALPHA", 0.0, 100.0, 20.0),
    ]
    if market == "us":
        rows.append(("2026-01-08", "buy", OPTION, 2.0, 3.0, -6.0))
    return rows


@pytest.fixture
def run_pipeline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Quantity normalization still runs, with a deterministic empty split history.
    monkeypatch.setattr(us_portfolio.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    def run(market, provider, *, rows=None, history=None):
        rows = _sample_rows(market) if rows is None else rows
        frame = pd.DataFrame(rows, columns=["Date", "Action", "Symbol", "Quantity", "Price", "Amount"])
        frame["Name"] = frame["Symbol"] + " name"
        frame.to_csv(f"{market}_train.csv", index=False, encoding="utf-8-sig")
        history = {"ALPHA": 100.0, "BETA": 200.0, "CLOSED": 50.0, OPTION: 3.0} if history is None else history

        def daily_prices(symbols, start_date, end_date, *, is_tw):
            assert is_tw == (market == "tw")
            assert OPTION not in symbols
            dates = pd.bdate_range(start_date, end_date)
            suffix = ".TW" if is_tw else ""
            return pd.DataFrame({symbol + suffix: history.get(symbol, np.nan) for symbol in symbols}, index=dates)

        kwargs = dict(
            clean_currency=clean_currency,
            build_cash_ledgers=build_cash_ledgers,
            fix_share_sign=fix_share_sign,
            get_daily_price=daily_prices,
            get_latest_available_price=get_latest_available_price,
            calculate_total_pnl_for_closed_position=calculate_total_pnl_for_closed_position,
        )
        if market == "tw":
            def fx_history(start_date, end_date):
                rates = pd.Series(40.0, index=pd.bdate_range(start_date, end_date))
                rates.iloc[0] = 20.0
                return rates

            return process_tw_data(
                **kwargs,
                get_usd_twd_history=fx_history,
                align_fx_series=align_fx_series,
                get_current_price_yf=provider,
                record_calibrations=lambda count: None,
            )
        return us_portfolio.process_us_data(
            **kwargs,
            build_option_history_series=lambda symbol, dates: pd.Series(history.get(symbol, np.nan), index=dates),
            resolve_market_price=provider,
        )

    return run


def test_tw_valuation_preserves_fx_costs_dividends_and_return_contract(run_pipeline):
    calls = []

    def provider(symbol, *, is_tw, fallback_price):
        calls.append((symbol, is_tw, fallback_price))
        return {"ALPHA": 120.0, "BETA": 240.0}[symbol]

    result = run_pipeline("tw", provider)
    positions = result["portfolio_df"].set_index("Symbol")
    assert set(result) == COMMON_KEYS | {"price_data_twd", "usd_twd_series"}
    assert calls[:2] == [("ALPHA", True, 100.0), ("BETA", True, 200.0)]
    assert all(call in calls[:2] for call in calls)
    assert result["portfolio_snapshot"] == 24.0
    assert result["portfolio_value"].iloc[0] == 10.0
    assert result["final_portfolio_value"] == 20.0
    assert result["invested_capital"] == result["total_investment"] == 26.25
    assert result["total_profit"] == -6.25
    assert result["total_profit_pct"] == pytest.approx(-6.25 / 26.25 * 100)
    assert positions.loc["ALPHA", ["Quantity_now", "Price", "Price_Total", "Cost", "Price PnL", "Dividend", "Dividend_TWD", "Total PnL", "Total PnL(%)"]].tolist() == [2, 3, 6, 9.5, -3.5, .5, 20, -3, -30]
    assert positions.loc["BETA", ["Price", "Price_Total", "Cost", "Total PnL"]].tolist() == [6, 18, 15, 3]
    assert positions.loc["CLOSED", ["Quantity_now", "Cost", "Price PnL", "Total PnL"]].tolist() == [0, 1.25, .25, .25]
    assert pd.isna(positions.loc["CLOSED", "Price"])
    assert result["price_data_twd"]["ALPHA"].iloc[-1] == 100.0
    assert result["price_data"]["ALPHA"].iloc[-1] == 2.5


def test_us_valuation_preserves_option_history_costs_and_return_contract(run_pipeline):
    calls = []

    def provider(symbol, *, history_series, is_tw):
        assert is_tw is False
        assert history_series.iloc[-1] == {"ALPHA": 100.0, "BETA": 200.0, OPTION: 3.0}[symbol]
        calls.append(symbol)
        return {"ALPHA": 120.0, "BETA": 240.0, OPTION: 4.0}[symbol]

    result = run_pipeline("us", provider)
    positions = result["portfolio_df"].set_index("Symbol")
    assert set(result) == COMMON_KEYS | {"cash_transactions"}
    assert calls[:3] == ["ALPHA", "BETA", OPTION]
    assert "CLOSED" not in calls
    assert result["portfolio_snapshot"] == 968.0
    assert result["portfolio_value"].iloc[0] == 200.0
    assert result["final_portfolio_value"] == 806.0
    assert result["invested_capital"] == result["total_investment"] == 856.0
    assert result["total_profit"] == -50.0
    assert result["total_profit_pct"] == pytest.approx(-50 / 856 * 100)
    assert positions.loc["ALPHA", ["Quantity_now", "Price", "Price_Total", "Cost", "Price PnL", "Dividend", "Total PnL", "Total PnL(%)"]].tolist() == [2, 120, 240, 180, 60, 20, 80, 40]
    assert positions.loc[OPTION, ["Quantity_now", "Price", "Price_Total", "Cost", "Total PnL"]].tolist() == [2, 4, 8, 6, 2]
    assert positions.loc["CLOSED", ["Quantity_now", "Cost", "Total PnL"]].tolist() == [0, 50, 10]
    assert pd.isna(positions.loc["CLOSED", "Price"])
    assert result["cash_transactions"]["Amount"].sum() == -776.0


@pytest.mark.parametrize("market", ["tw", "us"])
@pytest.mark.parametrize("quote", [None, 0.0, np.nan], ids=["none", "zero", "nan"])
def test_unavailable_quote_semantics(run_pipeline, market, quote):
    rows = [("2026-01-05", "buy", "ALPHA", 2.0, 100.0, -200.0)]
    result = run_pipeline(market, lambda *args, **kwargs: quote, rows=rows)
    position = result["portfolio_df"].iloc[0]
    if quote is None and market == "tw":
        # Historical USD fallback is used when the TW provider has no quote.
        assert position["Price"] == 2.5
        assert position["Price_Total"] == result["portfolio_snapshot"] == 5.0
    elif quote is None:
        assert pd.isna(position["Price"])
        assert pd.isna(position["Price_Total"])
        assert result["portfolio_snapshot"] == 0
    elif pd.isna(quote):
        assert pd.isna(position["Price"])
        assert pd.isna(position["Price_Total"])
        assert pd.isna(result["portfolio_snapshot"])
    else:
        assert position["Price"] == position["Price_Total"] == result["portfolio_snapshot"] == 0


def test_tw_missing_history_and_quote_remain_unavailable(run_pipeline):
    calls = []

    def provider(symbol, *, is_tw, fallback_price):
        calls.append((symbol, is_tw, fallback_price))
        return None

    rows = [("2026-01-05", "buy", "ALPHA", 2.0, 100.0, -200.0)]
    result = run_pipeline("tw", provider, rows=rows, history={})
    assert all(call == ("ALPHA", True, None) for call in calls)
    assert result["portfolio_snapshot"] == 0
    assert pd.isna(result["portfolio_df"].iloc[0]["Price"])
    assert result["final_portfolio_value"] == 0


@pytest.mark.parametrize("market", ["tw", "us"])
def test_first_snapshot_resolution_error_propagates_in_symbol_order(run_pipeline, market):
    calls = []

    def provider(symbol, **kwargs):
        calls.append(symbol)
        raise RuntimeError("quote unavailable")

    with pytest.raises(RuntimeError, match="quote unavailable"):
        run_pipeline(market, provider)
    assert calls == ["ALPHA"]


@pytest.mark.parametrize("market", ["tw", "us"])
@pytest.mark.parametrize(
    "quantities, table_only",
    [([.1, .2, -.3], True), ([.1, .3, -.2, -.2], False)],
    ids=["table-only", "snapshot-only"],
)
def test_fractional_aggregation_preserves_lookup_path(run_pipeline, market, quantities, table_only):
    dates = pd.bdate_range("2026-01-05", periods=len(quantities))
    rows = [(date, "buy" if quantity > 0 else "sell", "ALPHA", quantity, 100.0, -quantity * 100) for date, quantity in zip(dates, quantities)]
    calls = []

    def provider(symbol, **kwargs):
        calls.append(symbol)
        return 120.0

    result = run_pipeline(market, provider, rows=rows)
    quantities = result["df"]["Quantity"]
    grouped = result["df"].groupby("Symbol")["Quantity"].sum().iloc[0]
    sequential = sum(quantities)
    assert (grouped == 0) == table_only
    assert (sequential != 0) == table_only
    assert calls == ["ALPHA"]
    position = result["portfolio_df"].iloc[0]
    assert position["Quantity_now"] == sequential
    if table_only:
        assert result["portfolio_snapshot"] == 0
        assert position["Price"] == (3.0 if market == "tw" else 120.0)
    else:
        assert result["portfolio_snapshot"] == grouped * (3.0 if market == "tw" else 120.0)
        assert pd.isna(position["Price"])


@pytest.mark.parametrize("market", ["tw", "us"])
def test_table_only_resolution_error_keeps_historical_fallback(run_pipeline, capsys, market):
    quantities = [.1, .2, -.3]
    rows = [(date, "buy" if quantity > 0 else "sell", "ALPHA", quantity, 100.0, -quantity * 100) for date, quantity in zip(pd.bdate_range("2026-01-05", periods=3), quantities)]

    def provider(symbol, **kwargs):
        raise RuntimeError("table-only failure")

    result = run_pipeline(market, provider, rows=rows)
    assert result["portfolio_snapshot"] == 0
    assert result["portfolio_df"].iloc[0]["Price"] == (2.5 if market == "tw" else 100.0)
    assert "Error fetching data for ALPHA: table-only failure" in capsys.readouterr().out


@pytest.mark.parametrize("market", ["tw", "us"])
def test_reuses_first_quote_within_run_and_refreshes_next_run(run_pipeline, market):
    calls = []
    counts = {}
    base_prices = {"ALPHA": 120.0, "BETA": 240.0, OPTION: 4.0}
    expected_symbols = ["ALPHA", "BETA"] + ([OPTION] if market == "us" else [])

    def provider(symbol, **kwargs):
        calls.append(symbol)
        counts[symbol] = counts.get(symbol, 0) + 1
        return base_prices[symbol] + 40.0 * (counts[symbol] - 1)

    first = run_pipeline(market, provider)
    assert first["portfolio_snapshot"] == first["portfolio_df"]["Price_Total"].sum()
    assert calls == expected_symbols
    first_prices = first["portfolio_df"].set_index("Symbol")["Price"]
    divisor = 40.0 if market == "tw" else 1.0
    for symbol in expected_symbols:
        assert first_prices[symbol] == base_prices[symbol] / divisor

    second = run_pipeline(market, provider)
    assert second["portfolio_snapshot"] == second["portfolio_df"]["Price_Total"].sum()
    assert calls == expected_symbols * 2
    second_prices = second["portfolio_df"].set_index("Symbol")["Price"]
    for symbol in expected_symbols:
        assert second_prices[symbol] == (base_prices[symbol] + 40.0) / divisor


@pytest.mark.parametrize("market", ["tw", "us"])
@pytest.mark.parametrize("quote", [None, 0.0, np.nan], ids=["none", "zero", "nan"])
def test_reuses_unavailable_results_by_symbol_membership(run_pipeline, market, quote):
    calls = []

    def provider(symbol, **kwargs):
        calls.append(symbol)
        return quote

    rows = [("2026-01-05", "buy", "ALPHA", 2.0, 100.0, -200.0)]
    run_pipeline(market, provider, rows=rows, history={})
    assert calls == ["ALPHA"]
