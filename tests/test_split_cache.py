from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

import portfolio.cache as cache
from portfolio import us_portfolio


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize("column_order", ["flat", "field-first", "symbol-first"])
def test_split_history_preserves_valid_ratios_and_column_formats(monkeypatch, column_order):
    dates = pd.date_range("2026-01-01", periods=6)
    history = pd.DataFrame({"Stock Splits": [0.0, 2.0, 1.0, -1.0, float("nan"), 0.1]}, index=dates)
    if column_order == "field-first":
        history.columns = pd.MultiIndex.from_tuples([("Stock Splits", "CLOSED")])
    elif column_order == "symbol-first":
        history.columns = pd.MultiIndex.from_tuples([("CLOSED", "Stock Splits")])
    monkeypatch.setattr(us_portfolio.yf, "download", lambda *args, **kwargs: history)

    result = us_portfolio._fetch_us_split_events(["CLOSED"], dates[0], dates[-1])

    assert result == {"CLOSED": [(dates[1], 2.0), (dates[5], 0.1)]}


def test_split_normalization_preserves_closed_position_history(monkeypatch):
    history = pd.DataFrame({"Stock Splits": [2.0]}, index=pd.to_datetime(["2026-01-09"]))
    requested = []

    def download(symbol, **kwargs):
        requested.append(symbol)
        return history

    monkeypatch.setattr(us_portfolio.yf, "download", download)
    transactions = pd.DataFrame({
        "Symbol": ["CLOSED", "CLOSED", "QQQ270115P00350000"],
        "Date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "Quantity": [3.0, -3.0, 100.0],
        "Action": ["buy", "sell", "buy"],
    })

    result = us_portfolio._normalize_us_quantities_to_split_adjusted_basis(transactions)

    assert requested == ["CLOSED"]
    assert result["Quantity"].tolist() == [6.0, -6.0, 100.0]
    assert result.loc[result["Symbol"] == "CLOSED", "Quantity"].sum() == 0.0
    assert transactions["Quantity"].tolist() == [3.0, -3.0, 100.0]


def test_split_history_cache_is_stable_intraday_and_separates_symbol_and_dates(monkeypatch):
    calls = []

    def download(symbol, **kwargs):
        calls.append(symbol)
        return pd.DataFrame({"Stock Splits": [2.0]}, index=pd.to_datetime(["2026-01-09"]))

    monkeypatch.setattr(us_portfolio.yf, "download", download)
    fetch = us_portfolio._fetch_us_split_events
    start = pd.Timestamp("2026-01-01")
    first = fetch(["CLOSED"], start, pd.Timestamp("2026-01-20 01:00"))
    repeated = fetch(["CLOSED"], start, pd.Timestamp("2026-01-20 18:00"))
    assert first == repeated == {"CLOSED": [(pd.Timestamp("2026-01-09"), 2.0)]}
    assert calls == ["CLOSED"]
    fetch(["CLOSED"], start, pd.Timestamp("2026-01-21"))
    fetch(["CLOSED"], start + pd.Timedelta(days=1), pd.Timestamp("2026-01-20"))
    fetch(["OTHER"], start, pd.Timestamp("2026-01-20"))
    assert calls == ["CLOSED", "CLOSED", "CLOSED", "OTHER"]


def test_valid_history_without_splits_is_cached_and_expires(monkeypatch, tmp_path):
    calls = []

    def download(symbol, **kwargs):
        calls.append(symbol)
        return pd.DataFrame({"Stock Splits": [0.0]}, index=pd.to_datetime(["2026-01-09"]))

    monkeypatch.setattr(us_portfolio.yf, "download", download)
    args = (["CLOSED"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-20"))
    assert us_portfolio._fetch_us_split_events(*args) == {}
    assert us_portfolio._fetch_us_split_events(*args) == {}
    assert calls == ["CLOSED"]
    files = list(tmp_path.rglob("*.pkl"))
    assert len(files) == 1
    expired = time.time() - 86401
    os.utime(files[0], (expired, expired))
    assert us_portfolio._fetch_us_split_events(*args) == {}
    assert calls == ["CLOSED", "CLOSED"]


@pytest.mark.parametrize("failure", [RuntimeError("network failure"), pd.DataFrame(), None])
def test_failed_split_download_is_retried_without_poisoning_cache(monkeypatch, tmp_path, failure):
    calls = []

    def download(symbol, **kwargs):
        calls.append(symbol)
        if len(calls) == 1:
            if isinstance(failure, Exception):
                raise failure
            return failure
        return pd.DataFrame({"Stock Splits": [2.0]}, index=pd.to_datetime(["2026-01-09"]))

    monkeypatch.setattr(us_portfolio.yf, "download", download)
    args = (["CLOSED"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-20"))
    assert us_portfolio._fetch_us_split_events(*args) == {}
    assert list(tmp_path.iterdir()) == []
    expected = {"CLOSED": [(pd.Timestamp("2026-01-09"), 2.0)]}
    assert us_portfolio._fetch_us_split_events(*args) == expected
    assert us_portfolio._fetch_us_split_events(*args) == expected
    assert calls == ["CLOSED", "CLOSED"]
