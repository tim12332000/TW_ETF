from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest

from portfolio.transactions import build_combined_cash_ledger
from portfolio.performance import calculate_twr_series


def ledger(tw, us, rates=None):
    dates = pd.bdate_range('2026-01-05', periods=4)
    tw_df = pd.DataFrame(tw, columns=['Date', 'Amount_TWD'])
    us_df = pd.DataFrame(us, columns=['Date', 'Amount'])
    fx = pd.Series(rates or [30.0] * 4, index=dates)
    return build_combined_cash_ledger(tw_df, us_df, dates, fx)


def test_cross_market_sale_funds_us_purchase_without_new_deposit():
    result = ledger([('2026-01-05', -1000), ('2026-01-06', 600)],
                    [('2026-01-06', -10), ('2026-01-07', -15)])
    assert result.cash_twd.tolist() == [0, 300, 0, 0]
    assert result.external_flow_twd.tolist() == [-1000, 0, -150, 0]
    assert result.invested_twd.iloc[-1] == 1150


def test_same_day_netting_ignores_csv_order():
    rows = [('2026-01-05', -100), ('2026-01-06', -150), ('2026-01-06', 200)]
    pd.testing.assert_frame_equal(ledger(rows, []), ledger(rows[::-1], []))
    assert ledger(rows, []).cash_twd.iloc[-1] == 50


def test_dividends_and_taxes_are_internal_and_fx_is_date_specific():
    result = ledger([('2026-01-05', -1000)],
                    [('2026-01-06', 10), ('2026-01-06', -3), ('2026-01-07', -5)],
                    [30, 32, 31, 30])
    assert result.cash_twd.tolist() == [0, 224, 69, 69]
    assert result.invested_twd.iloc[-1] == 1000


def test_securities_sale_and_repurchase_do_not_create_twr_loss():
    result = ledger([('2026-01-05', -100), ('2026-01-06', 60), ('2026-01-07', -60)], [])
    securities = pd.Series([100, 40, 100, 100], index=result.index)
    total = securities + result.cash_twd
    twr = calculate_twr_series(total, list(result.external_flow_twd.items()))
    assert twr.tolist() == [0, 0, 0, 0]
    assert result.external_flow_twd.sum() == -100


def test_empty_ledger_and_accounting_identity():
    result = ledger([], [])
    assert (result == 0).all().all()
    result = ledger([('2026-01-05', -100), ('2026-01-06', 120)], [])
    assert result.cash_twd.iloc[-1] == result.transaction_net_twd.sum() - result.external_flow_twd.sum()
    assert result.invested_twd.iloc[-1] == 100


def test_missing_fx_does_not_use_future_quote():
    with pytest.raises(ValueError, match='FX'):
        ledger([], [('2026-01-02', -10)])
