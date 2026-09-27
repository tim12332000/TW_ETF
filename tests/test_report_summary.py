from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from portfolio.report_summary import render_summary_report


def test_summary_has_one_currency_cash_and_five_core_charts():
    benchmark = pd.DataFrame([['My Portfolio', 1200, 200, 20, 10, 15, 25, .8]],
        columns=['Asset', 'Final Value (TWD)', 'Profit (TWD)', 'Profit %', 'XIRR %', 'AnnVol %', 'MaxDD %', 'Sharpe'])
    text = render_summary_report(
        invested=1000, total=1200, cash=100, benchmark=benchmark,
        holdings=pd.Series({'ETF': 1100, '現金（推算）': 100}), puts=[],
        first_date='2021-01-01', last_date='2026-09-28',
    )
    assert '1,200' in text and '8.33%' in text
    assert text.count('![') == 5
    assert 'report_details.md' in text
    assert '未持有 Put' in text
    assert 'Execution Summary' not in text
    assert 'Funding Ratio' not in text
    assert 'Quantity_now' not in text
    assert 'USD)' not in text
    assert '現金（推算）' in text
