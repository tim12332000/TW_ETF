from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import portfolio.benchmarking as benchmarking


def test_nontrading_day_funding_stays_cash_until_next_quote(monkeypatch):
    px = pd.Series([10.0, 20.0], index=pd.to_datetime(['2026-01-02', '2026-01-05']))
    monkeypatch.setattr(benchmarking, 'get_cached_data', lambda *args: px)
    dates = pd.to_datetime(['2026-01-02', '2026-01-03', '2026-01-06'])
    values, shares = benchmarking.simulate_stock_full(list(zip(dates, [-100, -100, -50])), 'TEST')
    assert values.loc['2026-01-03'] == 200
    assert shares.loc['2026-01-03'] == 10
    assert values.loc['2026-01-05'] == 300
    assert shares.loc['2026-01-05'] == 15
    assert values.loc['2026-01-06'] == 350
    assert shares.loc['2026-01-06'] == 15
