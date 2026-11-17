from pathlib import Path
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.pyplot as plt
plt.close('all')
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image
import pytest

import portfolio.app as app
import portfolio.cache as cache
import portfolio.reporting as reporting


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path))


def test_reset_put_selection_preserves_expiry_strike_price_and_input(monkeypatch):
    puts = pd.DataFrame([
        {'strike': 45.0, 'ask': 1.0, 'bid': 0.5, 'lastPrice': 0.8},
        {'strike': 50.0, 'ask': 2.0, 'bid': 1.5, 'lastPrice': 1.8},
        {'strike': 55.0, 'ask': 0.0, 'bid': 0.0, 'lastPrice': 0.0},
    ])
    original = puts.copy(deep=True)
    ticker = SimpleNamespace(options=('2026-12-01', '2027-11-19'),
                             option_chain=lambda expiry: SimpleNamespace(puts=puts))
    monkeypatch.setattr(app.yf, 'Ticker', lambda symbol: ticker)

    expiry = app._recommended_put_expiry('QQQ', pd.Timestamp('2026-11-18'))
    selected = app._find_reset_put_hedge('QQQ', expiry, 100.0, 1.2, 10000.0)

    assert expiry == pd.Timestamp('2027-11-19')
    assert selected['strike'] == 50.0
    assert selected['contracts'] == 2
    assert selected['buy_price'] == 2.0
    assert selected['gross_cost'] == 400.0
    assert selected['protection_rate'] == pytest.approx(8000 / 9000 * 100)
    pd.testing.assert_frame_equal(puts, original)


def test_cache_supports_short_lifetime_without_changing_default(monkeypatch, tmp_path):
    calls = []
    fetch = lambda: calls.append(True) or len(calls)
    assert cache.get_cached_data('short.pkl', fetch, ttl_seconds=10) == 1
    filename = cache.get_cache_path('short.pkl')
    mtime = filename.stat().st_mtime
    monkeypatch.setattr(cache.time, 'time', lambda: mtime + 9)
    assert cache.get_cached_data('short.pkl', fetch, ttl_seconds=10) == 1
    monkeypatch.setattr(cache.time, 'time', lambda: mtime + 11)
    assert cache.get_cached_data('short.pkl', fetch, ttl_seconds=10) == 2
    os.utime(filename, (mtime, mtime))
    assert cache.get_cached_data('short.pkl', fetch) == 2
    assert len(calls) == 2


def test_put_queries_reuse_successful_responses_and_expire(monkeypatch, tmp_path):
    counts = {'options': 0, 'chain': 0}

    class Ticker:
        @property
        def options(self):
            counts['options'] += 1
            return ('2027-11-19',)

        def option_chain(self, expiry):
            counts['chain'] += 1
            return SimpleNamespace(puts=pd.DataFrame([
                {'strike': 50.0, 'ask': 2.0, 'bid': 1.5, 'lastPrice': 1.8}]))

    monkeypatch.setattr(app.yf, 'Ticker', lambda symbol: Ticker())
    for _ in range(2):
        expiry = app._recommended_put_expiry('QQQ', pd.Timestamp('2026-11-18'))
        app._find_reset_put_hedge('QQQ', expiry, 100.0, 1.0, 10000.0)
    assert counts == {'options': 1, 'chain': 1}
    for file in tmp_path.rglob('*.pkl'):
        old_time = cache.time.time() - 901
        os.utime(file, (old_time, old_time))
    app._find_reset_put_hedge('QQQ', expiry, 100.0, 1.0, 10000.0)
    assert counts['chain'] == 2


def test_failed_put_queries_are_retried(monkeypatch):
    calls = []

    def unavailable(symbol):
        calls.append(symbol)
        raise RuntimeError('temporary outage')

    monkeypatch.setattr(app.yf, 'Ticker', unavailable)
    today = pd.Timestamp('2026-11-18')
    for _ in range(2):
        assert app._recommended_put_expiry('QQQ', today) == pd.Timestamp('2027-11-18')
        assert app._find_reset_put_hedge('QQQ', today, 100.0, 1.0, 10000.0) is None
    assert calls == ['QQQ'] * 4


def test_fast_chart_save_preserves_pixels_dimensions_and_dpi(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [1, 4, 2], label='Portfolio')
    ax.legend()
    plt.tight_layout()
    baseline = tmp_path / 'baseline.png'
    optimized = tmp_path / 'optimized.png'
    try:
        plt.savefig(baseline, dpi=125)
        reporting.save_report_chart(optimized, dpi=125)
        with Image.open(baseline) as before, Image.open(optimized) as after:
            assert before.size == after.size == (500, 375)
            assert before.info['dpi'] == after.info['dpi']
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    finally:
        plt.close(fig)


def test_put_stress_values_and_chart_pixels_match_with_cached_quotes(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda: None)
    (tmp_path / 'output/charts').mkdir(parents=True)
    calls = []

    def history(**kwargs):
        calls.append(True)
        return pd.DataFrame({'Close': [600.0]})

    monkeypatch.setattr(reporting.yf, 'Ticker', lambda symbol: SimpleNamespace(history=history))
    positions = pd.DataFrame([
        {'Symbol': 'QQQ270115P00350000', 'Quantity_now': 100, 'Price_Total': 1000, 'Cost': 1100},
        {'Symbol': 'QLD', 'Quantity_now': 10, 'Price_Total': 2000, 'Cost': 1500},
    ])
    save_chart = reporting.save_report_chart
    get_cached_data = reporting.get_cached_data
    monkeypatch.setattr(reporting, 'save_report_chart', plt.savefig)
    monkeypatch.setattr(reporting, 'get_cached_data', lambda key, fetch, **kwargs: fetch())
    reporting.analyze_put_protection(positions)
    original_text = capsys.readouterr().out
    filenames = ['total_asset_protection.png', 'put_stock_hedge_effect.png']
    original_pixels = {name: np.array(Image.open(tmp_path / 'output/charts' / name)) for name in filenames}

    monkeypatch.setattr(reporting, 'save_report_chart', save_chart)
    monkeypatch.setattr(reporting, 'get_cached_data', get_cached_data)
    for _ in range(2):
        reporting.analyze_put_protection(positions)
        assert capsys.readouterr().out == original_text
        for name in filenames:
            with Image.open(tmp_path / 'output/charts' / name) as image:
                np.testing.assert_array_equal(np.asarray(image), original_pixels[name])
    assert len(calls) == 2
