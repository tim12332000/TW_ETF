from pathlib import Path
import hashlib
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import portfolio.cache as cache


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path))


@pytest.mark.parametrize('key, category', [
    ('current_price_QQQ.pkl', 'quotes'),
    ('daily_price_abc.pkl', 'prices'),
    ('risk_QQQ_1y.pkl', 'prices'),
    ('opt_price_QQQ270115P00350000.pkl', 'options'),
    ('put_chain_QQQ_2027-01-15.pkl', 'options'),
    ('put_expiries_QQQ.pkl', 'options'),
    ('put_underlying_history_QQQ.pkl', 'options'),
    ('usd_twd_hist_2026-11-18.pkl', 'fx'),
    ('rate_twd_usd.pkl', 'fx'),
    ('bench_twr_QQQ_2021-01-12.pkl', 'benchmarks'),
    ('sim_hist_QQQ_2021-01-12_2026-11-18.pkl', 'benchmarks'),
    ('us_split_history_QLD_2021-01-12_2026-11-19.pkl', 'splits'),
    ('custom.pkl', 'misc'),
])
def test_cache_keys_are_grouped_without_changing_filenames(key, category, tmp_path):
    path = cache.get_cache_path(key)

    assert path == tmp_path / category / key
    assert not path.parent.exists()


@pytest.mark.parametrize('key', ['../../secret.pkl', r'C:\outside\secret.pkl', '/', '', '.', '..'])
def test_cache_key_cannot_escape_its_category(key, tmp_path):
    path = cache.get_cache_path(key)

    assert path.parent == tmp_path / 'misc'
    assert path.name not in ('', '.', '..')
    assert path.resolve().is_relative_to(tmp_path.resolve())


def test_long_cache_key_keeps_existing_hash_rule(tmp_path):
    key = 'daily_price_' + 'Q' * 220 + '.pkl'
    expected_name = key[:150] + '_' + hashlib.md5(key.encode()).hexdigest()

    assert cache.get_cache_path(key) == tmp_path / 'prices' / expected_name


def test_grouped_cache_preserves_arguments_values_and_expiry(tmp_path):
    calls = []

    def fetch(value, *, increment):
        calls.append(value)
        return {'value': value + increment}

    assert cache.get_cached_data('current_price_QQQ.pkl', fetch, 40, increment=2) == {'value': 42}
    assert cache.get_cached_data('current_price_QQQ.pkl', fetch, 99, increment=1) == {'value': 42}
    path = cache.get_cache_path('current_price_QQQ.pkl')
    assert path.is_file()
    assert list(tmp_path.glob('*.pkl')) == []
    expired = time.time() - 11
    os.utime(path, (expired, expired))
    assert cache.get_cached_data('current_price_QQQ.pkl', fetch, 99, increment=1, ttl_seconds=10) == {'value': 100}
    assert calls == [40, 99]


def test_failed_fetch_does_not_create_cache_directories(tmp_path):
    def unavailable():
        raise RuntimeError('temporary outage')

    with pytest.raises(RuntimeError, match='temporary outage'):
        cache.get_cached_data('put_chain_QQQ.pkl', unavailable)

    assert list(tmp_path.iterdir()) == []
