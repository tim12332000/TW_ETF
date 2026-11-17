from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import portfolio.reporting as reporting
from portfolio.reporting import black_scholes_put, estimate_put_scenario_value


def scalar_put_reference(S, K, T, r, sigma):
    """Freeze the pre-vectorization formula and boundary precedence."""
    if K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    if S <= 0:
        return K * np.exp(-r * T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


@pytest.mark.parametrize('S,K,T,r,sigma', [
    (100, 90, 0.75, 0.045, 0.42),
    (10, 200, 1.5, -0.01, 1.0),
    (200, 10, 0.01, 0.04, 0.1),
    (100, 0, 0.5, 0.04, 0.2),
    (-10, -1, -1, float('nan'), -1),
    (-10, 100, 0, 0.04, 0.2),
    (0, 100, 0.5, 0.04, 0.2),
    (-10, 100, 0.5, 0.04, 0.2),
    (100, 120, -1, 0.04, 0.2),
    (100, 120, 0.5, 0.04, 0),
    (float('nan'), 0, 0.5, 0.04, 0.2),
    (float('nan'), 100, 0, 0.04, 0.2),
    (100, float('nan'), 0.5, 0.04, 0.2),
    (0, 100, float('nan'), 0.04, 0.2),
    (100, 100, 0.5, float('nan'), 0.2),
    (100, 100, 0.5, 0.04, float('nan')),
])
def test_scalar_put_preserves_formula_and_boundary_precedence(S, K, T, r, sigma):
    actual = black_scholes_put(S, K, T, r, sigma)
    expected = scalar_put_reference(S, K, T, r, sigma)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-13, equal_nan=True)


def test_scalar_scenario_reference_is_market_anchored_and_nonnegative():
    info = {'s0': 100.0, 'strike': 70.0, 'expiry': datetime(2027, 1, 15),
            'contracts': 1.5, 'market_value': 250.0}
    years = (datetime(2027, 1, 15) - datetime(2026, 8, 1)).days / 365.0
    base = scalar_put_reference(100.0, 70.0, years, 0.045, 0.20)
    for drop in (-1.5, -1.0, -0.5, 0.0, 0.2, 1.0):
        stressed = scalar_put_reference(max(100 * (1 + drop), 0), 70.0, years,
                                        0.045, 0.20 + max(-drop, 0) * 0.8)
        expected = max(250.0 + (stressed - base) * 150, 0)
        actual = estimate_put_scenario_value(info, drop, today='2026-08-01')
        assert actual == pytest.approx(expected, rel=1e-14, abs=1e-12)


def test_vector_put_matches_scalar_reference_with_broadcasting():
    spots = np.array([np.nan, -10.0, 0.0, 50.0, 100.0, 250.0])[:, None]
    strikes = np.array([0.0, -1.0, 70.0, 100.0, np.nan])
    years = np.array([0.0, 1.0, 0.5, -1.0, 0.5])
    volatility = np.array([0.0, np.nan, 0.2, 0.4, -0.1])
    arrays = np.broadcast_arrays(spots, strikes, years, 0.045, volatility)
    expected = np.array([scalar_put_reference(*values) for values in zip(
        *(array.ravel() for array in arrays))]).reshape(arrays[0].shape)

    actual = black_scholes_put(spots, strikes, years, 0.045, volatility)

    assert actual.shape == (6, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-13, equal_nan=True)


def test_historical_vector_put_matches_thousand_scalar_prices():
    rng = np.random.default_rng(1618)
    spots = rng.uniform(1, 800, 1000)
    years = np.maximum(np.linspace(2, 0, 1000), 1 / 365)
    expected = [scalar_put_reference(float(spot), 350.0, float(year), 0.045, 0.42)
                for spot, year in zip(spots, years)]

    actual = black_scholes_put(spots, 350.0, years, 0.045, 0.42)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_vector_scenarios_match_individual_market_anchored_scenarios():
    info = {'s0': 100.0, 'strike': 70.0, 'expiry': datetime(2027, 1, 15),
            'contracts': 1.5, 'market_value': 250.0}
    drops = np.array([-1.5, -1.0, -0.5, 0.0, 0.2, 1.0, np.nan])
    expected = [estimate_put_scenario_value(info, drop, today='2026-08-01') for drop in drops]

    actual = estimate_put_scenario_value(info, drops, today='2026-08-01')

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-12, equal_nan=True)
    assert actual[3] == 250.0
    np.testing.assert_array_equal(
        estimate_put_scenario_value(info, drops, today='2027-01-15'), np.zeros(len(drops)))


def test_scalar_pricing_and_scenarios_return_floats():
    for args in [(100, 0, 1, 0, 0.2), (-10, 100, 0, 0, 0.2),
                 (0, 100, 1, 0, 0.2), (100, 100, 1, 0, 0.2)]:
        assert type(black_scholes_put(*args)) is float


def test_stress_report_and_svg_match_frozen_scalar_formula(monkeypatch, tmp_path, capsys):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 18, 8)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reporting, 'datetime', FrozenDatetime)
    monkeypatch.setattr(reporting, 'get_cached_data', lambda key, fetch, **kwargs:
                        pd.DataFrame({'Close': [600.0 if 'QQQ' in key else 300.0]}))
    positions = pd.DataFrame([
        {'Symbol': 'QQQ270115P00350000', 'Quantity_now': 100, 'Price_Total': 1000, 'Cost': 1100},
        {'Symbol': 'TSM270115P00220000', 'Quantity_now': 100, 'Price_Total': 400, 'Cost': 500},
        {'Symbol': 'QLD', 'Quantity_now': 10, 'Price_Total': 2000, 'Cost': 1500},
        {'Symbol': 'EDV', 'Quantity_now': 5, 'Price_Total': 500, 'Cost': 600},
    ])
    optimized = reporting.black_scholes_put
    monkeypatch.setattr(reporting, 'black_scholes_put', np.vectorize(scalar_put_reference, otypes=[float]))
    reporting.analyze_put_protection(positions)
    reference_text = capsys.readouterr().out
    filenames = ['total_asset_protection.svg', 'put_stock_hedge_effect.svg']
    reference_svg = {name: (tmp_path / 'output/charts' / name).read_bytes() for name in filenames}

    monkeypatch.setattr(reporting, 'black_scholes_put', optimized)
    reporting.analyze_put_protection(positions)

    assert capsys.readouterr().out == reference_text
    for name in filenames:
        assert (tmp_path / 'output/charts' / name).read_bytes() == reference_svg[name]
