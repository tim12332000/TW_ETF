from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from portfolio.reporting import black_scholes_put, implied_volatility_for_put


def test_put_implied_volatility_round_trip():
    expected_iv = 0.42
    market_price = black_scholes_put(S=100.0, K=90.0, T=0.75, r=0.045, sigma=expected_iv)

    actual_iv = implied_volatility_for_put(market_price, S=100.0, K=90.0, T=0.75, r=0.045)

    assert actual_iv == pytest.approx(expected_iv, abs=1e-8)
