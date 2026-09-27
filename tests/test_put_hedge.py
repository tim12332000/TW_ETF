from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from portfolio.reporting import estimate_put_scenario_value


@pytest.fixture
def put_info():
    return {
        's0': 100.0,
        'strike': 70.0,
        'expiry': datetime(2027, 1, 15),
        'contracts': 1.0,
        'market_value': 250.0,
    }


def test_put_scenario_curve_is_anchored_to_current_market_value(put_info):
    value = estimate_put_scenario_value(put_info, 0.0, today='2026-08-01')

    assert value == pytest.approx(put_info['market_value'])


def test_put_gains_value_in_a_large_downside_scenario(put_info):
    current = estimate_put_scenario_value(put_info, 0.0, today='2026-08-01')
    stressed = estimate_put_scenario_value(put_info, -0.50, today='2026-08-01')

    assert stressed > current
