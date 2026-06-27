from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from portfolio.reporting import print_rebalance_recommendation


def test_rebalance_groups_00646_with_sp500_bucket(capsys):
    portfolio_df = pd.DataFrame(
        [
            {"Symbol": "SPLG", "Quantity_now": 1, "Price_Total": 100.0},
            {"Symbol": "SPYM", "Quantity_now": 1, "Price_Total": 200.0},
            {"Symbol": "00646", "Quantity_now": 40, "Price_Total": 50.0},
            {"Symbol": "QLD", "Quantity_now": 1, "Price_Total": 150.0},
            {"Symbol": "00631L", "Quantity_now": 1, "Price_Total": 100.0},
            {"Symbol": "006208", "Quantity_now": 1, "Price_Total": 400.0},
        ]
    )

    print_rebalance_recommendation(portfolio_df, usd_to_twd=30.0)

    output = capsys.readouterr().out
    assert "SPLG / SPYM / 00646" in output
    assert "10,500" in output
