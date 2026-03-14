from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import portfolio.app as app


def assert_close(name, left, right, atol=1e-6, rtol=1e-6):
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if not np.allclose(left_arr, right_arr, atol=atol, rtol=rtol, equal_nan=True):
        diff = np.nanmax(np.abs(left_arr - right_arr))
        raise AssertionError(f"{name} mismatch: max diff={diff}")


def main():
    tw = app.process_tw_data()
    us = app.process_us_data()

    date_index = tw["portfolio_value"].index.union(us["portfolio_value"].index).sort_values()
    usd_twd = app.get_usd_twd_history(date_index.min(), date_index.max())
    fx_on_date = app.align_fx_series(date_index, usd_twd)
    latest_usd_twd = float(usd_twd.iloc[-1])

    tw_non_null = tw["df"].dropna(subset=["Amount", "Amount_TWD", "USD_TWD"]).copy()
    reconstructed_twd = tw_non_null["Amount"] * tw_non_null["USD_TWD"]
    assert_close("TW cashflow FX roundtrip", reconstructed_twd, tw_non_null["Amount_TWD"], atol=1e-4, rtol=1e-6)

    if tw["symbols"]:
        sample_symbol = tw["symbols"][0]
        px_usd = pd.to_numeric(tw["price_data"][sample_symbol], errors="coerce")
        px_twd = pd.to_numeric(tw["price_data_twd"][sample_symbol], errors="coerce")
        valid = px_usd.notna() & px_twd.notna() & (px_usd != 0)
        if valid.any():
            derived_fx = (px_twd[valid] / px_usd[valid]).tail(20)
            expected_fx = fx_on_date.reindex(derived_fx.index)
            assert_close("TW daily valuation FX", derived_fx, expected_fx, atol=1e-4, rtol=1e-5)

    portfolio_value_us = tw["portfolio_value"].reindex(date_index, method="ffill").fillna(0) + us["portfolio_value"].reindex(date_index, method="ffill").fillna(0)
    portfolio_value_twd = portfolio_value_us * fx_on_date
    assert_close("Portfolio TWD valuation", portfolio_value_twd, portfolio_value_us * fx_on_date, atol=1e-6, rtol=1e-6)

    combined_external_cashflows = tw["external_cashflows"] + us["external_cashflows"]
    combined_external_cashflows_twd = app.convert_cashflows_to_twd(combined_external_cashflows, usd_twd)
    twr_usd = app.calculate_twr_series(portfolio_value_us, combined_external_cashflows)
    twr_twd = app.calculate_twr_series(portfolio_value_twd, combined_external_cashflows_twd)
    risk_usd = app.calc_risk_metrics_from_twr(twr_usd)
    risk_twd = app.calc_risk_metrics_from_twr(twr_twd)

    if np.isnan(np.asarray(risk_usd[:3], dtype=float)).all():
        raise AssertionError("USD risk metrics are all NaN")
    if np.isnan(np.asarray(risk_twd[:3], dtype=float)).all():
        raise AssertionError("TWD risk metrics are all NaN")

    constant_rate = latest_usd_twd
    portfolio_value_twd_fixed = portfolio_value_us * constant_rate
    max_diff_twd = float((portfolio_value_twd - portfolio_value_twd_fixed).abs().max())
    if max_diff_twd <= 1e-6:
        raise AssertionError("Historical FX produced no difference versus fixed FX")

    print("Historical FX validation passed.")
    print(f"Latest USD/TWD: {latest_usd_twd:.4f}")
    print(f"Max portfolio TWD diff vs fixed FX: {max_diff_twd:,.2f}")
    print(f"USD TWR final: {twr_usd.iloc[-1]:.4f}%")
    print(f"TWD TWR final: {twr_twd.iloc[-1]:.4f}%")
    print(f"USD risk metrics: AnnVol={risk_usd[0]:.4f} MaxDD={risk_usd[1]:.4f} Sharpe={risk_usd[2]:.4f}")
    print(f"TWD risk metrics: AnnVol={risk_twd[0]:.4f} MaxDD={risk_twd[1]:.4f} Sharpe={risk_twd[2]:.4f}")


if __name__ == "__main__":
    main()
