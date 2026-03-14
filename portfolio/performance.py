import numpy as np
import pandas as pd
from scipy.optimize import newton


def xnpv(rate, cashflows):
    t0 = min(date for date, _ in cashflows)
    return sum(cf / ((1 + rate) ** ((date - t0).days / 365.0)) for date, cf in cashflows)


def xirr(cashflows, guess=0.1):
    return newton(lambda r: xnpv(r, cashflows), guess)


def calculate_twr_series(portfolio_value_series, cashflow_list):
    cf_df = pd.DataFrame(cashflow_list, columns=['Date', 'Amt'])
    cf_df['Date'] = pd.to_datetime(cf_df['Date']).dt.normalize()
    daily_cf = cf_df.groupby('Date')['Amt'].sum()

    df = pd.DataFrame({'PV': portfolio_value_series})
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(daily_cf, how='left').fillna(0)

    returns = []
    prev_pv = 0
    started = False
    for _, row in df.iterrows():
        pv = row['PV']
        cf = row['Amt']
        if not started:
            if pv > 0:
                started = True
                prev_pv = pv
            returns.append(0.0)
            continue
        daily_r = 0.0 if prev_pv == 0 else (pv + cf - prev_pv) / prev_pv
        returns.append(daily_r)
        prev_pv = pv

    twr_series = pd.Series(returns, index=df.index)
    return ((1 + twr_series).cumprod() - 1) * 100


def twr_to_daily_returns(twr_pct_series):
    if twr_pct_series.empty:
        return pd.Series(dtype=float)
    wealth_index = 1 + (twr_pct_series / 100.0)
    return wealth_index.pct_change().dropna()


def calc_risk_metrics_from_twr(twr_pct_series, risk_free_rate=0.02):
    ret = twr_to_daily_returns(twr_pct_series)
    if ret.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    ann_vol = ret.std() * np.sqrt(252)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_ret = ret - daily_rf
    sharpe = np.nan if ret.std() == 0 else np.sqrt(252) * (excess_ret.mean() / ret.std())

    downside_returns = excess_ret.copy()
    downside_returns[downside_returns > 0] = 0
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) if len(downside_returns) > 0 else np.nan
    sortino = ((excess_ret.mean() / downside_deviation) * np.sqrt(252) if pd.notna(downside_deviation) and downside_deviation != 0 else np.nan)

    wealth_index = 1 + (twr_pct_series / 100.0)
    run_max = wealth_index.cummax()
    max_dd = 0 if run_max.max() == 0 else abs(((wealth_index - run_max) / run_max).min())

    total_years = len(ret) / 252
    if total_years <= 0 or wealth_index.iloc[-1] <= 0:
        ann_return = np.nan
    else:
        ann_return = wealth_index.iloc[-1] ** (1 / total_years) - 1
    calmar = ann_return / max_dd if pd.notna(ann_return) and max_dd != 0 else np.nan
    return ann_vol * 100, max_dd * 100, sharpe, sortino, calmar
