import pandas as pd
import yfinance as yf

from .cache import get_cached_data


def simulate_stock_full(cashflows, ticker='^SP500TR'):
    """Simulate buying or selling a benchmark using the portfolio cashflow stream."""
    cf_df = pd.DataFrame(cashflows, columns=['Date', 'Amt']).assign(
        Date=lambda d: pd.to_datetime(d['Date']).dt.normalize()
    )
    start = cf_df['Date'].min()
    end = pd.Timestamp.today().normalize()

    def _fetch_sim_history():
        return yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)['Close']

    key = f"sim_hist_{ticker}_{start.date()}_{end.date()}.pkl"
    px = get_cached_data(key, _fetch_sim_history)
    px = px.sort_index().ffill().bfill()
    if hasattr(px.index, 'tz'):
        px.index = px.index.tz_localize(None)

    # Keep deposits as cash until a market price is available. In particular,
    # a deposit after the last quote must not buy retroactively at index -1.
    daily_cf = cf_df.groupby('Date')['Amt'].sum()
    index = px.index.union(daily_cf.index).sort_values()
    marks = px.reindex(index).ffill()

    port_list, shares_list = [], []
    shares = 0.0
    cash = 0.0
    for dt in index:
        if dt in daily_cf.index:
            cash -= daily_cf.loc[dt]
        if dt in px.index:
            price = px.loc[dt]
            delta = cash / price
            if shares + delta < -1e-9:
                raise ValueError('Benchmark withdrawal exceeds available assets')
            shares += delta
            cash = 0.0
        value = shares * marks.loc[dt] if shares else 0.0
        port_list.append(value + cash)
        shares_list.append(shares)

    portfolio = pd.Series(port_list, index=index, name=f'{ticker}_Value')
    shares_ts = pd.Series(shares_list, index=index, name=f'{ticker}_Shares')
    return portfolio, shares_ts
