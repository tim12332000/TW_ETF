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

    cf_df['TradeDate'] = cf_df['Date'].apply(
        lambda d: px.index[px.index.get_indexer([d], method='bfill')[0]]
    )
    daily_cf = cf_df.groupby('TradeDate')['Amt'].sum()

    port_list, shares_list = [], []
    shares = 0.0
    for dt in px.index:
        price = px.loc[dt]
        if dt in daily_cf.index:
            cash = daily_cf.loc[dt]
            delta = -cash / price
            if shares + delta < 0:
                delta = -shares
            shares += delta
        port_list.append(shares * price)
        shares_list.append(shares)

    portfolio = pd.Series(port_list, index=px.index, name=f'{ticker}_Value')
    shares_ts = pd.Series(shares_list, index=px.index, name=f'{ticker}_Shares')
    return portfolio, shares_ts
