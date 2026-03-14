import hashlib
import re
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from .cache import get_cached_data
from .transactions import convert_ticker


def get_daily_price(stock_symbol, start_date, end_date, is_tw=True):
    try:
        if is_tw:
            if isinstance(stock_symbol, list):
                stock_symbol = [convert_ticker(s) for s in stock_symbol]
            else:
                stock_symbol = convert_ticker(stock_symbol)

        def _fetch_daily():
            return yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=False, actions=True)

        sym_str = str(sorted(stock_symbol)) if isinstance(stock_symbol, list) else str(stock_symbol)
        key = f"daily_price_{hashlib.md5(sym_str.encode()).hexdigest()}_{str(start_date)[:10]}_{str(end_date)[:10]}.pkl"
        data = get_cached_data(key, _fetch_daily)

        try:
            if 'Stock Splits' in data.columns:
                splits = data['Stock Splits'].fillna(0)
                close_data = data['Close'].copy()

                def unadjust_series(price_s, split_s):
                    split_events = split_s[split_s > 0]
                    for split_date, ratio in split_events.items():
                        mask = price_s.index < split_date
                        price_s.loc[mask] *= ratio
                    return price_s

                if isinstance(data['Close'], pd.DataFrame):
                    for col in close_data.columns:
                        if col in splits.columns:
                            close_data[col] = unadjust_series(close_data[col], splits[col])
                    return close_data
                return unadjust_series(close_data, splits)
        except Exception as e:
            print(f"[WARN] Price un-adjustment failed: {e}")

        return data['Close']
    except Exception as e:
        print(f"?? {stock_symbol} ????: {e}")
        return pd.DataFrame()


def get_option_price(occ_symbol):
    try:
        match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", occ_symbol)
        if not match:
            return None

        underlying = match.group(1)
        date_str = match.group(2)
        opt_type = match.group(3)
        year = int("20" + date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiry = f"{year}-{month:02d}-{day:02d}"

        def _fetch_opt_price():
            ticker = yf.Ticker(underlying)
            try:
                chain = ticker.option_chain(expiry)
            except Exception:
                return None
            opts = chain.calls if opt_type == 'C' else chain.puts
            match_row = opts[opts['contractSymbol'] == occ_symbol]
            if not match_row.empty:
                return float(match_row['lastPrice'].iloc[0])
            return None

        return get_cached_data(f"opt_price_{occ_symbol}.pkl", _fetch_opt_price)
    except Exception as e:
        print(f"????? {occ_symbol} ????: {e}")
        return None


def parse_occ_symbol(occ_symbol):
    match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", str(occ_symbol))
    if not match:
        return None
    return {
        'underlying': match.group(1),
        'expiry': pd.Timestamp(datetime.strptime(match.group(2), "%y%m%d")),
        'option_type': match.group(3),
        'strike': float(match.group(4)) / 1000.0,
    }


def build_option_history_series(occ_symbol, date_index):
    meta = parse_occ_symbol(occ_symbol)
    if meta is None or len(date_index) == 0:
        return pd.Series(index=date_index, dtype=float)

    underlying_prices = get_daily_price(
        meta['underlying'],
        date_index.min(),
        date_index.max() + pd.Timedelta(days=1),
        is_tw=False,
    )
    if isinstance(underlying_prices, pd.DataFrame):
        if meta['underlying'] in underlying_prices.columns:
            underlying_prices = underlying_prices[meta['underlying']]
        else:
            underlying_prices = underlying_prices.iloc[:, 0]

    if underlying_prices is None or len(underlying_prices) == 0:
        return pd.Series(index=date_index, dtype=float)

    underlying_prices = underlying_prices.reindex(date_index).ffill()
    if meta['option_type'] == 'P':
        option_series = (meta['strike'] - underlying_prices).clip(lower=0)
    else:
        option_series = (underlying_prices - meta['strike']).clip(lower=0)

    option_series = option_series.astype(float)
    option_series[date_index > meta['expiry']] = 0.0

    current_price = get_option_price(occ_symbol)
    if current_price is not None and pd.notna(current_price) and not option_series.empty:
        option_series.iloc[-1] = max(float(current_price), float(option_series.iloc[-1]))

    return option_series


def get_latest_available_price(price_series, fallback=None):
    try:
        if price_series is not None:
            valid = pd.Series(price_series).dropna()
            if not valid.empty:
                return float(valid.iloc[-1])
    except Exception:
        pass
    return fallback


def resolve_market_price(ticker, history_series=None, is_tw=True, prefer_history=True):
    fallback_price = get_latest_available_price(history_series)
    if prefer_history and fallback_price is not None and pd.notna(fallback_price):
        return float(fallback_price)
    live_price = get_current_price_yf(ticker, is_tw=is_tw, fallback_price=fallback_price)
    if live_price is not None and pd.notna(live_price):
        return float(live_price)
    return fallback_price


def get_current_price_yf(ticker, is_tw=True, fallback_price=None):
    try:
        if is_tw:
            ticker = convert_ticker(ticker)
        elif re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", str(ticker)):
            option_price = get_option_price(ticker)
            if option_price is not None:
                return option_price

        def _fetch_current():
            ticker_obj = yf.Ticker(ticker)
            price = None
            try:
                fast_info = getattr(ticker_obj, 'fast_info', None)
                if fast_info is not None:
                    price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
            except Exception:
                price = None
            if price is None:
                try:
                    price = ticker_obj.info.get('regularMarketPrice')
                except Exception:
                    price = None
            if price is None:
                history = ticker_obj.history(period='1d')
                if not history.empty:
                    price = history['Close'].iloc[-1]
            return price

        price = get_cached_data(f"current_price_{ticker}.pkl", _fetch_current)
        return price if price is not None else fallback_price
    except Exception as e:
        print(f"?? {ticker} ????: {e}")
        return fallback_price


def _normalize_history_series(series_like):
    if isinstance(series_like, pd.DataFrame):
        if 'Close' in series_like.columns:
            series_like = series_like['Close']
        else:
            series_like = series_like.iloc[:, 0]
    series = pd.Series(series_like).copy()
    if hasattr(series.index, 'tz') and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = pd.to_datetime(series.index).normalize()
    return pd.to_numeric(series, errors='coerce').sort_index()


def get_usd_twd_history(start_date=None, end_date=None):
    start_ts = pd.Timestamp(start_date).normalize() if start_date is not None else None
    end_ts = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    fetch_start = (start_ts - pd.Timedelta(days=10)) if start_ts is not None else None
    fetch_end = end_ts + pd.Timedelta(days=1)

    def _fetch_fx():
        hist = yf.download('TWD=X', start=fetch_start, end=fetch_end, auto_adjust=False, progress=False)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            if 'Close' in hist.columns:
                return hist['Close']
            return hist.iloc[:, 0]
        return pd.Series(dtype=float)

    start_key = fetch_start.date() if fetch_start is not None else 'full'
    fx = _normalize_history_series(get_cached_data(f"usd_twd_hist_{start_key}_{fetch_end.date()}.pkl", _fetch_fx))
    if fx.empty:
        fx = _normalize_history_series(_fetch_fx())
    if fx.empty:
        try:
            ticker = yf.Ticker('TWD=X')
            rate = ticker.fast_info.get('lastPrice') if getattr(ticker, 'fast_info', None) is not None else None
            if rate is None:
                rate = ticker.info.get('regularMarketPrice')
            if rate is None:
                history = ticker.history(period='5d')
                if not history.empty:
                    rate = history['Close'].iloc[-1]
            if rate is not None:
                fx = pd.Series([float(rate)], index=[end_ts], dtype=float)
        except Exception:
            fx = pd.Series(dtype=float)

    if fx.empty:
        fallback = 30.0
        idx = pd.date_range(start=start_ts or end_ts, end=end_ts, freq='B')
        if idx.empty:
            idx = pd.DatetimeIndex([end_ts])
        return pd.Series(fallback, index=idx, dtype=float, name='USD_TWD')

    target_start = start_ts if start_ts is not None else fx.index.min()
    idx = pd.date_range(start=target_start, end=end_ts, freq='B')
    if idx.empty:
        idx = pd.DatetimeIndex([end_ts])
    fx = fx.reindex(idx).ffill().bfill()
    fx.name = 'USD_TWD'
    return fx


def align_fx_series(index_like, fx_series):
    idx = pd.DatetimeIndex(pd.to_datetime(index_like)).normalize()
    aligned = fx_series.reindex(idx).ffill().bfill()
    aligned.index = idx
    return aligned


def convert_cashflows_to_twd(cashflows_usd, fx_series):
    if not cashflows_usd:
        return []
    cf_df = pd.DataFrame(cashflows_usd, columns=['Date', 'Amount_USD'])
    cf_df['Date'] = pd.to_datetime(cf_df['Date']).dt.normalize()
    cf_df['USD_TWD'] = align_fx_series(cf_df['Date'], fx_series).values
    cf_df['Amount_TWD'] = cf_df['Amount_USD'] * cf_df['USD_TWD']
    return list(cf_df[['Date', 'Amount_TWD']].itertuples(index=False, name=None))


def get_twd_to_usd_rate():
    try:
        rate = float(get_usd_twd_history(end_date=pd.Timestamp.today().normalize()).iloc[-1])
        return 1.0 / rate
    except Exception as e:
        print('?? TWD/USD ????:', e)
        return 1 / 30.0


def compute_risk_return(stock_code, is_tw=True, period='1y'):
    ticker = convert_ticker(stock_code) if is_tw else stock_code

    def _fetch_risk():
        return yf.Ticker(ticker).history(period=period)

    data = get_cached_data(f"risk_{ticker}_{period}.pkl", _fetch_risk)
    if data.empty:
        return np.nan, np.nan
    prices = data['Close']
    daily_returns = prices.pct_change().dropna()
    ann_return = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(daily_returns)) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    return ann_vol, ann_return
