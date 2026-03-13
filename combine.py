import pandas as pd
import sys
import re


import numpy as np
import yfinance as yf
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Set CWD to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import hashlib
from scipy.optimize import newton
from scipy.stats import norm
from matplotlib import rcParams
from tabulate import tabulate

# Global Counters for logging
CACHE_LOADS = 0
CACHE_MISSES = 0
CALIBRATIONS = 0
# =============================================================================
# Cache Settings
# =============================================================================
CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cached_data(key_name, fetch_func, *args, **kwargs):
    """
    Generic caching function.
     Checks for 'cache/{key_name}'.
     If exists and < 1 day old, load via pickle.
     Else, call fetch_func(*args, **kwargs), save, and return.
    """
    # Sanitize key_name to be a valid filename
    safe_key = "".join([c if c.isalnum() or c in ('-','_','.') else '_' for c in key_name])
    if len(safe_key) > 200: # Truncate if too long, using hash for uniqueness
        safe_key = safe_key[:150] + "_" + hashlib.md5(key_name.encode()).hexdigest()
    
    filepath = os.path.join(CACHE_DIR, safe_key)
    
    global CACHE_LOADS, CACHE_MISSES
    # Check cache
    if os.path.exists(filepath):
        # Check age
        mtime = os.path.getmtime(filepath)
        if (time.time() - mtime) < 86400: # 1 day in seconds
            try:
                with open(filepath, 'rb') as f:
                    CACHE_LOADS += 1
                    return pickle.load(f)
            except Exception as e:
                pass # Silent fallback
        else:
             pass # Expired
    
    # Fetch data
    CACHE_MISSES += 1
    data = fetch_func(*args, **kwargs)
    
    # Save to cache
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[CACHE] Error saving {filepath}: {e}")
        
    return data

# =============================================================================
# 全域設定：設定中文字型與正確顯示負號
# =============================================================================
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
import os
if not os.path.exists('output'):
    os.makedirs('output')

# =============================================================================
# Logging Class
# =============================================================================
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
        # ANSI escape sequence regex
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.log_filtered = False

    def write(self, message):
        # Write to terminal (original content with colors)
        self.terminal.write(message)
        
        # Filter logic for file output
        # 1. Check for noisy tags
        if "[CACHE]" in message or "[CAL]" in message:
            self.log_filtered = True
            return
        
        # 2. Check if this is a newline following a filtered log
        if message == '\n' and self.log_filtered:
            self.log_filtered = False
            return
            
        # Reset filter flag for normal messages
        self.log_filtered = False
        
        # 3. Strip ANSI codes
        clean_message = self.ansi_escape.sub('', message)
        
        # 4. Write to file
        self.log.write(clean_message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# =============================================================================
# 共用功能函式
# =============================================================================
def clean_currency(x):
    if pd.isnull(x) or str(x).strip() == "":
        return None
    try:
        return float(str(x).replace("NT$", "").replace("$", "").replace(",", "").strip())
    except Exception as e:
        print(f"轉換 {x} 失敗: {e}")
        return None

def fix_share_sign(row):
    action = str(row['Action']).lower()
    if (action == '賣' or action == 'sell') and (row['Quantity'] > 0):
        row['Quantity'] = -row['Quantity']
    return row

def convert_ticker(ticker):
    if '.' not in ticker:
        return ticker + '.TW'
    return ticker

def get_daily_price(stock_symbol, start_date, end_date, is_tw=True):
    try:
        if is_tw:
            if isinstance(stock_symbol, list):
                stock_symbol = [convert_ticker(s) for s in stock_symbol]
            else:
                stock_symbol = convert_ticker(stock_symbol)
        # Define the actual fetch operation
        def _fetch_daily():
            return yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=False, actions=True)
            
        # Create a unique key
        # Use hash for stock_symbol if it's a list to avoid huge filenames
        if isinstance(stock_symbol, list):
             sym_str = str(sorted(stock_symbol))
        else:
             sym_str = str(stock_symbol)
        
        # We assume start_date/end_date are convertible to string representation
        # Safe key generation handled inside get_cached_data to some extent, but let's be descriptive
        key = f"daily_price_{hashlib.md5(sym_str.encode()).hexdigest()}_{str(start_date)[:10]}_{str(end_date)[:10]}.pkl"
        
        data = get_cached_data(key, _fetch_daily)
        
        # 嘗試進行價格還原 (Un-adjust)，解決 yfinance Close 已呈現分割調整後價格的問題
        try:
            if 'Stock Splits' in data.columns:
                splits = data['Stock Splits'].fillna(0)
                close_data = data['Close'].copy()
                
                # 定義還原函式
                def unadjust_series(price_s, split_s):
                    split_events = split_s[split_s > 0]
                    # 由後往前或由前往後皆可，只要對 split date 之前的價格乘上 ratio
                    for split_date, ratio in split_events.items():
                        # 找到該日期前的所有索引
                        mask = price_s.index < split_date
                        price_s.loc[mask] *= ratio
                    return price_s
                
                # 判斷是單一股票還是多檔
                if isinstance(data['Close'], pd.DataFrame):
                    # 多檔股票 (columns 為 Ticker)
                    # 注意：stock_symbol 傳進來可能是 list，也可能是 string (但 download return DF)
                    # 需要對應 column name
                    cols = close_data.columns
                    for col in cols:
                        # yfinance MultiIndex columns usually match symbol
                        if col in splits.columns:
                            close_data[col] = unadjust_series(close_data[col], splits[col])
                    return close_data
                else:
                    # 單一股票 (Series)
                    return unadjust_series(close_data, splits)
        except Exception as e:
            print(f"[WARN] Price un-adjustment failed: {e}")

        return data['Close']
    except Exception as e:
        print(f"下載 {stock_symbol} 價格失敗: {e}")
        return 0

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
            except Exception as e:
                return None
            opts = chain.calls if opt_type == 'C' else chain.puts
            match_row = opts[opts['contractSymbol'] == occ_symbol]
            if not match_row.empty:
                return float(match_row['lastPrice'].iloc[0])
            return None
            
        key = f"opt_price_{occ_symbol}.pkl"
        return get_cached_data(key, _fetch_opt_price)
    except Exception as e:
        print(f"取得選擇權 {occ_symbol} 價格失敗: {e}")
        return None

def get_current_price_yf(ticker, is_tw=True):
    try:
        if is_tw:
            ticker = convert_ticker(ticker)
        else:
            if re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", str(ticker)):
                p = get_option_price(ticker)
                if p is not None:
                    return p
            
        def _fetch_current():
            d = yf.Ticker(ticker)
            p = d.info.get("regularMarketPrice")
            if p is None:
                h = d.history(period="1d")
                if not h.empty:
                    p = h["Close"].iloc[-1]
            return p

        # Cache key
        key = f"current_price_{ticker}.pkl"
        price = get_cached_data(key, _fetch_current)
        
        return price
    except Exception as e:
        print(f"取得 {ticker} 價格失敗: {e}")
        return None

def xnpv(rate, cashflows):
    t0 = min(date for date, _ in cashflows)
    return sum(cf / ((1 + rate) ** ((date - t0).days / 365.0))
               for date, cf in cashflows)

def xirr(cashflows, guess=0.1):
    return newton(lambda r: xnpv(r, cashflows), guess)

def calculate_realized_gain(symbol, df):
    df_sym = df[df['Symbol'] == symbol]
    total_buy = -df_sym[df_sym['Action'].str.lower().isin(['買', 'buy'])]['Amount'].sum()
    total_sell = df_sym[df_sym['Action'].str.lower().isin(['賣', 'sell'])]['Amount'].sum()
    realized_gain = total_sell - total_buy
    realized_gain_pct = (realized_gain / total_buy * 100) if total_buy != 0 else 0
    return total_buy, realized_gain, realized_gain_pct

def get_twd_to_usd_rate():
    try:
        def _fetch_rate():
            t = yf.Ticker('TWD=X')
            r = t.info.get("regularMarketPrice")
            if r is None:
                h = t.history(period="1d")
                if not h.empty:
                    r = h["Close"].iloc[-1]
            return r

        rate = get_cached_data("rate_twd_usd.pkl", _fetch_rate)
        
        return 1.0 / rate
    except Exception as e:
        print("取得 TWD/USD 匯率失敗:", e)
        return 1/30.0

# =============================================================================
# 新增：計算風險報酬 (以最近1年資料)
# =============================================================================
def compute_risk_return(stock_code, is_tw=True, period='1y'):
    if is_tw:
        ticker = convert_ticker(stock_code)
    else:
        ticker = stock_code
        
    def _fetch_risk():
         return yf.Ticker(ticker).history(period=period)
         
    key = f"risk_{ticker}_{period}.pkl"
    data = get_cached_data(key, _fetch_risk)
    if data.empty:
        return np.nan, np.nan
    prices = data['Close']
    daily_returns = prices.pct_change().dropna()
    ann_return = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(daily_returns)) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    return ann_vol, ann_return

def calculate_twr_series(portfolio_value_series, cashflow_list):
    """
    計算時間加權報酬率 (Time-Weighted Return, TWR) 累積走勢
    Formula: r_t = (EndValue_t - NetInflow_t) / EndValue_{t-1} - 1
                 = (PV_t + Cashflow_t) / PV_{t-1} - 1
    (Because Cashflow < 0 means Inflow)
    """
    # 1. Align cashflows to daily index
    cf_df = pd.DataFrame(cashflow_list, columns=['Date', 'Amt'])
    cf_df['Date'] = pd.to_datetime(cf_df['Date']).dt.normalize()
    daily_cf = cf_df.groupby('Date')['Amt'].sum()
    
    # 2. Join with Portfolio Value
    df = pd.DataFrame({'PV': portfolio_value_series})
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(daily_cf, how='left').fillna(0)
    
    returns = []
    prev_pv = 0
    started = False
    
    for date, row in df.iterrows():
        pv = row['PV']
        cf = row['Amt'] # -In, +Out
        
        if not started:
            if pv > 0:
                started = True
                prev_pv = pv
                returns.append(0.0)
            else:
                returns.append(0.0)
            continue
            
        if prev_pv == 0:
            daily_r = 0.0
        else:
            # TWR: exclude expected flow effect from denominator
            daily_r = (pv + cf - prev_pv) / prev_pv
            
        returns.append(daily_r)
        prev_pv = pv
        
    twr_series = pd.Series(returns, index=df.index)
    cum_twr = (1 + twr_series).cumprod() - 1
    return cum_twr * 100

# =============================================================================
# 處理台股資料 (轉換為 USD 計價)
# =============================================================================
def process_tw_data():
    df_tw = pd.read_csv('tw_train.csv', encoding='utf-8-sig')
    df_tw.rename(columns={
        '交易日': 'Date',
        '交易別': 'Action',
        '股票代號': 'Symbol',
        '股票名稱': 'Name',
        '股數': 'Quantity',
        '單價': 'Price',
        '進帳/出帳': 'Amount'
    }, inplace=True)
    df_tw['Date'] = pd.to_datetime(df_tw['Date'])
    df_tw.sort_values('Date', inplace=True)
    df_tw['Quantity'] = pd.to_numeric(df_tw['Quantity'], errors='coerce')
    df_tw = df_tw.apply(fix_share_sign, axis=1)
    df_tw["Amount"] = df_tw["Amount"].apply(clean_currency)

    twd_to_usd = get_twd_to_usd_rate()
    df_tw["Amount"] = df_tw["Amount"] * twd_to_usd

    # ----------- 真實淨投入資金計算 (模擬帳戶現金流) -----------
    df_tw_sorted = df_tw.sort_values("Date")
    account_cash = 0
    net_invested = 0
    for _, row in df_tw_sorted.iterrows():
        amt = row["Amount"]
        if amt > 0:
            account_cash += amt
        else:
            needed = -amt
            if account_cash >= needed:
                account_cash -= needed
            else:
                net_invested += (needed - account_cash)
                account_cash = 0
    invested_capital_tw = net_invested

    start_date = df_tw['Date'].min()
    end_date = pd.Timestamp.today()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')

    pivot = df_tw.pivot_table(index='Date', columns='Symbol', values='Quantity', aggfunc='sum')
    pivot = pivot.reindex(date_range, fill_value=0).fillna(0)
    cum_holdings = pivot.cumsum()
    symbols_tw = cum_holdings.columns.tolist()

    price_data_tw = get_daily_price(symbols_tw, start_date, end_date, is_tw=True)
    if price_data_tw is not None and not price_data_tw.empty:
        price_data_tw.columns = [col.split('.')[0] for col in price_data_tw.columns]
    price_data_tw = price_data_tw.reindex(date_range).ffill().bfill()
    # === Calibration: align yfinance Close to first trade price per symbol (TWD scale) ===
    try:
        _cal_info = []
        for _s in symbols_tw:
            _rows = df_tw[df_tw['Symbol'] == _s].sort_values('Date')
            if _rows.empty:
                continue
            _d0 = _rows.iloc[0]['Date']
            _px_csv = _rows.iloc[0]['Price']
            if pd.isna(_px_csv):
                continue
            # yfinance Close on first trade date
            try:
                _px_yf = float(price_data_tw.loc[_d0, _s])
            except Exception:
                continue
            if _px_yf and _px_yf > 0:
                _factor = float(_px_csv) / _px_yf
                price_data_tw[_s] = price_data_tw[_s] * _factor
                _cal_info.append((_s, str(_d0.date()), float(_px_csv), float(_px_yf), float(_factor)))
        if _cal_info:
            global CALIBRATIONS
            CALIBRATIONS += len(_cal_info)
            _cal_info = [] # clear after counting
    except Exception as _e:
        pass

    price_data_tw = price_data_tw * twd_to_usd

    portfolio_value_tw = (cum_holdings * price_data_tw).sum(axis=1).fillna(0)

    cashflows_tw = list(df_tw[['Date', 'Amount']].itertuples(index=False, name=None))
    net_holdings_tw = df_tw.groupby('Symbol')['Quantity'].sum()
    portfolio_snapshot_tw = 0
    for stock, shares in net_holdings_tw.items():
        if shares != 0:
            price = get_current_price_yf(stock, is_tw=True)
            if price is not None:
                portfolio_snapshot_tw += shares * (price * twd_to_usd)
    today = pd.Timestamp.today().normalize()
    # cashflows_tw.append((today, portfolio_snapshot_tw))

    total_investment_tw = -df_tw[df_tw['Amount'] < 0]['Amount'].sum()
    final_portfolio_value_tw = portfolio_value_tw.iloc[-1]
    total_profit_tw = final_portfolio_value_tw - invested_capital_tw
    total_profit_pct_tw = (total_profit_tw / invested_capital_tw) * 100 if invested_capital_tw != 0 else 0
    stock_counts_tw = {}
    for idx, row in df_tw.iterrows():
        stock_code = row['Symbol']
        stock_name = row['Name'] if pd.notna(row['Name']) else stock_code
        count = row['Quantity']
        cost = float(row['Amount'])
        if pd.isna(stock_code) or pd.isna(count):
            continue
        if stock_code not in stock_counts_tw:
            stock_counts_tw[stock_code] = {'stock_name': stock_name, 'Quantity_now': 0, 'cost': 0}
        stock_counts_tw[stock_code]['Quantity_now'] += count
        stock_counts_tw[stock_code]['cost'] += cost
    data_list_tw = []
    for stock_code, data_dict in stock_counts_tw.items():
        name = data_dict['stock_name']
        count = data_dict['Quantity_now']
        aggregated_cost = -data_dict['cost']
        if count != 0:
            try:
                ticker_obj = yf.Ticker(convert_ticker(stock_code))
                current_price = ticker_obj.history(period='1d')['Close'].iloc[-1] * twd_to_usd
            except Exception as e:
                print(f"Error fetching data for {stock_code}: {e}")
                current_price = 0
            current_value = current_price * count
            gain = current_value - aggregated_cost
            gain_per = (gain / aggregated_cost) * 100 if aggregated_cost != 0 else 0
        else:
            total_buy, realized_gain, realized_gain_pct = calculate_realized_gain(stock_code, df_tw)
            current_price = np.nan
            current_value = np.nan
            aggregated_cost = total_buy
            gain = realized_gain
            gain_per = realized_gain_pct
        data_list_tw.append([stock_code, name, count, current_price, current_value, aggregated_cost, gain, gain_per])
    headers = ['Symbol', 'Name', 'Quantity_now', 'Price', 'Price_Total', 'Cost', 'Gain', 'Gain(%)']
    portfolio_df_tw = pd.DataFrame(data_list_tw, columns=headers)

    return {
        'df': df_tw,
        'date_range': date_range,
        'portfolio_value': portfolio_value_tw,
        'cashflows': cashflows_tw,
        'total_investment': total_investment_tw,
        'invested_capital': invested_capital_tw,
        'final_portfolio_value': final_portfolio_value_tw,
        'total_profit': total_profit_tw,
        'total_profit_pct': total_profit_pct_tw,
        'portfolio_df': portfolio_df_tw,
        'portfolio_snapshot': portfolio_snapshot_tw,
        'price_data': price_data_tw,
        'symbols': symbols_tw
    }
# =============================================================================
# 處理美股資料 (以 USD 計價)
# =============================================================================
def process_us_data():
    df_us = pd.read_csv('us_train.csv', encoding='utf-8-sig')
    df_us = df_us.copy()

    df_us['Date'] = pd.to_datetime(df_us['Date'])
    df_us.sort_values('Date', inplace=True)
    df_us = df_us.apply(fix_share_sign, axis=1)
    df_us["Amount"] = df_us["Amount"].apply(clean_currency)

    # ----------- 真實淨投入資金計算 (模擬帳戶現金流) -----------
    df_us_sorted = df_us.sort_values("Date")
    account_cash = 0
    net_invested = 0
    for _, row in df_us_sorted.iterrows():
        amt = row["Amount"]
        if amt > 0:
            account_cash += amt
        else:
            needed = -amt
            if account_cash >= needed:
                account_cash -= needed
            else:
                net_invested += (needed - account_cash)
                account_cash = 0
    invested_capital_us = net_invested

    start_date = df_us['Date'].min()
    end_date = pd.Timestamp.today()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')

    pivot = df_us.pivot_table(index='Date', columns='Symbol', values='Quantity', aggfunc='sum')
    pivot = pivot.reindex(date_range, fill_value=0).fillna(0)
    cum_holdings = pivot.cumsum()
    symbols_us = cum_holdings.columns.tolist()

    # Filter out option symbols for daily price fetching to prevent yfinance delisted warnings
    symbols_to_fetch_us = [s for s in symbols_us if not re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", str(s))]
    
    price_data_us = pd.DataFrame(index=date_range)
    if symbols_to_fetch_us:
        fetched_price_data = get_daily_price(symbols_to_fetch_us, start_date, end_date, is_tw=False)
        fetched_price_data = fetched_price_data.reindex(date_range).ffill().bfill()
        if isinstance(fetched_price_data, pd.Series):
             price_data_us[symbols_to_fetch_us[0]] = fetched_price_data
        else:
             price_data_us = fetched_price_data

    # Add missing option columns with 0 or last known values so pandas aligned multiply works
    for s in symbols_us:
        if s not in price_data_us.columns:
            opt_price = get_option_price(s)
            price_data_us[s] = opt_price if opt_price is not None else 0.0

    portfolio_value_us = (cum_holdings * price_data_us).sum(axis=1).fillna(0)

    cashflows_us = list(df_us[['Date', 'Amount']].itertuples(index=False, name=None))
    net_holdings_us = df_us.groupby('Symbol')['Quantity'].sum()
    portfolio_snapshot_us = 0
    for stock, shares in net_holdings_us.items():
        if shares != 0:
            price = get_current_price_yf(stock, is_tw=False)
            if price is not None:
                portfolio_snapshot_us += shares * price
    today = pd.Timestamp.today().normalize()
    # cashflows_us.append((today, portfolio_snapshot_us))

    total_investment_us = -df_us[df_us['Amount'] < 0]['Amount'].sum()
    final_portfolio_value_us = portfolio_value_us.iloc[-1]
    total_profit_us = final_portfolio_value_us - invested_capital_us
    total_profit_pct_us = (total_profit_us / invested_capital_us) * 100 if invested_capital_us != 0 else 0

    # 組裝 portfolio_df_us
    stock_counts_us = {}
    for idx, row in df_us.iterrows():
        stock_code = row['Symbol']
        stock_name = row['Symbol']
        count = row['Quantity']
        cost = float(row['Amount'])
        if pd.isna(stock_code) or pd.isna(count):
            continue
        if stock_code not in stock_counts_us:
            stock_counts_us[stock_code] = {'stock_name': stock_name, 'Quantity_now': 0, 'cost': 0}
        stock_counts_us[stock_code]['Quantity_now'] += count
        stock_counts_us[stock_code]['cost'] += cost
    data_list_us = []
    for stock_code, data_dict in stock_counts_us.items():
        name = data_dict['stock_name']
        count = data_dict['Quantity_now']
        aggregated_cost = -data_dict['cost']
        if count != 0:
            try:
                if re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", str(stock_code)):
                    current_price = get_option_price(stock_code)
                    if current_price is None:
                        current_price = 0
                else:
                    ticker_obj = yf.Ticker(stock_code)
                    current_price = ticker_obj.history(period='1d')['Close'].iloc[-1]
            except Exception as e:
                print(f"Error fetching data for {stock_code}: {e}")
                current_price = 0
            current_value = current_price * count
            gain = current_value - aggregated_cost
            gain_per = (gain / aggregated_cost) * 100 if aggregated_cost != 0 else 0
        else:
            total_buy, realized_gain, realized_gain_pct = calculate_realized_gain(stock_code, df_us)
            current_price = np.nan
            current_value = np.nan
            aggregated_cost = total_buy
            gain = realized_gain
            gain_per = realized_gain_pct
        data_list_us.append([stock_code, name, count, current_price, current_value, aggregated_cost, gain, gain_per])
    headers = ['Symbol', 'Name', 'Quantity_now', 'Price', 'Price_Total', 'Cost', 'Gain', 'Gain(%)']
    portfolio_df_us = pd.DataFrame(data_list_us, columns=headers)

    return {
        'df': df_us,
        'date_range': date_range,
        'portfolio_value': portfolio_value_us,
        'cashflows': cashflows_us,
        'total_investment': total_investment_us,
        'invested_capital': invested_capital_us,
        'final_portfolio_value': final_portfolio_value_us,
        'total_profit': total_profit_us,
        'total_profit_pct': total_profit_pct_us,
        'portfolio_df': portfolio_df_us,
        'portfolio_snapshot': portfolio_snapshot_us,
        'price_data': price_data_us,
        'symbols': symbols_us
    }


def simulate_stock_full(cashflows, ticker='^SP500TR'):
    """根據現金流買 / 賣指定 ticker（預設 S&P 500 Total Return）。

    Parameters
    ----------
    cashflows : list[(Timestamp, float)]
        正數 = 提領 / 賣出，負數 = 投入 / 買入。
    ticker : str, default '^SP500TR'
        Yahoo Finance 代碼；可替換成 ETF、個股或其他指數。

    Returns
    -------
    portfolio : pd.Series
        每日市值 (USD)。
    shares_ts : pd.Series
        每日持有份額。"""

    # --- 整理現金流 ---
    cf_df = (pd.DataFrame(cashflows, columns=['Date', 'Amt'])
                .assign(Date=lambda d: pd.to_datetime(d['Date']).dt.normalize()))
    start = cf_df['Date'].min()
    end   = pd.Timestamp.today().normalize()

    # --- 下載價格（Series）---
    # --- 下載價格（Series）---
    def _fetch_sim_history():
        return yf.Ticker(ticker).history(start=start, end=end,auto_adjust=True)['Close']
        
    key = f"sim_hist_{ticker}_{start.date()}_{end.date()}.pkl"
    px = get_cached_data(key, _fetch_sim_history)
    px = px.sort_index().ffill().bfill()
    if hasattr(px.index, 'tz'):
        px.index = px.index.tz_localize(None)

    # --- 把日期對齊到交易日 ---
    cf_df['TradeDate'] = cf_df['Date'].apply(
        lambda d: px.index[px.index.get_indexer([d], method='bfill')[0]]
    )
    daily_cf = cf_df.groupby('TradeDate')['Amt'].sum()

    # --- 逐日模擬：先算市值，再調份額 ---
    port_list, shares_list = [], []
    shares = 0.0
    for dt in px.index:
        price = px.loc[dt]
        port_list.append(shares * price)
        if dt in daily_cf.index:
            cash  = daily_cf.loc[dt]
            delta = -cash / price
            if shares + delta < 0:
                delta = -shares
            shares += delta
        shares_list.append(shares)

    portfolio = pd.Series(port_list,   index=px.index, name=f'{ticker}_Value')
    shares_ts = pd.Series(shares_list, index=px.index, name=f'{ticker}_Shares')
    return portfolio, shares_ts


def plot_stock_performance(tw_res, us_res):
    """
    Generate a chart showing the cumulative return % (TWR) for every individual stock.
    X: Time, Y: Return %
    """
    plt.figure(figsize=(14, 8))
    
    # Process TW stocks
    _process_stock_twr(tw_res, "TW", plt)
    
    # Process US stocks
    _process_stock_twr(us_res, "US", plt)
    
    plt.title("Individual Stock Performance (TWR %)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/stock_performance.png')
    plt.show()
    plt.close()
    print("Stock performance chart saved to output/stock_performance.png")

def _process_stock_twr(res_data, region, plt_obj):
    df_tx = res_data['df']
    price_df = res_data['price_data']
    date_range = res_data['date_range']
    symbols = res_data['symbols']
    
    for sym in symbols:
        # Filter transactions for this symbol
        sym_tx = df_tx[df_tx['Symbol'] == sym].copy()
        if sym_tx.empty:
            continue
            
        # 1. Build Shares Series
        daily_quantity = sym_tx.groupby('Date')['Quantity'].sum()
        shares_series = daily_quantity.reindex(date_range).fillna(0).cumsum()
        
        # 2. Get Price Series
        price_col = None
        if sym in price_df.columns:
            price_col = sym
        elif str(sym).split('.')[0] in price_df.columns:
             price_col = str(sym).split('.')[0]
        
        if price_col is None:
            continue
            
        price_s = price_df[price_col]
        
        # 3. Calculate Portfolio Value (PV) = Shares * Price
        combined_df = pd.DataFrame({'Shares': shares_series, 'Price': price_s})
        combined_df = combined_df.dropna(subset=['Price']) 
        combined_df['Shares'] = combined_df['Shares'].ffill().fillna(0)
        
        pv_series = combined_df['Shares'] * combined_df['Price']
        
        # 4. Prepare Cashflows
        cfs = list(sym_tx[['Date', 'Amount']].itertuples(index=False, name=None))
        
        # 5. Calculate TWR
        try:
            twr = calculate_twr_series(pv_series, cfs)
            
            # 6. Plot
            if not twr.empty and not (twr == 0).all():
                non_zero = twr[twr != 0]
                if not non_zero.empty:
                    first_date = non_zero.index[0]
                    plot_data = twr.loc[first_date:]
                    
                    # Style: dashed if sold out
                    current_shares = shares_series.iloc[-1]
                    alpha = 1.0 if abs(current_shares) > 0.001 else 0.4
                    linestyle = '-' if abs(current_shares) > 0.001 else ':'
                    label = f"{sym} ({region})"
                    
                    plt_obj.plot(plot_data.index, plot_data, label=label, alpha=alpha, linestyle=linestyle, linewidth=1.5)
                    
        except Exception as e:
            print(f"Error calculating TWR for {sym}: {e}")



# =============================================================================
# 目標配置與再平衡建議
# =============================================================================
def print_rebalance_recommendation(portfolio_df_combined, usd_to_twd):
    """
    Calculates and prints the target portfolio allocation vs current.
    Target: QLD(30%), SPLG/SPYM(25%), 0050正2 [00631L.TW](15%), 台灣50 [0050.TW](30%)
    Excludes Puts from computation.
    """
    targets = {
        'QLD': 0.30,
        'SPLG / SPYM': 0.25,
        '00631L': 0.15,
        '006208': 0.30
    }
    
    current_values_usd = {k: 0.0 for k in targets.keys()}
    
    total_pool_usd = 0.0
    
    for _, row in portfolio_df_combined.iterrows():
        sym = str(row['Symbol'])
        qty = float(row['Quantity_now'])
        val_usd = float(pd.to_numeric(row['Price_Total'], errors='coerce'))
        
        if pd.isna(val_usd) or qty == 0:
            continue
            
        # Add to total ONLY if it matches our target assets (excluding Puts and Other)
        if 'QLD' in sym:
            current_values_usd['QLD'] += val_usd
            total_pool_usd += val_usd
        elif 'SPLG' in sym or 'SPYM' in sym:
            current_values_usd['SPLG / SPYM'] += val_usd
            total_pool_usd += val_usd
        elif '00631L' in sym:
            current_values_usd['00631L'] += val_usd
            total_pool_usd += val_usd
        elif '006208' in sym:
            current_values_usd['006208'] += val_usd
            total_pool_usd += val_usd
            
    print("\n=== 目標配置與再平衡建議 (排除 Puts / 其他) ===")
    
    total_pool_twd = total_pool_usd * usd_to_twd
    print(f"再平衡總資金池 (僅計股票部位): {total_pool_twd:,.0f} TWD")
    
    table_data = []
    
    for key, target_pct in targets.items():
        curr_usd = current_values_usd[key]
        curr_twd = curr_usd * usd_to_twd
        
        target_twd = total_pool_twd * target_pct
        diff_twd = target_twd - curr_twd
        
        curr_pct = (curr_twd / total_pool_twd) * 100 if total_pool_twd > 0 else 0
        
        action_str = f"+{diff_twd:,.0f}" if diff_twd > 0 else f"{diff_twd:,.0f}"
        if diff_twd > 0:
            action_str = f"\033[92m{action_str}\033[0m" # Green
        elif diff_twd < 0:
            action_str = f"\033[91m{action_str}\033[0m" # Red
            
        table_data.append([
            key, 
            f"{target_pct*100:.1f}%", 
            f"{curr_pct:.1f}%", 
            f"{curr_twd:,.0f}", 
            f"{target_twd:,.0f}", 
            action_str
        ])
        
    headers = ["標的", "目標佔比", "當前佔比", "當前金額 (TWD)", "目標金額 (TWD)", "建議動作 (TWD)"]
    print(tabulate(table_data, headers=headers, tablefmt="psql", stralign="right"))

# =============================================================================
# 主程式：整合 TW 與 US 資料，產出 USD 與 TWD 版本報告、圖表及風險報酬散點圖
# =============================================================================

# =============================================================================
# 新增：QQQ Put 保護力分析
# =============================================================================
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def analyze_put_protection(portfolio_df):
    """
    Analyzes Put protection based on current portfolio composition.
    Supports multiple Puts (e.g. QQQ, TSM).
    portfolio_df: DataFrame with columns ['Symbol', 'Quantity_now', 'Price_Total', ...]
                  Price_Total should be in USD.
    """
    # 1. Identify Assets
    high_beta_tickers = ['QLD', '00631L.TW', 'TQQQ', 'SOXL', 'TECL', 'NVDL']
    hedge_tickers = ['EDV', 'TLT', 'TMF', 'ZROZ', 'UBT']
    
    # Normalize symbols for matching
    df = portfolio_df.copy()
    
    high_beta_val = 0
    market_beta_val = 0
    hedge_val = 0
    total_stock_val = 0
    
    put_contracts = []
    
    for _, row in df.iterrows():
        sym = str(row['Symbol'])
        val = float(row['Price_Total'])
        qty = float(row['Quantity_now'])
        
        # Check if Put
        # Pattern: Underlying + YYMMDD + P + Strike
        if 'P' in sym and any(c.isdigit() for c in sym) and qty > 0:
            m = re.match(r"^([A-Z]+)(\d{6})P(\d{8})$", sym)
            if m:
                put_contracts.append(row)
                continue
            # Backup: if it contains QQQ or TSM and P but different format
            if ('QQQ' in sym or 'TSM' in sym) and 'P' in sym: 
                put_contracts.append(row)
                continue
            
        total_stock_val += val
        
        # Categorize
        is_high = False
        for hb in high_beta_tickers:
            if hb.split('.')[0] in sym:
                high_beta_val += val
                is_high = True
                break
        
        if is_high: continue
        
        is_hedge = False
        for h in hedge_tickers:
            if h in sym:
                hedge_val += val
                is_hedge = True
                break
        
        if is_hedge: continue
        
        # Default to market beta
        market_beta_val += val

    # Setup Put Details
    if not put_contracts:
        return

    puts_info = []
    for put in put_contracts:
        sym = put['Symbol']
        m = re.match(r"^([A-Z]+)(\d{6})P(\d{8})$", sym)
        if m:
            underlying = m.group(1)
            date_str = m.group(2)
            strike_str = m.group(3)
        else:
            p_index = sym.find('P')
            base = sym[:p_index]
            date_str = base[-6:]
            underlying = base[:-6]
            strike_str = sym[p_index+1:]
        
        try:
            strike = float(strike_str) / 1000.0
            expiry = datetime.strptime(date_str, "%y%m%d")
            contracts = float(put['Quantity_now']) / 100.0 
            cost_basis = float(put['Cost'])
            
            # Underlying price
            try:
                hist = yf.Ticker(underlying).history(period="1d")
                s0 = hist['Close'].iloc[-1]
            except:
                s0 = 610.0 if underlying == 'QQQ' else 200.0 # fallback
                
            puts_info.append({
                'sym': sym,
                'underlying': underlying,
                'strike': strike,
                'expiry': expiry,
                'contracts': contracts,
                'cost_basis': cost_basis,
                's0': s0
            })
        except:
            continue

    if not puts_info:
        return

    # Parameters
    today = datetime.now()
    r = 0.045 # Risk free
    
    # Total Portfolio Value (Use calculated sum)
    PORTFOLIO_TOTAL = total_stock_val
    
    # Scenarios
    scenarios = [
        {"drop": 0.0, "desc": "Current (目前)"},
        {"drop": -0.10, "desc": "Correction (回調)"},
        {"drop": -0.20, "desc": "Bear Market (熊市)"},
        {"drop": -0.30, "desc": "Crash (崩盤)"},
        {"drop": -0.40, "desc": "Crisis (金融危機)"},
        {"drop": -0.50, "desc": "Collapse (毀滅)"},
    ]

    print("\n=== Put 保護力分析 (總資產預估) ===")
    for p_info in puts_info:
        print(f"Put: {p_info['sym']}, Strike ${p_info['strike']}, Exp {p_info['expiry'].strftime('%Y-%m-%d')} (Underlying: {p_info['underlying']} @ ${p_info['s0']:.2f})")
    print(f"總資產 (股票): ${PORTFOLIO_TOTAL:,.0f} USD")
    print("| 情境 | 基準跌幅 | 總資產 (無保) | 總資產 (有保) | Puts 總價值 | 保護效果 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    # For Charting
    drops = np.linspace(0, -1.00, 100)
    wealth_no_hedge = []
    wealth_with_hedge = []

    # Table Loop
    for sc in scenarios:
        drop = sc['drop']
        
        # Portfolio Loss
        drop_2x = drop * 2.0 # Simplify
        if drop_2x < -0.99: drop_2x = -0.99
        
        new_high = high_beta_val * (1 + drop_2x)
        
        bond_change = 0.0
        if drop <= -0.20: bond_change = 0.05
        if drop <= -0.30: bond_change = 0.15
        new_hedge = hedge_val * (1 + bond_change)
        
        new_market = market_beta_val * (1 + drop)
        
        new_total_no_put = new_high + new_hedge + new_market
        
        # Total Puts Value
        total_put_val = 0
        for p in puts_info:
            if p['expiry'] <= today:
                continue
            days_to_exp = (p['expiry'] - today).days
            T = days_to_exp / 365.0
            
            # Assuming the underlying drops by the same 'drop' percentage
            new_u = p['s0'] * (1 + drop)
            vol = 0.20 + abs(drop) * 0.8
            put_val_share = black_scholes_put(new_u, p['strike'], T, r, vol)
            total_put_val += put_val_share * 100 * p['contracts']
            
        new_total_with_put = new_total_no_put + total_put_val
        diff = new_total_with_put - new_total_no_put
        
        print(f"| {sc['desc']} | {drop*100:.0f}% | ${new_total_no_put:,.0f} | **${new_total_with_put:,.0f}** | ${total_put_val:,.0f} | +${diff:,.0f} |")

    # Chart Data Loop
    for drop in drops:
        drop_2x = drop * 2.0
        if drop_2x < -0.99: drop_2x = -0.99
        new_high = high_beta_val * (1 + drop_2x)
        bond_ch = 0.0
        if drop <= -0.20: bond_ch = 0.05
        if drop <= -0.30: bond_ch = 0.15
        new_hedge = hedge_val * (1 + bond_ch)
        new_market = market_beta_val * (1 + drop)
        
        val_no = new_high + new_hedge + new_market
        
        total_pv = 0
        for p in puts_info:
            if p['expiry'] <= today: continue
            days_to_exp = (p['expiry'] - today).days
            T = days_to_exp / 365.0
            new_u = p['s0'] * (1 + drop)
            vol = 0.20 + abs(drop) * 0.8
            pv_share = black_scholes_put(new_u, p['strike'], T, r, vol)
            total_pv += pv_share * 100 * p['contracts']
            
        wealth_no_hedge.append(val_no)
        wealth_with_hedge.append(val_no + total_pv)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(drops * 100, wealth_no_hedge, label='Total Assets (No Puts)', color='red', linewidth=2)
    ax1.plot(drops * 100, wealth_with_hedge, label='Total Assets (With Puts)', color='blue', linewidth=2)
    ax1.fill_between(drops * 100, wealth_no_hedge, wealth_with_hedge, color='green', alpha=0.1, label='Protection')
    
    ax1.set_title('Total Asset Protection (Market Drop)')
    ax1.set_xlabel('Market Drop (%)')
    ax1.set_ylabel('Total Asset Value (USD)')
    ax1.grid(True, alpha=0.3)
    
    payouts = np.array(wealth_with_hedge) - np.array(wealth_no_hedge)
    
    # Secondary axis for Puts Payout
    ax2 = ax1.twinx()
    ax2.plot(drops * 100, payouts, label='Puts Payout Value', color='purple', linestyle='--', linewidth=2)
    ax2.set_ylabel('Puts Value (USD)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Annotate
    try:
        idx = next(x for x, val in enumerate(payouts) if val > 1000)
        start_drop = drops[idx] * 100
        ax1.axvline(x=start_drop, color='orange', linestyle=':')
        ax1.text(start_drop - 2, min(wealth_no_hedge)+5000, f'Starts ~{abs(start_drop):.0f}%', rotation=90, color='orange')
    except:
        pass
        
    plt.tight_layout()
    plt.savefig('output/total_asset_protection.png')
    plt.show() 
    plt.close()
    print("Chart saved to output/total_asset_protection.png")

def main():
    # =================================================================
    # Phase 1: 資料準備（所有計算）
    # =================================================================

    # --- 1-1. Logger 初始化 ---
    original_stdout = sys.stdout
    sys.stdout = DualLogger('output/report.txt')

    # --- 1-2. 匯率 + TW/US 原始資料 ---
    twd_to_usd = get_twd_to_usd_rate()
    usd_to_twd = 1 / twd_to_usd
    tw_result = process_tw_data()
    us_result = process_us_data()

    # --- 1-3. 合併投組市值、現金流、投入資金 ---
    date_index = tw_result['portfolio_value'].index.union(us_result['portfolio_value'].index).sort_values()
    portfolio_value_tw = tw_result['portfolio_value'].reindex(date_index, method='ffill').fillna(0)
    portfolio_value_us = us_result['portfolio_value'].reindex(date_index, method='ffill').fillna(0)
    combined_portfolio_value_us = portfolio_value_tw + portfolio_value_us

    combined_cashflows = tw_result['cashflows'] + us_result['cashflows']
    total_investment_us = tw_result['total_investment'] + us_result['total_investment']
    invested_capital_us = tw_result['invested_capital'] + us_result['invested_capital']
    final_portfolio_value_us = combined_portfolio_value_us.iloc[-1]
    total_profit_us = final_portfolio_value_us - invested_capital_us
    total_profit_pct_us = (total_profit_us / invested_capital_us) * 100 if invested_capital_us != 0 else 0

    # TWD 版本
    combined_portfolio_value_twd = combined_portfolio_value_us * usd_to_twd
    total_investment_twd = total_investment_us * usd_to_twd
    invested_capital_twd = invested_capital_us * usd_to_twd
    final_portfolio_value_twd = final_portfolio_value_us * usd_to_twd
    total_profit_twd = total_profit_us * usd_to_twd

    # --- 1-4. transactions_df + 累積投入資金線（提前建立，後續多張圖需要） ---
    transactions_df = pd.concat([tw_result['df'], us_result['df']])
    transactions_df['Date'] = pd.to_datetime(transactions_df['Date']).dt.normalize()
    cf_df = transactions_df[['Date','Amount']].sort_values('Date')
    daily_cf = (cf_df.groupby('Date')['Amount']
                   .sum()
                   .reindex(date_index, fill_value=0)
                   .cumsum())
    daily_invested_capital = (-daily_cf).clip(lower=0)
    daily_invested_capital_twd = daily_invested_capital * usd_to_twd

    # --- 1-5. 風險指標：Sortino / Calmar / Max Drawdown ---
    annual_rf = 0.02
    daily_rf = annual_rf / 252
    daily_returns = combined_portfolio_value_us.pct_change().dropna()
    excess_returns = daily_returns - daily_rf
    downside_returns = excess_returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) if len(downside_returns) > 0 else np.nan
    sortino_ratio = (excess_returns.mean() / downside_deviation) * np.sqrt(252) if downside_deviation != 0 else np.nan

    wealth_factor = combined_portfolio_value_us / invested_capital_us
    running_max = wealth_factor.cummax()
    drawdown_series = (wealth_factor - running_max) / running_max
    max_drawdown = drawdown_series.min() * 100
    annual_return = (final_portfolio_value_us / combined_portfolio_value_us.iloc[0])**(252/len(daily_returns)) - 1
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # --- 1-6. XIRR ---
    total_snapshot = tw_result['portfolio_snapshot'] + us_result['portfolio_snapshot']
    xirr_cashflows = combined_cashflows + [(pd.Timestamp.today(), total_snapshot)]
    combined_irr = None
    try:
        combined_irr = xirr(xirr_cashflows)
    except Exception as e:
        print("綜合 XIRR 計算失敗:", e)

    # --- 1-7. 個股明細表處理 ---
    portfolio_df_combined = pd.concat([tw_result['portfolio_df'], us_result['portfolio_df']], ignore_index=True)
    portfolio_df_combined = portfolio_df_combined.fillna(0)
    portfolio_df_combined.rename(columns={'Gain': 'Gain(USD)'}, inplace=True)
    portfolio_df_combined['Gain(TWD)'] = portfolio_df_combined['Gain(USD)'] * usd_to_twd
    portfolio_df_combined['Gain(USD)'] = portfolio_df_combined['Gain(USD)'].apply(
        lambda x: f"\033[92m{float(x):,.2f}\033[0m" if float(x) > 0
                  else (f"\033[91m{float(x):,.2f}\033[0m" if float(x) < 0 else f"{float(x):,.2f}")
    )
    portfolio_df_combined['Gain(TWD)'] = portfolio_df_combined['Gain(TWD)'].apply(
        lambda x: f"\033[92m{float(x):,.2f}\033[0m" if float(x) > 0
                  else (f"\033[91m{float(x):,.2f}\033[0m" if float(x) < 0 else f"{float(x):,.2f}")
    )
    portfolio_df_combined['Gain(%)'] = portfolio_df_combined['Gain(%)'].apply(
        lambda x: f"\033[92m{float(x):,.2f}%\033[0m" if float(x) > 0
                  else (f"\033[91m{float(x):,.2f}%\033[0m" if float(x) < 0 else f"{float(x):,.2f}%")
    )

    # 計算平均成本 (每股成本)
    portfolio_df_combined['AvgCost'] = portfolio_df_combined.apply(
        lambda r: r['Cost'] / r['Quantity_now'] if r['Quantity_now'] != 0 else np.nan,
        axis=1
    )
    portfolio_df_combined['AvgCost'] = portfolio_df_combined['AvgCost'].map(
        lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
    )

    # 計算持股佔比
    portfolio_df_combined['Alloc_Price_Total'] = pd.to_numeric(portfolio_df_combined['Price_Total'], errors='coerce').fillna(0)
    total_val_for_alloc = portfolio_df_combined.loc[portfolio_df_combined['Alloc_Price_Total'] > 0, 'Alloc_Price_Total'].sum()

    portfolio_df_combined['Alloc(%)'] = 0.0
    if total_val_for_alloc > 0:
        mask = portfolio_df_combined['Alloc_Price_Total'] > 0
        portfolio_df_combined.loc[mask, 'Alloc(%)'] = (portfolio_df_combined.loc[mask, 'Alloc_Price_Total'] / total_val_for_alloc) * 100

    portfolio_df_combined['Alloc(%)'] = portfolio_df_combined.apply(
        lambda row: f"{row['Alloc(%)']:.2f}% ({row['Price_Total'] * usd_to_twd:,.0f})" if row['Alloc_Price_Total'] > 0 else "0.00% (0)",
        axis=1
    )

    portfolio_df_combined = portfolio_df_combined[
        ['Symbol', 'Name', 'Quantity_now', 'Price', 'AvgCost', 'Price_Total', 'Cost', 'Gain(USD)', 'Gain(TWD)', 'Gain(%)', 'Alloc(%)']
    ]

    # --- 1-8. Benchmark 模擬 ---
    COMPARE_TICKERS = ['SPY','QQQ','EWT']

    sim_portfolios = {}
    for tk in COMPARE_TICKERS:
        sim_portfolios[tk], _ = simulate_stock_full(combined_cashflows, ticker=tk)

    idx = combined_portfolio_value_us.index.copy()
    for p in sim_portfolios.values():
        idx = idx.union(p.index)
    idx = idx.sort_values()

    my_us = combined_portfolio_value_us.reindex(idx).ffill()
    sims  = {tk: p.reindex(idx).ffill() for tk, p in sim_portfolios.items()}

    # --- 1-9. TWR + Benchmark TWR ---
    twr_series = calculate_twr_series(combined_portfolio_value_us, combined_cashflows)

    bench_twr = {}
    valid_idx = twr_series[twr_series != 0].index
    if not valid_idx.empty:
        start_date = valid_idx[0]
    else:
        start_date = twr_series.index[0]

    for tk in COMPARE_TICKERS:
        try:
            def _fetch_bench():
                return yf.download(tk, start=start_date, end=None, progress=False, auto_adjust=True)

            key = f"bench_twr_{tk}_{start_date.date()}.pkl"
            _px = get_cached_data(key, _fetch_bench)

            if isinstance(_px, pd.DataFrame):
                if 'Close' in _px.columns:
                     _px = _px['Close']
                else:
                     _px = _px.iloc[:, 0]

            if isinstance(_px, pd.DataFrame):
                _px = _px.iloc[:, 0]

            _px.index = _px.index.tz_localize(None)
            _px = _px.reindex(twr_series.index).ffill().bfill()

            start_val = _px.loc[start_date]
            if start_val > 0:
                 bench_twr[tk] = (_px / start_val - 1) * 100
        except Exception as e:
             print(f"Skipping benchmark TWR for {tk}: {e}")

    # --- 1-10. Benchmark 對照表計算 ---
    today = pd.Timestamp.today().normalize()
    base_cf = [(d, amt) for (d, amt) in combined_cashflows if d < today]

    def last_valid(series):
        return series.dropna().iloc[-1] if series.dropna().size else np.nan

    def calc_risk_metrics_from_twr(twr_pct_series, risk_free_rate=0.03):
        if twr_pct_series.empty:
            return np.nan, np.nan, np.nan
        wealth_index = 1 + (twr_pct_series / 100.0)
        ret = wealth_index.pct_change().dropna()
        if ret.empty:
            return np.nan, np.nan, np.nan
        ann_vol = ret.std() * np.sqrt(252)
        daily_rf_inner = (1 + risk_free_rate) ** (1/252) - 1
        excess_ret = ret - daily_rf_inner
        if ret.std() == 0:
            sharpe = np.nan
        else:
            sharpe = np.sqrt(252) * (excess_ret.mean() / ret.std())
        run_max = wealth_index.cummax()
        if run_max.max() == 0:
            max_dd = 0
        else:
            drawdown = (wealth_index - run_max) / run_max
            max_dd = abs(drawdown.min())
        return ann_vol * 100, max_dd * 100, sharpe

    benchmark_rows = []

    # My Portfolio
    p_my = combined_portfolio_value_us
    ann_vol_my, max_dd_my, sharpe_my = calc_risk_metrics_from_twr(twr_series)

    benchmark_rows.append([
        'My Portfolio',
        p_my.iloc[-1] * usd_to_twd,
        (p_my.iloc[-1] * usd_to_twd) - invested_capital_twd,
        total_profit_pct_us,
        combined_irr * 100 if combined_irr is not None else np.nan,
        ann_vol_my,
        max_dd_my,
        sharpe_my
    ])

    # Benchmarks
    for tk, p_raw in sims.items():
        p = p_raw.copy()
        final_us = last_valid(p)
        if np.isnan(final_us):
            print(f'[warning] {tk} 無可用資料，已略過')
            continue
        final_twd  = final_us * usd_to_twd
        profit_twd = final_twd - invested_capital_twd
        profit_pct = (profit_twd / invested_capital_twd) * 100
        cf_sim = base_cf + [(today, final_us)]
        try:
            sim_irr = xirr(cf_sim) * 100
        except Exception:
            sim_irr = np.nan
        if tk in bench_twr:
            sim_vol, max_dd, sim_sharpe = calc_risk_metrics_from_twr(bench_twr[tk])
        else:
            sim_vol, max_dd, sim_sharpe = np.nan, np.nan, np.nan
        benchmark_rows.append([
            tk, final_twd, profit_twd, profit_pct, sim_irr, sim_vol, max_dd, sim_sharpe
        ])

    bench_headers = [
        'Asset', 'Final Value (TWD)', 'Profit (TWD)',
        'Profit %', 'XIRR %', 'AnnVol %', 'MaxDD %', 'Sharpe'
    ]
    bench_df = pd.DataFrame(benchmark_rows, columns=bench_headers)

    # =================================================================
    # Phase 2: 文字報告（一次性全部印出）
    # =================================================================

    # --- 2-1. 綜合資產報告 (USD) ---
    print("\n=== 綜合資產配置報告 (單位: USD) ===")
    print(f"累積買入金額：{total_investment_us:,.2f} USD")
    print(f"實際淨投入資金：{invested_capital_us:,.2f} USD")
    print(f"最終組合市值：{final_portfolio_value_us:,.2f} USD")
    print(f"總獲利：{total_profit_us:,.2f} USD")
    print(f"總獲利百分比：{total_profit_pct_us:.2f}%")
    print(f"Sortino Ratio：{sortino_ratio:.2f}")
    print(f"Calmar Ratio：{calmar_ratio:.2f}")
    if combined_irr is not None:
        print(f"綜合 XIRR: {combined_irr:.2%}")

    # --- 2-2. 綜合資產報告 (TWD) ---
    print("\n=== 綜合資產配置報告 (單位: TWD) ===")
    print(f"累積買入金額：{total_investment_twd:,.2f} TWD")
    print(f"實際淨投入資金：{invested_capital_twd:,.2f} TWD")
    print(f"最終組合市值：{final_portfolio_value_twd:,.2f} TWD")
    print(f"總獲利：{total_profit_twd:,.2f} TWD")
    print(f"總獲利百分比：{total_profit_pct_us:.2f}%")
    if combined_irr is not None:
        print(f"綜合 XIRR: {combined_irr:.2%}")

    # --- 2-3. 個股明細表 ---
    print("\n=== 綜合投資組合股票明細 (TWD) ===")
    print(tabulate(portfolio_df_combined, headers='keys', tablefmt='psql', showindex=False))

    # --- 2-4. 投組 vs Benchmark 總表 ---
    print("\n=== 投組 vs. Benchmark 總表 (TWD) ===")
    print(tabulate(
        bench_df,
        headers='keys',
        tablefmt='psql',
        showindex=False,
        floatfmt='.2f'
    ))

    # --- 2-5. 目標配置與再平衡建議 ---
    print_rebalance_recommendation(portfolio_df_combined, usd_to_twd)

    # =================================================================
    # Phase 3: 圖表（由概觀到細節）
    # =================================================================

    # --- 3-1. 資產走勢圖 + 累積投入資金線（全局概覽） ---
    plt.figure(figsize=(10, 6))
    plt.plot(combined_portfolio_value_twd.index, combined_portfolio_value_twd.values, label='資產走勢圖')
    plt.plot(daily_invested_capital_twd.index, daily_invested_capital_twd.values, label='累積投入資金', linestyle='--')
    plt.title('資產走勢圖')
    plt.xlabel('日期')
    plt.ylabel('組合市值 (TWD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/asset_trend.png')
    plt.show()
    plt.close()

    # --- 3-2. Funding Ratio ---
    _den = daily_invested_capital_twd.replace(0, np.nan)
    ratio = (combined_portfolio_value_twd / _den).dropna()

    plt.figure(figsize=(10, 5))
    plt.plot(ratio.index, ratio.values, label='Funding Ratio (資產/累積投入)')
    plt.axhline(1.0, linestyle='--', alpha=0.6, label='=1（打平）')
    plt.title('Funding Ratio（資產 ÷ 累積投入，TWD）')
    plt.xlabel('日期'); plt.ylabel('倍數')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('output/funding_ratio.png')
    plt.show()
    plt.close()

    # --- 3-3. 累積報酬率比較 (TWR) ---
    plt.figure(figsize=(12, 6))
    plt.plot(twr_series.index, twr_series, label='My Portfolio (TWR)', linewidth=2, color='blue')
    plt.legend()
    plt.title('Portfolio Cumulative TWR Return')
    plt.grid(True)
    plt.savefig('output/twr_chart.png')

    for tk, ser in bench_twr.items():
        plt.plot(ser.index, ser, label=f'{tk}', alpha=0.7)

    plt.title('累積報酬率比較 (Time-Weighted Return)')
    plt.xlabel('日期')
    plt.ylabel('累積報酬 (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/cumulative_return_comparison.png')
    plt.show()
    plt.close()

    # --- 3-4. My Portfolio vs Benchmark (USD 絕對市值) ---
    plt.figure(figsize=(11, 6))
    plt.plot(my_us, label='My Portfolio', linewidth=2)
    for tk, p in sims.items():
        plt.plot(p, label=f'{tk} 模擬')
    plt.plot(daily_invested_capital.reindex(idx).ffill(), label='累積投入資金 (USD)', linestyle='--', linewidth=1.5)
    plt.title('My Portfolio vs. 多重 Benchmark (USD)')
    plt.xlabel('日期'); plt.ylabel('市值 / 指數 (USD)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('output/portfolio_vs_benchmark_usd.png')
    plt.show()
    plt.close()

    # --- 3-5. Drawdown 水下圖 ---
    wealth_index = combined_portfolio_value_us / invested_capital_us
    running_max_dd = wealth_index.cummax()
    drawdown = (wealth_index - running_max_dd) / running_max_dd

    plt.figure(figsize=(10, 6))
    plt.fill_between(drawdown.index, drawdown * 100, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown * 100, label='Drawdown (%)')
    plt.title('最大回撤（Drawdown）水下圖')
    plt.xlabel('日期')
    plt.ylabel('回撤 (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/drawdown_underwater.png')
    plt.show()
    plt.close()

    # --- 3-6. 資產圓餅圖 ---
    combined_df_chart = portfolio_df_combined.dropna(subset=['Price_Total'])
    combined_df_chart = combined_df_chart[combined_df_chart['Price_Total'] > 0]
    combined_df_chart['Price_Total'] = pd.to_numeric(combined_df_chart['Price_Total'], errors='coerce')
    combined_df_chart['Price_Total_TWD'] = combined_df_chart['Price_Total'] * usd_to_twd

    total_pie_twd = combined_df_chart['Price_Total_TWD'].sum()
    pie_labels = combined_df_chart.apply(
        lambda row: f"{row['Name']} {row['Price_Total_TWD']/total_pie_twd*100:.1f}% ({row['Price_Total_TWD']:,.0f})", axis=1
    )

    plt.figure(figsize=(10, 8))
    plt.pie(combined_df_chart['Price_Total_TWD'], labels=pie_labels, startangle=140)
    plt.title('資產圓餅圖')
    plt.axis('equal')
    plt.savefig('output/asset_pie_chart.png')
    plt.show()
    plt.close()

    # --- 3-7. 每月投入資產 ---
    daily_net = transactions_df.groupby('Date')['Amount'].sum()
    daily_injection = daily_net[daily_net < 0].abs()
    monthly_investment = daily_injection.groupby(daily_injection.index.to_period('M')).sum()
    monthly_investment.index = monthly_investment.index.to_timestamp()

    plt.figure(figsize=(10, 6))
    plt.bar(monthly_investment.index, monthly_investment.values, width=20)
    plt.title("每月投入資產 (USD)")
    plt.xlabel("月份")
    plt.ylabel("投入金額 (USD)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('output/monthly_investment.png')
    plt.show()
    plt.close()

    # --- 3-8. 個股績效 (TWR) ---
    plot_stock_performance(tw_result, us_result)

    # --- 3-9. Put 保護力分析 ---
    analyze_put_protection(portfolio_df_combined)

    # =================================================================
    # Phase 4: 執行摘要
    # =================================================================
    print("\n==================================================")
    print("Execution Summary:")
    print(f"  - Cache Loads: {CACHE_LOADS}")
    print(f"  - Cache Misses (Network Fetches): {CACHE_MISSES}")
    print(f"  - Price Calibrations Applied: {CALIBRATIONS}")
    print("==================================================")


if __name__ == '__main__':
    main()

