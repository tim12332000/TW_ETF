import pandas as pd
import sys
import re


import numpy as np
import yfinance as yf
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pickle
import hashlib
from scipy.optimize import newton
from matplotlib import rcParams
from tabulate import tabulate

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
    
    # Check cache
    if os.path.exists(filepath):
        # Check age
        mtime = os.path.getmtime(filepath)
        if (time.time() - mtime) < 86400: # 1 day in seconds
            try:
                with open(filepath, 'rb') as f:
                    print(f"[CACHE] Loaded {key_name}")
                    return pickle.load(f)
            except Exception as e:
                print(f"[CACHE] Error reading {filepath}: {e}, refreshing...")
        else:
             print(f"[CACHE] {key_name} expired, refreshing...")
    
    # Fetch data
    print(f"[FETCH] Fetching {key_name}...")
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

def get_current_price_yf(ticker, is_tw=True):
    try:
        if is_tw:
            ticker = convert_ticker(ticker)
            
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
                print(f"[CAL] TWD price calibration applied: {_cal_info[:5]} ...")
    except Exception as _e:
        print("[CAL][WARN]", _e)

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

    price_data_us = get_daily_price(symbols_us, start_date, end_date, is_tw=False)
    price_data_us = price_data_us.reindex(date_range).ffill().bfill()
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
# 主程式：整合 TW 與 US 資料，產出 USD 與 TWD 版本報告、圖表及風險報酬散點圖
# =============================================================================
def main():
    # Redirect stdout to both terminal and a file
    sys.stdout = DualLogger('output/report.txt')

    twd_to_usd = get_twd_to_usd_rate()    
    usd_to_twd = 1 / twd_to_usd             
    
    tw_result = process_tw_data()  
    us_result = process_us_data()  

    # ----------------------
    # USD 版本報告與績效指標
    # ----------------------
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

    # 額外績效指標：Sortino Ratio 與 Calmar Ratio
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

    print("\n=== 綜合資產配置報告 (單位: USD) ===")
    print(f"累積買入金額：{total_investment_us:,.2f} USD")
    print(f"實際淨投入資金：{invested_capital_us:,.2f} USD")
    print(f"最終組合市值：{final_portfolio_value_us:,.2f} USD")
    print(f"總獲利：{total_profit_us:,.2f} USD")
    print(f"總獲利百分比：{total_profit_pct_us:.2f}%")
    print(f"Sortino Ratio：{sortino_ratio:.2f}")
    print(f"Calmar Ratio：{calmar_ratio:.2f}")
    try:
        # XIRR 需要包含最終市值作為一筆正向現金流
        total_snapshot = tw_result['portfolio_snapshot'] + us_result['portfolio_snapshot']
        xirr_cashflows = combined_cashflows + [(pd.Timestamp.today(), total_snapshot)]
        combined_irr = xirr(xirr_cashflows)
        print(f"綜合 XIRR: {combined_irr:.2%}")
    except Exception as e:
        print("綜合 XIRR 計算失敗:", e)

    # ----------------------
    # TWD 版本報告
    # ----------------------
    combined_portfolio_value_twd = combined_portfolio_value_us * usd_to_twd
    total_investment_twd = total_investment_us * usd_to_twd
    invested_capital_twd = invested_capital_us * usd_to_twd
    final_portfolio_value_twd = final_portfolio_value_us * usd_to_twd
    total_profit_twd = total_profit_us * usd_to_twd

    print("\n=== 綜合資產配置報告 (單位: TWD) ===")
    print(f"累積買入金額：{total_investment_twd:,.2f} TWD")
    print(f"實際淨投入資金：{invested_capital_twd:,.2f} TWD")
    print(f"最終組合市值：{final_portfolio_value_twd:,.2f} TWD")
    print(f"總獲利：{total_profit_twd:,.2f} TWD")
    print(f"總獲利百分比：{total_profit_pct_us:.2f}%")
    try:
        # TWD XIRR 也需要包含最終市值 (轉換後)
        total_snapshot_twd = (tw_result['portfolio_snapshot'] + us_result['portfolio_snapshot']) * usd_to_twd
        xirr_cashflows_twd = combined_cashflows + [(pd.Timestamp.today(), total_snapshot_twd)]
        # Note: combined_cashflows here are presumably in USD scale from previous logic? 
        # Wait, cashflows_tw came from 'Amount' * twd_to_usd. So combined_cashflows is in USD.
        # But for TWD report, we probably want to calculate XIRR on TWD cashflows?
        # The original code: combined_irr_twd = xirr(combined_cashflows) uses USD cashflows!
        # XIRR % should be roughly same for USD or TWD if currency fluctuation isn't huge, 
        # but technically valid to just check the variable usage.
        # Original code used 'combined_cashflows' for both. Let's stick to that but add snapshot.
        # Actually, if we use USD cashflows for the second print, we just reuse xirr_cashflows.
        # BUT the print says "Units: TWD".
        # Let's double check if we need to convert flows back to TWD.
        # effectively xirr should be dimension-less (%) so calculating on USD flows is fine.
        combined_irr_twd = xirr(xirr_cashflows) 
        print(f"綜合 XIRR: {combined_irr_twd:.2%}")
    except Exception as e:
        print("綜合 XIRR 計算失敗:", e)

    # ----------------------
    # 個股明細 (合併後，僅保留三個獲利欄位：Gain(USD)、Gain(TWD) 與 Gain(%))
    # ----------------------
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
	
	# 新增：計算平均成本 (每股成本)
    portfolio_df_combined['AvgCost'] = portfolio_df_combined.apply(
        lambda r: r['Cost'] / r['Quantity_now'] if r['Quantity_now'] != 0 else np.nan,
        axis=1
    )
    portfolio_df_combined['AvgCost'] = portfolio_df_combined['AvgCost'].map(
        lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
    )

    # 最後選欄位（將 AvgCost 插入 Price 之後）
    portfolio_df_combined = portfolio_df_combined[
        ['Symbol', 'Name', 'Quantity_now', 'Price', 'AvgCost', 'Price_Total', 'Cost', 'Gain(USD)', 'Gain(TWD)', 'Gain(%)']
    ]
	

    print("\n=== 綜合投資組合股票明細 (TWD) ===")
    print(tabulate(portfolio_df_combined, headers='keys', tablefmt='psql', showindex=False))
    # ~~ 此區塊已後移至 transactions_df/cashline 之後 ~~
# === 2. 想比較的指數／ETF／個股清單（可自行增刪） ===
    COMPARE_TICKERS = ['SPY','QQQ','EWT']#['SPY','QQQ','QLD','TQQQ','SQQQ','VT','EWT','GLD','TLT','SHY','BRK-B']

    # 建立 dict 存放「按照你的現金流模擬」之結果
    sim_portfolios = {}
    for tk in COMPARE_TICKERS:
        sim_portfolios[tk], _ = simulate_stock_full(combined_cashflows, ticker=tk)

    # === 3. 合併所有索引（我的投組 + 各模擬組合） ===
    idx = combined_portfolio_value_us.index.copy()
    for p in sim_portfolios.values():
        idx = idx.union(p.index)
    idx = idx.sort_values()

    # === 4. 對齊、補值 ===
    my_us = combined_portfolio_value_us.reindex(idx).ffill()
    sims  = {tk: p.reindex(idx).ffill() for tk, p in sim_portfolios.items()}
    # ~~ 此區塊已後移至 transactions_df/cashline 之後（4a 絕對市值） ~~
    # ----------------------
    # 累積報酬率走勢圖 (TWR - 不受本金進出影響)
    # ----------------------
    twr_series = calculate_twr_series(combined_portfolio_value_us, combined_cashflows)

    # Benchmarks TWR (Pure Price Return)
    bench_twr = {}
    
    # 找出 My Portfolio 開始的第一天
    valid_idx = twr_series[twr_series != 0].index
    if not valid_idx.empty:
        start_date = valid_idx[0]
    else:
        start_date = twr_series.index[0]
        
    # 下載 Benchmark 價格並歸一化
    for tk in COMPARE_TICKERS:
        try:
            def _fetch_bench():
                return yf.download(tk, start=start_date, end=None, progress=False, auto_adjust=True)
                
            key = f"bench_twr_{tk}_{start_date.date()}.pkl"
            _px = get_cached_data(key, _fetch_bench)
            
            if isinstance(_px, pd.DataFrame):
                # Handle MultiIndex columns if present
                if 'Close' in _px.columns:
                     _px = _px['Close']
                else:
                     # Fallback if structure is different
                     _px = _px.iloc[:, 0]
            
            # Additional check for Series vs DataFrame
            if isinstance(_px, pd.DataFrame):
                _px = _px.iloc[:, 0]

            _px.index = _px.index.tz_localize(None)
            _px = _px.reindex(twr_series.index).ffill().bfill()
            
            # 歸一化: (Pt / P0) - 1
            start_val = _px.loc[start_date]
            if start_val > 0:
                 bench_twr[tk] = (_px / start_val - 1) * 100
        except Exception as e:
             print(f"Skipping benchmark TWR for {tk}: {e}")

    # ----------------------  Benchmark 對照表 (TWD) + 風險指標  ----------------------
    today = pd.Timestamp.today().normalize()

    # 只保留 today 以前的現金流，避免雙重快照
    base_cf = [(d, amt) for (d, amt) in combined_cashflows if d < today]

    benchmark_rows = []

    # ---------------------------------------------------------------------
    # 工具函式
    # ---------------------------------------------------------------------
    def last_valid(series):
        """最後一筆非 NaN 值；若整列皆 NaN 回傳 np.nan"""
        return series.dropna().iloc[-1] if series.dropna().size else np.nan

    def calc_risk_metrics_from_twr(twr_pct_series, risk_free_rate=0.03):
        """
        計算風險指標 (AnnVol %, MaxDD %, Sharpe)
        twr_pct_series: 累積報酬率 (%) Series (Time-Weighted)
                        e.g. 0, 1.2, -0.5, 10.5 ...
        """
        if twr_pct_series.empty:
            return np.nan, np.nan, np.nan

        # 還原成淨值走勢 (Wealth Index)
        wealth_index = 1 + (twr_pct_series / 100.0)
        
        # 日報酬率
        ret = wealth_index.pct_change().dropna()
        
        if ret.empty:
            return np.nan, np.nan, np.nan

        # 年化波動率
        ann_vol = ret.std() * np.sqrt(252)
        
        # 夏普值 (Sharpe Ratio)
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_ret = ret - daily_rf
        if ret.std() == 0:
            sharpe = np.nan
        else:
            sharpe = np.sqrt(252) * (excess_ret.mean() / ret.std())

        # 最大回撤 (Max Drawdown)
        run_max = wealth_index.cummax()
        if run_max.max() == 0:
            max_dd = 0
        else:
            drawdown = (wealth_index - run_max) / run_max
            max_dd = abs(drawdown.min())

        return ann_vol * 100, max_dd * 100, sharpe

    # ---------------------------------------------------------------------
    # 你的投組
    # ---------------------------------------------------------------------
    p_my = combined_portfolio_value_us
    # 改用 TWR Series 計算風險指標
    ann_vol_my, max_dd_my, sharpe_my = calc_risk_metrics_from_twr(twr_series)

    benchmark_rows.append([
        'My Portfolio',
        p_my.iloc[-1] * usd_to_twd,                           # Final Value (TWD)
        (p_my.iloc[-1] * usd_to_twd) - invested_capital_twd,  # Profit (TWD)
        total_profit_pct_us,                                  # Profit %
        combined_irr_twd * 100,                               # XIRR %
        ann_vol_my,                                           # AnnVol %
        max_dd_my,                                            # MaxDD %
        sharpe_my                                             # Sharpe
    ])

    # ---------------------------------------------------------------------
    # 各 Benchmark
    # ---------------------------------------------------------------------
    for tk, p_raw in sims.items():
        p = p_raw.copy()

        final_us = last_valid(p)
        if np.isnan(final_us):
            print(f'[warning] {tk} 無可用資料，已略過')
            continue

        final_twd  = final_us * usd_to_twd
        profit_twd = final_twd - invested_capital_twd
        profit_pct = (profit_twd / invested_capital_twd) * 100

        # IRR (today 一次性清算)
        cf_sim = base_cf + [(today, final_us)]
        try:
            sim_irr = xirr(cf_sim) * 100
        except Exception:
            sim_irr = np.nan

        # 風險指標 (改用 Benchmark TWR)
        if tk in bench_twr:
            sim_vol, max_dd, sim_sharpe = calc_risk_metrics_from_twr(bench_twr[tk])
        else:
            sim_vol, max_dd, sim_sharpe = np.nan, np.nan, np.nan

        benchmark_rows.append([
            tk, final_twd, profit_twd, profit_pct, sim_irr, sim_vol, max_dd, sim_sharpe
        ])

    # ---------------------------------------------------------------------
    # 列印結果
    # ---------------------------------------------------------------------
    bench_headers = [
        'Asset', 'Final Value (TWD)', 'Profit (TWD)',
        'Profit %', 'XIRR %', 'AnnVol %', 'MaxDD %', 'Sharpe'
    ]
    bench_df = pd.DataFrame(benchmark_rows, columns=bench_headers)

    print("\n=== 投組 vs. Benchmark 總表 (TWD) ===")
    print(tabulate(
        bench_df,
        headers='keys',
        tablefmt='psql',
        showindex=False,
        floatfmt='.2f'
    ))

    # 畫圖
    plt.figure(figsize=(12, 6))
    plt.plot(twr_series.index, twr_series, label='My Portfolio (TWR)', linewidth=2, color='blue')
    plt.legend()
    plt.title('Portfolio Cumulative TWR Return')
    plt.grid(True)
    plt.savefig('output/twr_chart.png')
    # plt.close() # Keep open or close depending on usage? Original code didn't have close() maybe? 
    # Ah, original code ended with plt.plot(...). It didn't save or show in the snippet I saw?
    # Wait, the snippet showed:
    # 799:     plt.figure(figsize=(12, 6))
    # 800:     plt.plot(twr_series.index, twr_series, label='My Portfolio (TWR)', linewidth=2, color='blue')
    # It didn't show the end of main(). 
    # But I am replacing from line 800.
    # The snippet cut off at line 800.
    # I should be careful not to delete subsequent code if I don't know what it is.
    # I will just INSERT the call at the end of main, or after the TWR plotting block.
    # I need to see more lines to be safe.
    # Let's abort this chunk and view more lines first?
    # No, I can assume the standard pattern or just append to end of file if main calls it?
    # Wait, I am replacing line 800 "plt.plot(twr_series...)"
    # I should instead READ the end of the file first.
    # I will remove this chunk from this call and do a separate view_file then edit.
    pass

    
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


    # ----------------------
    # 資產圓餅圖 (以 TWD 為基準)
    # ----------------------
    combined_df_chart = portfolio_df_combined.dropna(subset=['Price_Total'])
    combined_df_chart = combined_df_chart[combined_df_chart['Price_Total'] > 0]
    combined_df_chart['Price_Total'] = pd.to_numeric(combined_df_chart['Price_Total'], errors='coerce')
    combined_df_chart['Price_Total_TWD'] = combined_df_chart['Price_Total'] * usd_to_twd
    plt.figure(figsize=(10,8))
    plt.pie(combined_df_chart['Price_Total_TWD'], labels=combined_df_chart['Name'], 
            autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '', startangle=140)
    plt.title('資產圓餅圖')
    plt.axis('equal')
    plt.axis('equal')
    plt.savefig('output/asset_pie_chart.png')
    plt.show()
    plt.close()

    # ----------------------
    # 新增：每月投入資產圖表 (扣除同日賣出金額)
    # ----------------------
    # 先依日期計算淨現金流（買進為負、賣出為正）
    transactions_df = pd.concat([tw_result['df'], us_result['df']])

    # ==== PATCH: tx cashline (after transactions_df defined) BEGIN ====
    # 用 transactions_df 作為現金流來源，建立累積淨投入資金線
    transactions_df['Date'] = pd.to_datetime(transactions_df['Date']).dt.normalize()
    cf_df = transactions_df[['Date','Amount']].sort_values('Date')
    daily_cf = (cf_df.groupby('Date')['Amount']
                   .sum()
                   .reindex(date_index, fill_value=0)
                   .cumsum())
    daily_invested_capital = (-daily_cf).clip(lower=0)
    daily_invested_capital_twd = daily_invested_capital * usd_to_twd
    # ==== PATCH: tx cashline (after transactions_df defined) END ====

    # === Funding Ratio：資產 / 累積投入（TWD） ===
    _den = daily_invested_capital_twd.replace(0, np.nan)
    ratio = (combined_portfolio_value_twd / _den).dropna()

    plt.figure(figsize=(10, 5))
    plt.plot(ratio.index, ratio.values, label='Funding Ratio (資產/累積投入)')
    plt.axhline(1.0, linestyle='--', alpha=0.6, label='=1（打平）')
    plt.title('Funding Ratio（資產 ÷ 累積投入，TWD）')
    plt.xlabel('日期'); plt.ylabel('倍數')
    plt.legend(); plt.grid(True); plt.tight_layout();
    plt.savefig('output/funding_ratio.png')
    plt.show()
    plt.close()


# ----------------------
    # 資產走勢圖 (以 TWD 為基準)
    # ----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(combined_portfolio_value_twd.index, combined_portfolio_value_twd.values, label='資產走勢圖')
    plt.plot(daily_invested_capital_twd.index, daily_invested_capital_twd.values, label='累積投入資金', linestyle='--')
    plt.title('資產走勢圖')
    plt.xlabel('日期')
    plt.ylabel('組合市值 (TWD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('output/asset_trend.png')
    plt.show()
    plt.close()

# ---------------- 4a. 絕對市值 ----------------
    plt.figure(figsize=(11,6))
    plt.plot(my_us, label='My Portfolio', linewidth=2)
    for tk, p in sims.items():
        plt.plot(p, label=f'{tk} 模擬')
    plt.plot(daily_invested_capital.reindex(idx).ffill(), label='累積投入資金 (USD)', linestyle='--', linewidth=1.5)
    plt.title('My Portfolio vs. 多重 Benchmark (USD)')
    plt.xlabel('日期'); plt.ylabel('市值 / 指數 (USD)')
    plt.legend(); plt.grid(True); plt.tight_layout();
    plt.savefig('output/portfolio_vs_benchmark_usd.png')
    plt.show()
    plt.close()


    daily_net = transactions_df.groupby('Date')['Amount'].sum()
    # 僅取淨現金流為負的日期（代表實際新增資金投入），再取絕對值
    daily_injection = daily_net[daily_net < 0].abs()
    monthly_investment = daily_injection.groupby(daily_injection.index.to_period('M')).sum()
    monthly_investment.index = monthly_investment.index.to_timestamp()
    plt.figure(figsize=(10,6))
    plt.bar(monthly_investment.index, monthly_investment.values, width=20)
    plt.title("每月投入資產 (USD)")
    plt.xlabel("月份")
    plt.ylabel("投入金額 (USD)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('output/monthly_investment.png')
    plt.show()
    plt.close()
	
	# ----------------------
    # 最大回撤（Drawdown）可視化──水下圖
    # ----------------------
    # wealth_index：累積殖利率（含本金）
    wealth_index = combined_portfolio_value_us / invested_capital_us
    # running_max：至今最高值
    running_max = wealth_index.cummax()
    # drawdown：現值與最高值的相對落後
    drawdown = (wealth_index - running_max) / running_max

    plt.figure(figsize=(10, 6))
    # 填滿水下區域（多乘 100 轉成％）
    plt.fill_between(drawdown.index, drawdown * 100, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown * 100, label='Drawdown (%)')
    plt.title('最大回撤（Drawdown）水下圖')
    plt.xlabel('日期')
    plt.ylabel('回撤 (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('output/drawdown_underwater.png')
    plt.show()
    plt.close()

    # Output individual stock performance
    plot_stock_performance(tw_result, us_result)







if __name__ == '__main__':
    main()
