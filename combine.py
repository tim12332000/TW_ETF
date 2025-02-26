import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import newton
from matplotlib import rcParams
from tabulate import tabulate

# =============================================================================
# 全域設定：設定中文字型與正確顯示負號
# =============================================================================
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# =============================================================================
# 共用功能函式
# =============================================================================

def clean_currency(x):
    """
    清理貨幣字串，移除可能的 "NT$", "$" 與逗號，並轉換成 float。
    """
    if pd.isnull(x) or str(x).strip() == "":
        return None
    try:
        return float(str(x).replace("NT$", "").replace("$", "").replace(",", "").strip())
    except Exception as e:
        print(f"轉換 {x} 失敗: {e}")
        return None

def fix_share_sign(row):
    """
    若交易別為賣出（中文「賣」或英文 "sell"），且股數為正，則將股數轉為負值。
    """
    action = str(row['Action']).lower()
    if (action == '賣' or action == 'sell') and (row['Quantity'] > 0):
        row['Quantity'] = -row['Quantity']
    return row

def convert_ticker(ticker):
    """
    將台股股票代號轉換成 Yahoo Finance 格式（加上 .TW 後綴）
    """
    if '.' not in ticker:
        return ticker + '.TW'
    return ticker

def get_daily_price(stock_symbol, start_date, end_date, is_tw=True):
    """
    下載指定股票（或股票列表）每日收盤價資料，回傳 'Close' 欄位。
    若 is_tw 為 True，則先將 ticker 轉換成台股格式（加上 .TW）。
    """
    try:
        if is_tw:
            if isinstance(stock_symbol, list):
                stock_symbol = [convert_ticker(s) for s in stock_symbol]
            else:
                stock_symbol = convert_ticker(stock_symbol)
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        return data['Close']
    except Exception as e:
        print(f"下載 {stock_symbol} 價格失敗: {e}")
        return 0

def get_current_price_yf(ticker, is_tw=True):
    """
    查詢單一股票的最新價格。
    若 is_tw 為 True，則先轉換成台股格式。
    """
    try:
        if is_tw:
            ticker = convert_ticker(ticker)
        data = yf.Ticker(ticker)
        price = data.info.get("regularMarketPrice")
        if price is None:
            hist = data.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
        return price
    except Exception as e:
        print(f"取得 {ticker} 價格失敗: {e}")
        return None

def xnpv(rate, cashflows):
    """
    計算不規則現金流的淨現值 (XNPV)
    """
    t0 = min(date for date, _ in cashflows)
    return sum(cf / ((1 + rate) ** ((date - t0).days / 365.0))
               for date, cf in cashflows)

def xirr(cashflows, guess=0.1):
    """
    利用牛頓法求解不規則現金流的內部報酬率 (XIRR)
    """
    return newton(lambda r: xnpv(r, cashflows), guess)

def calculate_realized_gain(symbol, df):
    """
    計算已平倉標的的實現盈虧及報酬率。
      - total_buy：買入總成本（轉正）
      - realized_gain：賣出總金額減去買入成本
      - realized_gain_pct：報酬率
    """
    df_sym = df[df['Symbol'] == symbol]
    total_buy = -df_sym[df_sym['Action'].str.lower().isin(['買', 'buy'])]['Amount'].sum()
    total_sell = df_sym[df_sym['Action'].str.lower().isin(['賣', 'sell'])]['Amount'].sum()
    realized_gain = total_sell - total_buy
    realized_gain_pct = (realized_gain / total_buy * 100) if total_buy != 0 else 0
    return total_buy, realized_gain, realized_gain_pct

def get_twd_to_usd_rate():
    """
    透過 yfinance 取得台幣兌美金匯率，回傳轉換因子：
    NT$ 轉 USD = 1 / (TWD per USD)
    """
    try:
        ticker = yf.Ticker('TWD=X')
        rate = ticker.info.get("regularMarketPrice")
        if rate is None:
            hist = ticker.history(period="1d")
            if not hist.empty:
                rate = hist["Close"].iloc[-1]
        return 1.0 / rate
    except Exception as e:
        print("取得 TWD/USD 匯率失敗:", e)
        return 1/30.0  # 預設約 1 USD ≒ 30 NT$

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
    
    # 匯率轉換：NT$ -> USD
    twd_to_usd = get_twd_to_usd_rate()
    df_tw["Amount"] = df_tw["Amount"] * twd_to_usd

    start_date = df_tw['Date'].min()
    end_date = pd.Timestamp.today()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    pivot = df_tw.pivot_table(index='Date', columns='Symbol', values='Quantity', aggfunc='sum')
    pivot = pivot.reindex(date_range, fill_value=0).fillna(0)
    cum_holdings = pivot.cumsum()
    symbols_tw = cum_holdings.columns.tolist()

    # 取得台股每日收盤價 (以 NT$ 表示，再換算成 USD)
    price_data_tw = get_daily_price(symbols_tw, start_date, end_date, is_tw=True)
    if price_data_tw is not None and not price_data_tw.empty:
        price_data_tw.columns = [col.split('.')[0] for col in price_data_tw.columns]
    price_data_tw = price_data_tw.reindex(date_range).ffill().bfill()
    price_data_tw = price_data_tw * twd_to_usd  # 轉換價格到 USD

    portfolio_value_tw = (cum_holdings * price_data_tw).sum(axis=1).fillna(0)

    # 建立 TW 現金流（均已換算成 USD）
    cashflows_tw = list(df_tw[['Date', 'Amount']].itertuples(index=False, name=None))
    net_holdings_tw = df_tw.groupby('Symbol')['Quantity'].sum()
    portfolio_snapshot_tw = 0
    for stock, shares in net_holdings_tw.items():
        if shares != 0:
            price = get_current_price_yf(stock, is_tw=True)
            if price is not None:
                portfolio_snapshot_tw += shares * (price * twd_to_usd)
    today = pd.Timestamp.today().normalize()
    cashflows_tw.append((today, portfolio_snapshot_tw))
    
    total_investment_tw = -df_tw[df_tw['Amount'] < 0]['Amount'].sum()
    cash_balance_tw = df_tw['Amount'].sum()
    invested_capital_tw = -cash_balance_tw
    final_portfolio_value_tw = portfolio_value_tw.iloc[-1]
    total_profit_tw = final_portfolio_value_tw - invested_capital_tw
    total_profit_pct_tw = (total_profit_tw / invested_capital_tw) * 100 if invested_capital_tw != 0 else 0

    # 個股明細 (均以 USD 計價)
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
        'portfolio_df': portfolio_df_tw
    }

# =============================================================================
# 處理美股資料 (本身以 USD 計價)
# =============================================================================

def process_us_data():
    df_us = pd.read_csv('us_train.csv', encoding='utf-8-sig')
    df_us['Date'] = pd.to_datetime(df_us['Date'])
    df_us.sort_values('Date', inplace=True)
    df_us = df_us.apply(fix_share_sign, axis=1)
    df_us["Amount"] = df_us["Amount"].apply(clean_currency)

    start_date = df_us['Date'].min()
    end_date = pd.Timestamp.today()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    pivot = df_us.pivot_table(index='Date', columns='Symbol', values='Quantity', aggfunc='sum')
    pivot = pivot.reindex(date_range, fill_value=0).fillna(0)
    cum_holdings = pivot.cumsum()
    symbols_us = cum_holdings.columns.tolist()

    # 美股不需轉換 ticker
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
    cashflows_us.append((today, portfolio_snapshot_us))

    total_investment_us = -df_us[df_us['Amount'] < 0]['Amount'].sum()
    cash_balance_us = df_us['Amount'].sum()
    invested_capital_us = -cash_balance_us
    final_portfolio_value_us = portfolio_value_us.iloc[-1]
    total_profit_us = final_portfolio_value_us - invested_capital_us
    total_profit_pct_us = (total_profit_us / invested_capital_us) * 100 if invested_capital_us != 0 else 0

    stock_counts_us = {}
    for idx, row in df_us.iterrows():
        stock_code = row['Symbol']
        stock_name = row['Symbol']  # 可根據需要調整為股票名稱
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
        'portfolio_df': portfolio_df_us
    }

# =============================================================================
# 主程式：整合 TW 與 US 資料，並以 USD 計價
# =============================================================================

def main():
    tw_result = process_tw_data()
    us_result = process_us_data()

    # 統一日期範圍（取 TW 與 US 的聯集後排序）
    date_index = tw_result['portfolio_value'].index.union(us_result['portfolio_value'].index)
    date_index = date_index.sort_values()

    portfolio_value_tw = tw_result['portfolio_value'].reindex(date_index, method='ffill').fillna(0)
    portfolio_value_us = us_result['portfolio_value'].reindex(date_index, method='ffill').fillna(0)
    combined_portfolio_value = portfolio_value_tw + portfolio_value_us

    # 合併現金流（TW 與 US 均已換算成 USD）
    combined_cashflows = tw_result['cashflows'] + us_result['cashflows']

    total_investment = tw_result['total_investment'] + us_result['total_investment']
    invested_capital = tw_result['invested_capital'] + us_result['invested_capital']
    final_portfolio_value = combined_portfolio_value.iloc[-1]
    total_profit = final_portfolio_value - invested_capital
    total_profit_pct = (total_profit / invested_capital) * 100 if invested_capital != 0 else 0

    print("\n=== 綜合資產配置報告 (單位: USD) ===")
    print(f"累積買入金額：{total_investment:,.2f} USD")
    print(f"實際淨投入資金：{invested_capital:,.2f} USD")
    print(f"最終組合市值：{final_portfolio_value:,.2f} USD")
    print(f"總獲利：{total_profit:,.2f} USD")
    print(f"總獲利百分比：{total_profit_pct:.2f}%")
    try:
        combined_irr = xirr(combined_cashflows)
        print(f"綜合 XIRR: {combined_irr:.2%}")
    except Exception as e:
        print("綜合 XIRR 計算失敗:", e)

    # 合併個股明細（TW 與 US）
    portfolio_df_combined = pd.concat([tw_result['portfolio_df'], us_result['portfolio_df']], ignore_index=True)
    # 將所有 nan 值替換為 0
    portfolio_df_combined = portfolio_df_combined.fillna(0)
    # 根據 Gain 欄位數值，正數綠色，負數紅色
    portfolio_df_combined['Gain'] = portfolio_df_combined['Gain'].apply(
        lambda x: f"\033[92m{float(x):,.2f}\033[0m" if float(x) > 0 
                  else (f"\033[91m{float(x):,.2f}\033[0m" if float(x) < 0 
                        else f"{float(x):,.2f}")
    )
    # 同理，根據 Gain(%) 欄位數值正負顯示不同顏色
    portfolio_df_combined['Gain(%)'] = portfolio_df_combined['Gain(%)'].apply(
        lambda x: f"\033[92m{float(x):,.2f}%\033[0m" if float(x) > 0 
                  else (f"\033[91m{float(x):,.2f}%\033[0m" if float(x) < 0 
                        else f"{float(x):,.2f}%")
    )
    
    print("\n=== 綜合投資組合股票明細 ===")
    print(tabulate(portfolio_df_combined, headers='keys', tablefmt='psql', showindex=False))

    # 繪製綜合資產市值走勢圖
    plt.figure(figsize=(10, 6))
    plt.plot(combined_portfolio_value.index, combined_portfolio_value.values, label='Combined Portfolio Value')
    plt.title('Combined Portfolio Value Over Time (USD)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # 圓餅圖：展示各股票現值比例 (USD)
    # ------------------------------
    combined_df = portfolio_df_combined.dropna(subset=['Price_Total'])
    combined_df = combined_df[combined_df['Price_Total'] > 0]
    plt.figure(figsize=(10,8))
    plt.pie(combined_df['Price_Total'], 
            labels=combined_df['Name'], 
            autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '', 
            startangle=140)
    plt.title('綜合投資組合中各股票的現值比例 (USD)')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()
