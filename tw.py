import pandas as pd
df = pd.read_csv('tw_train.csv')
df


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
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微軟正黑體
rcParams['axes.unicode_minus'] = False

# =============================================================================
# 功能函式
# =============================================================================

def clean_currency(x):
    """
    清理貨幣字串，移除 'NT$', '$' 與逗號，並轉換成 float。
    若轉換失敗則回傳 None。
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
    若交易別為 "賣"（或英文 sell）且 Quantity 為正，則將 Quantity 轉為負值
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

def get_daily_price(stock_symbol, start_date, end_date):
    """
    利用 yfinance 下載指定股票（或股票列表）每日收盤價資料。
    若為台股，請自動加上 .TW 後綴
    回傳資料中的 'Close' 欄位
    """
    try:
        if isinstance(stock_symbol, list):
            stock_symbol = [convert_ticker(s) for s in stock_symbol]
        else:
            stock_symbol = convert_ticker(stock_symbol)
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        return data['Close']
    except Exception as e:
        print(f"下載 {stock_symbol} 價格失敗: {e}")
        return 0

def get_current_price_yf(ticker):
    """
    利用 yfinance 查詢單一股票的最新價格，
    優先從 info 取得 regularMarketPrice，若無則從 history 中取得最新收盤價
    注意：台股 ticker 需轉換格式
    """
    try:
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
    :param rate: 折現率
    :param cashflows: [(date, cashflow), ...]
    """
    t0 = min(date for date, _ in cashflows)
    return sum(cf / ((1 + rate) ** ((date - t0).days / 365.0))
               for date, cf in cashflows)

def xirr(cashflows, guess=0.1):
    """
    利用牛頓法求解不規則現金流的內部報酬率 (XIRR)
    :param cashflows: [(date, cashflow), ...]
    :param guess: 初始猜測值
    """
    return newton(lambda r: xnpv(r, cashflows), guess)

def calculate_realized_gain(symbol, df):
    """
    計算已平倉標的的實現盈虧及實現盈虧百分比
    根據原始交易記錄計算：
      - total_buy：買入的總成本（轉為正值）
      - realized_gain：賣出總金額減去買入總成本
      - realized_gain_pct：報酬率
    """
    df_sym = df[df['Symbol'] == symbol]
    total_buy = -df_sym[df_sym['Action'].str.lower().isin(['買', 'buy'])]['Amount'].sum()
    total_sell = df_sym[df_sym['Action'].str.lower().isin(['賣', 'sell'])]['Amount'].sum()
    realized_gain = total_sell - total_buy
    realized_gain_pct = (realized_gain / total_buy * 100) if total_buy != 0 else 0
    return total_buy, realized_gain, realized_gain_pct

# =============================================================================
# 主程式：計算投資組合績效、輸出文字報告，再顯示圖表
# =============================================================================

def main():
    # ------------------------------
    # 1. 讀取並前置處理交易資料
    # ------------------------------
    # 假設 CSV 檔案中有欄位： '交易日', '交易別', '股票代號', '股票名稱', '股數', '單價', '進帳/出帳'
    df = pd.read_csv('tw_train.csv', encoding='utf-8-sig')
    df.rename(columns={
        '交易日': 'Date',
        '交易別': 'Action',
        '股票代號': 'Symbol',
        '股票名稱': 'Name',
        '股數': 'Quantity',
        '單價': 'Price',
        '進帳/出帳': 'Amount'
    }, inplace=True)
    
    # 轉換日期格式
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    # 將數值欄位轉為數值型態，確保運算正確（例如股數）
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    
    # 將賣出時的股數轉為負值
    df = df.apply(fix_share_sign, axis=1)
    
    # 清理金額字串，轉換 Amount 欄位
    df["Amount"] = df["Amount"].apply(clean_currency)

    # 建立日期範圍（從最早交易日到今日，僅包含商業日）
    start_date = df['Date'].min()
    end_date = pd.Timestamp.today()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # ------------------------------
    # 2. 建立每日累積持股數與組合市值
    # ------------------------------
    pivot = df.pivot_table(index='Date', columns='Symbol', values='Quantity', aggfunc='sum')
    pivot = pivot.reindex(date_range, fill_value=0).fillna(0)
    cum_holdings = pivot.cumsum()
    
    # 取得所有股票代號
    symbols = cum_holdings.columns.tolist()
    
    # 取得各標的每日調整後收盤價資料
    price_data = get_daily_price(symbols, start_date, end_date)
    # ★★ 關鍵：調整價格資料欄位，將「.TW」後綴移除，使之與 cum_holdings 的欄位名稱一致 ★★
    if price_data is not None and not price_data.empty:
        price_data.columns = [col.split('.')[0] for col in price_data.columns]
    
    # 補值：缺失值以前值填充，再以後值填充
    price_data = price_data.reindex(date_range).ffill().bfill()
    
    # 計算每日投資組合市值
    portfolio_value = (cum_holdings * price_data).sum(axis=1)
    # 若有 NaN 可選擇填 0
    portfolio_value = portfolio_value.fillna(0)
    
    # 畫出投資組合市值走勢圖
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_value.index, portfolio_value.values, label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (NT$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # ------------------------------
    # 正確版本：以資金流模擬方式計算實際淨投入資金
    # ------------------------------
    df_filtered = df[["Date", "Action", "Amount"]].copy()
    df_filtered = df_filtered[pd.to_numeric(df_filtered["Amount"], errors="coerce").notnull()]
    df_filtered["Amount"] = df_filtered["Amount"].astype(float)
    df_filtered = df_filtered.sort_values("Date")

    account_cash = 0
    net_invested = 0

    for _, row in df_filtered.iterrows():
        amt = row["Amount"]
        if amt > 0:
            account_cash += amt  # 賣出進帳
        else:
            needed = -amt
            if account_cash >= needed:
                account_cash -= needed
            else:
                net_invested += (needed - account_cash)
                account_cash = 0

    # ------------------------------
    # 3. 投資績效數值計算與文字輸出（假設所有賣出均再投資）
    # ------------------------------
    
    total_investment = -df[df['Amount'] < 0]['Amount'].sum()
    cash_balance = df['Amount'].sum()
    invested_capital = -cash_balance
    final_portfolio_value = portfolio_value.iloc[-1]
    total_profit = final_portfolio_value - invested_capital
    total_profit_pct = (total_profit / invested_capital) * 100
    
    print("\n=== 總結報告 ===")
    print(f"累積買入金額：{total_investment:,.2f} 元")
    print(f"淨現金流(真實計算)：{net_invested:,.2f} 元")
    print(f"淨現金流(cashflow)：{cash_balance:,.2f} 元")
    print(f"實際淨投入資金：{invested_capital:,.2f} 元")
    print(f"最終組合市值（現有持股）：{final_portfolio_value:,.2f} 元")
    print(f"總獲利：{total_profit:,.2f} 元")
    print(f"總獲利百分比：{total_profit_pct:.2f}%")
    
    # XIRR 計算：現金流包含所有交易與最新組合市值（以今日為日期）
    cashflows = list(df[['Date', 'Amount']].itertuples(index=False, name=None))
    net_holdings = df.groupby('Symbol')['Quantity'].sum()
    portfolio_snapshot = 0
    for stock, shares in net_holdings.items():
        if shares != 0:
            price = get_current_price_yf(stock)
            if price is not None:
                portfolio_snapshot += shares * price
    today = pd.Timestamp.today().normalize()
    cashflows.append((today, portfolio_snapshot))
    
    try:
        irr = xirr(cashflows)
    except Exception as e:
        irr = None
        print("XIRR 計算失敗:", e)
    
    daily_cash_flow = df.groupby('Date')['Amount'].sum().reindex(date_range, fill_value=0).cumsum()
    daily_invested_capital = -daily_cash_flow  # 正值表示累積投入資金
    performance_ratio = pd.Series(np.nan, index=date_range)
    mask = (daily_invested_capital > 0)
    performance_ratio[mask] = portfolio_value[mask] / daily_invested_capital[mask]
    profit_pct_series = (performance_ratio - 1) * 100

    wealth_factor = performance_ratio.dropna()
    if len(wealth_factor) > 0:
        daily_IRR = wealth_factor.iloc[-1] ** (1 / len(wealth_factor)) - 1
        annual_IRR = (1 + daily_IRR) ** 252 - 1
    else:
        daily_IRR = None
        annual_IRR = None

    daily_return = performance_ratio.pct_change()
    annual_rf = 0.02
    daily_rf = annual_rf / 252
    excess_return = daily_return - daily_rf
    sharpe_ratio = np.sqrt(252) * (excess_return.mean() / excess_return.std())
    
    pr_for_dd = profit_pct_series / 100 + 1
    running_max = pr_for_dd.cummax()
    drawdown = pr_for_dd / running_max - 1
    max_drawdown = drawdown.min() * 100

    if irr is not None:
        print(f"XIRR: {irr:.2%}")
    else:
        print("XIRR: 計算失敗")
    
    if daily_IRR is not None and annual_IRR is not None:
        print("最終累積績效: {:.2f}%".format((wealth_factor.iloc[-1] - 1) * 100))
        print(f"每日 IRR: {daily_IRR:.4%}")
        print(f"年化 IRR: {annual_IRR:.2%}")
    else:
        print("IRR: 無法計算")
    
    print(f"夏普值: {sharpe_ratio:.4f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    
    # ------------------------------
    # 4. 整合個股現值與投資組合股票明細
    # ------------------------------
    stock_counts = {}
    for idx, row in df.iterrows():
        stock_code = row['Symbol']
        stock_name = row['Name'] if pd.notna(row['Name']) else stock_code
        count = row['Quantity']
        cost = float(row['Amount'])
        if pd.isna(stock_code) or pd.isna(count):
            continue
        if stock_code not in stock_counts:
            stock_counts[stock_code] = {'stock_name': stock_name, 'Quantity_now': 0, 'cost': 0}
        stock_counts[stock_code]['Quantity_now'] += count
        stock_counts[stock_code]['cost'] += cost
    
    data_list = []
    for stock_code, data_dict in stock_counts.items():
        name = data_dict['stock_name']
        count = data_dict['Quantity_now']
        aggregated_cost = -data_dict['cost']
        if count != 0:
            try:
                ticker_obj = yf.Ticker(convert_ticker(stock_code))
                current_price = ticker_obj.history(period='1d')['Close'].iloc[-1]
            except Exception as e:
                print(f"Error fetching data for {stock_code}: {e}")
                current_price = 0
            current_value = current_price * count
            gain = current_value - aggregated_cost
            gain_per = (gain / aggregated_cost) * 100 if aggregated_cost != 0 else 0
        else:
            total_buy, realized_gain, realized_gain_pct = calculate_realized_gain(stock_code, df)
            current_price = np.nan
            current_value = np.nan
            aggregated_cost = total_buy
            gain = realized_gain
            gain_per = realized_gain_pct
        data_list.append([stock_code, name, count, current_price, current_value, aggregated_cost, gain, gain_per])
    
    headers = ['Symbol', 'Name', 'Quantity_now', 'Price', 'Price_Total', 'Cost', 'Gain', 'Gain(%)']
    portfolio_df = pd.DataFrame(data_list, columns=headers)
    filtered_df = portfolio_df[portfolio_df['Price_Total'] > 0]
    
    print("\n=== 投資組合股票明細 ===")
    print(tabulate(portfolio_df, headers='keys', tablefmt='psql', showindex=False))
    
    # ------------------------------
    # 5. 建立圖表
    # ------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(profit_pct_series.index, profit_pct_series.values, label='我的獲利%', color='purple')
    plt.xlabel('日期')
    plt.ylabel('報酬 (%)')
    plt.title('我的獲利%')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 0 else ''
    
    plt.figure(figsize=(10, 8))
    plt.pie(filtered_df['Price_Total'], 
            labels=filtered_df['Name'], 
            autopct=autopct_format, 
            startangle=140)
    plt.title('投資組合中各股票的現值總價比例')
    plt.axis('equal')
    
    plt.show()
    
    # ------------------------------
    # 6. Debug 部分：查詢特定期間市值
    # ------------------------------
    debug_start = pd.Timestamp('2023-02-10')
    debug_end = pd.Timestamp('2023-02-15')
    selected_values = portfolio_value.loc[debug_start:debug_end]
    print("2023/2/10 到 2023/2/15 的市值:")
    print(selected_values)
    
    query_date_range = pd.date_range(debug_start, debug_end, freq='D')
    daily_pf = portfolio_value.reindex(query_date_range, method='ffill')
    print("\n查詢期間每日市值:")
    print(daily_pf)

if __name__ == '__main__':
    main()
