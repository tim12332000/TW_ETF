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
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        return data['Close']
    except Exception as e:
        print(f"下載 {stock_symbol} 價格失敗: {e}")
        return 0

def get_current_price_yf(ticker, is_tw=True):
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
        ticker = yf.Ticker('TWD=X')
        rate = ticker.info.get("regularMarketPrice")
        if rate is None:
            hist = ticker.history(period="1d")
            if not hist.empty:
                rate = hist["Close"].iloc[-1]
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
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        return np.nan, np.nan
    prices = data['Close']
    daily_returns = prices.pct_change().dropna()
    ann_return = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(daily_returns)) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    return ann_vol, ann_return

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
    cashflows_tw.append((today, portfolio_snapshot_tw))
    
    total_investment_tw = -df_tw[df_tw['Amount'] < 0]['Amount'].sum()
    cash_balance_tw = df_tw['Amount'].sum()
    invested_capital_tw = -cash_balance_tw
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
        'portfolio_df': portfolio_df_tw
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
        'portfolio_df': portfolio_df_us
    }

# =============================================================================
# 主程式：整合 TW 與 US 資料，產出 USD 與 TWD 版本報告、圖表及風險報酬散點圖
# =============================================================================
def main():
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
        combined_irr = xirr(combined_cashflows)
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
        combined_irr_twd = xirr(combined_cashflows)
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
    portfolio_df_combined = portfolio_df_combined[['Symbol', 'Name', 'Quantity_now', 'Price', 'Price_Total', 'Cost', 'Gain(USD)', 'Gain(TWD)', 'Gain(%)']]
    
    print("\n=== 綜合投資組合股票明細 (TWD) ===")
    print(tabulate(portfolio_df_combined, headers='keys', tablefmt='psql', showindex=False))

    # ----------------------
    # 資產走勢圖 (以 TWD 為基準)
    # ----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(combined_portfolio_value_twd.index, combined_portfolio_value_twd.values, label='資產走勢圖')
    plt.title('資產走勢圖')
    plt.xlabel('日期')
    plt.ylabel('組合市值 (TWD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ----------------------
    # 獲利走勢圖 (Profit %)
    # ----------------------
    cf_df = pd.DataFrame(combined_cashflows, columns=['Date', 'CashFlow'])
    cf_df['Date'] = pd.to_datetime(cf_df['Date'])
    cf_df = cf_df.sort_values('Date')
    daily_cf = cf_df.groupby('Date')['CashFlow'].sum().reindex(date_index, fill_value=0).cumsum()
    daily_invested_capital = -daily_cf
    profit_pct_series = pd.Series(np.nan, index=date_index)
    mask = daily_invested_capital > 0
    profit_pct_series[mask] = (combined_portfolio_value_us[mask] / daily_invested_capital[mask] - 1) * 100
    plt.figure(figsize=(12,6))
    plt.plot(profit_pct_series.index, profit_pct_series.values, label='獲利走勢圖', color='purple')
    plt.xlabel('日期')
    plt.ylabel('獲利 (%)')
    plt.title('獲利走勢圖')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    plt.show()

    # ----------------------
    # 新增：每月投入資產圖表 (扣除同日賣出金額)
    # ----------------------
    # 先依日期計算淨現金流（買進為負、賣出為正）
    transactions_df = pd.concat([tw_result['df'], us_result['df']])
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
    plt.show()

    # ----------------------
    # 風險報酬散點圖 (以最近 1 年資料計算)
    # ----------------------
    risks = []
    returns = []
    labels = []
    for idx, row in portfolio_df_combined.iterrows():
        stock_code = row['Symbol']
        is_tw = stock_code.isdigit()
        ann_vol, ann_return = compute_risk_return(stock_code, is_tw=is_tw, period='1y')
        if not np.isnan(ann_vol) and not np.isnan(ann_return):
            risks.append(ann_vol)
            returns.append(ann_return)
            labels.append(stock_code)
    plt.figure(figsize=(10,6))
    plt.scatter(risks, returns, color='blue')
    for i, txt in enumerate(labels):
        plt.annotate(txt, (risks[i], returns[i]), textcoords="offset points", xytext=(5,5), fontsize=9)
    plt.xlabel("年化波動率 (風險)")
    plt.ylabel("年化報酬率 (回報)")
    plt.title("風險報酬散點圖")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
