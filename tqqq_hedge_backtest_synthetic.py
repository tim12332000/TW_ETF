"""
TQQQ + 年化避險成本壓力測試 (含模擬 2000/2008 泡沫)
使用 QQQ 日報酬 x3 模擬 TQQQ，可回溯至 1999 年
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import datetime

# 設定 Matplotlib 風格
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def black_scholes_put_price(S, K, T, r, sigma):
    """計算 Put Option 價格 (Black-Scholes 模型)"""
    if T <= 0:
        return max(K - S, 0.0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def find_strike_for_budget(S, T, r, sigma, budget, target_quantity):
    """尋找履約價 K，使得 (Put Price * Quantity) = Budget"""
    if target_quantity <= 0 or budget <= 0:
        return 0.0
    target_price = budget / target_quantity
    
    def objective(K):
        if K <= 0: return -target_price
        return black_scholes_put_price(S, K, T, r, sigma) - target_price
    
    try:
        k_sol = brentq(objective, S * 0.01, S * 2.0)
        return k_sol
    except:
        return 0.0

def create_synthetic_tqqq(qqq_prices):
    """
    使用 QQQ 日報酬模擬 3x 槓桿 ETF (Synthetic TQQQ)
    
    公式: Daily Return = 3 * QQQ_Daily_Return - Daily_Expense
    TQQQ Expense Ratio ≈ 0.95% 年化 → 約 0.0038% 日化
    """
    daily_expense = 0.0095 / 252  # 年化費用率轉日化
    
    qqq_returns = qqq_prices.pct_change().fillna(0)
    
    # 3x 槓桿日報酬 (扣除費用)
    leveraged_returns = 3 * qqq_returns - daily_expense
    
    # 累積報酬 (模擬價格，起始 = 100)
    synthetic_price = (1 + leveraged_returns).cumprod() * 100
    
    return synthetic_price

def run_backtest():
    print("正在下載資料 (QQQ, ^VXN, ^TNX)...")
    # 只下載 QQQ (從 1999 開始)
    tickers = ['QQQ', '^VXN', '^TNX']
    data = yf.download(tickers, start='1999-03-10', progress=False)
    
    # 處理 Data Access
    try:
        if isinstance(data.columns, pd.MultiIndex):
            df = pd.DataFrame()
            df['QQQ_Price'] = data['Close']['QQQ']
            df['VXN'] = data['Close']['^VXN']
            df['TNX'] = data['Close']['^TNX']
        else:
            print("警告: 下載資料格式非預期")
            return
            
    except KeyError as e:
        print(f"資料讀取錯誤: {e}")
        return

    # 資料前處理
    df = df.dropna()
    df['Rate'] = df['TNX'] / 100.0
    df['Sigma'] = df['VXN'] / 100.0
    
    # 創建 Synthetic TQQQ
    print("創建 Synthetic TQQQ (3x QQQ)...")
    df['Synthetic_TQQQ'] = create_synthetic_tqqq(df['QQQ_Price'])
    
    print(f"資料範圍: {df.index[0].date()} ~ {df.index[-1].date()}, 共 {len(df)} 筆")
    
    # 設定回測參數
    hedge_ratios = [0.02, 0.04, 0.08]
    results = {}
    initial_wealth = 10000.0
    
    # Benchmark: Synthetic TQQQ Buy and Hold
    df['Benchmark_TQQQ'] = (df['Synthetic_TQQQ'] / df['Synthetic_TQQQ'].iloc[0]) * initial_wealth

    print("開始回測迴圈...")
    
    for hedge_cost_pct in hedge_ratios:
        print(f"模擬避險成本: {hedge_cost_pct*100}%")
        
        portfolio_value = initial_wealth
        cash = portfolio_value
        tqqq_shares = 0
        
        option_holdings = {
            'quantity': 0,
            'strike': 0,
            'expiry_date': None
        }
        
        value_history = []
        next_rebalance_year = df.index[0].year
        
        for date, row in df.iterrows():
            current_tqqq_price = row['Synthetic_TQQQ']
            current_qqq_price = row['QQQ_Price'] 
            r = row['Rate']
            sigma = row['Sigma']
            
            # 計算當前選擇權價值
            option_val = 0.0
            if option_holdings['quantity'] > 0:
                days_to_expiry = (option_holdings['expiry_date'] - date).days
                T_remain = max(days_to_expiry / 365.25, 0.0)
                
                if T_remain <= 0:
                    intrinsic_val = max(option_holdings['strike'] - current_qqq_price, 0.0)
                    option_val = intrinsic_val * option_holdings['quantity']
                else:
                    option_val = black_scholes_put_price(
                        S=current_qqq_price,
                        K=option_holdings['strike'],
                        T=T_remain,
                        r=r,
                        sigma=sigma
                    ) * option_holdings['quantity']
            
            # 判斷是否需要再平衡 (每年年初)
            if date.year >= next_rebalance_year:
                current_total_equity = (tqqq_shares * current_tqqq_price) + option_val + (cash if date == df.index[0] else 0)
                
                cash = current_total_equity
                tqqq_shares = 0
                option_val = 0
                
                budget = cash * hedge_cost_pct
                investable_for_tqqq = cash - budget
                
                tqqq_shares = investable_for_tqqq / current_tqqq_price
                
                expiry_date = date + datetime.timedelta(days=365)
                T_new = 1.0
                
                quantity_needed = (3 * (tqqq_shares * current_tqqq_price)) / current_qqq_price
                
                strike = find_strike_for_budget(
                    S=current_qqq_price,
                    T=T_new,
                    r=r,
                    sigma=sigma,
                    budget=budget,
                    target_quantity=quantity_needed
                )
                
                option_holdings = {
                    'quantity': quantity_needed,
                    'strike': strike,
                    'expiry_date': expiry_date
                }
                cash = 0
                
                next_rebalance_year += 1
                option_val = budget
            
            current_total_val = (tqqq_shares * current_tqqq_price) + option_val + cash
            value_history.append(current_total_val)
            
        results[f'Hedge {hedge_cost_pct*100}%'] = pd.Series(value_history, index=df.index)

    # 合併結果並繪圖
    res_df = pd.DataFrame(results)
    res_df['Synthetic TQQQ (No Hedge)'] = df['Benchmark_TQQQ']
    
    # 繪製圖表
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 上圖：完整時間序列
    ax1 = axes[0]
    ax1.plot(res_df.index, res_df['Synthetic TQQQ (No Hedge)'], label='Synthetic TQQQ (No Hedge)', linewidth=2, color='black', alpha=0.7)
    ax1.plot(res_df.index, res_df['Hedge 2.0%'], label='TQQQ + 2% Hedge', linewidth=2)
    ax1.plot(res_df.index, res_df['Hedge 4.0%'], label='TQQQ + 4% Hedge', linewidth=2)
    ax1.plot(res_df.index, res_df['Hedge 8.0%'], label='TQQQ + 8% Hedge', linewidth=2)
    
    ax1.set_title('Synthetic TQQQ (3x QQQ) + Annualized QQQ Put Hedge\n(1999-Present, Includes 2000 & 2008 Crashes)', fontsize=14)
    ax1.set_ylabel('Portfolio Value (USD, Log Scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Add annotation for final value
    for line in ax1.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        if len(x_data) > 0:
            final_x = x_data[-1]
            final_y = y_data[-1]
            ax1.annotate(f'${final_y:,.0f}', xy=(final_x, final_y), xytext=(5, 0), 
                         textcoords='offset points', verticalalignment='center', fontsize=9, fontweight='bold')
    
    # 標註重大事件
    ax1.axvline(pd.Timestamp('2000-03-10'), color='red', linestyle='--', alpha=0.5, label='Dot-com Peak')
    ax1.axvline(pd.Timestamp('2002-10-09'), color='green', linestyle='--', alpha=0.5, label='Dot-com Bottom')
    ax1.axvline(pd.Timestamp('2008-09-15'), color='red', linestyle='--', alpha=0.5, label='Lehman Collapse')
    ax1.axvline(pd.Timestamp('2009-03-09'), color='green', linestyle='--', alpha=0.5, label='2009 Bottom')
    ax1.axvline(pd.Timestamp('2020-03-23'), color='orange', linestyle='--', alpha=0.5, label='COVID Bottom')
    
    # 下圖：Drawdown
    ax2 = axes[1]
    for col in res_df.columns:
        roll_max = res_df[col].cummax()
        drawdown = (res_df[col] - roll_max) / roll_max * 100
        ax2.plot(res_df.index, drawdown, label=col, linewidth=1.5, alpha=0.8)
    
    ax2.set_title('Drawdown Comparison (%)', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(y=-50, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(y=-80, color='darkred', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    output_file = 'tqqq_hedge_backtest_synthetic_result.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n回測完成，圖表已儲存至 {output_file}")
    
    # 計算統計數據
    print("\n" + "=" * 70)
    print("回測統計 (1999 - Present)")
    print("=" * 70)
    print(f"{'Strategy':<30} {'CAGR':>10} {'Max DD':>12} {'Final Value':>15}")
    print("-" * 70)
    
    for col in res_df.columns:
        final_val = res_df[col].iloc[-1]
        start_val = res_df[col].iloc[0]
        years = (res_df.index[-1] - res_df.index[0]).days / 365.25
        cagr = (final_val / start_val) ** (1/years) - 1
        
        roll_max = res_df[col].cummax()
        drawdown = (res_df[col] - roll_max) / roll_max
        max_dd = drawdown.min()
        
        print(f"{col:<30} {cagr*100:>9.2f}% {max_dd*100:>11.2f}% ${final_val:>14,.0f}")
    
    # 分段分析
    print("\n" + "=" * 70)
    print("分段分析: 2000 泡沫 (2000-03 ~ 2002-10)")
    print("=" * 70)
    
    bubble2000_start = pd.Timestamp('2000-03-10')
    bubble2000_end = pd.Timestamp('2002-10-09')
    
    for col in res_df.columns:
        try:
            start_idx = res_df.index.get_indexer([bubble2000_start], method='nearest')[0]
            end_idx = res_df.index.get_indexer([bubble2000_end], method='nearest')[0]
            
            period_data = res_df[col].iloc[start_idx:end_idx+1]
            start_val = period_data.iloc[0]
            end_val = period_data.iloc[-1]
            total_return = (end_val / start_val - 1) * 100
            
            roll_max = period_data.cummax()
            max_dd = ((period_data - roll_max) / roll_max).min() * 100
            
            print(f"{col:<30}: Return = {total_return:>8.1f}%, Max DD = {max_dd:>8.1f}%")
        except:
            pass
    
    print("\n" + "=" * 70)
    print("分段分析: 2008 金融海嘯 (2007-10 ~ 2009-03)")
    print("=" * 70)
    
    crisis2008_start = pd.Timestamp('2007-10-09')
    crisis2008_end = pd.Timestamp('2009-03-09')
    
    for col in res_df.columns:
        try:
            start_idx = res_df.index.get_indexer([crisis2008_start], method='nearest')[0]
            end_idx = res_df.index.get_indexer([crisis2008_end], method='nearest')[0]
            
            period_data = res_df[col].iloc[start_idx:end_idx+1]
            start_val = period_data.iloc[0]
            end_val = period_data.iloc[-1]
            total_return = (end_val / start_val - 1) * 100
            
            roll_max = period_data.cummax()
            max_dd = ((period_data - roll_max) / roll_max).min() * 100
            
            print(f"{col:<30}: Return = {total_return:>8.1f}%, Max DD = {max_dd:>8.1f}%")
        except:
            pass

if __name__ == "__main__":
    run_backtest()
