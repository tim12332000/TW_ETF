import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 設定 Matplotlib 風格
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_synthetic_vt2x(vt_prices):
    """
    使用 VT 日報酬模擬 2x 槓桿 ETF (Synthetic VT 2x)
    公式: Daily Return = 2 * VT_Daily_Return - Daily_Expense
    簡單假設年化成本約 1.5% (內扣+借貸)
    """
    daily_expense = 0.015 / 252  # 年化費用率轉日化
    
    vt_returns = vt_prices.pct_change().fillna(0)
    
    # 2x 槓桿日報酬 (扣除費用)
    leveraged_returns = 2 * vt_returns - daily_expense
    
    # 累積報酬 (模擬價格，起始 = 100)
    synthetic_price = (1 + leveraged_returns).cumprod() * 100
    
    return synthetic_price

def main():
    print("正在下載資料 (VT)...")
    data = yf.download('VT', progress=False)
    
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['VT_Price'] = data['Close']['VT']
    else:
        df['VT_Price'] = data['Close']
        
    df = df.dropna()
    print("創建 Synthetic VT正2 (2x VT)...")
    df['Synthetic_VT2x'] = create_synthetic_vt2x(df['VT_Price'])
    
    initial_wealth = 10000.0
    
    # Benchmark
    df['Benchmark_VT'] = (df['VT_Price'] / df['VT_Price'].iloc[0]) * initial_wealth
    df['Synthetic_VT2x_Value'] = (df['Synthetic_VT2x'] / df['Synthetic_VT2x'].iloc[0]) * initial_wealth
    
    # 繪製圖表
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    ax1 = axes[0]
    ax1.plot(df.index, df['Benchmark_VT'], label='VT (1x)', linewidth=2, color='black')
    ax1.plot(df.index, df['Synthetic_VT2x_Value'], label='VT正2 (2x VT Simulated)', linewidth=2, color='red')
    
    ax1.set_title('VT vs VT正2 (2x VT) 歷史績效比較', fontsize=16)
    ax1.set_ylabel('Portfolio Value (USD, Log Scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add annotation for final value
    for line in ax1.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        if len(x_data) > 0:
            final_x = x_data[-1]
            final_y = y_data[-1]
            ax1.annotate(f'${final_y:,.0f}', xy=(final_x, final_y), xytext=(5, 0), 
                         textcoords='offset points', verticalalignment='center', fontsize=10, fontweight='bold')
    
    # 下圖：Drawdown
    ax2 = axes[1]
    
    cols = [('VT (1x)', 'Benchmark_VT'), ('VT正2 (2x VT Simulated)', 'Synthetic_VT2x_Value')]
    
    for label, col in cols:
        roll_max = df[col].cummax()
        drawdown = (df[col] - roll_max) / roll_max * 100
        ax2.plot(df.index, drawdown, label=label, linewidth=1.5, alpha=0.8)
    
    ax2.set_title('Drawdown Comparison (%)', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(y=-20, color='orange', linestyle=':', alpha=0.5)
    ax2.axhline(y=-50, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    output_file = 'vt_vs_vt2x.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n圖表已儲存至 {output_file}")
    
    # 統計
    print("\n" + "=" * 70)
    print("回測統計 (VT 成立以來)")
    print("=" * 70)
    print(f"{'Strategy':<30} {'CAGR':>10} {'Max DD':>12} {'Final Value':>15}")
    print("-" * 70)
    
    for label, col in cols:
        final_val = df[col].iloc[-1]
        start_val = df[col].iloc[0]
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25 if days > 0 else 1
        cagr = (final_val / start_val) ** (1/years) - 1
        
        roll_max = df[col].cummax()
        drawdown = (df[col] - roll_max) / roll_max
        max_dd = drawdown.min()
        
        print(f"{label:<30} {cagr*100:>9.2f}% {max_dd*100:>11.2f}% ${final_val:>14,.0f}")

if __name__ == '__main__':
    main()
