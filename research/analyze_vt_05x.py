import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_synthetic_leveraged(prices, leverage, annual_expense):
    daily_expense = annual_expense / 252
    returns = prices.pct_change().fillna(0)
    leveraged_returns = leverage * returns - daily_expense
    return (1 + leveraged_returns).cumprod() * 100

def main():
    print("正在下載資料 (VT)...")
    data = yf.download('VT', progress=False)
    
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['VT_Price'] = data['Close']['VT']
    else:
        df['VT_Price'] = data['Close']
        
    df = df.dropna()
    print("創建 0.5x, 1x, 2x...")
    
    # 0.5x (相當於 50% VT + 50% 現金)
    # 不計現金利息與摩擦成本
    df['VT_0_5x'] = create_synthetic_leveraged(df['VT_Price'], 0.5, 0.0)
    # 2x
    df['VT_2x'] = create_synthetic_leveraged(df['VT_Price'], 2, 0.015)
    
    initial_wealth = 10000.0
    
    df['Benchmark_1x'] = (df['VT_Price'] / df['VT_Price'].iloc[0]) * initial_wealth
    df['Simulated_0_5x'] = (df['VT_0_5x'] / df['VT_0_5x'].iloc[0]) * initial_wealth
    df['Simulated_2x'] = (df['VT_2x'] / df['VT_2x'].iloc[0]) * initial_wealth
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    ax1 = axes[0]
    ax1.plot(df.index, df['Benchmark_1x'], label='VT (1x)', linewidth=2, color='black')
    ax1.plot(df.index, df['Simulated_2x'], label='VT正2 (2x VT, 1.5% 成本)', linewidth=2, color='red', alpha=0.5)
    ax1.plot(df.index, df['Simulated_0_5x'], label='VT 0.5倍 (50% VT + 50% 現金)', linewidth=2, color='green')
    
    ax1.set_title('0.5倍、1倍 與 2倍 VT 歷史績效比較', fontsize=16)
    ax1.set_ylabel('Portfolio Value (USD, Log Scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    for line in ax1.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        if len(x_data) > 0:
            final_x = x_data[-1]
            final_y = y_data[-1]
            ax1.annotate(f'${final_y:,.0f}', xy=(final_x, final_y), xytext=(5, 0), 
                         textcoords='offset points', verticalalignment='center', fontsize=10, fontweight='bold')
    
    ax2 = axes[1]
    
    cols = [
        ('1. VT (1x)', 'Benchmark_1x'), 
        ('2. VT 0.5倍 (50%股 50%現)', 'Simulated_0_5x'),
        ('3. VT正2 (2x)', 'Simulated_2x')
    ]
    
    for label, col in cols:
        roll_max = df[col].cummax()
        drawdown = (df[col] - roll_max) / roll_max * 100
        color = 'black' if '1x' in label else ('green' if '0.5' in label else 'red')
        line_style = '-'
        alpha = 0.5 if '2x' in label else 0.8
        ax2.plot(df.index, drawdown, label=label, linewidth=1.5, alpha=alpha, color=color, linestyle=line_style)
    
    ax2.set_title('Drawdown Comparison (%)', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_file = 'vt_0_5x.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n圖表已儲存至 {output_file}")
    
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
