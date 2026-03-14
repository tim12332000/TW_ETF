import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def simulate_insurance():
    print("正在下載資料 (QQQ 與 VUSTX長天期公債)...")
    # VUSTX 為 Vanguard 長期公債基金，歷史悠久，可完美代理 TLT (20+年以上美債)
    data = yf.download(['QQQ', 'VUSTX'], start='1999-01-01', progress=False, auto_adjust=True)
    
    df = pd.DataFrame()
    df['QQQ'] = data['Close']['QQQ']
    df['Bonds'] = data['Close']['VUSTX']
    df = df.dropna()
    
    ret_qqq = df['QQQ'].pct_change().fillna(0)
    ret_bonds = df['Bonds'].pct_change().fillna(0)
    
    # 計算 2 倍槓桿 (含 1.5% 借貸成本)
    daily_cost_2x = 0.015 / 252
    
    ret_qqq_2x = 2.0 * ret_qqq - daily_cost_2x
    ret_qqq_2x = np.clip(ret_qqq_2x, -0.9999, None)
    
    ret_bonds_2x = 2.0 * ret_bonds - daily_cost_2x
    ret_bonds_2x = np.clip(ret_bonds_2x, -0.9999, None)
    
    # 策略 1: 裸奔的納指正2 (100% 2x QQQ)
    strat_1_naked_2x = (1 + ret_qqq_2x).cumprod()
    
    # 策略 2: 選擇權保險概念 (每年固定扣 4% 當作買 Put，但躲過 2000 股災的最深跌幅)
    # 這裡我們用您之前的 tqqq_hedge 經驗，簡單以每年固定扣血 4% + 將最大跌幅鎖在 -50% (理想狀態) 來模擬
    # 但實際上 Put 非常昂貴且不完美，所以我們來跑更務實的「資產配置保險」
    
    # 策略 3: 公債避險法 (HFEA 風險平價概念) - 60% 納指正2 + 40% 長債正2，每日再平衡
    ret_6040 = 0.6 * ret_qqq_2x + 0.4 * ret_bonds_2x
    strat_3_hfea = (1 + ret_6040).cumprod()
    
    # 策略 4: 均線停損法 (Trend Following) - 跌破 200 日線賣出換現金
    ma200 = df['QQQ'].rolling(200).mean()
    # 訊號: 昨天收盤若 > MA200，今天持有 2x QQQ；否則持有現金 (報酬為 0)
    signal = (df['QQQ'].shift(1) > ma200.shift(1)).astype(int)
    # 前 200 天沒均線，預設持有
    signal.iloc[:200] = 1 
    
    ret_trend = signal * ret_qqq_2x
    strat_4_trend = (1 + ret_trend).cumprod()
    
    # 畫圖
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    ax1 = axes[0]
    
    wealth_initial = 1_000_000
    ax1.plot(df.index, strat_1_naked_2x * wealth_initial, label='裸奔的 2 倍納指 (100% QQQ正2)', color='red', alpha=0.5, linewidth=1.5)
    ax1.plot(df.index, strat_3_hfea * wealth_initial, label='配公債保險 (60% QQQ正2 + 40% 長債正2)', color='blue', linewidth=2)
    ax1.plot(df.index, strat_4_trend * wealth_initial, label='均線停損保險 (跌破 200 日線全閃人)', color='green', linewidth=2)
    
    ax1.set_title('大股災防護罩：「買保險的槓桿」 vs 「裸奔的槓桿」 (1999-至今, 起始 100 萬)', fontsize=16)
    ax1.set_ylabel('總資產 (USD, Log Scale)', fontsize=12)
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
    
    def plot_dd(series, label, color, alpha=0.8):
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max * 100
        ax2.plot(df.index, drawdown, label=label, color=color, alpha=alpha, linewidth=1.5)
        return drawdown.min()
        
    dd1 = plot_dd(strat_1_naked_2x, '裸奔的 2 倍納指', 'red', 0.4)
    dd3 = plot_dd(strat_3_hfea, '配公債保險 (60/40)', 'blue')
    dd4 = plot_dd(strat_4_trend, '均線停損保險', 'green')
    
    ax2.set_title('最大回撤 (Max Drawdown) 比較 - 2000年網路泡沫是主戰場', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_file = 'leverage_insurance_qqq.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n圖表已儲存至 {output_file}")
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 1
    
    print("\n" + "=" * 70)
    print("納指槓桿買保險策略 (1999 至今，包含 2000 大魔王)")
    print("=" * 70)
    print(f"{'策略':<30}  {'CAGR':>8} {'Max DD':>10}")
    print("-" * 70)
    
    def print_stat(name, series, dd):
        cagr = (series.iloc[-1] ** (1/years) - 1) * 100
        print(f"{name:<30} {cagr:>8.2f}% {dd:>9.2f}%")
        
    print_stat('1. 裸奔的 2 倍納指', strat_1_naked_2x, dd1)
    print_stat('2. 配公債保險 (60/40 股債平價)', strat_3_hfea, dd3)
    print_stat('3. 均線停損保險 (跌破200MA)', strat_4_trend, dd4)

if __name__ == '__main__':
    simulate_insurance()
