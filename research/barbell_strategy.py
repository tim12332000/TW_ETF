import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def simulate_barbell():
    print("正在下載資料 (QQQ 作為風險端代表)...")
    data = yf.download('QQQ', start='1999-01-01', progress=False, auto_adjust=True)
    
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['QQQ'] = data['Close']['QQQ']
    else:
        df['QQQ'] = data['Close']
    df = df.dropna()
    
    ret_qqq = df['QQQ'].pct_change().fillna(0)
    
    # 假設極度安全的資產端 (現金/定存/短債) 每年提供穩定的 3% 殖利率
    ret_cash = 0.03 / 252
    
    # 建立極度風險端：3 倍槓桿納指 (模擬 TQQQ，扣除 2% 借貸+摩擦成本)
    ret_3x = 3.0 * ret_qqq - (0.02 / 252)
    ret_3x = np.clip(ret_3x, -0.9999, None)
    
    # 策略 1: 100% QQQ 原型 (一般大眾的風險暴露)
    strat_100_qqq = (1 + ret_qqq).cumprod()
    
    # 策略 2: 中庸之道 (50% QQQ + 50% 現金，每日再平衡)
    strat_50_50 = (1 + (0.5 * ret_qqq + 0.5 * ret_cash)).cumprod()
    
    # 策略 3: 槓鈴策略 (Barbell) - 85% 極度安全(現金) + 15% 極度風險(3x 納指)
    # 這裡採用「買入持有、不去再平衡」的塔雷伯正宗精神。
    # 因為再平衡會不斷把安全的錢輸血給暴跌的風險端，而槓鈴精神是「買彩票，輸光就算了」
    safe_part = 0.85 * (1 + pd.Series(ret_cash, index=df.index)).cumprod()
    risk_part = 0.15 * (1 + ret_3x).cumprod()
    strat_barbell_bh = safe_part + risk_part
    
    # 策略 4: 槓鈴策略 (每日再平衡版) 作為對照
    strat_barbell_rebal = (1 + (0.85 * pd.Series(ret_cash, index=df.index) + 0.15 * ret_3x)).cumprod()
    
    # 繪圖
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    ax1 = axes[0]
    
    initial = 1_000_000
    ax1.plot(df.index, strat_100_qqq * initial, label='100% 納指原型 (承擔 100% 市場風險)', color='black', alpha=0.7)
    ax1.plot(df.index, strat_50_50 * initial, label='中庸之道 (50% 納指 + 50% 現金)', color='gray', linestyle='--')
    ax1.plot(df.index, strat_barbell_bh * initial, label='槓鈴策略 (85% 現金 + 15% 納指正3, 買入持有)', color='red', linewidth=2.5)
    ax1.plot(df.index, strat_barbell_rebal * initial, label='槓鈴策略 (85% 現金 + 15% 納指正3, 每日再平衡)', color='orange', linewidth=1.5, alpha=0.8)
    
    ax1.set_title('塔雷伯「槓鈴策略 (Barbell Strategy)」驗證 (1999-至今, 起始 100 萬)', fontsize=16)
    ax1.set_ylabel('總資產 (USD, Log Scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    for line in ax1.get_lines():
        x = line.get_xdata()
        y = line.get_ydata()
        if len(x) > 0:
            ax1.annotate(f'${y[-1]:,.0f}', xy=(x[-1], y[-1]), xytext=(5, 0), 
                         textcoords='offset points', va='center', fontsize=10, fontweight='bold')
    
    ax2 = axes[1]
    
    def plot_dd(series, label, color, lw=1.5, ls='-'):
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max * 100
        ax2.plot(df.index, drawdown, label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.8)
        return drawdown.min()
        
    dd1 = plot_dd(strat_100_qqq, '100% 納指原型', 'black')
    dd2 = plot_dd(strat_50_50, '中庸之道 (50/50)', 'gray', ls='--')
    dd3 = plot_dd(strat_barbell_bh, '槓鈴策略 (85/15 買入持有)', 'red', lw=2.5)
    dd4 = plot_dd(strat_barbell_rebal, '槓鈴策略 (85/15 每日再平衡)', 'orange')
    
    ax2.set_title('最大回撤 (Max Drawdown) - 2000年網路泡沫防護力', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_file = 'barbell_strategy.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n圖表已儲存至 {output_file}")
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 1
    
    print("\n" + "=" * 70)
    print("槓鈴策略 (Barbell) 歷史回測 (1999 至今，經歷 2000 泡沫化)")
    print("=" * 70)
    print(f"{'策略':<35} {'CAGR':>8} {'Max DD':>10} {'期末餘額':>12}")
    print("-" * 70)
    
    def print_stat(name, series, dd):
        cagr = (series.iloc[-1] ** (1/years) - 1) * 100
        final = series.iloc[-1] * initial
        print(f"{name:<35} {cagr:>8.2f}% {dd:>9.2f}% ${final:>11,.0f}")
        
    print_stat('1. 100% 納指原型 (全承擔風險)', strat_100_qqq, dd1)
    print_stat('2. 中庸之道 (50% QQQ/50% 現金)', strat_50_50, dd2)
    print_stat('3. 槓鈴 (85現/15%正3, 再平衡)', strat_barbell_rebal, dd4)
    print_stat('4. 槓鈴 (85現/15%正3, 死抱不放)', strat_barbell_bh, dd3)

if __name__ == '__main__':
    simulate_barbell()
