import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def get_optimal_leverage():
    print("正在下載資料 (^TWII 台灣加權指數)...")
    # 使用 ^TWII 作為台股大盤，歷史紀錄回溯至 1997
    data = yf.download('^TWII', start='1999-01-01', progress=False)
    
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['Price'] = data['Close']['^TWII']
    else:
        df['Price'] = data['Close']
        
    df = df.dropna()
    
    # 台灣加權指數是Price Return (除息會掉下來)，台股平均殖利率大約 4% = 0.04/252
    daily_dividend_yield = 0.04 / 252
    returns = df['Price'].pct_change().fillna(0) + daily_dividend_yield
    
    leverages = np.arange(0.1, 4.1, 0.1)
    cagrs = []
    max_dds = []
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 1
    
    for lev in leverages:
        # 使用 1.5% 作為基準與借貸摩擦成本 (00631L等正2的隱含成本也大約 1%~1.5%)
        annual_cost = max(0, lev - 1) * 0.015
        daily_cost = annual_cost / 252
        
        lev_returns = lev * returns - daily_cost
        lev_returns = np.clip(lev_returns, -0.9999, None)
        cum_returns = (1 + lev_returns).cumprod()
        
        if cum_returns.iloc[-1] <= 0 or np.isnan(cum_returns.iloc[-1]):
             cagr = -1.0
        else:
             cagr = (cum_returns.iloc[-1]) ** (1/years) - 1
             
        roll_max = cum_returns.cummax()
        drawdown = (cum_returns - roll_max) / roll_max
        max_dd = drawdown.min()
        
        cagrs.append(cagr * 100)
        max_dds.append(max_dd * 100)
        
    # 尋找最佳結果
    max_cagr_idx = np.argmax(cagrs)
    optimal_lev = leverages[max_cagr_idx]
    optimal_cagr = cagrs[max_cagr_idx]
    optimal_dd = max_dds[max_cagr_idx]
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = 'tab:blue'
    ax1.set_xlabel('槓桿倍數 (Leverage Ratio)', fontsize=12)
    ax1.set_ylabel('年化報酬率 CAGR (%)', color=color, fontsize=12)
    ax1.plot(leverages, cagrs, color=color, linewidth=3, label='CAGR (%)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 標註最高點
    ax1.plot(optimal_lev, optimal_cagr, 'ro', markersize=10)
    ax1.annotate(f'最優解: {optimal_lev:.1f}x\nCAGR: {optimal_cagr:.2f}%', 
                 xy=(optimal_lev, optimal_cagr), xytext=(15, -15),
                 textcoords='offset points', color='darkred', fontsize=12, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='darkred', lw=1.5))
                 
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('最大回撤 Max Drawdown (%)', color=color, fontsize=12)  
    ax2.plot(leverages, max_dds, color=color, linewidth=2, linestyle='--', alpha=0.6, label='Max DD (%)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 標註 1x
    idx_1x = np.where(np.isclose(leverages, 1.0))[0][0]
    ax1.plot(1.0, cagrs[idx_1x], 'bo', markersize=8)
    ax1.annotate(f'台股原型含息 (1.0x)\nCAGR: {cagrs[idx_1x]:.2f}%', 
                 xy=(1.0, cagrs[idx_1x]), xytext=(-30, 20),
                 textcoords='offset points', color='blue', fontsize=10)
                 
    try:
        idx_2x = np.where(np.isclose(leverages, 2.0))[0][0]
        ax1.plot(2.0, cagrs[idx_2x], 'mo', markersize=8)
        ax1.annotate(f'台股正2 (2.0x)\nCAGR: {cagrs[idx_2x]:.2f}%', 
                     xy=(2.0, cagrs[idx_2x]), xytext=(-30, 20),
                     textcoords='offset points', color='purple', fontsize=10)
    except:
        pass
    
    fig.tight_layout()  
    plt.title(f'台灣加權指數(含息模擬, 1999-至今) 尋找數學最優解', fontsize=16)
    
    # 合併圖例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=12)
    
    output_file = 'tw_0050_optimal_leverage_twii.png'
    plt.savefig(output_file, dpi=150)
    
    print("\n" + "=" * 50)
    print("台灣加權指數 (含 4% 殖利率模擬，代表 0050 總報酬) 1999 至今回測")
    print("=" * 50)
    for lev in [1.0, 1.5, 2.0, 3.0]:
        try:
            idx = np.where(np.isclose(leverages, lev))[0][0]
            print(f"{lev:.1f}x 槓桿: CAGR = {cagrs[idx]:.2f}%, Max DD = {max_dds[idx]:.2f}%")
        except:
            pass
            
    print(f"\n最佳槓桿倍數: {optimal_lev:.1f}x (年化報酬 {optimal_cagr:.2f}%, 最大回撤 {optimal_dd:.2f}%)")
    print(f"圖表已儲存至 {output_file}")

if __name__ == '__main__':
    get_optimal_leverage()
