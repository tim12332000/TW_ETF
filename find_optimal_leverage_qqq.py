import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def get_optimal_leverage():
    print("正在下載資料 (QQQ)...")
    #使用 QQQ 作為那斯達克 100 指數的代理，資料從 1999 年開始
    data = yf.download('QQQ', start='1999-03-10', progress=False)
    
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['Price'] = data['Close']['QQQ']
    else:
        df['Price'] = data['Close']
        
    df = df.dropna()
    returns = df['Price'].pct_change().fillna(0)
    
    leverages = np.arange(0.1, 4.1, 0.1)
    cagrs = []
    max_dds = []
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 1
    
    for lev in leverages:
        # 假設借貸成本為 (槓桿率 - 1) * 1.5% (若 lev > 1) 加上一些摩擦成本
        # 這裡為了保守估計，我們設 2% 以應付科技股較高的借貸/互換成本也可以，但維持 1.5% 作為比較基準
        annual_cost = max(0, lev - 1) * 0.015
        daily_cost = annual_cost / 252
        
        lev_returns = lev * returns - daily_cost
        
        # 避免破產導致的錯誤 (如果單日跌幅超過 100/lev)
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
    
    # 標註特定槓桿的 DD 點
    ax2.plot(optimal_lev, optimal_dd, 'ro', markersize=6)
    ax2.annotate(f'Max DD:\n{optimal_dd:.1f}%', 
                 xy=(optimal_lev, optimal_dd), xytext=(15, 5),
                 textcoords='offset points', color='darkred', fontsize=10)
    
    # 標註 1x
    idx_1x = np.where(np.isclose(leverages, 1.0))[0][0]
    ax1.plot(1.0, cagrs[idx_1x], 'bo', markersize=8)
    ax1.annotate(f'QQQ 原型 (1.0x)\nCAGR: {cagrs[idx_1x]:.2f}%', 
                 xy=(1.0, cagrs[idx_1x]), xytext=(-30, 20),
                 textcoords='offset points', color='blue', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='blue', lw=1))
    
    fig.tight_layout()  
    plt.title('那斯達克 QQQ (1999-至今) 槓桿倍數之「CAGR 報酬」尋找數學最優解', fontsize=16)
    
    # 合併圖例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=12)
    
    output_file = 'qqq_optimal_leverage.png'
    plt.savefig(output_file, dpi=150)
    
    print("\n" + "=" * 50)
    print("QQQ (那斯達克 100) 1999 至今槓桿回測")
    print("=" * 50)
    for lev in [1.0, 2.0, 3.0]:
        idx = np.where(np.isclose(leverages, lev))[0][0]
        print(f"{lev:.1f}x 槓桿: CAGR = {cagrs[idx]:.2f}%, Max DD = {max_dds[idx]:.2f}%")
        
    print(f"\n最佳槓桿倍數: {optimal_lev:.1f}x (年化報酬 {optimal_cagr:.2f}%, 最大回撤 {optimal_dd:.2f}%)")
    print(f"圖表已儲存至 {output_file}")

if __name__ == '__main__':
    get_optimal_leverage()
