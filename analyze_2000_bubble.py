import yfinance as yf
import pandas as pd
import numpy as np

def analyze_crash(ticker, name, start_date, peak_date, end_date):
    print(f"正在分析 {name} ({ticker}) 在 2000 年網路泡沫的表現...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['Price'] = data['Close'][ticker]
    else:
        df['Price'] = data['Close']
    df = df.dropna()
    
    returns = df['Price'].pct_change().fillna(0)
    
    # 找到最高點和最低點附近的區間
    # 假設我們在 peak_date 買在最高點，抱到 2002 年 10 月 (底部)
    try:
        peak_idx = df.index.get_loc(pd.to_datetime(peak_date), method='nearest')
        trough_idx = df.index.get_loc(pd.to_datetime('2002-10-09'), method='nearest')
    except:
        return
        
    df_crash = df.iloc[peak_idx:trough_idx+1].copy()
    crash_returns = df_crash['Price'].pct_change().fillna(0)
    
    results = {}
    for lev in [1.0, 1.5, 2.0, 3.0]:
        annual_cost = max(0, lev - 1) * 0.015
        daily_cost = annual_cost / 252
        
        lev_returns = lev * crash_returns - daily_cost
        lev_returns = np.clip(lev_returns, -0.9999, None)
        
        # 假設在最高點投入 100 萬
        wealth = 1000000 * (1 + lev_returns).cumprod()
        
        lowest_point = wealth.min()
        max_loss_pct = (1000000 - lowest_point) / 1000000 * 100
        
        results[lev] = {
            'lowest_val': lowest_point,
            'max_loss': max_loss_pct
        }
        
    print(f"\n| {name} 買在 2000 年最高點 (投入 100 萬) | 跌到 2002 年底剩下多少錢？ | 財產縮水率 |")
    print(f"|:---|:---|:---|")
    for lev in [1.0, 1.5, 2.0, 3.0]:
        val = results[lev]['lowest_val']
        loss = results[lev]['max_loss']
        print(f"| **{lev:.1f}x 槓桿** | 剩下 **{val:,.0f}** 元 | **-{loss:.2f}%** |")

if __name__ == '__main__':
    # QQQ 高點大約在 2000-03-24
    analyze_crash('QQQ', '納斯達克', '1999-01-01', '2000-03-24', '2005-01-01')
    # TWII 高點大約在 2000-02-18
    analyze_crash('^TWII', '台灣加權', '1999-01-01', '2000-02-18', '2005-01-01')
    # SPY (標普500) 作為對比，高點大約在 2000-03-24
    analyze_crash('SPY', '標普500', '1999-01-01', '2000-03-24', '2005-01-01')
