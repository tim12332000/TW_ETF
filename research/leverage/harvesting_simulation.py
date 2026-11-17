import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def simulate_volatility_harvesting(ticker, market_name):
    data = yf.download(ticker, progress=False, auto_adjust=True)
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['Price'] = data['Close'][ticker]
    else:
        df['Price'] = data['Close']
    df = df.dropna()
    
    returns = df['Price'].pct_change().fillna(0)
    
    # 策略 1: 100% 買入持有
    strat_100_bh = (1 + returns).cumprod()
    
    # 策略 2: 50% 股票 / 50% 現金 (買入後死抱不平衡)
    # 起始本金為 1，一半放現金不變(假設0利息)，一半跟著股票走
    strat_50_bh = 0.5 + 0.5 * strat_100_bh
    
    # 策略 3: 0.5倍槓桿 (每日再平衡 50/50) -> 這就是所謂的波動收割
    strat_50_rebal = (1 + 0.5 * returns).cumprod()
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 1
    
    def get_stats(series):
        cagr = (series.iloc[-1]) ** (1/years) - 1
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        max_dd = drawdown.min()
        return cagr * 100, max_dd * 100
        
    cagr_100, dd_100 = get_stats(strat_100_bh)
    cagr_50bh, dd_50bh = get_stats(strat_50_bh)
    cagr_50rebal, dd_50rebal = get_stats(strat_50_rebal)
    
    print(f"\n{'='*50}")
    print(f"{market_name} ({ticker}) 波動收割效應分析")
    print(f"==================================================")
    print(f"{'策略':<20} | {'CAGR (%)':>8} | {'Max DD (%)':>10}")
    print("-" * 50)
    print(f"{'1. 原型 (100% 買入持有)':<20} | {cagr_100:>8.2f} | {dd_100:>10.2f}")
    print(f"{'2. 50/50 不平衡 (裝死)':<20} | {cagr_50bh:>8.2f} | {dd_50bh:>10.2f}")
    print(f"{'3. 0.5x 每日再平衡':<20} | {cagr_50rebal:>8.2f} | {dd_50rebal:>10.2f}")
    print("-" * 50)
    print(f"波動收割超額紅利 = {cagr_50rebal - cagr_50bh:.2f}% (再平衡 vs 不平衡)")
    if cagr_100 > 0:
        print(f"50%配置的獲利比例 = {cagr_50rebal/cagr_100*100:.1f}% (只用一半資金，卻拿到原型多少比例的報酬)")

if __name__ == '__main__':
    simulate_volatility_harvesting('FXI', '中國')
    simulate_volatility_harvesting('EWJ', '日本')
