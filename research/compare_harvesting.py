import yfinance as yf
import pandas as pd
import numpy as np

def get_harvest_stats(ticker, name, is_twii=False):
    data = yf.download(ticker, progress=False, auto_adjust=True)
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['Price'] = data['Close'][ticker]
    else:
        df['Price'] = data['Close']
    df = df.dropna()
    
    returns = df['Price'].pct_change().fillna(0)
    if is_twii:
        returns += 0.04 / 252
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 1
    
    # 50% Buy & Hold (Half cash, half ETF)
    stock_val = 0.5 * (1 + returns).cumprod()
    cash_val = 0.5
    total_bh = stock_val + cash_val
    cagr_bh = (total_bh.iloc[-1])**(1/years) - 1
    
    roll_max_bh = total_bh.cummax()
    dd_bh = (total_bh - roll_max_bh) / roll_max_bh
    max_dd_bh = dd_bh.min()
    
    # 50% Rebalanced (0.5x leverage)
    total_rebal = (1 + 0.5 * returns).cumprod()
    cagr_rebal = (total_rebal.iloc[-1])**(1/years) - 1
    
    roll_max_rebal = total_rebal.cummax()
    dd_rebal = (total_rebal - roll_max_rebal) / roll_max_rebal
    max_dd_rebal = dd_rebal.min()
    
    # 100% Stock
    total_100 = (1 + returns).cumprod()
    cagr_100 = (total_100.iloc[-1])**(1/years) - 1
    
    return {
        'name': name,
        'year': df.index[0].date().year,
        'cagr_100': cagr_100 * 100,
        'cagr_bh': cagr_bh * 100,
        'dd_bh': max_dd_bh * 100,
        'cagr_rebal': cagr_rebal * 100,
        'dd_rebal': max_dd_rebal * 100,
        'bonus': (cagr_rebal - cagr_bh) * 100
    }

print('Calculating...')
markets = [
    get_harvest_stats('^TWII', '台灣(^TWII)', True),
    get_harvest_stats('VT', '全球(VT)'),
    get_harvest_stats('QQQ', '納指(QQQ)'),
    get_harvest_stats('EWJ', '日本(EWJ)'),
    get_harvest_stats('FXI', '中國(FXI)')
]

print('\n| 市場 (統計起點) | 1. 100% 全押原型 | 2. 50/50 死抱不平衡 | 3. 0.5x 每日再平衡 | 波動收割紅利 |')
print('|:---|:---|:---|:---|:---|')
for m in markets:
    res100 = f"{m['cagr_100']:.2f}%"
    resbh = f"{m['cagr_bh']:.2f}% ({m['dd_bh']:.2f}%)"
    resreb = f"{m['cagr_rebal']:.2f}% ({m['dd_rebal']:.2f}%)"
    bonus = f"+{m['bonus']:.2f}%"
    
    print(f"| **{m['name']} ({m['year']})** | {res100:>10} | {resbh:>18} | {resreb:>18} | **{bonus:>10}** |")
