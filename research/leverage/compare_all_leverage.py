import yfinance as yf
import pandas as pd
import numpy as np

def get_stats(ticker, name, is_twii=False, leverages=[0.1, 0.5, 1.0, 1.2, 1.5, 1.6, 2.0, 2.5, 3.0]):
    data = yf.download(ticker, progress=False, auto_adjust=True)
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['Price'] = data['Close'][ticker]
    else:
        df['Price'] = data['Close']
    df = df.dropna()
    
    returns = df['Price'].pct_change().fillna(0)
    if is_twii:
        returns += 0.04 / 252 # 台股大盤加上平均4%殖利率模擬總報酬
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 1
    
    results = {}
    for lev in leverages:
        annual_cost = max(0, lev - 1) * 0.015
        daily_cost = annual_cost / 252
        
        lev_returns = lev * returns - daily_cost
        lev_returns = np.clip(lev_returns, -0.9999, None)
        cum_returns = (1 + lev_returns).cumprod()
        
        if cum_returns.iloc[-1] <= 0 or np.isnan(cum_returns.iloc[-1]):
            cagr = -100.0
        else:
            cagr = ((cum_returns.iloc[-1]) ** (1/years) - 1) * 100
            
        roll_max = cum_returns.cummax()
        drawdown = (cum_returns - roll_max) / roll_max
        max_dd = drawdown.min() * 100
        
        results[lev] = {'CAGR': cagr, 'MaxDD': max_dd}
        
    return results, df.index[0].date().year

print("正在為您計算各大市場所有槓桿倍數的回測數據 (包含波動耗損及1.5%借貸成本)...\n")
try:
    twii_res, twii_year = get_stats('^TWII', '台灣(^TWII)', is_twii=True)
    vt_res, vt_year = get_stats('VT', '全球(VT)')
    qqq_res, qqq_year = get_stats('QQQ', '納斯達克(QQQ)')
    fxi_res, fxi_year = get_stats('FXI', '中國(FXI)')
except Exception as e:
    print(f"下載失敗: {e}")

leverages = [0.1, 0.5, 1.0, 1.2, 1.5, 1.6, 2.0, 2.5, 3.0]

print(f"{'槓桿':^6}│ {'台灣 (TWII, 自'+str(twii_year)+')':^17}│ {'全球 (VT, 自'+str(vt_year)+')':^17}│ {'納指 (QQQ, 自'+str(qqq_year)+')':^17}│ {'中國 (FXI, 自'+str(fxi_year)+')':^17}")
print("─" * 92)

for lev in leverages:
    def format_res(res):
        c = res[lev]['CAGR']
        d = res[lev]['MaxDD']
        # 標出特高或特低
        return f"{c:>6.2f}% / {d:>7.2f}%"

    row = (f" {lev:<4.1f}x │ "
           f"{format_res(twii_res)} │ "
           f"{format_res(vt_res)} │ "
           f"{format_res(qqq_res)} │ "
           f"{format_res(fxi_res)}")
    print(row)
print("─" * 92)
print("* 說明: 左側為 CAGR(年化報酬)、右側為 MaxDD(最大回撤)")
print("* 台灣數據採用 ^TWII 加權指數並補回每年平均 4% 殖利率以模擬總報酬")
