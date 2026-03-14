import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def analyze_turkey():
    print("正在下載 土耳其 ETF (TUR, 美元計價) 與 伊斯坦堡100指數 (XU100.IS, 里拉計價)...")
    data = yf.download(['TUR', 'XU100.IS'], start='2008-05-01', progress=False)
    
    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['TUR_USD'] = data['Close']['TUR']
        df['BIST100_TRY'] = data['Close']['XU100.IS']
    else:
        df['TUR_USD'] = data['Close']['TUR']
        df['BIST100_TRY'] = data['Close']['XU100.IS']
        
    df = df.dropna()
    
    # 標準化到 100
    df['TUR_Normalized'] = df['TUR_USD'] / df['TUR_USD'].iloc[0] * 100
    df['BIST100_Normalized'] = df['BIST100_TRY'] / df['BIST100_TRY'].iloc[0] * 100
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(df.index, df['BIST100_Normalized'], label='土耳其大盤 (里拉計價 XU100) - 狂噴!', color='red', linewidth=2)
    ax1.plot(df.index, df['TUR_Normalized'], label='土耳其 ETF (美元計價 TUR) - 慘跌!', color='blue', linewidth=2)
    
    ax1.set_title('通膨幻覺：土耳其股市「本國貨幣」vs「強勢美元」的真實差距', fontsize=16)
    ax1.set_ylabel('累積報酬 (基期 100, Log Scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    for line in ax1.get_lines():
        x = line.get_xdata()
        y = line.get_ydata()
        if len(x) > 0:
            ax1.annotate(f'{y[-1]:,.0f}', xy=(x[-1], y[-1]), xytext=(5, 0), 
                         textcoords='offset points', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = 'turkey_currency_illusion.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n圖表已儲存至 {output_file}")
    
    # Print stats
    ret_try = df['BIST100_Normalized'].iloc[-1] / 100 - 1
    ret_usd = df['TUR_Normalized'].iloc[-1] / 100 - 1
    
    print("\n" + "=" * 50)
    print("土耳其股市的「匯率幻覺」 (2008年至今)")
    print("=" * 50)
    print(f"1. 里拉計價的大盤 (當地人視角): 報酬率 {ret_try * 100:,.2f}% (漲了 {ret_try+1:,.1f} 倍！)")
    print(f"2. 美元計價的 ETF (外國人視角): 報酬率 {ret_usd * 100:,.2f}% (直接倒賠...)")
    print(f"\n=> 股市雖然噴上天，但匯率貶值速度比股市漲幅還快，吃光了所有的獲利！")

if __name__ == '__main__':
    analyze_turkey()
