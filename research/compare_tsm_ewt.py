
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def analyze_tsm():
    # 1. Download Data (5 Years)
    # TSM: US ADR
    # EWT: Taiwan ETF (USD)
    # ^TWII: Taiwan Index (TWD)
    # 2330.TW: Taiwan TSMC (TWD)
    tickers = ['TSM', 'EWT', '^TWII', '2330.TW']
    data = yf.download(tickers, period="5y", auto_adjust=True)['Close']
    
    # Fill NA
    data = data.ffill().bfill()
    
    # Normalize to 100 at start
    norm_data = data / data.iloc[0] * 100
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Indices
    p1, = ax1.plot(norm_data.index, norm_data['^TWII'], label='Taiwan TAIEX (^TWII - TWD)', color='blue', linewidth=2, alpha=0.6)
    p2, = ax1.plot(norm_data.index, norm_data['EWT'], label='EWT (USD)', color='gray', linestyle='--', linewidth=1.5)
    p3, = ax1.plot(norm_data.index, norm_data['TSM'], label='TSM (US ADR - USD)', color='red', linewidth=2.5)
    p4, = ax1.plot(norm_data.index, norm_data['2330.TW'], label='2330.TW (TWD)', color='orange', linestyle=':', linewidth=1.5)

    ax1.set_ylabel('Performance (Base=100)')
    ax1.grid(True, alpha=0.3)
    
    plt.title('Is TSM a Good Proxy for Taiwan? (TSM vs EWT vs TAIEX)')
    lines = [p1, p2, p3, p4]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/tsm_vs_taiwan.png')
    # plt.show()
    print("Chart saved to output/tsm_vs_taiwan.png")
    
    # Correlation Matrix
    print("\nCorrelation Matrix (Daily Returns):")
    returns = data.pct_change().dropna()
    corr = returns.corr()
    print(corr)

if __name__ == "__main__":
    analyze_tsm()
