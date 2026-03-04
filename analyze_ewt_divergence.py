
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def analyze_divergence():
    # 1. Download Data (5 Years)
    tickers = ['EWT', '^TWII', 'TWD=X']
    data = yf.download(tickers, period="5y", auto_adjust=True)['Close']
    
    # Fill NA
    data = data.ffill().bfill()
    
    # Normalize to 100 at start
    norm_data = data / data.iloc[0] * 100
    
    # Calculate EWT in TWD (approx) = EWT(USD) * TWD=X
    # This shows what EWT would be if stripped of currency effect
    data['EWT_in_TWD'] = data['EWT'] * data['TWD=X']
    norm_ewt_twd = data['EWT_in_TWD'] / data['EWT_in_TWD'].iloc[0] * 100

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Indices
    p1, = ax1.plot(norm_data.index, norm_data['^TWII'], label='Taiwan TAIEX (^TWII)', color='blue', linewidth=2)
    p2, = ax1.plot(norm_data.index, norm_data['EWT'], label='EWT (USD)', color='red', linewidth=2)
    p3, = ax1.plot(norm_ewt_twd.index, norm_ewt_twd, label='EWT (Adjusted to TWD)', color='green', linestyle='--', alpha=0.7)

    ax1.set_ylabel('Performance (Base=100)')
    ax1.grid(True, alpha=0.3)
    
    # Plot Currency on secondary axis
    ax2 = ax1.twinx()
    p4, = ax2.plot(norm_data.index, data['TWD=X'], label='USD/TWD Rate', color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('USD/TWD Exchange Rate')
    
    plt.title('Why EWT Diverges from TAIEX? (Effect of Currency)')
    lines = [p1, p2, p3, p4]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/ewt_vs_taiex.png')
    # plt.show()
    print("Chart saved to output/ewt_vs_taiex.png")

if __name__ == "__main__":
    analyze_divergence()
