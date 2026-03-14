
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def black_scholes_put_value(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def simulate_over_hedge():
    # --- Assumptions ---
    US_Assets = 33000
    TW_Assets = 34000
    
    QQQ_S0 = 610.0
    QQQ_K = 350.0
    TSM_S0 = 349.0
    TSM_K = 150.0
    
    T = 1.0; r = 0.045
    
    drops = np.linspace(0, -0.60, 100)
    
    wealth_2q_2t = [] 
    wealth_1q_2t = [] 
    wealth_no_hedge = []
    
    print("Simulating 2+2 vs 1+2 Comparison...")
    
    for drop in drops:
        # Correlation Assumption
        us_loss_pct = drop * 1.8 
        if us_loss_pct < -0.99: us_loss_pct = -0.99
        tw_loss_pct = drop * 1.2
        if tw_loss_pct < -0.99: tw_loss_pct = -0.99
        
        # Asset Values
        port_raw = US_Assets * (1 + us_loss_pct) + TW_Assets * (1 + tw_loss_pct)
        
        # Option Pricing
        vol_qqq = 0.20 + abs(drop) * 0.8
        vol_tsm = 0.35 + abs(drop) * 1.0 
        
        qqq_price_new = QQQ_S0 * (1 + drop)
        tsm_price_new = TSM_S0 * (1 + drop * 1.2)
        
        one_qqq_val = black_scholes_put_value(qqq_price_new, QQQ_K, T, r, vol_qqq) * 100
        one_tsm_val = black_scholes_put_value(tsm_price_new, TSM_K, T, r, vol_tsm) * 100
        
        wealth_no_hedge.append(port_raw)
        
        # Strategies 
        # 2Q + 2T ($1450)
        w2 = port_raw + (2 * one_qqq_val) + (2 * one_tsm_val) - (1450)
        wealth_2q_2t.append(w2)
        
        # 1Q + 2T ($1090)
        w3 = port_raw + one_qqq_val + (2 * one_tsm_val) - (1090)
        wealth_1q_2t.append(w3)

    # Plot
    plt.figure(figsize=(10, 6))
    
    # 1. No Hedge (Baseline)
    plt.plot(drops*100, wealth_no_hedge, label='No Hedge (Risk of Ruin)', color='red', linestyle='--', linewidth=1.5)
    
    # 2. 1Q + 2T (Recommended)
    plt.plot(drops*100, wealth_1q_2t, label='1Q + 2T (Recommended)', color='green', linewidth=3)
    
    # 3. 2Q + 2T (Overkill)
    plt.plot(drops*100, wealth_2q_2t, label='2Q + 2T (Highest Protection)', color='purple', linewidth=2, linestyle='-.')
    
    plt.title('Strategy Comparison: 1Q+2T vs 2Q+2T')
    plt.xlabel('QQQ Drop (%)')
    plt.ylabel('Total Net Worth (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotation
    plt.text(-45, 40000, "Both strategies survive\nthe crash comfortably", color='black', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('output/strategy_comparison.png')
    print("Chart saved to output/strategy_comparison.png")

if __name__ == "__main__":
    simulate_over_hedge()
