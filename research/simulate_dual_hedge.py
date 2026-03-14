
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

def simulate_dual_hedge():
    # --- Assumptions ---
    # Portfolio ($67k Total)
    # US Tech (QQQ/QLD): $33k
    # Taiwan (006208/00631L): $34k
    US_Assets = 33000
    TW_Assets = 34000
    
    # Options
    # QQQ Put: Strike 350, Exp Jan 27. Current QQQ ~610. Cost ~360 (Already paid)
    # TSM Put: Strike 150, Exp Jan 27. Current TSM ~349. Cost ~365 (Proposed)
    QQQ_S0 = 610.0
    QQQ_K = 350.0
    TSM_S0 = 349.0
    TSM_K = 150.0
    
    T = 1.0 # Approx 1 year
    r = 0.045
    
    # Scenarios (QQQ Drop as baseline)
    drops = np.linspace(0, -0.60, 100)
    
    wealth_no_hedge = []
    wealth_qqq_only = []
    wealth_dual_hedge = []
    
    print("Simulating Dual Hedge (QQQ + TSM)...")
    print(f"Portfolio: US ${US_Assets} + TW ${TW_Assets} = ${US_Assets+TW_Assets}")
    print("\nScenario | US Asset | TW Asset | QQQ Put | TSM Put | Total Wealth | % Saved")
    print("--- | --- | --- | --- | --- | --- | ---")
    
    checkpoints = [0.0, -0.10, -0.30, -0.50]
    
    for drop in drops:
        # Correlation Assumption:
        # If QQQ drops X%, TSM drops 1.2 * X% (Higher Beta + Currency Risk in crisis)
        # US Assets (2x Lev on half) -> 1.5x drop aggregate?
        # Let's simplify: 
        # US Funds: (QLD+QQQ) -> approx 1.8x QQQ drop logic used before
        us_drop_mult = 1.8
        tw_drop_mult = 1.2 # TSM/EWT drops more than QQQ usually
        
        us_loss_pct = drop * us_drop_mult
        if us_loss_pct < -0.99: us_loss_pct = -0.99
        
        tw_loss_pct = drop * tw_drop_mult
        if tw_loss_pct < -0.99: tw_loss_pct = -0.99
        
        # Asset Values
        us_new = US_Assets * (1 + us_loss_pct)
        tw_new = TW_Assets * (1 + tw_loss_pct)
        total_assets_raw = us_new + tw_new
        
        # Option Pricing
        # Volatility spikes
        vol_qqq = 0.20 + abs(drop) * 0.8
        vol_tsm = 0.35 + abs(drop) * 1.0 # TSM more volatile
        
        qqq_price_new = QQQ_S0 * (1 + drop)
        tsm_price_new = TSM_S0 * (1 + drop * 1.2) # TSM drops more
        
        qqq_put_val = black_scholes_put_value(qqq_price_new, QQQ_K, T, r, vol_qqq) * 100
        tsm_put_val = black_scholes_put_value(tsm_price_new, TSM_K, T, r, vol_tsm) * 100
        
        # Wealth calcs
        # 1. No Hedge
        wealth_no_hedge.append(total_assets_raw)
        
        # 2. QQQ Only (Current)
        wealth_qqq_only.append(total_assets_raw + qqq_put_val)
        
        # 3. Dual Hedge (+ TSM Put, - Cost $365)
        # Note: Cost is sunk for QQQ, but TSM is new money (-365)
        wealth_dual_hedge.append(total_assets_raw + qqq_put_val + tsm_put_val - 365)

        # Print
        for cp in checkpoints:
            if abs(drop - cp) < 0.005:
                saved = (total_assets_raw + qqq_put_val + tsm_put_val - 365) - total_assets_raw
                print(f"QQQ {drop*100:.0f}% | ${us_new:.0f} | ${tw_new:.0f} | ${qqq_put_val:.0f} | ${tsm_put_val:.0f} | ${wealth_dual_hedge[-1]:.0f} | +${saved:.0f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(drops*100, wealth_no_hedge, label='No Hedge (Red)', color='red', linestyle='--')
    plt.plot(drops*100, wealth_qqq_only, label='QQQ Put Only (Current)', color='orange')
    plt.plot(drops*100, wealth_dual_hedge, label='Dual Shield (QQQ + TSM)', color='blue', linewidth=2.5)
    
    plt.fill_between(drops*100, wealth_qqq_only, wealth_dual_hedge, color='blue', alpha=0.1, label='TSM Protection Boost')
    
    plt.title('Performance of "Dual Shield" (QQQ + TSM Puts)')
    plt.xlabel('QQQ Drop (%)')
    plt.ylabel('Total Net Worth (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/dual_hedge_simulation.png')
    print("Chart saved to output/dual_hedge_simulation.png")

if __name__ == "__main__":
    simulate_dual_hedge()
