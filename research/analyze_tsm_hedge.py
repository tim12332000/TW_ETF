
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

def analyze_tsm_trade():
    # TSM Data
    S0 = 349.0  # Current Price
    K = 150.0   # Strike
    Price = 3.65 # Premium per share
    Cost = Price * 100 # $365
    Expiry = "Jan 2027"
    T = 340 / 365.0 # Approx 1 year
    r = 0.045
    
    # Portfolio
    Portfolio_Val = 35000.0 # Approx $35k USD exposure
    
    # Scenarios: TSM Drops
    drops = np.linspace(0, -0.90, 100) # 0% to -90% drop
    
    wealth_no_hedge = []
    wealth_with_hedge = []
    put_values = []
    
    print(f"Analysis of TSM Put: Strike ${K}, Cost ${Cost}")
    print(f"Break-even at Expiry: ${K - Price:.2f} (Drop: {(K-Price-S0)/S0*100:.1f}%)")
    
    print("\nScenario | TSM Price | Return % | Put Value | Net Benefit")
    print("--- | --- | --- | --- | ---")
    
    checkpoints = [-0.10, -0.30, -0.50, -0.60, -0.70, -0.80]
    
    for drop in drops:
        S_new = S0 * (1 + drop)
        
        # Portfolio Loss
        port_new = Portfolio_Val * (1 + drop)
        
        # Put Value
        # Volatility usually spikes as price drops
        # Base vol 30%, increasing with drop
        vol = 0.35 + abs(drop) * 1.0 # Significant skew for crash
        if drop > -0.2: vol = 0.35 # Normal vol
        
        put_val = black_scholes_put_value(S_new, K, T, r, vol) * 100
        
        net_wealth = port_new + put_val - Cost
        
        wealth_no_hedge.append(port_new)
        wealth_with_hedge.append(net_wealth)
        put_values.append(put_val)

        # Print Checkpoints
        for cp in checkpoints:
            if abs(drop - cp) < 0.005:
                benefit = put_val - Cost
                print(f"{drop*100:.0f}% | ${S_new:.1f} | {benefit/Cost*100:.0f}% | ${put_val:.0f} | ${benefit:.0f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(drops*100, wealth_no_hedge, label='Taiwan Assets (No Hedge)', color='red', linestyle='--')
    plt.plot(drops*100, wealth_with_hedge, label='Taiwan Assets + TSM Put', color='blue', linewidth=2)
    
    plt.title(f'TSM Strike ${K} Put Protection (Cost ${Cost})')
    plt.xlabel('TSM Price Drop (%)')
    plt.ylabel('Portfolio Value (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Text
    plt.text(-60, 20000, "War Scenario (-60%)\nProtection Kicks In", color='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/tsm_put_analysis.png')
    print("Chart saved to output/tsm_put_analysis.png")

if __name__ == "__main__":
    analyze_tsm_trade()
