import numpy as np
from scipy.stats import norm

def black_scholes_put(S, K, T, r, sigma):
    """
    S: Spot Price
    K: Strike Price
    T: Time to Maturity (years)
    r: Risk-free rate
    sigma: Volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_scenario():
    # --- Assumptions ---
    # Index: QQQ
    S0 = 100.0  # Current Price
    
    # Strategy: Buy 1.5% worth of Puts (LEAPS)
    # Portfolio: $1,000,000
    # QLD Exposure: $1,000,000
    PORTFOLIO_VAL = 1_000_000
    BUDGET_PCT = 0.015
    BUDGET = PORTFOLIO_VAL * BUDGET_PCT 
    
    # Option Selection
    # OTM 20% (Strike = 80)
    # Expiry 1 year (1.0) - LEAPS
    # Current Vol (VIX equivalent for QQQ) = 20% (0.2)
    K = 80
    T_initial = 1.0
    r = 0.04
    sigma_initial = 0.20
    
    put_price_initial = black_scholes_put(S0, K, T_initial, r, sigma_initial)
    contracts_num = BUDGET / put_price_initial # Number of "shares" of keys covered
    
    print(f"--- Initial State ---")
    print(f"Index Price: {S0}")
    print(f"Put Strike: {K} (20% OTM)")
    print(f"Put Price (Premium): {put_price_initial:.4f}")
    print(f"Contracts Bought: {contracts_num:.2f} units (covering {contracts_num:.2f} index units)")
    print(f"Cost: ${BUDGET:.2f} ({BUDGET_PCT*100}% of Portfolio)")
    
    # --- Crash Scenarios ---
    scenarios = [
        {"desc": "Correction (-10%)", "drop": -0.10, "vol_spike": 0.25}, # Vol goes to 25%
        {"desc": "Bear Market (-20%)", "drop": -0.20, "vol_spike": 0.35}, # Vol goes to 35%
        {"desc": "Crash (-30%)", "drop": -0.30, "vol_spike": 0.50}, # Vol goes to 50% (Panic)
        {"desc": "Covid Style (-40%)", "drop": -0.40, "vol_spike": 0.60},
    ]
    
    print(f"\n--- Payoff Analysis ---")
    print(f"{'Scenario':<20} | {'QLD Loss (Est)':<15} | {'Put Value (New)':<15} | {'Put Profit':<12} | {'ROI':<8} | {'Coverage %':<10}")
    print("-" * 100)
    
    for sc in scenarios:
        s_new = S0 * (1 + sc['drop'])
        sigma_new = sc['vol_spike']
        t_new = T_initial - 0.08 # Assume crash happens in 1 month (0.08 yr)
        
        # QLD Loss Estimation (2x Leverage approx)
        # If QQQ -10%, QLD ~ -20%. Compometing effects matter but let's use 2x multiplier for simple estimate
        qld_loss_pct = sc['drop'] * 2
        portfolio_loss = PORTFOLIO_VAL * abs(qld_loss_pct)
        
        # New Put Value
        put_price_new = black_scholes_put(s_new, K, t_new, r, sigma_new)
        if put_price_new < 0.01: put_price_new = 0 # min value
        
        total_put_value = contracts_num * put_price_new
        put_profit = total_put_value - BUDGET
        
        roi = (put_profit / BUDGET) * 100
        coverage = (put_profit / portfolio_loss) * 100 if portfolio_loss > 0 else 0
        
        print(f"{sc['desc']:<20} | ${portfolio_loss:,.0f} ({qld_loss_pct*100:.0f}%) | ${total_put_value:,.0f} | ${put_profit:,.0f} | {roi:>.0f}% | {coverage:.1f}%")

if __name__ == "__main__":
    calculate_scenario()
