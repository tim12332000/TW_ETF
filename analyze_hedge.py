import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def analyze():
    # Market Data
    QQQ_PRICE = 610.0
    RISK_FREE_RATE = 0.045
    TODAY = datetime(2026, 2, 8)
    EXPIRY = datetime(2027, 1, 15)
    DAYS_TO_EXPIRY = (EXPIRY - TODAY).days
    T = DAYS_TO_EXPIRY / 365.0
    
    # Portfolio Data (Approx from report)
    HIGH_BETA_VAL = 30800.0  # QLD, 00631L (2x Leveraged)
    MARKET_BETA_VAL = 30300.0 # SPYM, 006208, Stocks (1x, highly correlated)
    HEDGE_VAL = 4900.0       # EDV (Bonds, low/neg correlation)
    CASH_VAL = 66737 - (HIGH_BETA_VAL + MARKET_BETA_VAL + HEDGE_VAL)
    if CASH_VAL < 0: CASH_VAL = 0
    PORTFOLIO_TOTAL = HIGH_BETA_VAL + MARKET_BETA_VAL + HEDGE_VAL + CASH_VAL
    
    # Put Option
    STRIKE = 350.0
    CONTRACTS = 1 
    TOTAL_COST = 360.0 
    
    # --- 1. Print Markdown Table ---
    scenarios = [
        {"drop": 0.0, "desc": "Current (目前)"},
        {"drop": -0.10, "desc": "Correction (回調)"},
        {"drop": -0.20, "desc": "Bear Market (熊市)"},
        {"drop": -0.30, "desc": "Crash (崩盤)"},
        {"drop": -0.40, "desc": "Crisis (金融危機)"},
        {"drop": -0.50, "desc": "Collapse (毀滅)"},
    ]

    print("\n### 你的總資產預估 (Total Asset Projection)")
    print("| 情境 (Scenario) | QQQ 跌幅 | 總資產 (原本) | 總資產 (災難後) | 你的 Put 變成 | **最終總資產** | 少賠了多少? |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for sc in scenarios:
        drop = sc['drop']
        new_qqq = QQQ_PRICE * (1 + drop)
        
        # Portfolio Loss
        drop_2x = drop * 2
        if drop_2x <= -1.0: drop_2x = -0.99
        new_high_beta = HIGH_BETA_VAL * (1 + drop_2x)
        new_market_beta = MARKET_BETA_VAL * (1 + drop)
        
        bond_change = 0.0
        if drop <= -0.20: bond_change = 0.05
        if drop <= -0.30: bond_change = 0.15
        new_hedge = HEDGE_VAL * (1 + bond_change)
        
        new_total_port_assets = new_high_beta + new_market_beta + new_hedge + CASH_VAL
        
        # Put Profit
        vol = 0.20 + abs(drop) * 0.8
        put_val_share = black_scholes_put(new_qqq, STRIKE, T, RISK_FREE_RATE, vol)
        total_put_val = put_val_share * 100 * CONTRACTS
        
        final_total_wealth = new_total_port_assets + total_put_val
        protection_amt = total_put_val 
        
        print(f"| **{sc['desc']}** | {drop*100:.0f}% | ${PORTFOLIO_TOTAL:,.0f} | ${new_total_port_assets:,.0f} | ${total_put_val:,.0f} | **${final_total_wealth:,.0f}** | +${total_put_val:,.0f} |")

    # --- 2. Generate Chart ---
    drops = np.linspace(0, -0.60, 100)
    
    wealth_no_hedge = []
    wealth_with_hedge = []
    
    for drop in drops:
        # Portfolio Loss
        drop_2x = drop * 2
        if drop_2x <= -1.0: drop_2x = -0.99
        new_high_beta = HIGH_BETA_VAL * (1 + drop_2x)
        new_market_beta = MARKET_BETA_VAL * (1 + drop)
        
        bond_change = 0.0
        if drop <= -0.20: bond_change = 0.05
        if drop <= -0.30: bond_change = 0.15
        new_hedge = HEDGE_VAL * (1 + bond_change)
        
        new_total_port = new_high_beta + new_market_beta + new_hedge + CASH_VAL
        
        # Put Value
        vol = 0.20 + abs(drop) * 0.8
        new_qqq = QQQ_PRICE * (1 + drop)
        put_val_share = black_scholes_put(new_qqq, STRIKE, T, RISK_FREE_RATE, vol)
        total_put_val = put_val_share * 100 * CONTRACTS
        
        wealth_no_hedge.append(new_total_port)
        wealth_with_hedge.append(new_total_port + total_put_val)

    plt.figure(figsize=(10, 6))
    
    # Plot Unhedged Wealth (Red)
    plt.plot(drops * 100, wealth_no_hedge, label='Total Assets (No Insurance)', color='red', linewidth=2)
    
    # Plot Hedged Wealth (Blue)
    plt.plot(drops * 100, wealth_with_hedge, label='Total Assets (With Put)', color='blue', linewidth=2)
    
    # Fill between
    plt.fill_between(drops * 100, wealth_no_hedge, wealth_with_hedge, color='green', alpha=0.1, label='Insurance Payout')
    
    plt.title('Total Asset Protection (QQQ Drop)', fontsize=14)
    plt.xlabel('QQQ Drop Percentage (%)', fontsize=12)
    plt.ylabel('Total Asset Value ($ USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Text annotation for breakout point
    try:
        # Payout > 1000 index
        payouts = np.array(wealth_with_hedge) - np.array(wealth_no_hedge)
        idx = next(x for x, val in enumerate(payouts) if val > 1000)
        start_drop = drops[idx] * 100
        plt.axvline(x=start_drop, color='orange', linestyle=':')
        plt.text(start_drop - 2, 25000, f'Protection Starts ~{abs(start_drop):.0f}% Drop', rotation=90, color='orange')
    except:
        pass

    plt.tight_layout()
    plt.savefig(r'C:\Git\TW_ETF\output\total_asset_protection.png')
    print("\nChart saved to C:\\Git\\TW_ETF\\output\\total_asset_protection.png")

if __name__ == "__main__":
    analyze()
