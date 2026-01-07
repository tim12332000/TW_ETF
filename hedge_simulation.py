import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_simulation():
    # 1. Get Data (QLD) - 10 Years
    df = yf.download('QLD', period='10y')['Close']
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    
    df.name = 'QLD'
    # Start with $100
    initial_capital = 100.0
    
    # ---------------------------
    # Strategy 1: Cash Hedge (The Control)
    # 90% QLD, 10% Cash (Rebalanced Monthly or just static weight approximation)
    # For simplicity, let's assume daily rebalancing to constant 90/10 weight
    # Daily Return = 0.9 * QLD_Ret + 0.1 * 0
    # ---------------------------
    daily_ret = df.pct_change().fillna(0)
    strat_cash_ret = daily_ret * 0.9
    strat_cash_eq = (1 + strat_cash_ret).cumprod() * initial_capital
    
    # ---------------------------
    # Strategy 2: Protective Put (The "Big Short")
    # 98% QLD.
    # Cost: 2% per year (insurance premium) -> deducted daily approx.
    # Payoff: If QLD drops > 15% in a short period... 
    # SIMULATION LOGIC: 
    #   - Daily decay: -2% / 252
    #   - Crisis Alpha: value spikes if QLD daily return < -3% (Tail event)
    #   Let's approximate: 
    #   Hold 98% QLD.
    #   Spend 2% on Puts.
    #   Put Value Logic: Modeled as a "Tail Risk Fund"
    #     - Base decay: -0.01% per day
    #     - Explosion: If QLD < -3%, Put gains +50% (Convexity)
    #     - If QLD < -5%, Put gains +100%
    # ---------------------------
    
    # Simple Logic:
    # Portfolio = Stock_Part + Put_Part
    # Rebalance monthly to 98/2 ratio
    
    dates = daily_ret.index
    portfolio_put = [initial_capital]
    cash_alloc_put = 0.02
    stock_alloc_put = 0.98
    
    curr_val = initial_capital
    curr_stock = curr_val * stock_alloc_put
    curr_put = curr_val * cash_alloc_put
    
    put_vals = []
    
    for i in range(1, len(dates)):
        r_stock = daily_ret.iloc[i]
        
        # --- Stock Leg ---
        curr_stock *= (1 + r_stock)
        
        # --- Put Leg (Simulation) ---
        # 1. Time Decay (Theta)
        # Annual cost ~20-50% of the Put value itself (if OTM)
        # Let's say Put loses 0.2% of its value every day (approx 50% loss per year if nothing happens)
        put_ret = -0.002 
        
        # 2. Gamma / Convexity (Explosion)
        # If stock crashes, Put explodes
        if r_stock < -0.02: # -2% drop
            put_ret += abs(r_stock) * 5 # 5x leverage on drop
        if r_stock < -0.04: # -4% drop
            put_ret += abs(r_stock) * 10 # 10x leverage
            
        curr_put *= (1 + put_ret)
        
        # Daily Total
        curr_val = curr_stock + curr_put
        
        # Monthly Rebalance (Approx every 21 days)
        if i % 21 == 0:
            curr_stock = curr_val * stock_alloc_put
            curr_put = curr_val * cash_alloc_put
            
        portfolio_put.append(curr_val)
        
    strat_put_eq = pd.Series(portfolio_put, index=dates)

    # ---------------------------
    # Metrics
    # ---------------------------
    def get_stats(series):
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        cagr = (series.iloc[-1] / series.iloc[0]) ** (252/len(series)) - 1
        dd = (series / series.cummax()) - 1
        return cagr * 100, dd.min() * 100, total_ret * 100
        
    c_cagr, c_mdd, c_tot = get_stats(strat_cash_eq)
    p_cagr, p_mdd, p_tot = get_stats(strat_put_eq)
    base_cagr, base_mdd, base_tot = get_stats((1+daily_ret).cumprod() * initial_capital)
    
    print(f"\n=== Simulation Results (10 Years on QLD) ===")
    print(f"1. Baseline (100% QLD):          CAGR: {base_cagr:.2f}% | MaxDrawdown: {base_mdd:.2f}% | Total Return: {base_tot:.0f}%")
    print(f"2. Cash Hedge (90% QLD + Cash):  CAGR: {c_cagr:.2f}% | MaxDrawdown: {c_mdd:.2f}% | Total Return: {c_tot:.0f}%")
    print(f"3. Put Hedge_Sim (98% + Puts):   CAGR: {p_cagr:.2f}% | MaxDrawdown: {p_mdd:.2f}% | Total Return: {p_tot:.0f}%")

    print("\n[Comparison]")
    if p_cagr > c_cagr:
        print(">> 'Buying Puts' (Strategy A) OUTPERFORMS on Return (CAGR).")
    else:
        print(">> 'Cash Hedge' (Strategy B) OUTPERFORMS on Return (CAGR).")
        
    if abs(p_mdd) < abs(c_mdd):
        print(">> 'Buying Puts' (Strategy A) provides BETTER Protection (Lower MaxDD).")
    else:
        print(">> 'Cash Hedge' (Strategy B) provides BETTER Protection (Lower MaxDD).")

if __name__ == "__main__":
    run_simulation()
