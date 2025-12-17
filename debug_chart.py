
import pandas as pd
import numpy as np
import combine
import matplotlib.pyplot as plt

# Patching plt.show to do nothing so we don't block
def dummy_show():
    pass
plt.show = dummy_show
plt.savefig = dummy_show

print("Running combine.main() to capture data...")
try:
    # We might need to mock some inputs if main() is too heavy or interactive, 
    # but based on code it seems self-contained reading from CSVs.
    
    # Run user's logic
    twd_to_usd = combine.get_twd_to_usd_rate()
    usd_to_twd = 1 / twd_to_usd
    
    tw_result = combine.process_tw_data()
    us_result = combine.process_us_data()
    
    date_index = tw_result['portfolio_value'].index.union(us_result['portfolio_value'].index).sort_values()
    portfolio_value_tw = tw_result['portfolio_value'].reindex(date_index, method='ffill').fillna(0)
    portfolio_value_us = us_result['portfolio_value'].reindex(date_index, method='ffill').fillna(0)
    combined_portfolio_value_us = portfolio_value_tw + portfolio_value_us
    combined_cashflows = tw_result['cashflows'] + us_result['cashflows']
    
    twr_series = combine.calculate_twr_series(combined_portfolio_value_us, combined_cashflows)
    
    print("\n=== TWR Series Stats ===")
    print(twr_series.describe())
    print("\nHead:")
    print(twr_series.head(10))
    print("\nTail:")
    print(twr_series.tail(10))
    
    # Check for huge jumps
    daily_diff = twr_series.diff()
    print("\nMax Daily Jump:", daily_diff.max())
    print("Min Daily Jump:", daily_diff.min())
    
    # Benchmarks
    COMPARE_TICKERS = ['SPY','QQQ','EWT']
    
    valid_idx = twr_series[twr_series != 0].index
    if not valid_idx.empty:
        start_date = valid_idx[0]
    else:
        start_date = twr_series.index[0]
        
    print(f"\nStart Date for Benchmarks: {start_date}")

    import yfinance as yf
    bench_twr = {}
    for tk in COMPARE_TICKERS:
        try:
            print(f"Fetching {tk}...")
            # Using same logic as source
            def _fetch_bench():
                return yf.download(tk, start=start_date, end=None, progress=False, auto_adjust=True)
            
            # We can't easily access the inner cache logic of combine without modifying it 
            # or replicating `get_cached_data`. combine.get_cached_data is available.
            key = f"bench_twr_{tk}_{start_date.date()}.pkl"
            _px = combine.get_cached_data(key, _fetch_bench)
            
            if isinstance(_px, pd.DataFrame):
                 if 'Close' in _px.columns:
                     _px = _px['Close']
                 else:
                     _px = _px.iloc[:, 0]
            if isinstance(_px, pd.DataFrame):
                 _px = _px.iloc[:, 0]

            if hasattr(_px.index, 'tz'):
                 _px.index = _px.index.tz_localize(None)

            _px = _px.reindex(twr_series.index).ffill().bfill()
            
            start_val = _px.loc[start_date]
            if start_val > 0:
                 bench_twr[tk] = (_px / start_val - 1) * 100
                 print(f"{tk} Stats:")
                 print(bench_twr[tk].describe())
                 print(bench_twr[tk].head())
        except Exception as e:
             print(f"Error {tk}: {e}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()

