import yfinance as yf
import pandas as pd
import numpy as np

print("Downloading data to verify backfill logic...")
# Logic matched from notebook
data = yf.download(['QQQ', '^VXN', '^TNX'], start='1999-03-10', progress=False)

df = pd.DataFrame()
if isinstance(data.columns, pd.MultiIndex):
    df['QQQ_Price'] = data['Close']['QQQ']
    df['VXN'] = data['Close']['^VXN']
    df['TNX'] = data['Close']['^TNX']

print(f"Original start date (before dropna): {df.index[0]}")
print(f"VXN first valid index: {df['VXN'].first_valid_index()}")

# The Backfill Logic
print("Applying backfill...")
df['Hist_Vol'] = df['QQQ_Price'].pct_change().rolling(window=21).std() * np.sqrt(252) * 100
df['VXN'] = df['VXN'].fillna(df['Hist_Vol'])

print(f"VXN first valid index (after backfill): {df['VXN'].first_valid_index()}")

# The Check
df = df.dropna()
start_date = df.index[0]
print(f"Final Start Date (after dropna): {start_date}")

if start_date.year == 1999:
    print("SUCCESS: Data starts in 1999.")
else:
    print(f"FAILURE: Data starts in {start_date.year}")
