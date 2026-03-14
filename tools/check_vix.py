import yfinance as yf

print("Downloading VIX data...")
data = yf.download(['^VIX'], start='1990-01-01', progress=False)

print("\nStart Date:")
print(f"VIX: {data.index[0]}")
