import yfinance as yf

print("Downloading data...")
data = yf.download(['QQQ', '^VXN', '^TNX'], start='1990-01-01', progress=False)

print("\nStart Dates:")
for col in data['Close'].columns:
    first_valid = data['Close'][col].first_valid_index()
    print(f"{col}: {first_valid}")
