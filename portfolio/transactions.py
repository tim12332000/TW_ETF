import pandas as pd
import numpy as np


def build_combined_cash_ledger(tw_df, us_df, date_index, fx_series):
    """Net both markets daily in TWD, retain cash and infer only funding deficits."""
    parts = []
    for frame, column, is_us in [(tw_df, 'Amount_TWD', False), (us_df, 'Amount', True)]:
        rows = frame[['Date', column]].copy()
        rows['Date'] = pd.to_datetime(rows['Date']).dt.normalize()
        amounts = pd.to_numeric(rows[column], errors='raise').fillna(0).to_numpy(dtype=float)
        if not np.isfinite(amounts).all():
            raise ValueError('Transaction amounts must be finite')
        if is_us and len(rows):
            rates = fx_series.sort_index().reindex(pd.DatetimeIndex(rows['Date']), method='ffill').to_numpy(dtype=float)
            if not (np.isfinite(rates) & (rates > 0)).all():
                raise ValueError('Missing or invalid historical FX for USD transactions')
            amounts = amounts * rates
        parts.append(pd.Series(amounts, index=pd.DatetimeIndex(rows['Date']), dtype=float))
    flows = pd.concat(parts).groupby(level=0).sum()
    index = pd.DatetimeIndex(date_index).union(flows.index).sort_values()
    net = flows.reindex(index, fill_value=0.0)
    cumulative = net.cumsum()
    invested = (-cumulative.cummin()).clip(lower=0)
    external = -invested.diff().fillna(invested)
    return pd.DataFrame({
        'transaction_net_twd': net,
        'external_flow_twd': external,
        'cash_twd': (cumulative + invested).clip(lower=0),
        'invested_twd': invested,
    }, index=index)


def clean_currency(x):
    if pd.isnull(x) or str(x).strip() == "":
        return None
    try:
        return float(str(x).replace("NT$", "").replace("$", "").replace(",", "").strip())
    except Exception as e:
        print(f"Error parsing currency {x}: {e}")
        return None


def build_cash_ledgers(df):
    """
    Split source transaction amounts into transaction cashflows and external cashflows.
    Sign convention:
    negative = capital in, positive = capital out.
    """
    cash_df = df[["Date", "Amount"]].copy()
    cash_df["Date"] = pd.to_datetime(cash_df["Date"]).dt.normalize()
    cash_df["Amount"] = pd.to_numeric(cash_df["Amount"], errors="coerce")
    cash_df = cash_df.dropna(subset=["Amount"]).sort_values("Date")
    cash_df = cash_df[cash_df["Amount"] != 0]

    transaction_cashflows = list(cash_df[["Date", "Amount"]].itertuples(index=False, name=None))

    cash_balance = 0.0
    external_cashflows = []
    for row in cash_df.itertuples(index=False):
        amt = float(row.Amount)
        if amt > 0:
            cash_balance += amt
            continue

        needed = -amt
        if cash_balance >= needed:
            cash_balance -= needed
            continue

        contribution = needed - cash_balance
        external_cashflows.append((row.Date, -contribution))
        cash_balance = 0.0

    invested_capital = -sum(amount for _, amount in external_cashflows if amount < 0)
    return transaction_cashflows, external_cashflows, invested_capital


def fix_share_sign(row):
    action = str(row["Action"]).strip().lower()
    if action in {"\u8ce3", "\u8ce3\u51fa", "sell"} and row["Quantity"] > 0:
        row["Quantity"] = -row["Quantity"]
    return row


def convert_ticker(ticker):
    if "." not in ticker:
        return ticker + ".TW"
    return ticker
