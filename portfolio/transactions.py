import pandas as pd


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
