import numpy as np
import pandas as pd


TW_STOCK_SPLITS = {
    # 0050: 1 -> 4 split, resumed trading on 2025-06-18.
    "0050": [
        {"effective_date": "2025-06-18", "ratio": 4.0},
    ],
    # 00631L: 1 -> 22 split. Last pre-split trading day was 2026-03-24;
    # units were split during the 2026-03-25 to 2026-03-30 halt.
    "00631L": [
        {"effective_date": "2026-03-25", "ratio": 22.0},
    ],
}


def apply_tw_split_events(df):
    """
    Append synthetic split transactions so quantity changes at split date
    without introducing cashflow. This keeps price, quantity, and cashflow
    on a consistent basis for valuation and TWR.
    """
    adjusted = df.copy()
    required = {"Symbol", "Date", "Quantity", "Amount"}
    if not required.issubset(set(adjusted.columns)):
        return adjusted

    adjusted["Date"] = pd.to_datetime(adjusted["Date"])
    adjusted["Quantity"] = pd.to_numeric(adjusted["Quantity"], errors="coerce").fillna(0.0)
    adjusted["Amount"] = pd.to_numeric(adjusted["Amount"], errors="coerce").fillna(0.0)

    synthetic_rows = []
    for symbol, events in TW_STOCK_SPLITS.items():
        events_sorted = sorted(events, key=lambda x: pd.Timestamp(x["effective_date"]))
        for event in events_sorted:
            effective_date = pd.Timestamp(event["effective_date"])
            ratio = float(event["ratio"])
            if ratio <= 0:
                continue

            pos_before = adjusted.loc[
                (adjusted["Symbol"].astype(str) == symbol) & (adjusted["Date"] < effective_date),
                "Quantity",
            ].sum()
            if pos_before <= 0:
                continue

            delta_qty = pos_before * (ratio - 1.0)
            if abs(delta_qty) < 1e-12:
                continue

            synthetic_rows.append(
                {
                    "Date": effective_date,
                    "Action": "split",
                    "Symbol": symbol,
                    "Name": symbol,
                    "Quantity": delta_qty,
                    "Price": np.nan,
                    "Amount": 0.0,
                }
            )

            # Include event impact for subsequent split events on same symbol.
            adjusted = pd.concat(
                [adjusted, pd.DataFrame([synthetic_rows[-1]])],
                ignore_index=True,
                sort=False,
            )

    if not synthetic_rows:
        return adjusted.sort_values("Date")

    base_cols = list(adjusted.columns)
    syn_df = pd.DataFrame(synthetic_rows)
    for col in base_cols:
        if col not in syn_df.columns:
            syn_df[col] = np.nan
    syn_df = syn_df[base_cols]

    result = pd.concat([df, syn_df], ignore_index=True, sort=False)
    result["Date"] = pd.to_datetime(result["Date"])
    result = result.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return result


def tw_split_price_factors(symbol, date_index):
    """
    Return per-date multipliers that restore split-adjusted Yahoo closes to
    the transaction ledger's share basis before each synthetic split event.
    """
    base_symbol = str(symbol).split(".")[0]
    factors = pd.Series(1.0, index=pd.DatetimeIndex(pd.to_datetime(date_index)))

    for event in sorted(TW_STOCK_SPLITS.get(base_symbol, []), key=lambda x: pd.Timestamp(x["effective_date"])):
        ratio = float(event["ratio"])
        if ratio <= 0:
            continue
        effective_date = pd.Timestamp(event["effective_date"])
        factors.loc[factors.index < effective_date] *= ratio

    return factors
