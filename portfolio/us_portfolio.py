import re

import numpy as np
import pandas as pd
import yfinance as yf


def _is_split_like_action(action_value):
    action = str(action_value).strip().lower()
    if not action:
        return False
    return ("stock split" in action) or ("journaled shares" in action) or (action == "split")


def _append_us_split_events_from_yf(df):
    """
    Some brokers do not export split rows (or use non-standard action names).
    Pull split events from Yahoo actions and append synthetic split rows so the
    later normalizer can keep quantity/price basis consistent.
    """
    out = df.copy()
    required = {"Symbol", "Date", "Quantity", "Amount"}
    if not required.issubset(set(out.columns)):
        return out
    if out.empty:
        return out

    out["Date"] = pd.to_datetime(out["Date"])
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["Amount"] = pd.to_numeric(out["Amount"], errors="coerce")

    start_date = out["Date"].min() - pd.Timedelta(days=7)
    end_date = pd.Timestamp.today() + pd.Timedelta(days=1)

    option_pattern = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")
    symbols = [str(s) for s in out["Symbol"].dropna().unique() if not option_pattern.match(str(s))]
    if not symbols:
        return out

    existing_split_rows = out[out["Action"].map(_is_split_like_action).fillna(False)].copy()
    existing_split_rows["Date"] = pd.to_datetime(existing_split_rows["Date"]).dt.normalize()
    existing_pairs = set(zip(existing_split_rows["Symbol"].astype(str), existing_split_rows["Date"]))

    synthetic_rows = []
    for symbol in symbols:
        try:
            hist = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=True,
                progress=False,
            )
        except Exception:
            continue
        if not isinstance(hist, pd.DataFrame) or hist.empty:
            continue
        split_obj = None
        if "Stock Splits" in hist.columns:
            split_obj = hist["Stock Splits"]
        elif isinstance(hist.columns, pd.MultiIndex):
            if ("Stock Splits", symbol) in hist.columns:
                split_obj = hist[("Stock Splits", symbol)]
            elif (symbol, "Stock Splits") in hist.columns:
                split_obj = hist[(symbol, "Stock Splits")]
        if split_obj is None:
            continue
        if isinstance(split_obj, pd.DataFrame):
            if split_obj.shape[1] == 0:
                continue
            split_obj = split_obj.iloc[:, 0]

        splits = pd.to_numeric(split_obj, errors="coerce").dropna()
        splits = splits[splits != 0]
        if splits.empty:
            continue

        for idx, ratio_raw in splits.items():
            try:
                ratio = float(ratio_raw)
            except Exception:
                continue
            if not np.isfinite(ratio) or ratio <= 0 or ratio == 1.0:
                continue

            split_date = pd.Timestamp(idx).normalize()
            pair = (symbol, split_date)
            if pair in existing_pairs:
                continue

            pre_mask = (out["Symbol"].astype(str) == symbol) & (out["Date"] < split_date)
            pre_qty = out.loc[pre_mask, "Quantity"].sum(skipna=True)
            if pd.isna(pre_qty) or abs(pre_qty) < 1e-12:
                continue

            delta_qty = float(pre_qty) * (ratio - 1.0)
            if not np.isfinite(delta_qty) or abs(delta_qty) < 1e-12:
                continue

            synthetic_rows.append(
                {
                    "Date": split_date,
                    "Action": "Stock Split (YF)",
                    "Symbol": symbol,
                    "Quantity": delta_qty,
                    "Amount": 0.0,
                }
            )
            existing_pairs.add(pair)

    if not synthetic_rows:
        return out

    syn_df = pd.DataFrame(synthetic_rows)
    for col in out.columns:
        if col not in syn_df.columns:
            syn_df[col] = np.nan
    syn_df = syn_df[out.columns]
    return pd.concat([out, syn_df], ignore_index=True, sort=False).sort_values("Date")


def _fetch_us_split_events(symbols, start_date, end_date):
    events = {}
    for symbol in symbols:
        try:
            hist = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=True,
                progress=False,
            )
        except Exception:
            continue
        if not isinstance(hist, pd.DataFrame) or hist.empty:
            continue

        split_obj = None
        if "Stock Splits" in hist.columns:
            split_obj = hist["Stock Splits"]
        elif isinstance(hist.columns, pd.MultiIndex):
            if ("Stock Splits", symbol) in hist.columns:
                split_obj = hist[("Stock Splits", symbol)]
            elif (symbol, "Stock Splits") in hist.columns:
                split_obj = hist[(symbol, "Stock Splits")]
        if split_obj is None:
            continue
        if isinstance(split_obj, pd.DataFrame):
            if split_obj.shape[1] == 0:
                continue
            split_obj = split_obj.iloc[:, 0]

        split_series = pd.to_numeric(split_obj, errors="coerce").dropna()
        split_series = split_series[(split_series != 0) & (split_series != 1)]
        if split_series.empty:
            continue

        rows = []
        for idx, ratio_raw in split_series.items():
            try:
                ratio = float(ratio_raw)
            except Exception:
                continue
            if not np.isfinite(ratio) or ratio <= 0 or ratio == 1.0:
                continue
            rows.append((pd.Timestamp(idx).normalize(), ratio))
        if rows:
            events[str(symbol)] = sorted(rows, key=lambda x: x[0])
    return events


def _normalize_us_quantities_to_split_adjusted_basis(df):
    """
    Normalize transaction quantities to the same split-adjusted basis as Yahoo
    close prices by applying all Yahoo split ratios to prior transactions.
    This also handles symbols that were already fully closed before split dates.
    """
    out = df.copy()
    required = {"Symbol", "Date", "Quantity", "Action"}
    if not required.issubset(set(out.columns)) or out.empty:
        return out

    out["Date"] = pd.to_datetime(out["Date"])
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")

    option_pattern = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")
    symbols = [str(s) for s in out["Symbol"].dropna().unique() if not option_pattern.match(str(s))]
    if not symbols:
        return out

    start_date = out["Date"].min() - pd.Timedelta(days=7)
    end_date = pd.Timestamp.today() + pd.Timedelta(days=1)
    split_events = _fetch_us_split_events(symbols, start_date, end_date)

    for symbol, events in split_events.items():
        sym_mask = out["Symbol"].astype(str) == symbol
        if not sym_mask.any():
            continue
        for split_date, ratio in events:
            pre_mask = sym_mask & (out["Date"] < split_date)
            if pre_mask.any():
                out.loc[pre_mask, "Quantity"] = out.loc[pre_mask, "Quantity"] * float(ratio)

    # split/journal rows are synthetic for quantity basis; remove to avoid
    # double counting quantity moves after normalization.
    split_like_mask = out["Action"].map(_is_split_like_action).fillna(False)
    out = out.loc[~split_like_mask].copy()
    return out


def _normalize_us_split_rows(df):
    """
    Yahoo US close is split-adjusted. If transaction CSV also contains explicit
    stock-split quantity rows, we need to back-adjust historical quantities and
    drop split rows to avoid artificial jumps in valuation/TWR.
    """
    out = df.copy()
    if "Action" not in out.columns or "Symbol" not in out.columns or "Date" not in out.columns:
        return out

    split_mask = out["Action"].map(_is_split_like_action).fillna(False)
    if not split_mask.any():
        return out

    out["Date"] = pd.to_datetime(out["Date"])
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")

    split_rows = out.loc[split_mask, ["Date", "Symbol", "Quantity"]].sort_values("Date")
    for row in split_rows.itertuples(index=False):
        split_date = pd.Timestamp(row.Date)
        symbol = str(row.Symbol)
        delta_qty = float(row.Quantity) if pd.notna(row.Quantity) else np.nan
        if pd.isna(delta_qty):
            continue

        pre_mask = (out["Symbol"].astype(str) == symbol) & (out["Date"] < split_date)
        pre_qty = out.loc[pre_mask, "Quantity"].sum(skipna=True)
        if pre_qty == 0 or pd.isna(pre_qty):
            continue

        ratio = 1.0 + (delta_qty / float(pre_qty))
        if ratio <= 0 or not np.isfinite(ratio):
            continue

        out.loc[pre_mask, "Quantity"] = out.loc[pre_mask, "Quantity"] * ratio

    # split rows are synthetic share adjustments with no cashflow; remove them
    # after back-adjusting prior quantities so holdings/price basis stays consistent.
    out = out.loc[~split_mask].copy()
    return out


def process_us_data(
    *,
    clean_currency,
    build_cash_ledgers,
    fix_share_sign,
    get_daily_price,
    build_option_history_series,
    resolve_market_price,
    get_latest_available_price,
    calculate_total_pnl_for_closed_position,
):
    df_us = pd.read_csv("us_train.csv", encoding="utf-8-sig")
    df_us = df_us.copy()

    df_us["Date"] = pd.to_datetime(df_us["Date"])
    df_us.sort_values("Date", inplace=True)
    df_us = df_us.apply(fix_share_sign, axis=1)
    df_us["Quantity"] = pd.to_numeric(df_us["Quantity"], errors="coerce")
    df_us["Amount"] = df_us["Amount"].apply(clean_currency)
    df_us = _normalize_us_quantities_to_split_adjusted_basis(df_us)

    transaction_cashflows_us, external_cashflows_us, invested_capital_us = build_cash_ledgers(df_us)

    start_date = df_us["Date"].min()
    end_date = pd.Timestamp.today()
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    pivot = df_us.pivot_table(index="Date", columns="Symbol", values="Quantity", aggfunc="sum")
    pivot = pivot.reindex(date_range, fill_value=0).fillna(0)
    cum_holdings = pivot.cumsum()
    symbols_us = cum_holdings.columns.tolist()

    symbols_to_fetch_us = [s for s in symbols_us if not re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", str(s))]

    price_data_us = pd.DataFrame(index=date_range)
    if symbols_to_fetch_us:
        fetched_price_data = get_daily_price(symbols_to_fetch_us, start_date, end_date, is_tw=False)
        fetched_price_data = fetched_price_data.reindex(date_range).ffill().bfill()
        if isinstance(fetched_price_data, pd.Series):
            price_data_us[symbols_to_fetch_us[0]] = fetched_price_data
        else:
            price_data_us = fetched_price_data

    for symbol in symbols_us:
        if symbol not in price_data_us.columns:
            if re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", str(symbol)):
                price_data_us[symbol] = build_option_history_series(symbol, date_range)
            else:
                price_data_us[symbol] = np.nan

    portfolio_value_us = (cum_holdings * price_data_us).sum(axis=1).fillna(0)

    net_holdings_us = df_us.groupby("Symbol")["Quantity"].sum()
    portfolio_snapshot_us = 0
    for stock, shares in net_holdings_us.items():
        if shares != 0:
            price = resolve_market_price(stock, history_series=price_data_us.get(stock), is_tw=False)
            if price is not None:
                portfolio_snapshot_us += shares * price

    total_investment_us = -sum(amount for _, amount in external_cashflows_us if amount < 0)
    final_portfolio_value_us = portfolio_value_us.iloc[-1]
    total_profit_us = final_portfolio_value_us - invested_capital_us
    total_profit_pct_us = (total_profit_us / invested_capital_us) * 100 if invested_capital_us != 0 else 0

    stock_counts_us = {}
    for _, row in df_us.iterrows():
        stock_code = row["Symbol"]
        stock_name = row["Symbol"]
        count = row["Quantity"]
        cost = float(row["Amount"])
        if pd.isna(stock_code) or pd.isna(count):
            continue
        if stock_code not in stock_counts_us:
            stock_counts_us[stock_code] = {"stock_name": stock_name, "Quantity_now": 0, "cost": 0}
        stock_counts_us[stock_code]["Quantity_now"] += count
        stock_counts_us[stock_code]["cost"] += cost

    data_list_us = []
    for stock_code, data_dict in stock_counts_us.items():
        name = data_dict["stock_name"]
        count = data_dict["Quantity_now"]
        aggregated_cost = -data_dict["cost"]
        if count != 0:
            try:
                current_price = resolve_market_price(stock_code, history_series=price_data_us.get(stock_code), is_tw=False)
            except Exception as e:
                print(f"Error fetching data for {stock_code}: {e}")
                current_price = get_latest_available_price(price_data_us.get(stock_code))
            current_value = current_price * count if pd.notna(current_price) else np.nan
            gain = current_value - aggregated_cost
            gain_per = (gain / aggregated_cost) * 100 if aggregated_cost != 0 else 0
        else:
            total_buy, total_pnl, total_pnl_pct = calculate_total_pnl_for_closed_position(stock_code, df_us)
            current_price = np.nan
            current_value = np.nan
            aggregated_cost = total_buy
            gain = total_pnl
            gain_per = total_pnl_pct
        data_list_us.append([stock_code, name, count, current_price, current_value, aggregated_cost, gain, gain_per])

    headers = ["Symbol", "Name", "Quantity_now", "Price", "Price_Total", "Cost", "Total PnL", "Total PnL(%)"]
    portfolio_df_us = pd.DataFrame(data_list_us, columns=headers)

    return {
        "df": df_us,
        "date_range": date_range,
        "portfolio_value": portfolio_value_us,
        "transaction_cashflows": transaction_cashflows_us,
        "external_cashflows": external_cashflows_us,
        "total_investment": total_investment_us,
        "invested_capital": invested_capital_us,
        "final_portfolio_value": final_portfolio_value_us,
        "total_profit": total_profit_us,
        "total_profit_pct": total_profit_pct_us,
        "portfolio_df": portfolio_df_us,
        "portfolio_snapshot": portfolio_snapshot_us,
        "price_data": price_data_us,
        "symbols": symbols_us,
    }
