import numpy as np
import pandas as pd


def process_tw_data(
    *,
    clean_currency,
    build_cash_ledgers,
    fix_share_sign,
    get_usd_twd_history,
    align_fx_series,
    get_daily_price,
    get_latest_available_price,
    get_current_price_yf,
    calculate_total_pnl_for_closed_position,
    record_calibrations,
):
    df_tw = pd.read_csv("tw_train.csv", encoding="utf-8-sig")
    df_tw.rename(
        columns={
            "交易日": "Date",
            "交易別": "Action",
            "股票代號": "Symbol",
            "股票名稱": "Name",
            "股數": "Quantity",
            "單價": "Price",
            "進帳/出帳": "Amount",
        },
        inplace=True,
    )
    required_columns = ["Date", "Action", "Symbol", "Quantity", "Price", "Amount"]
    missing_columns = [col for col in required_columns if col not in df_tw.columns]
    if missing_columns:
        raise KeyError(f"tw_train.csv missing expected columns: {missing_columns}; actual columns: {list(df_tw.columns)}")
    df_tw["Date"] = pd.to_datetime(df_tw["Date"])
    df_tw.sort_values("Date", inplace=True)
    df_tw["Quantity"] = pd.to_numeric(df_tw["Quantity"], errors="coerce")
    df_tw = df_tw.apply(fix_share_sign, axis=1)
    start_date = df_tw["Date"].min()
    end_date = pd.Timestamp.today()
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    usd_twd_series = get_usd_twd_history(start_date, end_date)

    df_tw["Amount"] = df_tw["Amount"].apply(clean_currency)
    df_tw["Amount_TWD"] = df_tw["Amount"]
    df_tw["USD_TWD"] = align_fx_series(df_tw["Date"], usd_twd_series).values
    df_tw["Amount"] = df_tw["Amount_TWD"] / df_tw["USD_TWD"]

    transaction_cashflows_tw, external_cashflows_tw, invested_capital_tw = build_cash_ledgers(df_tw)

    pivot = df_tw.pivot_table(index="Date", columns="Symbol", values="Quantity", aggfunc="sum")
    pivot = pivot.reindex(date_range, fill_value=0).fillna(0)
    cum_holdings = pivot.cumsum()
    symbols_tw = cum_holdings.columns.tolist()

    price_data_tw = get_daily_price(symbols_tw, start_date, end_date, is_tw=True)
    if price_data_tw is not None and not price_data_tw.empty:
        price_data_tw.columns = [col.split(".")[0] for col in price_data_tw.columns]
    price_data_tw = price_data_tw.reindex(date_range).ffill().bfill()

    try:
        calibration_count = 0
        for symbol in symbols_tw:
            symbol_rows = df_tw[df_tw["Symbol"] == symbol].sort_values("Date")
            if symbol_rows.empty:
                continue
            first_trade_date = symbol_rows.iloc[0]["Date"]
            csv_price = symbol_rows.iloc[0]["Price"]
            if pd.isna(csv_price):
                continue
            try:
                yf_price = float(price_data_tw.loc[first_trade_date, symbol])
            except Exception:
                continue
            if yf_price and yf_price > 0:
                factor = float(csv_price) / yf_price
                price_data_tw[symbol] = price_data_tw[symbol] * factor
                calibration_count += 1
        if calibration_count:
            record_calibrations(calibration_count)
    except Exception:
        pass

    price_data_tw_twd = price_data_tw.copy()
    fx_on_date = align_fx_series(date_range, usd_twd_series)
    price_data_tw = price_data_tw_twd.div(fx_on_date, axis=0)

    portfolio_value_tw = (cum_holdings * price_data_tw).sum(axis=1).fillna(0)

    net_holdings_tw = df_tw.groupby("Symbol")["Quantity"].sum()
    portfolio_snapshot_tw = 0
    latest_usd_twd = float(usd_twd_series.iloc[-1])
    for stock, shares in net_holdings_tw.items():
        if shares != 0:
            fallback_price_usd = get_latest_available_price(price_data_tw.get(stock))
            live_price_twd = get_current_price_yf(stock, is_tw=True)
            price = (live_price_twd / latest_usd_twd) if live_price_twd is not None else fallback_price_usd
            if price is not None:
                portfolio_snapshot_tw += shares * price

    total_investment_tw = -sum(amount for _, amount in external_cashflows_tw if amount < 0)
    final_portfolio_value_tw = portfolio_value_tw.iloc[-1]
    total_profit_tw = final_portfolio_value_tw - invested_capital_tw
    total_profit_pct_tw = (total_profit_tw / invested_capital_tw) * 100 if invested_capital_tw != 0 else 0

    stock_counts_tw = {}
    for _, row in df_tw.iterrows():
        stock_code = row["Symbol"]
        stock_name = row["Name"] if pd.notna(row["Name"]) else stock_code
        count = row["Quantity"]
        cost = float(row["Amount"])
        if pd.isna(stock_code) or pd.isna(count):
            continue
        if stock_code not in stock_counts_tw:
            stock_counts_tw[stock_code] = {"stock_name": stock_name, "Quantity_now": 0, "cost": 0}
        stock_counts_tw[stock_code]["Quantity_now"] += count
        stock_counts_tw[stock_code]["cost"] += cost

    data_list_tw = []
    for stock_code, data_dict in stock_counts_tw.items():
        name = data_dict["stock_name"]
        count = data_dict["Quantity_now"]
        aggregated_cost = -data_dict["cost"]
        if count != 0:
            try:
                fallback_price_usd = get_latest_available_price(price_data_tw.get(stock_code))
                live_price_twd = get_current_price_yf(stock_code, is_tw=True)
                current_price = (live_price_twd / latest_usd_twd) if live_price_twd is not None else fallback_price_usd
            except Exception as e:
                print(f"Error fetching data for {stock_code}: {e}")
                current_price = get_latest_available_price(price_data_tw.get(stock_code))
            current_value = current_price * count if pd.notna(current_price) else np.nan
            gain = current_value - aggregated_cost
            gain_per = (gain / aggregated_cost) * 100 if aggregated_cost != 0 else 0
        else:
            total_buy, total_pnl, total_pnl_pct = calculate_total_pnl_for_closed_position(stock_code, df_tw)
            current_price = np.nan
            current_value = np.nan
            aggregated_cost = total_buy
            gain = total_pnl
            gain_per = total_pnl_pct
        data_list_tw.append([stock_code, name, count, current_price, current_value, aggregated_cost, gain, gain_per])

    headers = ["Symbol", "Name", "Quantity_now", "Price", "Price_Total", "Cost", "Total PnL", "Total PnL(%)"]
    portfolio_df_tw = pd.DataFrame(data_list_tw, columns=headers)

    return {
        "df": df_tw,
        "date_range": date_range,
        "portfolio_value": portfolio_value_tw,
        "transaction_cashflows": transaction_cashflows_tw,
        "external_cashflows": external_cashflows_tw,
        "total_investment": total_investment_tw,
        "invested_capital": invested_capital_tw,
        "final_portfolio_value": final_portfolio_value_tw,
        "total_profit": total_profit_tw,
        "total_profit_pct": total_profit_pct_tw,
        "portfolio_df": portfolio_df_tw,
        "portfolio_snapshot": portfolio_snapshot_tw,
        "price_data": price_data_tw,
        "price_data_twd": price_data_tw_twd,
        "symbols": symbols_tw,
        "usd_twd_series": usd_twd_series,
    }
