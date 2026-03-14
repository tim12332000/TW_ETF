import re

import numpy as np
import pandas as pd


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
    df_us["Amount"] = df_us["Amount"].apply(clean_currency)

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
