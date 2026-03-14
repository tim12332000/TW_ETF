def calculate_total_pnl_for_closed_position(symbol, df):
    df_sym = df[df['Symbol'] == symbol]
    total_buy = -df_sym[df_sym['Action'].str.lower().isin(['?', 'buy'])]['Amount'].sum()
    total_sell = df_sym[df_sym['Action'].str.lower().isin(['?', 'sell'])]['Amount'].sum()
    total_pnl = total_sell - total_buy
    total_pnl_pct = (total_pnl / total_buy * 100) if total_buy != 0 else 0
    return total_buy, total_pnl, total_pnl_pct
