def calculate_total_pnl_for_closed_position(symbol, df):
    df_sym = df[df['Symbol'] == symbol]
    total_buy = -df_sym[df_sym['Action'].str.lower().isin(['?', '\u8cb7', '\u8cb7\u9032', 'buy'])]['Amount'].sum()
    total_sell = df_sym[df_sym['Action'].str.lower().isin(['?', '\u8ce3', '\u8ce3\u51fa', 'sell'])]['Amount'].sum()
    total_pnl = total_sell - total_buy
    total_pnl_pct = (total_pnl / total_buy * 100) if total_buy != 0 else 0
    return total_buy, total_pnl, total_pnl_pct


def calculate_total_buy_for_position(symbol, df):
    df_sym = df[df['Symbol'] == symbol]
    buy_mask = df_sym['Action'].str.lower().isin(['?', '\u8cb7', '\u8cb7\u9032', 'buy'])
    return -df_sym[buy_mask]['Amount'].sum()


def is_cash_dividend_action(action):
    action_text = str(action).strip().lower()
    return action_text == '\u606f' or 'dividend' in action_text


def calculate_dividends_for_position(symbol, df, amount_col='Amount'):
    df_sym = df[df['Symbol'] == symbol].copy()
    if df_sym.empty:
        return 0.0
    dividend_mask = df_sym['Action'].apply(is_cash_dividend_action)
    amounts = df_sym.loc[dividend_mask, amount_col]
    return float(amounts.sum()) if not amounts.empty else 0.0
