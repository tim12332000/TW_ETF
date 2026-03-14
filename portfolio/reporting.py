import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import rcParams
from scipy.stats import norm
from tabulate import tabulate

from .performance import calculate_twr_series

rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


def plot_stock_performance(tw_res, us_res):
    plt.figure(figsize=(14, 8))
    _process_stock_twr(tw_res, 'TW', plt)
    _process_stock_twr(us_res, 'US', plt)
    plt.title('Individual Stock Performance (TWR %)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/stock_performance.png')
    plt.show()
    plt.close()
    print('Stock performance chart saved to output/stock_performance.png')


def _process_stock_twr(res_data, region, plt_obj):
    df_tx = res_data['df']
    price_df = res_data['price_data']
    date_range = res_data['date_range']
    symbols = res_data['symbols']

    for sym in symbols:
        sym_tx = df_tx[df_tx['Symbol'] == sym].copy()
        if sym_tx.empty:
            continue

        daily_quantity = sym_tx.groupby('Date')['Quantity'].sum()
        shares_series = daily_quantity.reindex(date_range).fillna(0).cumsum()

        price_col = None
        if sym in price_df.columns:
            price_col = sym
        elif str(sym).split('.')[0] in price_df.columns:
            price_col = str(sym).split('.')[0]
        if price_col is None:
            continue

        combined_df = pd.DataFrame({'Shares': shares_series, 'Price': price_df[price_col]})
        combined_df = combined_df.dropna(subset=['Price'])
        combined_df['Shares'] = combined_df['Shares'].ffill().fillna(0)
        pv_series = combined_df['Shares'] * combined_df['Price']
        cashflows = list(sym_tx[['Date', 'Amount']].itertuples(index=False, name=None))

        try:
            twr = calculate_twr_series(pv_series, cashflows)
            if twr.empty or (twr == 0).all():
                continue
            non_zero = twr[twr != 0]
            if non_zero.empty:
                continue
            first_date = non_zero.index[0]
            plot_data = twr.loc[first_date:]
            current_shares = shares_series.iloc[-1]
            alpha = 1.0 if abs(current_shares) > 0.001 else 0.4
            linestyle = '-' if abs(current_shares) > 0.001 else ':'
            plt_obj.plot(plot_data.index, plot_data, label=f'{sym} ({region})', alpha=alpha, linestyle=linestyle, linewidth=1.5)
        except Exception as e:
            print(f'Error calculating TWR for {sym}: {e}')


def print_rebalance_recommendation(portfolio_df_combined, usd_to_twd):
    targets = {
        'QLD': 0.30,
        'SPLG / SPYM': 0.25,
        '00631L': 0.15,
        '006208': 0.30,
    }
    current_values_usd = {k: 0.0 for k in targets}
    total_pool_usd = 0.0

    for _, row in portfolio_df_combined.iterrows():
        sym = str(row['Symbol'])
        qty = float(row['Quantity_now'])
        val_usd = float(pd.to_numeric(row['Price_Total'], errors='coerce'))
        if pd.isna(val_usd) or qty == 0:
            continue
        if 'QLD' in sym:
            current_values_usd['QLD'] += val_usd
            total_pool_usd += val_usd
        elif 'SPLG' in sym or 'SPYM' in sym:
            current_values_usd['SPLG / SPYM'] += val_usd
            total_pool_usd += val_usd
        elif '00631L' in sym:
            current_values_usd['00631L'] += val_usd
            total_pool_usd += val_usd
        elif '006208' in sym:
            current_values_usd['006208'] += val_usd
            total_pool_usd += val_usd

    print('\n=== Target Allocation And Rebalance Suggestion ===')
    total_pool_twd = total_pool_usd * usd_to_twd
    print(f'Rebalance capital pool (stocks only): {total_pool_twd:,.0f} TWD')

    rows = []
    for key, target_pct in targets.items():
        curr_usd = current_values_usd[key]
        curr_twd = curr_usd * usd_to_twd
        target_twd = total_pool_twd * target_pct
        diff_twd = target_twd - curr_twd
        curr_pct = (curr_twd / total_pool_twd) * 100 if total_pool_twd > 0 else 0
        action_str = f'+{diff_twd:,.0f}' if diff_twd > 0 else f'{diff_twd:,.0f}'
        rows.append([key, f'{target_pct*100:.1f}%', f'{curr_pct:.1f}%', f'{curr_twd:,.0f}', f'{target_twd:,.0f}', action_str])

    headers = ['Asset', 'Target %', 'Current %', 'Current Amount (TWD)', 'Target Amount (TWD)', 'Suggested Action (TWD)']
    print(tabulate(rows, headers=headers, tablefmt='psql', stralign='right'))


def black_scholes_put(S, K, T, r, sigma):
    if K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    if S <= 0:
        return K * np.exp(-r * T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def analyze_put_protection(portfolio_df):
    high_beta_tickers = ['QLD', '00631L.TW', 'TQQQ', 'SOXL', 'TECL', 'NVDL']
    hedge_tickers = ['EDV', 'TLT', 'TMF', 'ZROZ', 'UBT']
    df = portfolio_df.copy()
    high_beta_val = 0
    market_beta_val = 0
    hedge_val = 0
    total_stock_val = 0
    put_contracts = []

    for _, row in df.iterrows():
        sym = str(row['Symbol'])
        val = float(row['Price_Total'])
        qty = float(row['Quantity_now'])
        if 'P' in sym and any(c.isdigit() for c in sym) and qty > 0:
            match = re.match(r'^([A-Z]+)(\d{6})P(\d{8})$', sym)
            if match or (('QQQ' in sym or 'TSM' in sym) and 'P' in sym):
                put_contracts.append(row)
                continue
        total_stock_val += val
        is_high = False
        for hb in high_beta_tickers:
            if hb.split('.')[0] in sym:
                high_beta_val += val
                is_high = True
                break
        if is_high:
            continue
        is_hedge = False
        for hedge in hedge_tickers:
            if hedge in sym:
                hedge_val += val
                is_hedge = True
                break
        if is_hedge:
            continue
        market_beta_val += val

    if not put_contracts:
        return

    puts_info = []
    for put in put_contracts:
        sym = put['Symbol']
        match = re.match(r'^([A-Z]+)(\d{6})P(\d{8})$', sym)
        if match:
            underlying, date_str, strike_str = match.group(1), match.group(2), match.group(3)
        else:
            p_index = sym.find('P')
            base = sym[:p_index]
            date_str = base[-6:]
            underlying = base[:-6]
            strike_str = sym[p_index + 1:]
        try:
            strike = float(strike_str) / 1000.0
            expiry = datetime.strptime(date_str, '%y%m%d')
            contracts = float(put['Quantity_now']) / 100.0
            cost_basis = float(put['Cost'])
            try:
                hist = yf.Ticker(underlying).history(period='1d')
                s0 = hist['Close'].iloc[-1]
            except Exception:
                s0 = 610.0 if underlying == 'QQQ' else 200.0
            puts_info.append({
                'sym': sym,
                'underlying': underlying,
                'strike': strike,
                'expiry': expiry,
                'contracts': contracts,
                'cost_basis': cost_basis,
                's0': s0,
            })
        except Exception:
            continue
    if not puts_info:
        return

    today = datetime.now()
    r = 0.045
    portfolio_total = total_stock_val
    scenarios = [
        {'drop': 0.0, 'desc': 'Current'},
        {'drop': -0.10, 'desc': 'Correction'},
        {'drop': -0.20, 'desc': 'Bear Market'},
        {'drop': -0.30, 'desc': 'Crash'},
        {'drop': -0.40, 'desc': 'Crisis'},
        {'drop': -0.50, 'desc': 'Collapse'},
    ]

    print('\n=== Put Protection Analysis ===')
    for put_info in puts_info:
        print(f"Put: {put_info['sym']}, Strike ${put_info['strike']}, Exp {put_info['expiry'].strftime('%Y-%m-%d')} (Underlying: {put_info['underlying']} @ ${put_info['s0']:.2f})")
    print(f'Total stock assets: ${portfolio_total:,.0f} USD')
    print('| Scenario | Base Drop | Total Assets (Unhedged) | Total Assets (Hedged) | Total Puts Value | Protection |')
    print('| :--- | :--- | :--- | :--- | :--- | :--- |')

    drops = np.linspace(0, -1.00, 100)
    wealth_no_hedge = []
    wealth_with_hedge = []

    for scenario in scenarios:
        drop = scenario['drop']
        drop_2x = max(drop * 2.0, -0.99)
        new_high = high_beta_val * (1 + drop_2x)
        bond_change = 0.0
        if drop <= -0.20:
            bond_change = 0.05
        if drop <= -0.30:
            bond_change = 0.15
        new_hedge = hedge_val * (1 + bond_change)
        new_market = market_beta_val * (1 + drop)
        new_total_no_put = new_high + new_hedge + new_market
        total_put_val = 0
        for put_info in puts_info:
            if put_info['expiry'] <= today:
                continue
            days_to_exp = (put_info['expiry'] - today).days
            T = days_to_exp / 365.0
            new_u = put_info['s0'] * (1 + drop)
            vol = 0.20 + abs(drop) * 0.8
            put_val_share = black_scholes_put(new_u, put_info['strike'], T, r, vol)
            total_put_val += put_val_share * 100 * put_info['contracts']
        new_total_with_put = new_total_no_put + total_put_val
        diff = new_total_with_put - new_total_no_put
        print(f"| {scenario['desc']} | {drop*100:.0f}% | ${new_total_no_put:,.0f} | **${new_total_with_put:,.0f}** | ${total_put_val:,.0f} | +${diff:,.0f} |")

    for drop in drops:
        drop_2x = max(drop * 2.0, -0.99)
        new_high = high_beta_val * (1 + drop_2x)
        bond_change = 0.0
        if drop <= -0.20:
            bond_change = 0.05
        if drop <= -0.30:
            bond_change = 0.15
        new_hedge = hedge_val * (1 + bond_change)
        new_market = market_beta_val * (1 + drop)
        val_no = new_high + new_hedge + new_market
        total_pv = 0
        for put_info in puts_info:
            if put_info['expiry'] <= today:
                continue
            days_to_exp = (put_info['expiry'] - today).days
            T = days_to_exp / 365.0
            new_u = put_info['s0'] * (1 + drop)
            vol = 0.20 + abs(drop) * 0.8
            pv_share = black_scholes_put(new_u, put_info['strike'], T, r, vol)
            total_pv += pv_share * 100 * put_info['contracts']
        wealth_no_hedge.append(val_no)
        wealth_with_hedge.append(val_no + total_pv)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(drops * 100, wealth_no_hedge, label='Total Assets (No Puts)', color='red', linewidth=2)
    ax1.plot(drops * 100, wealth_with_hedge, label='Total Assets (With Puts)', color='blue', linewidth=2)
    ax1.fill_between(drops * 100, wealth_no_hedge, wealth_with_hedge, color='green', alpha=0.1, label='Protection')
    ax1.set_title('Total Asset Protection (Market Drop)')
    ax1.set_xlabel('Market Drop (%)')
    ax1.set_ylabel('Total Asset Value (USD)')
    ax1.grid(True, alpha=0.3)

    payouts = np.array(wealth_with_hedge) - np.array(wealth_no_hedge)
    ax2 = ax1.twinx()
    ax2.plot(drops * 100, payouts, label='Puts Payout Value', color='purple', linestyle='--', linewidth=2)
    ax2.set_ylabel('Puts Value (USD)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    try:
        idx = next(x for x, val in enumerate(payouts) if val > 1000)
        start_drop = drops[idx] * 100
        ax1.axvline(x=start_drop, color='orange', linestyle=':')
        ax1.text(start_drop - 2, min(wealth_no_hedge) + 5000, f'Starts ~{abs(start_drop):.0f}%', rotation=90, color='orange')
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig('output/total_asset_protection.png')
    plt.show()
    plt.close()
    print('Chart saved to output/total_asset_protection.png')
