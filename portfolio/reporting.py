import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import rcParams
from scipy.stats import norm
from tabulate import tabulate


rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


def color_signed(value, text):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return text
    if pd.isna(numeric) or numeric == 0:
        return text
    color = "#188038" if numeric > 0 else "#d93025"
    return f'<span style="color: {color}; font-weight: 600">{text}</span>'


def plot_stock_performance(tw_res, us_res):
    plt.figure(figsize=(14, 8))
    _process_stock_twr(tw_res, 'TW', plt)
    _process_stock_twr(us_res, 'US', plt)
    plt.title('Individual Stock Performance (Equity Return %)')
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
    option_pattern = re.compile(r'^[A-Z]{1,6}\d{6}[CP]\d{8}$')

    for sym in symbols:
        if option_pattern.match(str(sym)):
            continue
        sym_tx = df_tx[df_tx['Symbol'] == sym].copy()
        if sym_tx.empty:
            continue

        daily_quantity = sym_tx.groupby('Date')['Quantity'].sum()
        shares_series = daily_quantity.reindex(date_range).fillna(0).cumsum()
        daily_amount = pd.to_numeric(sym_tx.groupby('Date')['Amount'].sum(), errors='coerce')
        cash_series = daily_amount.reindex(date_range).fillna(0).cumsum()

        price_col = None
        if sym in price_df.columns:
            price_col = sym
        elif str(sym).split('.')[0] in price_df.columns:
            price_col = str(sym).split('.')[0]
        if price_col is None:
            continue

        combined_df = pd.DataFrame({'Shares': shares_series, 'Cash': cash_series, 'Price': price_df[price_col]})
        combined_df = combined_df.dropna(subset=['Price'])
        combined_df['Shares'] = combined_df['Shares'].ffill().fillna(0)
        combined_df['Cash'] = combined_df['Cash'].ffill().fillna(0)
        equity_series = (combined_df['Shares'] * combined_df['Price']) + combined_df['Cash']
        gross_buy = -pd.to_numeric(
            sym_tx.loc[pd.to_numeric(sym_tx['Amount'], errors='coerce') < 0, 'Amount'],
            errors='coerce',
        ).sum()
        if pd.isna(gross_buy) or gross_buy <= 0:
            continue

        try:
            perf = (equity_series / float(gross_buy)) * 100.0
            if perf.empty or perf.dropna().empty:
                continue
            non_zero = perf[perf != 0]
            if non_zero.empty:
                continue
            first_date = non_zero.index[0]
            plot_data = perf.loc[first_date:]
            current_shares = shares_series.iloc[-1]
            if abs(current_shares) <= 0.001:
                last_tx_date = pd.to_datetime(sym_tx['Date']).max()
                if pd.notna(last_tx_date):
                    plot_data = plot_data.loc[:pd.Timestamp(last_tx_date)]
                if plot_data.empty:
                    continue
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

    print('\n## Target Allocation And Rebalance Suggestion')
    total_pool_twd = total_pool_usd * usd_to_twd
    print(f'Rebalance capital pool (stocks only): {total_pool_twd:,.0f} TWD')

    rows = []
    for key, target_pct in targets.items():
        curr_usd = current_values_usd[key]
        curr_twd = curr_usd * usd_to_twd
        target_twd = total_pool_twd * target_pct
        diff_twd = target_twd - curr_twd
        curr_pct = (curr_twd / total_pool_twd) * 100 if total_pool_twd > 0 else 0
        action_text = f'+{diff_twd:,.0f}' if diff_twd > 0 else f'{diff_twd:,.0f}'
        action_str = color_signed(diff_twd, action_text)
        rows.append([key, f'{target_pct*100:.1f}%', f'{curr_pct:.1f}%', f'{curr_twd:,.0f}', f'{target_twd:,.0f}', action_str])

    headers = ['Asset', 'Target %', 'Current %', 'Current Amount (TWD)', 'Target Amount (TWD)', 'Suggested Action (TWD)']
    print(tabulate(rows, headers=headers, tablefmt='github', stralign='right'))


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
    df = portfolio_df.copy()
    put_contracts = []
    non_put_positions = []

    for _, row in df.iterrows():
        sym = str(row['Symbol'])
        val = float(row['Price_Total'])
        qty = float(row['Quantity_now'])
        if 'P' in sym and any(c.isdigit() for c in sym) and qty > 0:
            match = re.match(r'^([A-Z]+)(\d{6})P(\d{8})$', sym)
            if match or (('QQQ' in sym or 'TSM' in sym) and 'P' in sym):
                put_contracts.append(row)
                continue
        if qty != 0 and val > 0:
            non_put_positions.append({'sym': sym, 'value': val})

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
                'market_value': float(put['Price_Total']),
                's0': s0,
            })
        except Exception:
            continue
    if not puts_info:
        return

    def stress_assumption(symbol, drop):
        symbol = str(symbol).upper()
        if symbol in {'EDV', 'TLT', 'ZROZ'}:
            if drop <= -0.30:
                return 0.15, '長天期美債在崩跌情境 +15%'
            if drop <= -0.20:
                return 0.05, '長天期美債在熊市情境 +5%'
            return 0.0, '長天期美債不變'
        if symbol in {'TMF', 'UBT'}:
            if drop <= -0.30:
                return 0.30, '槓桿美債在崩跌情境 +30%'
            if drop <= -0.20:
                return 0.10, '槓桿美債在熊市情境 +10%'
            return 0.0, '槓桿美債不變'
        if symbol in {'QLD', '00631L'}:
            return max(drop * 2.0, -0.99), '2 倍股票曝險'
        if symbol == 'TQQQ':
            return max(drop * 3.0, -0.99), '3 倍股票曝險'
        if symbol in {'TSLA'}:
            return max(drop * 1.5, -0.99), '高 Beta 股票 1.5 倍'
        if symbol in {'UNH'}:
            return drop * 0.8, '防禦型股票 0.8 倍'
        return drop, '一般股票曝險'

    for put_info in puts_info:
        put_info['notional'] = put_info['s0'] * 100 * put_info['contracts']

    today = datetime.now()
    r = 0.045
    base_non_put_value = sum(pos['value'] for pos in non_put_positions)
    current_put_market_value = sum(put_info['market_value'] for put_info in puts_info)
    scenarios = [
        {'drop': 1.00, 'desc': '翻倍上漲'},
        {'drop': 0.90, 'desc': '大幅上漲'},
        {'drop': 0.80, 'desc': '強勢上漲'},
        {'drop': 0.70, 'desc': '明顯上漲'},
        {'drop': 0.60, 'desc': '延續上漲'},
        {'drop': 0.50, 'desc': '大漲'},
        {'drop': 0.40, 'desc': '強漲'},
        {'drop': 0.30, 'desc': '上漲'},
        {'drop': 0.20, 'desc': '溫和上漲'},
        {'drop': 0.10, 'desc': '小漲'},
        {'drop': 0.0, 'desc': '目前'},
        {'drop': -0.10, 'desc': '修正'},
        {'drop': -0.20, 'desc': '熊市'},
        {'drop': -0.30, 'desc': '崩跌'},
        {'drop': -0.40, 'desc': '危機'},
        {'drop': -0.50, 'desc': '重挫'},
        {'drop': -0.60, 'desc': '深度重挫'},
        {'drop': -0.70, 'desc': '極端崩跌'},
        {'drop': -0.80, 'desc': '系統性危機'},
        {'drop': -0.90, 'desc': '近乎歸零'},
        {'drop': -1.00, 'desc': '歸零壓力'},
    ]

    print('\n## Put 避險保護分析')
    print('這是全投組壓力測試：先依資產類型規則衝擊目前所有非 Put 持倉，再重估現有 Put 價值並加回總資產。')
    for put_info in puts_info:
        print(f"Put：{put_info['sym']}，履約價 ${put_info['strike']}，到期日 {put_info['expiry'].strftime('%Y-%m-%d')}（標的：{put_info['underlying']} @ ${put_info['s0']:.2f}）")
    print(f'目前非 Put 資產：${base_non_put_value:,.0f} USD')
    print(f'目前 Put 市值：${current_put_market_value:,.0f} USD')
    print('| Put | 標的 | 目前市值 | Put 名目金額 |')
    print('| :--- | :--- | ---: | ---: |')
    for put_info in puts_info:
        print(f"| {put_info['sym']} | {put_info['underlying']} | ${put_info['market_value']:,.0f} | ${put_info['notional']:,.0f} |")

    print('| 持倉 | 目前市值 | 壓力測試規則 |')
    print('| :--- | ---: | :--- |')
    for pos in sorted(non_put_positions, key=lambda item: item['value'], reverse=True):
        _, rule = stress_assumption(pos['sym'], -0.20)
        print(f"| {pos['sym']} | ${pos['value']:,.0f} | {rule} |")

    print('| 情境 | 基準漲跌幅 | 總資產（未避險） | 總資產（含 Put） | Put 總價值 | 保護效果 |')
    print('| :--- | :--- | :--- | :--- | :--- | :--- |')

    drops = np.linspace(1.00, -1.00, 200)
    wealth_no_hedge = []
    wealth_with_hedge = []

    for scenario in scenarios:
        drop = scenario['drop']
        new_total_no_put = sum(pos['value'] * (1 + stress_assumption(pos['sym'], drop)[0]) for pos in non_put_positions)
        total_put_val = 0
        for put_info in puts_info:
            if put_info['expiry'] <= today:
                continue
            if drop == 0:
                total_put_val += put_info['market_value']
            else:
                days_to_exp = (put_info['expiry'] - today).days
                T = days_to_exp / 365.0
                new_u = put_info['s0'] * (1 + drop)
                vol = 0.20 + max(-drop, 0) * 0.8
                put_val_share = black_scholes_put(new_u, put_info['strike'], T, r, vol)
                total_put_val += put_val_share * 100 * put_info['contracts']
        new_total_with_put = new_total_no_put + total_put_val
        diff = new_total_with_put - new_total_no_put
        protection = color_signed(diff, f"+${diff:,.0f}") if diff > 0 else f"${diff:,.0f}"
        print(f"| {scenario['desc']} | {drop*100:.0f}% | ${new_total_no_put:,.0f} | **${new_total_with_put:,.0f}** | ${total_put_val:,.0f} | {protection} |")

    for drop in drops:
        val_no = sum(pos['value'] * (1 + stress_assumption(pos['sym'], drop)[0]) for pos in non_put_positions)
        total_pv = 0
        for put_info in puts_info:
            if put_info['expiry'] <= today:
                continue
            if drop == 0:
                total_pv += put_info['market_value']
            else:
                days_to_exp = (put_info['expiry'] - today).days
                T = days_to_exp / 365.0
                new_u = put_info['s0'] * (1 + drop)
                vol = 0.20 + max(-drop, 0) * 0.8
                pv_share = black_scholes_put(new_u, put_info['strike'], T, r, vol)
                total_pv += pv_share * 100 * put_info['contracts']
        wealth_no_hedge.append(val_no)
        wealth_with_hedge.append(val_no + total_pv)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(drops * 100, wealth_no_hedge, label='全投組壓力測試（不含 Put）', color='red', linewidth=2)
    ax1.plot(drops * 100, wealth_with_hedge, label='全投組壓力測試（含 Put）', color='blue', linewidth=2)
    ax1.fill_between(drops * 100, wealth_no_hedge, wealth_with_hedge, color='green', alpha=0.1, label='保護效果')
    ax1.set_title('Put 避險保護全投組壓力測試')
    ax1.set_xlabel('基準股票漲跌幅 (%)')
    ax1.set_ylabel('總資產價值 (USD)')
    ax1.grid(True, alpha=0.3)

    payouts = np.array(wealth_with_hedge) - np.array(wealth_no_hedge)
    ax2 = ax1.twinx()
    ax2.plot(drops * 100, payouts, label='Put 保護價值', color='purple', linestyle='--', linewidth=2)
    ax2.set_ylabel('Put 價值 (USD)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    try:
        idx = next(x for x, val in enumerate(payouts) if val > 1000)
        start_drop = drops[idx] * 100
        ax1.axvline(x=start_drop, color='orange', linestyle=':')
        ax1.text(start_drop - 2, min(wealth_no_hedge) + 5000, f'約 {abs(start_drop):.0f}% 開始', rotation=90, color='orange')
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig('output/total_asset_protection.png')
    plt.show()
    plt.close()
    print('圖表已儲存至 output/total_asset_protection.png')
