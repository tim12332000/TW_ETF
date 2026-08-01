import pandas as pd
import sys

import numpy as np
import yfinance as yf
import os
import matplotlib.pyplot as plt

from . import (
    DualLogger,
    align_fx_series,
    analyze_put_protection,
    build_cash_ledgers,
    build_option_history_series,
    calc_risk_metrics_from_twr,
    calculate_total_pnl_for_closed_position,
    calculate_twr_series,
    clean_currency,
    fix_share_sign,
    convert_cashflows_to_twd,
    get_cache_stats,
    get_cached_data,
    simulate_stock_full,
    get_current_price_yf,
    get_daily_price,
    get_latest_available_price,
    get_twd_to_usd_rate,
    get_usd_twd_history,
    plot_stock_performance,
    print_rebalance_recommendation,
    process_tw_portfolio,
    process_us_portfolio,
    resolve_market_price,
    xirr,
)

# Set CWD to script directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hashlib
from matplotlib import rcParams
from tabulate import tabulate

# Global Counters for logging
CALIBRATIONS = 0

REPORT_CHARTS = [
    ('Asset Trend', 'asset_trend.png'),
    ('Daily Asset Allocation', 'asset_allocation_monthly.png'),
    ('Funding Ratio', 'funding_ratio.png'),
    ('Portfolio TWR', 'twr_chart.png'),
    ('Cumulative Return Comparison', 'cumulative_return_comparison.png'),
    ('Portfolio vs Benchmark USD', 'portfolio_vs_benchmark_usd.png'),
    ('Drawdown Underwater', 'drawdown_underwater.png'),
    ('Cashflow Drawdown Comparison', 'cashflow_drawdown_comparison.png'),
    ('Cashflow Drawdown Spread vs QQQ', 'cashflow_drawdown_spread_vs_qqq.png'),
    ('Asset Pie Chart', 'asset_pie_chart.png'),
    ('Monthly Investment', 'monthly_investment.png'),
    ('Stock Performance', 'stock_performance.png'),
    ('代理避險情境保護率', 'proxy_hedge_coverage.png'),
    ('Put 避險保護全投組壓力測試', 'total_asset_protection.png'),
]

# =============================================================================
# 全域設定：設定中文字型與正確顯示負號
# =============================================================================
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
if not os.path.exists('output'):
    os.makedirs('output')



# =============================================================================
# 處理台股資料 (轉換為 USD 計價)
def _record_calibrations(count):
    global CALIBRATIONS
    CALIBRATIONS += count


def print_chart_gallery():
    print("\n## Charts")
    printed = False
    for title, filename in REPORT_CHARTS:
        chart_path = os.path.join('output', filename)
        if os.path.exists(chart_path):
            print(f"\n### {title}")
            print(f"![{title}]({filename})")
            printed = True
    if not printed:
        print("No charts were generated.")


def print_metric_list(title, rows):
    print(f"\n## {title}\n")
    for label, value in rows:
        print(f"- **{label}**：{value}")


def color_signed(value, text):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return text
    if pd.isna(numeric) or numeric == 0:
        return text
    color = "#188038" if numeric > 0 else "#d93025"
    return f'<span style="color: {color}; font-weight: 600">{text}</span>'


def format_number(value, suffix="", decimals=2, color=False, force_sign=False):
    if pd.isna(value):
        return ""
    sign = "+" if force_sign and value > 0 else ""
    text = f"{sign}{value:,.{decimals}f}{suffix}"
    return color_signed(value, text) if color else text


def build_symbol_value_history(result, date_index, price_data, fx_series=None):
    df = result['df'].copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    holdings = df.pivot_table(index='Date', columns='Symbol', values='Quantity', aggfunc='sum')
    holdings = holdings.reindex(date_index, fill_value=0).fillna(0).cumsum()
    prices = price_data.reindex(date_index).ffill().bfill()
    values = holdings.reindex(columns=prices.columns, fill_value=0) * prices
    if fx_series is not None:
        values = values.mul(align_fx_series(values.index, fx_series), axis=0)
    return values.clip(lower=0)


def plot_monthly_asset_allocation(tw_result, us_result, date_index, usd_twd_series):
    tw_values = build_symbol_value_history(
        tw_result,
        date_index,
        tw_result.get('price_data_twd', tw_result['price_data']),
    )
    us_values = build_symbol_value_history(
        us_result,
        date_index,
        us_result['price_data'],
        fx_series=usd_twd_series,
    )
    values = pd.concat([tw_values, us_values], axis=1).fillna(0)
    values = values.loc[:, values.max() > 0]
    if values.empty:
        return

    daily_values = values.loc[values.sum(axis=1) > 0]
    if daily_values.empty:
        return

    daily_alloc = daily_values.div(daily_values.sum(axis=1), axis=0) * 100
    max_alloc = daily_alloc.max().sort_values(ascending=False)
    major_symbols = max_alloc[max_alloc >= 1.0].index.tolist()
    if len(major_symbols) > 12:
        major_symbols = max_alloc.head(12).index.tolist()
    daily_alloc = daily_alloc[major_symbols].copy()
    other = 100 - daily_alloc.sum(axis=1)
    if (other > 0.5).any():
        daily_alloc['Other'] = other.clip(lower=0)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.get_cmap('tab20').colors
    ax.stackplot(
        daily_alloc.index,
        [daily_alloc[col].values for col in daily_alloc.columns],
        labels=daily_alloc.columns,
        colors=colors[:len(daily_alloc.columns)],
        alpha=0.9,
    )
    ax.set_title('Daily Asset Allocation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Allocation (%)')
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig('output/asset_allocation_monthly.png')
    plt.show()
    plt.close()


def _weighted_exposure(value_history, weights):
    exposure = pd.Series(0.0, index=value_history.index)
    for symbol, weight in weights.items():
        if symbol in value_history.columns:
            exposure = exposure.add(value_history[symbol].fillna(0) * weight, fill_value=0)
    return exposure


def _put_proxy_inputs(result, date_index, put_symbol, proxy_symbol):
    df = result['df'].copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    qty = (
        df.loc[df['Symbol'] == put_symbol]
        .groupby('Date')['Quantity']
        .sum()
        .reindex(date_index, fill_value=0)
        .fillna(0)
        .cumsum()
    )
    contracts = (qty / 100.0).clip(lower=0)
    if contracts.max() <= 0:
        return contracts, pd.Series(np.nan, index=date_index), 0.0

    proxy_px = get_daily_price(proxy_symbol, date_index.min(), date_index.max() + pd.Timedelta(days=1), is_tw=False)
    if isinstance(proxy_px, pd.DataFrame):
        if proxy_symbol in proxy_px.columns:
            proxy_px = proxy_px[proxy_symbol]
        else:
            proxy_px = proxy_px.iloc[:, 0]
    proxy_px = pd.Series(proxy_px).copy()
    if hasattr(proxy_px.index, 'tz') and proxy_px.index.tz is not None:
        proxy_px.index = proxy_px.index.tz_localize(None)
    proxy_px.index = pd.to_datetime(proxy_px.index).normalize()
    proxy_px = proxy_px.reindex(date_index).ffill().bfill()

    put_rows = df.loc[df['Symbol'] == put_symbol].copy()
    put_rows['Amount'] = pd.to_numeric(put_rows['Amount'], errors='coerce')
    premium_paid = -put_rows.loc[put_rows['Amount'] < 0, 'Amount'].sum()
    return contracts, proxy_px, float(premium_paid)


def _put_entry_proxy_price(result, put_symbol, proxy_px):
    df = result['df'].copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    rows = df.loc[df['Symbol'] == put_symbol].copy()
    if rows.empty:
        return None, np.nan

    rows['Amount'] = pd.to_numeric(rows['Amount'], errors='coerce')
    buy_rows = rows.loc[rows['Amount'] < 0].sort_values('Date')
    if buy_rows.empty:
        buy_rows = rows.loc[pd.to_numeric(rows['Quantity'], errors='coerce') > 0].sort_values('Date')
    if buy_rows.empty:
        return None, np.nan

    buy_date = pd.Timestamp(buy_rows.iloc[0]['Date']).normalize()
    px = proxy_px.reindex(proxy_px.index.union([buy_date])).sort_index().ffill().bfill().loc[buy_date]
    return buy_date, float(px)


def _put_entry_cost(result, put_symbol):
    df = result['df'].copy()
    rows = df.loc[df['Symbol'] == put_symbol].copy()
    if rows.empty:
        return np.nan
    rows['Amount'] = pd.to_numeric(rows['Amount'], errors='coerce')
    return float(-rows.loc[rows['Amount'] < 0, 'Amount'].sum())


def _current_put_symbol(result, proxy_symbol, fallback_symbol):
    df = result['df'].copy()
    qty = pd.to_numeric(df.get('Quantity'), errors='coerce').fillna(0)
    symbols = df.get('Symbol').astype(str)
    candidates = symbols[
        symbols.str.startswith(proxy_symbol)
        & symbols.str.contains('P', regex=False)
    ].unique()
    open_symbols = []
    for symbol in candidates:
        symbol_qty = qty.loc[symbols == symbol].sum()
        if symbol_qty > 0:
            open_symbols.append(symbol)
    if not open_symbols:
        return fallback_symbol
    return max(open_symbols, key=lambda symbol: _parse_put_expiry(symbol))


def _scenario_protection_rate(contracts, proxy_px, strike, exposure, scenario):
    scenario_proxy_px = proxy_px * (1 + scenario)
    put_payout = (strike - scenario_proxy_px).clip(lower=0) * 100.0 * contracts
    pool_loss = exposure * abs(scenario)
    return (put_payout / pool_loss.replace(0, np.nan)) * 100.0


def _format_usd(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def _format_rate(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def _parse_put_expiry(occ_symbol):
    date_part = occ_symbol[-15:-9]
    return pd.to_datetime(date_part, format='%y%m%d')


def _parse_put_strike(occ_symbol):
    return float(str(occ_symbol)[-8:]) / 1000.0


def _recommended_put_expiry(proxy_symbol, today):
    fallback = (today + pd.DateOffset(months=12)).normalize()
    try:
        expiries = [pd.Timestamp(exp) for exp in yf.Ticker(proxy_symbol).options]
    except Exception:
        expiries = []
    candidates = [
        exp for exp in expiries
        if 270 <= (exp.normalize() - today.normalize()).days <= 395
    ]
    if not candidates:
        return fallback
    return min(candidates, key=lambda exp: abs((exp.normalize() - today.normalize()).days - 365))


def _option_buy_price(row):
    ask = float(row.get('ask') or 0)
    bid = float(row.get('bid') or 0)
    last = float(row.get('lastPrice') or 0)
    if ask > 0:
        return ask, 'ask'
    if bid > 0 and last > 0:
        return max(last, bid), 'last/bid'
    if last > 0:
        return last, 'last'
    return np.nan, 'N/A'


def _find_reset_put_hedge(proxy_symbol, expiry, proxy_px, contracts, exposure, target_moneyness=0.50, protection_scenario=-0.90):
    target_strike = proxy_px * target_moneyness
    try:
        chain = yf.Ticker(proxy_symbol).option_chain(expiry.strftime('%Y-%m-%d')).puts.copy()
    except Exception:
        return None

    candidates = []
    for _, row in chain.iterrows():
        strike = float(row['strike'])
        buy_price, price_basis = _option_buy_price(row)
        if pd.isna(buy_price) or buy_price <= 0:
            continue

        reset_contracts = max(int(np.ceil(contracts)), 1)
        scenario_px = proxy_px * (1 + protection_scenario)
        payout = max(strike - scenario_px, 0) * 100.0 * reset_contracts
        scenario_loss = exposure * abs(protection_scenario)
        protection_rate = (payout / scenario_loss) * 100.0 if scenario_loss > 0 else np.nan
        gross_cost = reset_contracts * buy_price * 100.0
        candidates.append({
            'strike': strike,
            'target_strike': target_strike,
            'moneyness': (strike / proxy_px) * 100.0 if proxy_px > 0 else np.nan,
            'contracts': reset_contracts,
            'buy_price': buy_price,
            'price_basis': price_basis,
            'gross_cost': gross_cost,
            'protection_rate': protection_rate,
            'scenario_px': scenario_px,
        })

    if not candidates:
        return None
    return min(candidates, key=lambda item: (abs(item['strike'] - target_strike), item['gross_cost']))


def _current_put_market_value(put_symbol, contracts):
    if contracts <= 0:
        return 0.0
    price = get_current_price_yf(put_symbol, is_tw=False, fallback_price=0)
    return float(price or 0) * 100.0 * contracts


def plot_proxy_hedge_coverage(tw_result, us_result, date_index, final_portfolio_value_us, invested_capital_us):
    us_values = build_symbol_value_history(us_result, date_index, us_result['price_data'])
    tw_values = build_symbol_value_history(tw_result, date_index, tw_result['price_data'])

    us_weights = {
        'SPYM': 1.0,
        'SPLG': 1.0,
        'VOO': 1.0,
        'QLD': 2.0,
        'TQQQ': 3.0,
        'TSLA': 1.5,
        'UNH': 0.8,
    }
    tw_weights = {
        '0050': 1.0,
        '006208': 1.0,
        '2330': 1.0,
        '2376': 1.0,
        '2884': 1.0,
        '0056': 1.0,
        '00646': 1.0,
        '00631L': 2.0,
    }

    us_exposure = _weighted_exposure(us_values, us_weights)
    tw_exposure = _weighted_exposure(tw_values, tw_weights)
    qqq_put_symbol = _current_put_symbol(us_result, 'QQQ', 'QQQ270115P00350000')
    tsm_put_symbol = _current_put_symbol(us_result, 'TSM', 'TSM270115P00200000')
    qqq_strike = _parse_put_strike(qqq_put_symbol)
    tsm_strike = _parse_put_strike(tsm_put_symbol)
    qqq_contracts, qqq_px, qqq_premium = _put_proxy_inputs(us_result, date_index, qqq_put_symbol, 'QQQ')
    tsm_contracts, tsm_px, tsm_premium = _put_proxy_inputs(us_result, date_index, tsm_put_symbol, 'TSM')
    scenarios = [-0.30, -0.50, -0.90]

    plot_cols = {}
    for scenario in scenarios:
        label_pct = f"{scenario:.0%}"
        plot_cols[f'QQQ Put / 美股 {label_pct}'] = _scenario_protection_rate(qqq_contracts, qqq_px, qqq_strike, us_exposure, scenario)
        plot_cols[f'TSM Put / 台股 {label_pct}'] = _scenario_protection_rate(tsm_contracts, tsm_px, tsm_strike, tw_exposure, scenario)
    plot_df = pd.DataFrame(plot_cols).dropna(how='all')
    plot_df = plot_df.loc[(plot_df > 0).any(axis=1)]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(ax=ax, linewidth=2)
    ax.axhline(30, color='orange', linestyle=':', linewidth=1, label='參考線 30%')
    ax.axhline(50, color='green', linestyle=':', linewidth=1, label='參考線 50%')
    ax.set_title('代理避險情境保護率')
    ax.set_xlabel('Date')
    ax.set_ylabel('Put 預估 payout / 資產池情境損失 (%)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('output/proxy_hedge_coverage.png')
    plt.show()
    plt.close()

    latest_us_exposure = float(us_exposure.replace(0, np.nan).dropna().iloc[-1])
    latest_tw_exposure = float(tw_exposure.replace(0, np.nan).dropna().iloc[-1])
    latest_q = float(qqq_px.dropna().iloc[-1])
    latest_t = float(tsm_px.dropna().iloc[-1])
    latest_q_contracts = float(qqq_contracts.iloc[-1])
    latest_t_contracts = float(tsm_contracts.iloc[-1])
    qqq_current_put_value = _current_put_market_value(qqq_put_symbol, latest_q_contracts)
    tsm_current_put_value = _current_put_market_value(tsm_put_symbol, latest_t_contracts)
    qqq_buy_date, qqq_buy_proxy_px = _put_entry_proxy_price(us_result, qqq_put_symbol, qqq_px)
    tsm_buy_date, tsm_buy_proxy_px = _put_entry_proxy_price(us_result, tsm_put_symbol, tsm_px)
    qqq_entry_cost = _put_entry_cost(us_result, qqq_put_symbol)
    tsm_entry_cost = _put_entry_cost(us_result, tsm_put_symbol)

    def current_rate(proxy_px, strike, contracts, exposure, scenario):
        payout = max(strike - proxy_px * (1 + scenario), 0) * 100.0 * contracts
        loss = exposure * abs(scenario)
        return (payout / loss) * 100.0 if loss > 0 else np.nan

    print('\n## 代理避險情境保護率')
    print('QQQ Put 對應美股風險曝險；TSM Put 對應台股風險曝險。保護率用履約價計算：Put payout / 資產池情境損失。')
    scenario_headers = ' | '.join(f'{s:.0%} 保護率' for s in scenarios)
    scenario_align = ' | '.join('---:' for _ in scenarios)
    print(f'| Proxy Put | 對應資產池 | 保費成本率 | {scenario_headers} |')
    print(f'| :--- | :--- | ---: | {scenario_align} |')
    q_cost_rate = (qqq_premium / latest_us_exposure) * 100 if latest_us_exposure > 0 else np.nan
    t_cost_rate = (tsm_premium / latest_tw_exposure) * 100 if latest_tw_exposure > 0 else np.nan
    q_rates = [current_rate(latest_q, qqq_strike, latest_q_contracts, latest_us_exposure, s) for s in scenarios]
    t_rates = [current_rate(latest_t, tsm_strike, latest_t_contracts, latest_tw_exposure, s) for s in scenarios]
    q_cells = ' | '.join(f'{rate:.1f}%' for rate in q_rates)
    t_cells = ' | '.join(f'{rate:.1f}%' for rate in t_rates)
    print(f"| QQQ Put | 美股風險曝險 | {q_cost_rate:.2f}% | {q_cells} |")
    print(f"| TSM Put | 台股風險曝險 | {t_cost_rate:.2f}% | {t_cells} |")

    total_put_premium = qqq_premium + tsm_premium
    total_put_market_value = qqq_current_put_value + tsm_current_put_value
    total_profit_us = final_portfolio_value_us - invested_capital_us
    print('\n### Put 成本紀錄')
    print('保費是成本紀錄；重設組合用履約價 / 今日標的價約 50% 的災難保險規則，並用即時 option chain 估算成本。')
    print('| 項目 | 數值 |')
    print('| :--- | ---: |')
    print(f"| 投組市值 | {_format_usd(final_portfolio_value_us)} |")
    print(f"| 累積獲利 | {_format_usd(total_profit_us)} |")
    print(f"| 目前 Put 已投入保費 | {_format_usd(total_put_premium)} |")
    print(f"| 目前 Put 市值（可抵扣重設成本） | {_format_usd(total_put_market_value)} |")
    print(f"| Put 已投入保費 / 投組市值 | {(total_put_premium / final_portfolio_value_us) * 100:.2f}% |")

    q_rate_map = dict(zip(scenarios, q_rates))
    t_rate_map = dict(zip(scenarios, t_rates))
    today = pd.Timestamp.today().normalize()
    qqq_reset_expiry = _recommended_put_expiry('QQQ', today)
    tsm_reset_expiry = _recommended_put_expiry('TSM', today)
    print('\n### Put 重設避險組合')
    print('數學規則：出清/抵扣現有 Put 後，重新建立履約價 / 今日標的價最接近 50% 的 Put；這是深價外災難保險，不保證 -50% 情境有固定保護率。')
    print('| Proxy Put | 對應資產池 | 原買入日 | 原標的價 | 原履約價 | 原履約價/原標的價 | 原 Put 成本 | 今日原 Put 市值 | 今日標的價 | 目前保護率 | 重設到期日 | 目標履約價 | 重設履約價 | 重設履約價/今日標的價 | 張數 | 重設 Put 現價 | 重設毛成本 | 淨重設成本 | 重設後 -90% 保護率 |')
    print('| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')

    reset_rows = [
        ('QQQ Put', '美股風險曝險', qqq_put_symbol, qqq_buy_date, qqq_buy_proxy_px, qqq_entry_cost, latest_q, latest_q_contracts, latest_us_exposure, q_rate_map, qqq_reset_expiry, qqq_current_put_value),
        ('TSM Put', '台股風險曝險', tsm_put_symbol, tsm_buy_date, tsm_buy_proxy_px, tsm_entry_cost, latest_t, latest_t_contracts, latest_tw_exposure, t_rate_map, tsm_reset_expiry, tsm_current_put_value),
    ]
    for name, pool, put_symbol, buy_date, buy_proxy_px, entry_cost, proxy_px, contracts, exposure, rate_map, reset_expiry, current_put_value in reset_rows:
        current_status = f"-50% {rate_map[-0.50]:.1f}%，-90% {rate_map[-0.90]:.1f}%"
        buy_date_text = buy_date.strftime('%Y-%m-%d') if buy_date is not None else 'N/A'
        original_strike = _parse_put_strike(put_symbol)
        original_moneyness = (original_strike / buy_proxy_px) * 100.0 if buy_proxy_px else np.nan
        reset_expiry_text = reset_expiry.strftime('%Y-%m-%d')
        reset = _find_reset_put_hedge(name.split()[0], reset_expiry, proxy_px, contracts, exposure)
        if reset is None:
            print(f"| {name} | {pool} | {buy_date_text} | {_format_usd(buy_proxy_px)} | {_format_usd(original_strike)} | {_format_rate(original_moneyness)} | {_format_usd(entry_cost)} | {_format_usd(current_put_value)} | {_format_usd(proxy_px)} | {current_status} | {reset_expiry_text} | {_format_usd(proxy_px * 0.50)} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        net_cost = reset['gross_cost'] - current_put_value
        print(
            f"| {name} | {pool} | {buy_date_text} | {_format_usd(buy_proxy_px)} | {_format_usd(original_strike)} | {_format_rate(original_moneyness)} | "
            f"{_format_usd(entry_cost)} | {_format_usd(current_put_value)} | {_format_usd(proxy_px)} | {current_status} | {reset_expiry_text} | "
            f"{_format_usd(reset['target_strike'])} | ${reset['strike']:,.0f} | {_format_rate(reset['moneyness'])} | {reset['contracts']} | "
            f"${reset['buy_price']:.2f} ({reset['price_basis']}) | "
            f"{_format_usd(reset['gross_cost'])} | {_format_usd(net_cost)} | {_format_rate(reset['protection_rate'])} |"
        )
    print('圖表已儲存至 output/proxy_hedge_coverage.png')


def process_tw_data():
    return process_tw_portfolio(
        clean_currency=clean_currency,
        build_cash_ledgers=build_cash_ledgers,
        fix_share_sign=fix_share_sign,
        get_usd_twd_history=get_usd_twd_history,
        align_fx_series=align_fx_series,
        get_daily_price=get_daily_price,
        get_latest_available_price=get_latest_available_price,
        get_current_price_yf=get_current_price_yf,
        calculate_total_pnl_for_closed_position=calculate_total_pnl_for_closed_position,
        record_calibrations=_record_calibrations,
    )


# =============================================================================
# èçç¾è¡è³æ (ä»¥ USD è¨å¹)
# =============================================================================
def process_us_data():
    return process_us_portfolio(
        clean_currency=clean_currency,
        build_cash_ledgers=build_cash_ledgers,
        fix_share_sign=fix_share_sign,
        get_daily_price=get_daily_price,
        build_option_history_series=build_option_history_series,
        resolve_market_price=resolve_market_price,
        get_latest_available_price=get_latest_available_price,
        calculate_total_pnl_for_closed_position=calculate_total_pnl_for_closed_position,
    )





def main():
    # =================================================================
    # Phase 1: 資料準備（所有計算）
    # =================================================================

    # --- 1-1. Logger 初始化 ---
    original_stdout = sys.stdout
    sys.stdout = DualLogger('output/report.md')

    # --- 1-2. 匯率 + TW/US 原始資料 ---
    tw_result = process_tw_data()
    us_result = process_us_data()

    # --- 1-3. 合併投組市值、現金流、投入資金 ---
    date_index = tw_result['portfolio_value'].index.union(us_result['portfolio_value'].index).sort_values()
    usd_twd_series = get_usd_twd_history(date_index.min(), date_index.max())
    latest_usd_twd = float(usd_twd_series.iloc[-1])
    twd_to_usd = 1 / latest_usd_twd
    usd_to_twd = latest_usd_twd
    portfolio_value_tw = tw_result['portfolio_value'].reindex(date_index, method='ffill').fillna(0)
    portfolio_value_us = us_result['portfolio_value'].reindex(date_index, method='ffill').fillna(0)
    combined_portfolio_value_us = portfolio_value_tw + portfolio_value_us

    combined_external_cashflows = tw_result['external_cashflows'] + us_result['external_cashflows']
    combined_external_cashflows_twd = convert_cashflows_to_twd(combined_external_cashflows, usd_twd_series)
    # The portfolio tracks securities only and does not retain an account cash
    # balance. Treat every trade/dividend amount as an external flow so sales
    # and subsequent purchases cannot create artificial TWR returns.
    combined_transaction_cashflows = tw_result['transaction_cashflows'] + us_result['transaction_cashflows']
    combined_transaction_cashflows_twd = convert_cashflows_to_twd(combined_transaction_cashflows, usd_twd_series)
    total_investment_us = tw_result['total_investment'] + us_result['total_investment']
    invested_capital_us = tw_result['invested_capital'] + us_result['invested_capital']
    final_portfolio_value_us = combined_portfolio_value_us.iloc[-1]
    total_profit_us = final_portfolio_value_us - invested_capital_us
    total_profit_pct_us = (total_profit_us / invested_capital_us) * 100 if invested_capital_us != 0 else 0

    # TWD 版本
    combined_portfolio_value_twd = combined_portfolio_value_us * align_fx_series(date_index, usd_twd_series)
    total_investment_twd = -sum(amount for _, amount in combined_external_cashflows_twd if amount < 0)
    invested_capital_twd = total_investment_twd
    final_portfolio_value_twd = combined_portfolio_value_twd.iloc[-1]
    total_profit_twd = final_portfolio_value_twd - invested_capital_twd
    total_profit_pct_twd = (total_profit_twd / invested_capital_twd) * 100 if invested_capital_twd != 0 else 0

    # --- 1-4. transactions_df + 累積投入資金線（提前建立，後續多張圖需要） ---
    transactions_df = pd.concat([tw_result['df'], us_result['df']])
    transactions_df['Date'] = pd.to_datetime(transactions_df['Date']).dt.normalize()
    cf_df = pd.DataFrame(combined_external_cashflows, columns=['Date', 'Amount']).sort_values('Date')
    if cf_df.empty:
        cf_df = pd.DataFrame({'Date': date_index, 'Amount': 0.0})
    daily_cf = (cf_df.groupby('Date')['Amount']
                   .sum()
                   .reindex(date_index, fill_value=0)
                   .cumsum())
    daily_invested_capital = (-daily_cf).clip(lower=0)
    cf_df_twd = pd.DataFrame(combined_external_cashflows_twd, columns=['Date', 'Amount']).sort_values('Date')
    if cf_df_twd.empty:
        cf_df_twd = pd.DataFrame({'Date': date_index, 'Amount': 0.0})
    daily_cf_twd = (cf_df_twd.groupby('Date')['Amount']
                       .sum()
                       .reindex(date_index, fill_value=0)
                       .cumsum())
    daily_invested_capital_twd = (-daily_cf_twd).clip(lower=0)

    # --- 1-5. 風險指標：以 cashflow-neutral TWR 路徑計算 ---
    twr_series = calculate_twr_series(combined_portfolio_value_us, combined_transaction_cashflows)
    twr_series_twd = calculate_twr_series(combined_portfolio_value_twd, combined_transaction_cashflows_twd)
    ann_vol_main, max_dd_main, sharpe_main, sortino_ratio, calmar_ratio = calc_risk_metrics_from_twr(
        twr_series,
        risk_free_rate=0.02
    )
    ann_vol_main_twd, max_dd_main_twd, sharpe_main_twd, sortino_ratio_twd, calmar_ratio_twd = calc_risk_metrics_from_twr(
        twr_series_twd,
        risk_free_rate=0.02
    )

    # --- 1-6. XIRR ---
    total_snapshot = tw_result['portfolio_snapshot'] + us_result['portfolio_snapshot']
    xirr_cashflows = combined_external_cashflows + [(pd.Timestamp.today(), total_snapshot)]
    combined_irr = None
    try:
        combined_irr = xirr(xirr_cashflows)
    except Exception as e:
        print("綜合 XIRR 計算失敗:", e)
    combined_irr_twd = None
    try:
        combined_irr_twd = xirr(combined_external_cashflows_twd + [(pd.Timestamp.today(), final_portfolio_value_twd)])
    except Exception as e:
        print("綜合 XIRR (TWD) 計算失敗:", e)

    # --- 1-7. 個股明細表處理 ---
    portfolio_df_combined = pd.concat([tw_result['portfolio_df'], us_result['portfolio_df']], ignore_index=True)
    portfolio_df_combined = portfolio_df_combined.fillna(0)
    portfolio_df_combined.rename(columns={
        'Price PnL': 'Price PnL(USD)',
        'Dividend': 'Dividend(USD)',
        'Total PnL': 'Total PnL(USD)',
    }, inplace=True)
    portfolio_df_combined['Price PnL(TWD)'] = portfolio_df_combined['Price PnL(USD)'] * latest_usd_twd
    portfolio_df_combined['Dividend(TWD)'] = np.where(
        portfolio_df_combined.get('Dividend_TWD', 0) != 0,
        portfolio_df_combined.get('Dividend_TWD', 0),
        portfolio_df_combined['Dividend(USD)'] * latest_usd_twd,
    )
    portfolio_df_combined['Total PnL(TWD)'] = portfolio_df_combined['Price PnL(TWD)'] + portfolio_df_combined['Dividend(TWD)']
    portfolio_df_combined['Price PnL(USD)'] = portfolio_df_combined['Price PnL(USD)'].apply(
        lambda x: format_number(float(x), color=True)
    )
    portfolio_df_combined['Price PnL(TWD)'] = portfolio_df_combined['Price PnL(TWD)'].apply(
        lambda x: format_number(float(x), color=True)
    )
    portfolio_df_combined['Dividend(USD)'] = portfolio_df_combined['Dividend(USD)'].apply(
        lambda x: format_number(float(x), color=True)
    )
    portfolio_df_combined['Dividend(TWD)'] = portfolio_df_combined['Dividend(TWD)'].apply(
        lambda x: format_number(float(x), color=True)
    )
    portfolio_df_combined['Total PnL(USD)'] = portfolio_df_combined['Total PnL(USD)'].apply(
        lambda x: format_number(float(x), color=True)
    )
    portfolio_df_combined['Total PnL(TWD)'] = portfolio_df_combined['Total PnL(TWD)'].apply(
        lambda x: format_number(float(x), color=True)
    )
    portfolio_df_combined['Total PnL(%)'] = portfolio_df_combined['Total PnL(%)'].apply(
        lambda x: format_number(float(x), suffix="%", color=True)
    )

    # 計算平均成本 (每股成本)
    portfolio_df_combined['AvgCost'] = portfolio_df_combined.apply(
        lambda r: r['Cost'] / r['Quantity_now'] if r['Quantity_now'] != 0 else np.nan,
        axis=1
    )
    portfolio_df_combined['AvgCost'] = portfolio_df_combined['AvgCost'].map(
        lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
    )

    # 計算持股佔比
    portfolio_df_combined['Alloc_Price_Total'] = pd.to_numeric(portfolio_df_combined['Price_Total'], errors='coerce').fillna(0)
    total_val_for_alloc = portfolio_df_combined.loc[portfolio_df_combined['Alloc_Price_Total'] > 0, 'Alloc_Price_Total'].sum()

    portfolio_df_combined['Alloc(%)'] = 0.0
    if total_val_for_alloc > 0:
        mask = portfolio_df_combined['Alloc_Price_Total'] > 0
        portfolio_df_combined.loc[mask, 'Alloc(%)'] = (portfolio_df_combined.loc[mask, 'Alloc_Price_Total'] / total_val_for_alloc) * 100

    portfolio_df_combined['Alloc(%)'] = portfolio_df_combined.apply(
        lambda row: f"{row['Alloc(%)']:.2f}% ({row['Price_Total'] * latest_usd_twd:,.0f})" if row['Alloc_Price_Total'] > 0 else "0.00% (0)",
        axis=1
    )

    portfolio_df_combined = portfolio_df_combined[
        [
            'Symbol', 'Name', 'Quantity_now', 'Price', 'AvgCost', 'Price_Total', 'Cost',
            'Price PnL(USD)', 'Price PnL(TWD)', 'Dividend(USD)', 'Dividend(TWD)',
            'Total PnL(USD)', 'Total PnL(TWD)', 'Total PnL(%)', 'Alloc(%)'
        ]
    ]

    # --- 1-8. Benchmark 模擬 ---
    COMPARE_TICKERS = ['SPY','QQQ','EWT']

    sim_portfolios = {}
    for tk in COMPARE_TICKERS:
        sim_portfolios[tk], _ = simulate_stock_full(combined_external_cashflows, ticker=tk)

    idx = combined_portfolio_value_us.index.copy()
    for p in sim_portfolios.values():
        idx = idx.union(p.index)
    idx = idx.sort_values()

    my_us = combined_portfolio_value_us.reindex(idx).ffill()
    sims  = {tk: p.reindex(idx).ffill() for tk, p in sim_portfolios.items()}

    # --- 1-9. TWR + Benchmark TWR ---

    bench_twr = {}
    bench_twr_twd = {}
    valid_idx = twr_series_twd[twr_series_twd != 0].index
    if not valid_idx.empty:
        start_date = valid_idx[0]
    else:
        start_date = twr_series_twd.index[0]

    for tk in COMPARE_TICKERS:
        try:
            def _fetch_bench():
                return yf.download(tk, start=start_date, end=None, progress=False, auto_adjust=True)

            key = f"bench_twr_{tk}_{start_date.date()}.pkl"
            _px = get_cached_data(key, _fetch_bench)

            if isinstance(_px, pd.DataFrame):
                if 'Close' in _px.columns:
                     _px = _px['Close']
                else:
                     _px = _px.iloc[:, 0]

            if isinstance(_px, pd.DataFrame):
                _px = _px.iloc[:, 0]

            _px.index = _px.index.tz_localize(None)
            _px = _px.reindex(twr_series_twd.index).ffill().bfill()

            start_val = _px.loc[start_date]
            if start_val > 0:
                 bench_twr[tk] = (_px / start_val - 1) * 100
                 bench_px_twd = _px * align_fx_series(_px.index, usd_twd_series)
                 bench_start_val_twd = bench_px_twd.loc[start_date]
                 if bench_start_val_twd > 0:
                     bench_twr_twd[tk] = (bench_px_twd / bench_start_val_twd - 1) * 100
        except Exception as e:
             print(f"Skipping benchmark TWR for {tk}: {e}")

    # --- 1-10. Benchmark 對照表計算 ---
    today = pd.Timestamp.today().normalize()
    base_cf = [(d, amt) for (d, amt) in combined_external_cashflows if d < today]
    base_cf_twd = [(d, amt) for (d, amt) in combined_external_cashflows_twd if d < today]

    def last_valid(series):
        return series.dropna().iloc[-1] if series.dropna().size else np.nan

    benchmark_rows = []

    # My Portfolio
    p_my = combined_portfolio_value_us
    ann_vol_my, max_dd_my, sharpe_my, _, _ = calc_risk_metrics_from_twr(twr_series_twd)

    benchmark_rows.append([
        'My Portfolio',
        final_portfolio_value_twd,
        final_portfolio_value_twd - invested_capital_twd,
        total_profit_pct_twd,
        combined_irr_twd * 100 if combined_irr_twd is not None else np.nan,
        ann_vol_my,
        max_dd_my,
        sharpe_my
    ])

    # Benchmarks
    for tk, p_raw in sims.items():
        p = p_raw.copy()
        final_us = last_valid(p)
        if np.isnan(final_us):
            print(f'[warning] {tk} 無可用資料，已略過')
            continue
        final_twd  = final_us * latest_usd_twd
        profit_twd = final_twd - invested_capital_twd
        profit_pct = (profit_twd / invested_capital_twd) * 100
        cf_sim = base_cf + [(today, final_us)]
        cf_sim_twd = base_cf_twd + [(today, final_twd)]
        try:
            sim_irr = xirr(cf_sim) * 100
        except Exception:
            sim_irr = np.nan
        try:
            sim_irr_twd = xirr(cf_sim_twd) * 100
        except Exception:
            sim_irr_twd = np.nan
        if tk in bench_twr_twd:
            sim_vol, max_dd, sim_sharpe, _, _ = calc_risk_metrics_from_twr(bench_twr_twd[tk])
        else:
            sim_vol, max_dd, sim_sharpe = np.nan, np.nan, np.nan
        benchmark_rows.append([
            tk, final_twd, profit_twd, profit_pct, sim_irr_twd, sim_vol, max_dd, sim_sharpe
        ])

    bench_headers = [
        'Asset', 'Final Value (TWD)', 'Profit (TWD)',
        'Profit %', 'XIRR %', 'AnnVol %', 'MaxDD %', 'Sharpe'
    ]
    bench_df = pd.DataFrame(benchmark_rows, columns=bench_headers)
    bench_display_df = bench_df.copy()
    bench_display_df['Final Value (TWD)'] = bench_display_df['Final Value (TWD)'].apply(
        lambda x: format_number(x)
    )
    bench_display_df['Profit (TWD)'] = bench_display_df['Profit (TWD)'].apply(
        lambda x: format_number(x, color=True)
    )
    bench_display_df['Profit %'] = bench_display_df['Profit %'].apply(
        lambda x: format_number(x, suffix="%", color=True)
    )
    bench_display_df['XIRR %'] = bench_display_df['XIRR %'].apply(
        lambda x: format_number(x, suffix="%", color=True)
    )
    bench_display_df['AnnVol %'] = bench_display_df['AnnVol %'].apply(
        lambda x: format_number(x, suffix="%")
    )
    bench_display_df['MaxDD %'] = bench_display_df['MaxDD %'].apply(
        lambda x: format_number(x, suffix="%")
    )
    bench_display_df['Sharpe'] = bench_display_df['Sharpe'].apply(
        lambda x: format_number(x)
    )

    # =================================================================
    # Phase 2: 文字報告（一次性全部印出）
    # =================================================================

    # --- 2-1. 綜合資產報告 (USD) ---
    usd_metrics = [
        ("累積外部投入金額", f"{total_investment_us:,.2f} USD"),
        ("實際淨投入資金", f"{invested_capital_us:,.2f} USD"),
        ("最終組合市值", f"{final_portfolio_value_us:,.2f} USD"),
        ("總獲利", format_number(total_profit_us, suffix=" USD", color=True)),
        ("總獲利百分比", format_number(total_profit_pct_us, suffix="%", color=True)),
        ("AnnVol", f"{ann_vol_main:.2f}%"),
        ("MaxDD", f"{max_dd_main:.2f}%"),
        ("Sharpe", f"{sharpe_main:.2f}"),
        ("Sortino Ratio", f"{sortino_ratio:.2f}"),
        ("Calmar Ratio", f"{calmar_ratio:.2f}"),
    ]
    if combined_irr is not None:
        usd_metrics.append(("綜合 XIRR", color_signed(combined_irr, f"{combined_irr:.2%}")))
    print_metric_list("綜合資產配置報告 (單位: USD)", usd_metrics)

    # --- 2-2. 綜合資產報告 (TWD) ---
    twd_metrics = [
        ("累積外部投入金額", f"{total_investment_twd:,.2f} TWD"),
        ("實際淨投入資金", f"{invested_capital_twd:,.2f} TWD"),
        ("最終組合市值", f"{final_portfolio_value_twd:,.2f} TWD"),
        ("總獲利", format_number(total_profit_twd, suffix=" TWD", color=True)),
        ("總獲利百分比", format_number(total_profit_pct_twd, suffix="%", color=True)),
        ("AnnVol", f"{ann_vol_main_twd:.2f}%"),
        ("MaxDD", f"{max_dd_main_twd:.2f}%"),
        ("Sharpe", f"{sharpe_main_twd:.2f}"),
        ("Sortino Ratio", f"{sortino_ratio_twd:.2f}"),
        ("Calmar Ratio", f"{calmar_ratio_twd:.2f}"),
    ]
    if combined_irr_twd is not None:
        twd_metrics.append(("綜合 XIRR", color_signed(combined_irr_twd, f"{combined_irr_twd:.2%}")))
    print_metric_list("綜合資產配置報告 (單位: TWD)", twd_metrics)

    # --- 2-3. 個股明細表 ---
    print("\n## 綜合投資組合股票明細 (TWD)")
    print(tabulate(portfolio_df_combined, headers='keys', tablefmt='github', showindex=False))

    # --- 2-4. 投組 vs Benchmark 總表 ---
    print("\n## 投組 vs. Benchmark 總表 (TWD)")
    print(tabulate(
        bench_display_df,
        headers='keys',
        tablefmt='github',
        showindex=False
    ))

    # --- 2-5. 目標配置與再平衡建議 ---
    print_rebalance_recommendation(portfolio_df_combined, usd_to_twd)

    # =================================================================
    # Phase 3: 圖表（由概觀到細節）
    # =================================================================

    # --- 3-1. 資產走勢圖 + 累積投入資金線（全局概覽） ---
    plt.figure(figsize=(10, 6))
    plt.plot(combined_portfolio_value_twd.index, combined_portfolio_value_twd.values, label='資產走勢圖')
    plt.plot(daily_invested_capital_twd.index, daily_invested_capital_twd.values, label='累積投入資金', linestyle='--')
    plt.title('資產走勢圖')
    plt.xlabel('日期')
    plt.ylabel('組合市值 (TWD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/asset_trend.png')
    plt.show()
    plt.close()

    # --- 3-1b. Monthly asset allocation ---
    plot_monthly_asset_allocation(tw_result, us_result, date_index, usd_twd_series)

    # --- 3-2. Funding Ratio ---
    _den = daily_invested_capital_twd.replace(0, np.nan)
    ratio = (combined_portfolio_value_twd / _den).dropna()

    plt.figure(figsize=(10, 5))
    plt.plot(ratio.index, ratio.values, label='Funding Ratio (資產/累積投入)')
    plt.axhline(1.0, linestyle='--', alpha=0.6, label='=1（打平）')
    plt.title('Funding Ratio（資產 ÷ 累積投入，TWD）')
    plt.xlabel('日期'); plt.ylabel('倍數')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('output/funding_ratio.png')
    plt.show()
    plt.close()

    # --- 3-3. 累積報酬率比較 (TWR) ---
    plt.figure(figsize=(12, 6))
    plt.plot(twr_series_twd.index, twr_series_twd, label='My Portfolio (TWR, TWD)', linewidth=2, color='blue')
    plt.legend()
    plt.title('Portfolio Cumulative TWR Return')
    plt.grid(True)
    plt.savefig('output/twr_chart.png')

    for tk, ser in bench_twr_twd.items():
        plt.plot(ser.index, ser, label=f'{tk}', alpha=0.7)

    plt.title('累積報酬率比較 (Time-Weighted Return)')
    plt.xlabel('日期')
    plt.ylabel('累積報酬 (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/cumulative_return_comparison.png')
    plt.show()
    plt.close()

    # --- 3-4. My Portfolio vs Benchmark (USD 絕對市值) ---
    plt.figure(figsize=(11, 6))
    plt.plot(my_us, label='My Portfolio', linewidth=2)
    for tk, p in sims.items():
        plt.plot(p, label=f'{tk} 模擬')
    plt.plot(daily_invested_capital.reindex(idx).ffill(), label='累積投入資金 (USD)', linestyle='--', linewidth=1.5)
    plt.title('My Portfolio vs. 多重 Benchmark (USD)')
    plt.xlabel('日期'); plt.ylabel('市值 / 指數 (USD)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('output/portfolio_vs_benchmark_usd.png')
    plt.show()
    plt.close()

    # --- 3-5. Drawdown 水下圖 ---
    wealth_index = combined_portfolio_value_twd / invested_capital_twd if invested_capital_twd != 0 else combined_portfolio_value_twd * np.nan
    running_max_dd = wealth_index.cummax()
    drawdown = (wealth_index - running_max_dd) / running_max_dd

    plt.figure(figsize=(10, 6))
    plt.fill_between(drawdown.index, drawdown * 100, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown * 100, label='Drawdown (%)')
    plt.title('最大回撤（Drawdown）水下圖')
    plt.xlabel('日期')
    plt.ylabel('回撤 (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/drawdown_underwater.png')
    plt.show()
    plt.close()

    # --- 3-5b. Same-Cashflow Drawdown vs QQQ ---
    qqq_sim = sims.get('QQQ')
    if qqq_sim is not None:
        qqq_drawdown = (qqq_sim / qqq_sim.cummax() - 1) * 100
        my_us_drawdown = (my_us / my_us.cummax() - 1) * 100

        plt.figure(figsize=(11, 6))
        plt.fill_between(my_us_drawdown.index, my_us_drawdown.values, 0, alpha=0.15, color='blue')
        plt.plot(my_us_drawdown.index, my_us_drawdown.values, label='My Portfolio', linewidth=2, color='blue')
        plt.plot(qqq_drawdown.index, qqq_drawdown.values, label='QQQ ??', linewidth=2, alpha=0.85, color='orange')
        plt.title('Same-Cashflow Drawdown vs. QQQ (USD)')
        plt.xlabel('??')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/cashflow_drawdown_comparison.png')
        plt.show()
        plt.close()

        drawdown_spread = my_us_drawdown - qqq_drawdown.reindex(my_us_drawdown.index).ffill()
        plt.figure(figsize=(11, 4.5))
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.fill_between(drawdown_spread.index, drawdown_spread.values, 0, where=(drawdown_spread.values >= 0), alpha=0.2, color='green')
        plt.fill_between(drawdown_spread.index, drawdown_spread.values, 0, where=(drawdown_spread.values < 0), alpha=0.2, color='red')
        plt.plot(drawdown_spread.index, drawdown_spread.values, color='black', linewidth=1.8, label='My Drawdown - QQQ Drawdown')
        plt.title('Drawdown Spread vs. QQQ (Positive = My Portfolio More Resilient)')
        plt.xlabel('??')
        plt.ylabel('Spread (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/cashflow_drawdown_spread_vs_qqq.png')
        plt.show()
        plt.close()

    # --- 3-6. 資產圓餅圖 ---
    combined_df_chart = portfolio_df_combined.dropna(subset=['Price_Total'])
    combined_df_chart = combined_df_chart[combined_df_chart['Price_Total'] > 0]
    combined_df_chart['Price_Total'] = pd.to_numeric(combined_df_chart['Price_Total'], errors='coerce')
    combined_df_chart['Price_Total_TWD'] = combined_df_chart['Price_Total'] * latest_usd_twd

    total_pie_twd = combined_df_chart['Price_Total_TWD'].sum()
    pie_labels = combined_df_chart.apply(
        lambda row: f"{row['Name']} {row['Price_Total_TWD']/total_pie_twd*100:.1f}% ({row['Price_Total_TWD']:,.0f})", axis=1
    )

    plt.figure(figsize=(10, 8))
    plt.pie(combined_df_chart['Price_Total_TWD'], labels=pie_labels, startangle=140)
    plt.title('資產圓餅圖')
    plt.axis('equal')
    plt.savefig('output/asset_pie_chart.png')
    plt.show()
    plt.close()

    # --- 3-7. 每月投入資產 ---
    daily_net = transactions_df.groupby('Date')['Amount'].sum()
    daily_injection = daily_net[daily_net < 0].abs()
    monthly_investment = daily_injection.groupby(daily_injection.index.to_period('M')).sum()
    monthly_investment.index = monthly_investment.index.to_timestamp()

    plt.figure(figsize=(10, 6))
    plt.bar(monthly_investment.index, monthly_investment.values, width=20)
    plt.title("每月投入資產 (USD)")
    plt.xlabel("月份")
    plt.ylabel("投入金額 (USD)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('output/monthly_investment.png')
    plt.show()
    plt.close()

    # --- 3-8. 個股績效 (TWR) ---
    plot_stock_performance(tw_result, us_result)

    # --- 3-8b. Proxy Put 覆蓋率 ---
    plot_proxy_hedge_coverage(
        tw_result,
        us_result,
        date_index,
        final_portfolio_value_us,
        invested_capital_us,
    )

    # --- 3-9. Put 保護力分析 ---
    analyze_put_protection(portfolio_df_combined)
    print_chart_gallery()

    # =================================================================
    # Phase 4: 執行摘要
    # =================================================================
    print("\n## Execution Summary")
    cache_stats = get_cache_stats()
    print(f"- Cache Loads: {cache_stats['loads']}")
    print(f"- Cache Misses (Network Fetches): {cache_stats['misses']}")
    print(f"- Price Calibrations Applied: {CALIBRATIONS}")

    # --- Cleanup: 關閉 Logger 並恢復 stdout ---
    if isinstance(sys.stdout, DualLogger):
        sys.stdout.close()
    sys.stdout = original_stdout


if __name__ == '__main__':
    main()

