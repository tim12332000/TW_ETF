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
    sys.stdout = DualLogger('output/report.txt')

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
    twr_series = calculate_twr_series(combined_portfolio_value_us, combined_external_cashflows)
    twr_series_twd = calculate_twr_series(combined_portfolio_value_twd, combined_external_cashflows_twd)
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
    portfolio_df_combined.rename(columns={'Total PnL': 'Total PnL(USD)'}, inplace=True)
    portfolio_df_combined['Total PnL(TWD)'] = portfolio_df_combined['Total PnL(USD)'] * latest_usd_twd
    portfolio_df_combined['Total PnL(USD)'] = portfolio_df_combined['Total PnL(USD)'].apply(lambda x: f"{float(x):,.2f}")
    portfolio_df_combined['Total PnL(TWD)'] = portfolio_df_combined['Total PnL(TWD)'].apply(lambda x: f"{float(x):,.2f}")
    portfolio_df_combined['Total PnL(%)'] = portfolio_df_combined['Total PnL(%)'].apply(lambda x: f"{float(x):,.2f}%")

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
        ['Symbol', 'Name', 'Quantity_now', 'Price', 'AvgCost', 'Price_Total', 'Cost', 'Total PnL(USD)', 'Total PnL(TWD)', 'Total PnL(%)', 'Alloc(%)']
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

    # =================================================================
    # Phase 2: 文字報告（一次性全部印出）
    # =================================================================

    # --- 2-1. 綜合資產報告 (USD) ---
    print("\n=== 綜合資產配置報告 (單位: USD) ===")
    print(f"累積外部投入金額：{total_investment_us:,.2f} USD")
    print(f"實際淨投入資金：{invested_capital_us:,.2f} USD")
    print(f"最終組合市值：{final_portfolio_value_us:,.2f} USD")
    print(f"總獲利：{total_profit_us:,.2f} USD")
    print(f"總獲利百分比：{total_profit_pct_us:.2f}%")
    print(f"AnnVol：{ann_vol_main:.2f}%")
    print(f"MaxDD：{max_dd_main:.2f}%")
    print(f"Sharpe：{sharpe_main:.2f}")
    print(f"Sortino Ratio：{sortino_ratio:.2f}")
    print(f"Calmar Ratio：{calmar_ratio:.2f}")
    if combined_irr is not None:
        print(f"綜合 XIRR: {combined_irr:.2%}")

    # --- 2-2. 綜合資產報告 (TWD) ---
    print("\n=== 綜合資產配置報告 (單位: TWD) ===")
    print(f"累積外部投入金額：{total_investment_twd:,.2f} TWD")
    print(f"實際淨投入資金：{invested_capital_twd:,.2f} TWD")
    print(f"最終組合市值：{final_portfolio_value_twd:,.2f} TWD")
    print(f"總獲利：{total_profit_twd:,.2f} TWD")
    print(f"總獲利百分比：{total_profit_pct_twd:.2f}%")
    print(f"AnnVol：{ann_vol_main_twd:.2f}%")
    print(f"MaxDD：{max_dd_main_twd:.2f}%")
    print(f"Sharpe：{sharpe_main_twd:.2f}")
    print(f"Sortino Ratio：{sortino_ratio_twd:.2f}")
    print(f"Calmar Ratio：{calmar_ratio_twd:.2f}")
    if combined_irr_twd is not None:
        print(f"綜合 XIRR: {combined_irr_twd:.2%}")

    # --- 2-3. 個股明細表 ---
    print("\n=== 綜合投資組合股票明細 (TWD) ===")
    print(tabulate(portfolio_df_combined, headers='keys', tablefmt='psql', showindex=False))

    # --- 2-4. 投組 vs Benchmark 總表 ---
    print("\n=== 投組 vs. Benchmark 總表 (TWD) ===")
    print(tabulate(
        bench_df,
        headers='keys',
        tablefmt='psql',
        showindex=False,
        floatfmt='.2f'
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

    # --- 3-9. Put 保護力分析 ---
    analyze_put_protection(portfolio_df_combined)

    # =================================================================
    # Phase 4: 執行摘要
    # =================================================================
    print("\n==================================================")
    print("Execution Summary:")
    cache_stats = get_cache_stats()
    print(f"  - Cache Loads: {cache_stats['loads']}")
    print(f"  - Cache Misses (Network Fetches): {cache_stats['misses']}")
    print(f"  - Price Calibrations Applied: {CALIBRATIONS}")
    print("==================================================")

    # --- Cleanup: 關閉 Logger 並恢復 stdout ---
    if isinstance(sys.stdout, DualLogger):
        sys.stdout.close()
    sys.stdout = original_stdout


if __name__ == '__main__':
    main()

