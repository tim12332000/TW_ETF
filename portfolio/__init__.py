from .cache import get_cache_stats, get_cached_data
from .logging_utils import DualLogger
from .market_data import (
    align_fx_series,
    build_option_history_series,
    compute_risk_return,
    convert_cashflows_to_twd,
    get_current_price_yf,
    get_daily_price,
    get_latest_available_price,
    get_option_price,
    get_twd_to_usd_rate,
    get_usd_twd_history,
    parse_occ_symbol,
    resolve_market_price,
)
from .performance import calc_risk_metrics_from_twr, calculate_twr_series, twr_to_daily_returns, xirr, xnpv
from .positions import calculate_total_pnl_for_closed_position
from .reporting import analyze_put_protection, black_scholes_put, plot_stock_performance, print_rebalance_recommendation
from .transactions import build_cash_ledgers, clean_currency, convert_ticker, fix_share_sign
from .tw_portfolio import process_tw_data
from .us_portfolio import process_us_data

process_tw_portfolio = process_tw_data
process_us_portfolio = process_us_data
from .benchmarking import simulate_stock_full
