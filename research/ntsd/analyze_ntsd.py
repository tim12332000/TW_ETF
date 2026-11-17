from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "research" / "ntsd"
START_DATE = "2013-01-01"
EXPENSE_RATIO = 0.0035
RISK_FREE_RATE = 0.02


@dataclass(frozen=True)
class SeriesStats:
    name: str
    start: str
    end: str
    total_return: float
    cagr: float
    ann_vol: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    worst_day: float
    best_day: float


def _download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if data.empty:
        raise RuntimeError("No market data downloaded from yfinance.")

    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.sort_index()


def _download_tbills(start: str, index: pd.DatetimeIndex) -> pd.Series:
    data = yf.download("^IRX", start=start, auto_adjust=False, progress=False)
    if data.empty or "Close" not in data:
        return pd.Series(0.0, index=index)
    rates = data["Close"]
    if isinstance(rates, pd.DataFrame):
        rates = rates.iloc[:, 0]
    rates.index = pd.to_datetime(rates.index).tz_localize(None).normalize()
    daily_rate = (pd.to_numeric(rates, errors="coerce") / 100.0 / 252.0).reindex(index).ffill().fillna(0.0)
    return daily_rate


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.dropna(how="all").pct_change(fill_method=None).dropna(how="all")


def _wealth_from_returns(returns: pd.Series) -> pd.Series:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    return (1.0 + clean).cumprod()


def _max_drawdown(wealth: pd.Series) -> float:
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def _stats(name: str, wealth: pd.Series, returns: pd.Series) -> SeriesStats:
    returns = returns.reindex(wealth.index).dropna()
    years = (wealth.index[-1] - wealth.index[0]).days / 365.25
    total_return = float(wealth.iloc[-1] - 1.0)
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(returns.std() * np.sqrt(252))
    daily_rf = (1.0 + RISK_FREE_RATE) ** (1.0 / 252.0) - 1.0
    excess = returns - daily_rf
    sharpe = float(np.sqrt(252) * excess.mean() / returns.std()) if returns.std() else np.nan
    downside = excess.clip(upper=0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else np.nan
    sortino = float(np.sqrt(252) * excess.mean() / downside_dev) if downside_dev else np.nan
    max_dd = _max_drawdown(wealth)
    calmar = float(cagr / abs(max_dd)) if max_dd else np.nan
    return SeriesStats(
        name=name,
        start=wealth.index[0].date().isoformat(),
        end=wealth.index[-1].date().isoformat(),
        total_return=total_return,
        cagr=cagr,
        ann_vol=ann_vol,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        worst_day=float(returns.min()),
        best_day=float(returns.max()),
    )


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_num(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def _stats_frame(stats: list[SeriesStats]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Strategy": item.name,
                "Start": item.start,
                "End": item.end,
                "Total Return": _format_pct(item.total_return),
                "CAGR": _format_pct(item.cagr),
                "AnnVol": _format_pct(item.ann_vol),
                "MaxDD": _format_pct(item.max_drawdown),
                "Sharpe": _format_num(item.sharpe),
                "Sortino": _format_num(item.sortino),
                "Calmar": _format_num(item.calmar),
                "Worst Day": _format_pct(item.worst_day),
                "Best Day": _format_pct(item.best_day),
            }
            for item in stats
        ]
    )


def _plot_wealth(wealth: pd.DataFrame, path: Path) -> None:
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 7))
    (wealth * 100).plot(ax=ax, linewidth=2)
    ax.set_title("NTSD Proxy Backtest: Growth of $100")
    ax.set_ylabel("Portfolio value")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_drawdown(wealth: pd.DataFrame, path: Path) -> None:
    drawdown = wealth / wealth.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(12, 6))
    (drawdown * 100).plot(ax=ax, linewidth=1.8)
    ax.set_title("NTSD Proxy Backtest: Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_key_comparison(wealth: pd.DataFrame, path: Path) -> None:
    selected = wealth[["NTSD financed proxy", "50% VXUS + 50% SSO"]].copy()
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(11, 6))
    (selected * 100).plot(ax=ax, linewidth=2.6)
    ax.set_title("Key Comparison: NTSD financed proxy vs 50% VXUS + 50% SSO")
    ax.set_ylabel("Growth of $100")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _build_report(
    stats_md: str,
    actual_stats_md: str,
    latest: pd.Series,
    synthetic_start: str,
    synthetic_end: str,
) -> str:
    latest_line = ", ".join(f"{k}: {v:.2f}" for k, v in latest.dropna().items())
    return f"""# NTSD ETF 分析與回測

產出日期：{pd.Timestamp.today().date().isoformat()}

## ETF 摘要

NTSD 是 WisdomTree Efficient U.S. Plus International Equity Fund，2026-03-19 上市，費用率 0.35%。官方描述是主動管理、模型式配置：以美國大型股作為核心，並透過已開發國際股指期貨取得國際股票曝險。這份回測用可交易 ETF 代理：

- 美國大型股：SPY
- 已開發市場股票：IEFA
- NTSD 90/60 proxy：90% SPY + 60% IEFA，每日再平衡，扣 0.35% 年費
- NTSD financed proxy：上述 90/60 proxy 再扣 50% 名目槓桿的 13 週美國國庫券殖利率，作為較保守的融資成本近似
- 對照組：SPY、IEFA、VT、60% SPY + 40% IEFA

資料來源：yfinance 調整收盤價；官方產品資料參考 [WisdomTree NTSD 產品頁](https://www.wisdomtree.com/us/products/capital-efficient/ntsd)、[WisdomTree 上市公告](https://www.businesswire.com/news/home/20260319818649/en/WisdomTree-Launches-Efficient-U.S.-Plus-International-Equity-Fund-NTSD)、[SEC 497K](https://www.sec.gov/Archives/edgar/data/1350487/000121465926003022/ntsd3926497k.htm)。最新下載價格：{latest_line}

## 長期合成回測

期間：{synthetic_start} 至 {synthetic_end}

{stats_md}

![NTSD proxy growth](ntsd_backtest_growth.png)

![NTSD proxy drawdown](ntsd_backtest_drawdown.png)

## NTSD 上市後實績

NTSD 實際掛牌歷史仍短，這段只能看早期跟蹤狀況，不能當作完整週期結論。

{actual_stats_md}

## 解讀

1. NTSD 的本質不是單純國際 ETF，而是「美股核心 + 國際股 overlay」的 150% 股票曝險產品。它會比 100% 全球股票更吃股票風險，也更依賴美股與國際股同時上漲。
2. 90/60 合成版本在多頭期間通常會放大報酬，但最大回撤也會放大。保守融資版本更接近投資人要承擔的實際槓桿成本壓力。
3. 它不提供債券緩衝，和 NTSX 那種 90% 股票 + 60% 債券不同。遇到全球股票同步下跌時，NTSD proxy 的保護力有限。
4. 適合把它當作「資本效率工具」：用較少資金取得美股加國際股曝險，空出現金配置其他低相關資產。若只是想買全球股市，VT 或 60/40 SPY/IEFA 的行為更直覺。

## 主要風險

- 實際基金歷史非常短，長期結果高度依賴代理假設。
- IEFA 不是期貨，費用、稅務、展期與融資成本都只是近似。
- 90/60 結構會把股票風險放大到約 1.5 倍，回撤可能比一般全球股票 ETF 更深。
- 國際曝險主要是已開發市場，並非完整全球市場；新興市場曝險可能不足。

## 產出檔案

- `output/research/ntsd/ntsd_backtest_stats.csv`
- `output/research/ntsd/ntsd_backtest_daily.csv`
- `output/research/ntsd/ntsd_backtest_growth.png`
- `output/research/ntsd/ntsd_backtest_drawdown.png`
- `output/research/ntsd/ntsd_report.md`
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = ["NTSD", "SPY", "SSO", "IEFA", "VXUS", "VT", "ACWI"]
    prices = _download_prices(tickers, START_DATE)
    returns = _daily_returns(prices)
    tbill_daily = _download_tbills(START_DATE, returns.index)
    daily_expense = EXPENSE_RATIO / 252.0

    required = returns[["SPY", "SSO", "IEFA", "VXUS", "VT"]].dropna()
    proxy_returns = pd.DataFrame(index=required.index)
    proxy_returns["SPY"] = required["SPY"]
    proxy_returns["SSO"] = required["SSO"]
    proxy_returns["IEFA"] = required["IEFA"]
    proxy_returns["VXUS"] = required["VXUS"]
    proxy_returns["VT"] = required["VT"]
    proxy_returns["60% SPY / 40% IEFA"] = 0.6 * required["SPY"] + 0.4 * required["IEFA"]
    proxy_returns["50% IEFA + 50% 2x SPY"] = required["SPY"] + 0.5 * required["IEFA"] - daily_expense
    proxy_returns["50% IEFA + 50% 2x SPY financed"] = (
        required["SPY"]
        + 0.5 * required["IEFA"]
        - daily_expense
        - 0.5 * tbill_daily.reindex(required.index).fillna(0.0)
    )
    proxy_returns["50% VXUS + 50% 2x SPY"] = required["SPY"] + 0.5 * required["VXUS"] - daily_expense
    proxy_returns["50% VXUS + 50% 2x SPY financed"] = (
        required["SPY"]
        + 0.5 * required["VXUS"]
        - daily_expense
        - 0.5 * tbill_daily.reindex(required.index).fillna(0.0)
    )
    proxy_returns["50% VXUS + 50% SSO"] = 0.5 * required["VXUS"] + 0.5 * required["SSO"]
    proxy_returns["NTSD proxy 90/60"] = 0.9 * required["SPY"] + 0.6 * required["IEFA"] - daily_expense
    proxy_returns["NTSD financed proxy"] = (
        0.9 * required["SPY"]
        + 0.6 * required["IEFA"]
        - daily_expense
        - 0.5 * tbill_daily.reindex(required.index).fillna(0.0)
    )

    wealth = proxy_returns.apply(_wealth_from_returns)
    stats = [_stats(name, wealth[name].dropna(), proxy_returns[name].dropna()) for name in wealth.columns]
    stats_df = _stats_frame(stats)

    actual = returns[[col for col in ["NTSD", "SPY", "IEFA", "VT"] if col in returns.columns]].dropna()
    actual_wealth = actual.apply(_wealth_from_returns)
    actual_stats = [_stats(name, actual_wealth[name].dropna(), actual[name].dropna()) for name in actual_wealth.columns]
    actual_stats_df = _stats_frame(actual_stats)

    combined_daily = pd.concat(
        {
            "returns": proxy_returns,
            "wealth": wealth,
        },
        axis=1,
    )
    combined_daily.to_csv(OUTPUT_DIR / "ntsd_backtest_daily.csv")
    stats_df.to_csv(OUTPUT_DIR / "ntsd_backtest_stats.csv", index=False)
    actual_stats_df.to_csv(OUTPUT_DIR / "ntsd_actual_since_inception_stats.csv", index=False)

    comparison_names = [
        "NTSD proxy 90/60",
        "NTSD financed proxy",
        "50% IEFA + 50% 2x SPY",
        "50% IEFA + 50% 2x SPY financed",
        "50% VXUS + 50% 2x SPY",
        "50% VXUS + 50% 2x SPY financed",
        "50% VXUS + 50% SSO",
    ]
    comparison_df = stats_df[stats_df["Strategy"].isin(comparison_names)].copy()
    comparison_df.to_csv(OUTPUT_DIR / "ntsd_spy2_comparison_stats.csv", index=False)

    _plot_wealth(wealth, OUTPUT_DIR / "ntsd_backtest_growth.png")
    _plot_drawdown(wealth, OUTPUT_DIR / "ntsd_backtest_drawdown.png")
    _plot_key_comparison(wealth, OUTPUT_DIR / "ntsd_key_comparison.png")

    report = _build_report(
        stats_df.to_markdown(index=False),
        actual_stats_df.to_markdown(index=False),
        prices.iloc[-1],
        wealth.index[0].date().isoformat(),
        wealth.index[-1].date().isoformat(),
    )
    (OUTPUT_DIR / "ntsd_report.md").write_text(report, encoding="utf-8")

    print(stats_df.to_string(index=False))
    print(f"\nWrote {(OUTPUT_DIR / 'ntsd_report.md').as_posix()}")


if __name__ == "__main__":
    main()
