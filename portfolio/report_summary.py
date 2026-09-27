"""Compact reader-facing report; detailed calculations remain in the appendix."""
from tabulate import tabulate


def render_summary_report(*, invested, total, cash, benchmark, holdings, puts,
                          first_date, last_date):
    cash_share = f'{cash / total:.2%}' if total > 0 else 'N/A'
    overview = [
        ['總資產（含現金）', f'{total:,.0f} 元'],
        ['累積投入', f'{invested:,.0f} 元'],
        ['總獲利', f'{total - invested:+,.0f} 元'],
        ['現金（推算）', f'{cash:,.0f} 元（{cash_share}）'],
    ]
    rows = []
    for _, row in benchmark.iterrows():
        rows.append([
            '我的投組' if row['Asset'] == 'My Portfolio' else row['Asset'],
            f"{row['Final Value (TWD)']:,.0f}", f"{row['Profit (TWD)']:+,.0f}",
            f"{row['XIRR %']:.2f}%", f"{row['MaxDD %']:.2f}%", f"{row['Sharpe']:.2f}",
        ])
    allocation = [[name, f'{value:,.0f}', f'{value / total:.2%}' if total > 0 else 'N/A']
                  for name, value in holdings.items() if value > 0]
    sections = [
        '# 投資組合報告',
        f'> 台美合併投組・金額以新台幣計｜估值期間 {first_date}～{last_date}',
        '## 資產總覽',
        tabulate(overview, headers=['項目', '金額'], tablefmt='github'),
        '## 與單買 ETF 比較',
        '使用相同外部投入日期與金額。XIRR 是資金加權年化報酬；最大回撤與 Sharpe 由 TWR 計算。',
        tabulate(rows, headers=['標的', '期末總資產', '總獲利', 'XIRR', '最大回撤', 'Sharpe'], tablefmt='github'),
        '![同投入資產比較](portfolio_vs_benchmark_twd.png)',
        '## 現在配置',
        tabulate(allocation, headers=['持倉', '市值', '占總資產'], tablefmt='github'),
        '![資產配置（含現金）](asset_pie_chart.png)',
        '## 歷史表現',
        '### 累積報酬比較',
        '排除外部投入的影響，觀察投資報酬本身。',
        '![累積報酬比較](cumulative_return_comparison.png)',
        '### 配置變化',
        '![歷史配置（含現金）](asset_allocation_monthly.png)',
        '### 距離歷史高點的跌幅',
        '![TWR 回撤](drawdown_underwater.png)',
        '## Put 持倉',
    ]
    if puts:
        sections.extend([
            tabulate(puts, headers=['標的', '到期日', '履約價（美元）', '市值（台幣）'], tablefmt='github'),
            'Put 的市值已計入上方總資產；到期給付與到期前市值不同。',
        ])
    else:
        sections.append('目前未持有 Put。')
    sections.extend([
        '## 詳細資料與計算前提',
        '[完整損益、再平衡與避險試算](report_details.md) · [逐日現金帳本](combined_cash_ledger.csv)',
        '- 台美共用推算現金，買入先扣現金，不足才算新投入；未記錄提款，故淨投入等於累積投入。現金以台幣記帳、不計息。',
        '- ETF 模擬含股息再投入，未扣個人股息稅及交易費；估值使用可得行情，並非全部即時報價。',
        '- Put 歷史估值資料有限，歷史風險與避險試算需配合詳細報表的模型假設判讀。',
    ])
    return '\n\n'.join(sections) + '\n'
