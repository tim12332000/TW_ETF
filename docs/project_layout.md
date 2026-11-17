# 專案目錄與整理紀錄

根目錄保留 `startup.ipynb`、`portfolio_report.py`、README、專案規則及設定。所有正式輸入、歷史資料、研究及產物分區保存。

## 正式使用

- 交易 CSV：`data/transactions/tw_train.csv`、`data/transactions/us_train.csv`。
- 產生報告：開啟根目錄 `startup.ipynb`，或在專案根目錄執行 `python portfolio_report.py`。
- 閱讀：`output/report.md`、`output/report_details.md`；兩者共用 `output/charts/`。
- 核對現金：`output/combined_cash_ledger.csv`。

## 歷史資料與舊檔

| 原位置 | 新位置 |
| --- | --- |
| `date_csv_src.csv`、`date_csv_dst.csv` | `data/imports/` |
| `transcribed_transactions.csv` | `data/imports/` |
| `merged_stock_data.csv`、`tw0050.csv`、`tw006208.csv` | `data/reference/` |
| `CSV_DateFormat.ipynb` | `tools/legacy/`，保留歷史日期轉換流程 |
| `qqq_qld_tqqq_hedge_backtest.ipynb` | `research/notebooks/` |
| `tw.py`、`us.py` | `research/legacy/` |
| 原 `research/*.py` | `research/leverage/`、`hedging/`、`comparisons/`、`ntsd/` |
| 原 `tools/*.py` | `tools/diagnostics/`、`maintenance/`、`legacy/` |
| `tests/test_plot.py` | `research/hedging/compare_payoff.py`，它是展示腳本而非斷言測試 |
| 根目錄研究 PNG | `output/research/` 下對應主題 |
| `output.xlsx` | `output/exports/` |
| `output/report.txt` | `output/archive/` |
| 歷次稽核 CSV、驗證 log、版面基準 | `output/diagnostics/` |
| Put 歷史試算、QQQ 避險與 NTSD 產物 | `output/research/hedging/`、`ntsd/` |
| 過期行情快照 | `cache/archive/年月/`；同名歷史副本在 `duplicates/` |

移除的根目錄 `report_details.md` 原本為 0 bytes，真正的報告保留在 `output/report_details.md`。

歷史 CSV 與行情快照保留原內容。舊版單市場程式及診斷工具使用舊的現金／API 假設，僅作參考；歷史 CSV 日期轉換工具沿用原日期格式設定，不自動執行或覆寫匯入檔。

## 保持目錄整齊

正式報告圖只寫入 `output/charts/`；研究圖與資料只寫入其主題產物目錄。測試使用 pytest 暫存目錄，不在根目錄保存圖或測試 CSV。市場快取按資料種類保存，過期封存不參與正常快取查詢。

`.omx/context/layout-moves.json` 記錄逐檔搬移與 SHA256；研究和工具的搬移／公式未變驗證紀錄也在 `.omx/context/`。重複的報告量測副本及舊驗證副本封存於 `.omx/archive/`，不屬於正式資料。
