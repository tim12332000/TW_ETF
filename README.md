# TW_ETF

台股、美股與 Put 的合併投資組合分析。日常使用只需開啟根目錄的 `startup.ipynb`，或執行：

```powershell
python portfolio_report.py
```

新下載專案後先執行一次入口，才會產生本機報告與圖表；`output/` 和 `cache/` 不隨版本上傳。

## 資料放哪裡

| 位置 | 內容 |
| --- | --- |
| `data/transactions/` | **目前使用的交易資料**：`tw_train.csv`、`us_train.csv` |
| `data/imports/` | 歷史券商匯出、日期轉換前後檔與人工抄錄交易，供核對 |
| `data/reference/` | 歷史股價快照，供研究參考 |
| `output/report.md` | 台幣摘要與 5 張核心圖 |
| `output/report_details.md` | 完整明細、美元指標與避險分析 |
| `output/combined_cash_ledger.csv` | 合併逐日現金帳本 |
| `output/charts/` | 目前報告的 19 張圖 |
| `output/research/` | 研究產物，按槓桿、避險、比較、NTSD 分類 |
| `output/diagnostics/` | 歷次驗證紀錄、基準與稽核快照 |
| `output/exports/`、`output/archive/` | 歷史試算表匯出與舊文字報告 |

主程式只讀取 `data/transactions/` 的兩份交易檔，不會使用歷史匯入或研究快照取代它們。整理時保留所有 CSV 內容。

## 程式與工具

```text
portfolio/                 正式計算、行情、報告與繪圖
research/
  leverage/                槓桿、配置與再平衡研究
  hedging/                 Put、保險及壓力測試研究
  comparisons/             市場與交易紀錄比較
  ntsd/                    NTSD 策略研究
  notebooks/               研究用 Notebook
  legacy/                  舊版單市場分析，保留作歷史參考
tools/
  diagnostics/             資料日期、VIX 與回補檢查
  maintenance/             研究 Notebook 維護
  legacy/                  舊模型診斷及歷史 CSV 轉換工具
tests/                     自動回歸測試，不產生根目錄圖檔
docs/                      模型、計算前提與目錄說明
```

研究程式以檔名直接執行，例如 `python research/leverage/find_optimal_leverage.py`；圖表寫入對應的 `output/research/` 分類。舊版程式不代表目前合併現金模型。

## 快取與測試暫存

`cache/` 按 `quotes`、`prices`、`options`、`fx`、`benchmarks`、`splits` 分類。過期快照保留在 `cache/archive/`，不會作為有效行情讀取。一般行情快取 24 小時，避險重設使用的選擇權資料快取 15 分鐘。

工具狀態及封存的驗證副本在 `.omx/`，與正式資料分開。產物、快取、Notebook 檢查點與測試暫存不應新增進版本控制；既有 Git 追蹤紀錄未重寫。

執行測試：`python -m pytest tests -q`。完整目錄與搬移說明見 [docs/project_layout.md](docs/project_layout.md)。
