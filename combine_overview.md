# combine.py 速查

## 使用原則

這份 md 不是最終真理，`combine.py` 跑出來的結果才是。

正確用法：

1. 先讀這份 md 快速定位
2. 再直接看 `combine.py` 對應函式
3. 真的有懷疑時，直接改 code / 加 log / 重跑驗證

如果 md 內容和實際執行結果衝突，以 code 與輸出為準，md 應該跟著更新。

## 它在做什麼

`combine.py` 是整個投資彙總腳本的總控：

1. 讀入台股交易 `tw_train.csv` 與美股交易 `us_train.csv`
2. 清洗交易欄位、修正買賣股數正負號、推導實際「外部投入資金」
3. 從 `yfinance` 抓歷史價格、即時價格、選擇權價格、USD/TWD 匯率
4. 把 TW / US 部位統一換成 USD，再額外計算一套 TWD 視角結果
5. 計算部位現值、PnL、XIRR、TWR、波動、MaxDD、Sharpe / Sortino / Calmar
6. 用同一組現金流模擬 `SPY` / `QQQ` / `EWT` benchmark
7. 產出文字報告 `output/report.txt` 與一批圖表 `output/*.png`
8. 額外做再平衡建議與 Put 保護情境分析

一句話版：它不是單純 combine，而是「交易紀錄 -> 持倉/績效/比較基準/風險圖表」的一條龍報表器。

## 主要輸入

- `tw_train.csv`
  - 台股交易紀錄
  - 原始金額先視為 TWD，再依歷史匯率轉成 USD 參與總帳
- `us_train.csv`
  - 美股/ETF/期權交易紀錄
  - 金額直接視為 USD
- `cache/*`
  - 價格、匯率、benchmark、選擇權價格快取
  - 1 天內優先讀 cache，避免重抓

## 主流程

### 1. 基礎工具層

- `get_cached_data`
  - 所有行情/匯率查詢的快取入口
- `clean_currency`
  - 清掉 `NT$`、`$`、`,`，轉成數字
- `build_cash_ledgers`
  - 從交易金額推導兩種 cashflow：
  - `transaction_cashflows`: 所有現金異動
  - `external_cashflows`: 真正來自帳外的投入/提出
  - 規則是「帳上現金不夠支付買入時，差額視為外部注資」
- `get_daily_price`
  - 抓股票/ETF 歷史價格
  - 台股代碼自動補 `.TW`
  - 會嘗試把 yfinance split 後價格還原回較接近原始成交價尺度
- `get_usd_twd_history`
  - 抓 USD/TWD 歷史匯率，供台股換算與 TWD 報表
- `build_option_history_series`
  - 對 OCC option symbol 建一條近似歷史價值線
  - 歷史部分用 intrinsic value，最後一天再用 option chain 現價補強

### 2. 台股處理 `process_tw_data()`

- 讀 `tw_train.csv`
- 欄位 rename、日期排序、股數正負修正
- `Amount_TWD` 保留原始台幣
- 用歷史匯率把 `Amount` 換成 USD
- 建持倉時間序列：
  - 每日成交量 pivot
  - `cumsum()` 得到每日持股數
- 抓台股歷史價格
- 做一次 calibration：
  - 用每個股票第一筆成交價對齊 yfinance 的第一天 close
  - 目的是修正拆股/資料尺度差異
- 算出：
  - 每日資產價值 `portfolio_value`
  - 當前持倉表 `portfolio_df`
  - 外部投入資金 `invested_capital`
  - 現值、總損益、報酬率

### 3. 美股處理 `process_us_data()`

- 讀 `us_train.csv`
- 清洗欄位與現金流
- 抓一般股票/ETF 的歷史價格
- 對 OCC 選擇權代碼不直接走一般價格抓取，改用：
  - `build_option_history_series`
  - `resolve_market_price`
- 算出和台股相同的一組結果：
  - `portfolio_value`
  - `portfolio_df`
  - `invested_capital`
  - 現值 / 損益 / 報酬率

### 4. 合併總帳 `main()`

- 合併 TW / US 的每日資產價值
- 用匯率把總資產再轉成 TWD 版本
- 合併 `external_cashflows`
- 計算：
  - USD / TWD 兩套總資產、總損益、報酬率
  - `XIRR`
  - `TWR`
  - `AnnVol`
  - `MaxDD`
  - `Sharpe / Sortino / Calmar`
- 合併持倉表 `portfolio_df_combined`
  - 顯示 PnL(USD)、PnL(TWD)、配置比重、平均成本

### 5. Benchmark / 風控 / 視覺化

- `simulate_stock_full`
  - 用「同一組 external cashflows」模擬如果全投入 `SPY` / `QQQ` / `EWT`
- `print_rebalance_recommendation`
  - 只看目標池：
  - `QLD`
  - `SPYM / SPLG`
  - `00631L`
  - `006208`
  - 算目前配置 vs 目標配置
- `analyze_put_protection`
  - 掃描持倉中像 `QQQ270115P00350000` 這類 put
  - 模擬市場下跌時，put 對總資產提供多少保護

## 主要輸出

- `output/report.txt`
  - 終端摘要報告落地檔
- `output/asset_trend.png`
  - 總資產 vs 累積投入資金
- `output/funding_ratio.png`
  - 資產 / 累積投入
- `output/twr_chart.png`
  - 自身 TWR
- `output/cumulative_return_comparison.png`
  - 自身 vs benchmark TWR
- `output/portfolio_vs_benchmark_usd.png`
  - 同現金流 benchmark 模擬比較
- `output/drawdown_underwater.png`
  - 回撤圖
- `output/asset_pie_chart.png`
  - 持倉圓餅圖
- `output/monthly_investment.png`
  - 每月投入
- `output/stock_performance.png`
  - 個股/資產績效線
- `output/total_asset_protection.png`
  - Put 保護情境圖

## 驗證方式

### 最小驗證

直接重跑主程式，確認報表與圖有沒有正常刷新。

建議看：

- `output/report.txt`
- `output/*.png`
- `cache/` 是否有新快取或命中舊快取

### 問題定位時怎麼驗

- 懷疑資產值錯：
  - 在 `process_tw_data()` / `process_us_data()` 裡加 `print(df.head())`、`print(portfolio_df)`、`print(portfolio_value.tail())`
- 懷疑投入資金錯：
  - 在 `build_cash_ledgers()` 印出 `transaction_cashflows`、`external_cashflows`
- 懷疑現價錯：
  - 在 `get_daily_price()`、`resolve_market_price()` 印出 symbol、最後一筆價格、fallback 來源
- 懷疑 option 錯：
  - 在 `parse_occ_symbol()`、`build_option_history_series()` 印出 strike、expiry、underlying、最後估值
- 懷疑 benchmark 錯：
  - 在 `simulate_stock_full()` 印出每次 cashflow 對應買到的股數與最終資產

### 改 code 才算確認

如果只是「看 md 覺得應該是這樣」，那還不算確認。

比較可靠的流程是：

1. 在懷疑的函式加最小量 log
2. 重跑一次
3. 確認資料流和預期是否一致
4. 再修改 code
5. 再重跑一次確認改動有生效

## 你要查問題時，先看這裡

### 1. 總資產不合理

先查：

- `process_tw_data()` / `process_us_data()`
- `build_cash_ledgers()`
- `get_daily_price()`
- `resolve_market_price()`

常見原因：

- 買賣方向正負號錯
- `Amount` 格式沒清乾淨
- 台股價格尺度被 split / calibration 影響
- 某檔最新價格抓不到，fallback 到舊價

### 2. 投入資金看起來怪

先查：

- `build_cash_ledgers()`

重點：

- 這份程式不是直接讀券商的入金/出金流水
- 它是從交易現金流反推「帳外投入」
- 所以如果原始匯出的 `Amount` 不完整，`invested_capital` 會偏掉

### 3. 台股報酬怪、和券商不一致

先查：

- `process_tw_data()` 內的匯率轉換
- `get_usd_twd_history()`
- calibration 那段第一筆成交價對齊邏輯

重點：

- 台股在內部主帳是先換成 USD
- 最後才再做 TWD 報表
- 所以券商純 TWD 視角和這裡的 USD 主帳，可能會有觀感差異

### 4. 選擇權市值怪

先查：

- `parse_occ_symbol()`
- `build_option_history_series()`
- `get_option_price()`
- `analyze_put_protection()`

重點：

- 歷史線不是完整 options OHLC
- 大多是 intrinsic value 近似
- 只在最後一天盡量用 option chain 現價修正
- 所以它適合總覽，不適合精準回測 Greeks / 真實盤中估值

### 5. Benchmark 結果怪

先查：

- `simulate_stock_full()`
- `bench_twr_*`

重點：

- benchmark 不是拿你的真實持股去比較
- 是拿「同一組外部現金流」去模擬全買某檔
- 所以它是在回答：
  - 「如果當時每次入金都買 SPY / QQQ / EWT，現在會怎樣？」

### 6. report.txt 和終端顯示不同

先查：

- `DualLogger`

重點：

- `[CACHE]`、`[CAL]` 這類訊息會被過濾，不一定寫進檔案
- ANSI 顏色碼會被剝掉

## 重要假設

- `external_cashflows` 的號號約定：
  - 負值 = 資金投入
  - 正值 = 資金提出
- 台股內部主帳會換成 USD 再和美股合併
- benchmark 比較使用 external cashflow，不是所有交易現金流
- option 歷史價值為近似，不是完整真實成交歷史
- cache 預設 1 天有效

## 建議的除錯順序

1. 先看 `output/report.txt`，確認是資產值錯、投入資金錯、還是 benchmark 錯
2. 再分 TW / US 看 `process_tw_data()` 或 `process_us_data()`
3. 若是金額面問題，先看 `build_cash_ledgers()`
4. 若是市值面問題，先看 `get_daily_price()` / `resolve_market_price()`
5. 若只影響 options，直接看 `build_option_history_series()` 與 `get_option_price()`
6. 若只影響比較基準，改查 `simulate_stock_full()`

## 建議優先改哪裡

如果之後要讓這支程式更容易自證，最值得優先補的是：

- 在 `main()` 開頭加一個 debug flag，能控制是否印出中間表
- 在 `build_cash_ledgers()` 補出更明確的 ledger dump
- 在 `process_tw_data()` / `process_us_data()` 補每檔持倉計算摘要
- 在 `resolve_market_price()` 標出價格來源是 history、live、還是 fallback
- 在 `simulate_stock_full()` 輸出 benchmark 現金流對應結果

## 備註

- 檔內有一些註解文字已經編碼亂掉，但不影響主邏輯閱讀。
- 這份 md 的定位是「幫你更快下刀改 code」，不是取代 code review。
- 這支腳本現在職責很多：資料清洗、估值、績效、比較、圖表、避險分析都在同一檔。未來若要更好維護，最值得先拆的是：
  - `data fetch / cache`
  - `transaction normalization`
  - `performance metrics`
  - `reporting / plotting`
