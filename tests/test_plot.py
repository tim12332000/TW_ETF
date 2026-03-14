import numpy as np
import matplotlib.pyplot as plt

# 參數：標的 100, 履約價 100, 權利金 6, 總資金 100
S = np.linspace(40, 160, 200)
K = 100

# 1. Buy Call + 剩餘現金 (假設現金放銀行不賠)
# 當 S < 100，Call 價值趨於 0，總資產 = 100 - 6 = 94
payoff_call = np.maximum(S - K, 0) - 6 + 94

# 2. 現貨 + Buy Put (考慮崩盤時 IV 飆升 30%~50%)
# 下跌時 Put 的增值會因 IV 噴發而出現「超額收益」(加成項)
iv_boost = np.where(S < K, 0.4 * (K - S)**1.15, 0) 
payoff_put_stock = S + (np.maximum(K - S, 0) - 6) + iv_boost

# 設定中文字型 (Windows 預設微軟正黑體)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 繪製對比
plt.figure(figsize=(10,6))
plt.plot(S, payoff_call, 'r--', label='Buy Call + Cash (底部躺平)')
plt.plot(S, payoff_put_stock, 'b-', label='Stock + Put (左側反轉上翹)')

plt.title('Buy Call + Cash vs Stock + Put 策略總資產對比')
plt.xlabel('標的價格 (S)')
plt.ylabel('總資產價值')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 儲存圖片
out_file = 'payoff_comparison.png'
plt.savefig(out_file, dpi=150)
print(f"執行完畢！圖片已成功儲存至 {out_file}")
