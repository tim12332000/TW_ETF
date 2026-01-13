import json
import os

nb_path = r'c:\Git\TW_ETF\qqq_qld_tqqq_hedge_backtest.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new logic to insert
new_logic = [
    "\n",
    "# Backfill VXN with Historical Volatility (21-day rolling) for pre-2001 data\n",
    "df['Hist_Vol'] = df['QQQ_Price'].pct_change().rolling(window=21).std() * np.sqrt(252) * 100\n",
    "df['VXN'] = df['VXN'].fillna(df['Hist_Vol'])\n",
    "\n"
]

target_line = "df = df.dropna()\n"
found = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Look for the cell containing the target line
        if target_line in source:
            # Check if logic already exists to avoid duplication
            if "df['Hist_Vol'] =" in "".join(source):
                print("Logic already present.")
                found = True
                break
                
            idx = source.index(target_line)
            cell['source'] = source[:idx] + new_logic + source[idx:]
            found = True
            break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4, ensure_ascii=False) # standard indent for nbs
    print(f"Successfully updated {nb_path}")
else:
    print("Could not find the target line in any code cell.")
