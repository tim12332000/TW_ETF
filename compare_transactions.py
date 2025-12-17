import pandas as pd
import json

def parse_shares(val):
    if pd.isna(val) or val == '':
        return 0
    if isinstance(val, str):
        # Remove commas and handle spaces
        clean_val = val.replace(',', '').strip()
        if not clean_val:
            return 0
        return float(clean_val)
    return float(val)

def normalize_date(date_val):
    try:
        return pd.to_datetime(date_val).strftime('%Y-%m-%d')
    except:
        return None

try:
    # Load transcribed
    df_trans = pd.read_csv('c:/Git/TW_ETF/transcribed_transactions.csv')
    
    # Load train
    # encoding='utf-8-sig' or 'cp950' usually for Traditional Chinese in CSV, try default first or specific if needed
    try:
        df_train = pd.read_csv('c:/Git/TW_ETF/tw_train.csv', encoding='utf-8')
    except UnicodeDecodeError:
         df_train = pd.read_csv('c:/Git/TW_ETF/tw_train.csv', encoding='cp950')

    # Normalize transcribed
    trans_records = []
    for idx, row in df_trans.iterrows():
        date = normalize_date(row['Date'])
        if not date: continue
        
        code = str(row['SecuritiesCode']).strip()
        
        # Determine type and shares
        shares = 0
        action = ''
        memo = str(row['Memo'])
        
        deposit = parse_shares(row['Deposit'])
        withdrawal = parse_shares(row['Withdrawal'])
        
        if deposit > 0:
            shares = deposit
            action = 'Buy'
        elif withdrawal > 0:
            shares = withdrawal
            action = 'Sell'
        
        # Overwrite action based on memo if specific
        if '賣出' in memo:
            action = 'Sell'
        # For simple buys, usually deposit > 0 aligns with 'Buy', but double check
        # '劃撥配發' -> StockDividend
        if '配發' in memo or '配股' in memo:
            action = 'StockDividend'
            
        trans_records.append({
            'ID': f"Trans-{idx+2}",
            'Date': date,
            'Code': code,
            'Shares': shares,
            'Action': action
        })

    # Normalize train
    train_records = []
    for idx, row in df_train.iterrows():
        if pd.isna(row['交易日']): continue
        
        date = normalize_date(row['交易日'])
        if not date: continue
            
        code = str(row['股票代號']).strip()
        shares = parse_shares(row['股數'])
        type_ = str(row['交易別'])
        
        action = ''
        if type_ == '買':
            action = 'Buy'
        elif type_ == '賣':
            action = 'Sell'
        elif type_ == '配股':
            action = 'StockDividend'
        else:
            continue # Ignore cash dividends '息' or others
            
        train_records.append({
            'ID': f"Train-{idx+2}",
            'Date': date,
            'Code': code,
            'Shares': shares,
            'Action': action,
            'LineNo': idx + 2
        })

    if not trans_records:
        print(json.dumps([{"Error": "No transcribed records parsed"}]))
        exit()

    # Range of transcribed dates
    min_date = min(r['Date'] for r in trans_records)
    max_date = max(r['Date'] for r in trans_records)

    # Filter train records to this range
    train_records_in_range = [r for r in train_records if min_date <= r['Date'] <= max_date]
    
    # Sort for deterministic matching (though set usage above didn't rely on sort)
    # Matching logic
    matched_train_indices = set()
    discrepancies = []

    # Map for easy lookup to check if a "perfect match" exists first
    # But since we have potential duplicates (same day same stock), greedy matching is tricky.
    # We will do greedy matching: Find exact match first.
    
    unmatched_trans = []
    
    # Pass 1: Exact matches
    for t_rec in trans_records:
        match_idx = -1
        for i, tr_rec in enumerate(train_records_in_range):
            if i in matched_train_indices: continue
            
            if (tr_rec['Date'] == t_rec['Date'] and 
                tr_rec['Code'] == t_rec['Code'] and
                tr_rec['Action'] == t_rec['Action'] and
                abs(tr_rec['Shares'] - t_rec['Shares']) < 0.001):
                match_idx = i
                break
        
        if match_idx != -1:
            matched_train_indices.add(match_idx)
        else:
            unmatched_trans.append(t_rec)
            
    # Pass 2: Check unmatched transcribed for discrepancies (same date/code/action, diff shares)
    for t_rec in unmatched_trans:
        candidate_idx = -1
        for i, tr_rec in enumerate(train_records_in_range):
            if i in matched_train_indices: continue
            
            if (tr_rec['Date'] == t_rec['Date'] and 
                tr_rec['Code'] == t_rec['Code'] and
                tr_rec['Action'] == t_rec['Action']):
                candidate_idx = i
                break # Take first candidate
        
        if candidate_idx != -1:
            tr_rec = train_records_in_range[candidate_idx]
            discrepancies.append({
                'Type': 'Share Mismatch',
                'Date': t_rec['Date'],
                'Code': t_rec['Code'],
                'Action': t_rec['Action'],
                'TranscribedShares': t_rec['Shares'],
                'TrainShares': tr_rec['Shares'],
                'TrainLine': tr_rec['LineNo']
            })
            matched_train_indices.add(candidate_idx)
        else:
            discrepancies.append({
                'Type': 'Missing in Train',
                'Date': t_rec['Date'],
                'Code': t_rec['Code'],
                'Action': t_rec['Action'],
                'TranscribedShares': t_rec['Shares'],
                'TrainShares': None,
                'TrainLine': None
            })

    # Pass 3: Check for extra rows in Train
    for i, tr_rec in enumerate(train_records_in_range):
        if i not in matched_train_indices:
            discrepancies.append({
                'Type': 'Extra in Train',
                'Date': tr_rec['Date'],
                'Code': tr_rec['Code'],
                'Action': tr_rec['Action'],
                'TranscribedShares': None,
                'TrainShares': tr_rec['Shares'],
                'TrainLine': tr_rec['LineNo']
            })

    print(json.dumps(discrepancies, indent=2, ensure_ascii=False))

except Exception as e:
    import traceback
    print(json.dumps([{"Error": str(e), "Traceback": traceback.format_exc()}]))
