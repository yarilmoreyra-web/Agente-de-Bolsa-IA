import os
import json
import csv
import pandas as pd
from typing import Dict, Any

def run_backtest_for_date(date_str: str) -> Dict[Any, Any]:
    history_path = f"data/history/{date_str}.json"
    if not os.path.exists(history_path):
        return {"error": f"No existe histórico para la fecha {date_str}"}
        
    with open(history_path, "r", encoding="utf-8") as f:
        history_data = json.load(f)
        
    picks = history_data.get("picks", [])
    results = []
    
    os.makedirs("data/backtests", exist_ok=True)
    
    for pick in picks:
        ticker = pick.get("ticker")
        csv_bar_path = f"data/intraday/{date_str}/{ticker}.csv"
        
        outcome = "NO_DATA"
        mfe = 0.0
        mae = 0.0
        
        if os.path.exists(csv_bar_path):
            df = pd.read_csv(csv_bar_path)
            # Simulación simple de barras intradía
            if not df.empty and "High" in df.columns and "Low" in df.columns:
                entry = pick.get("entry_zone_low") or pick.get("premarket_price")
                stop = pick.get("stop")
                t1 = pick.get("target_1")
                
                if entry and stop and t1:
                    hit_t1 = False
                    hit_stop = False
                    for _, row in df.iterrows():
                        high = row["High"]
                        low = row["Low"]
                        
                        if high >= t1:
                            hit_t1 = True
                        if low <= stop:
                            hit_stop = True
                            
                    if hit_t1 and hit_stop:
                        outcome = "AMBIGUOUS"
                    elif hit_t1:
                        outcome = "WIN_T1"
                    elif hit_stop:
                        outcome = "LOSS_STOP"
                    else:
                        outcome = "EXPIRED"
                        
        results.append({
            "ticker": ticker,
            "outcome": outcome
        })
        
    bt_result = {
        "date": date_str,
        "results": results
    }
    
    with open(f"data/backtests/{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(bt_result, f, indent=2, ensure_ascii=False)
        
    return bt_result