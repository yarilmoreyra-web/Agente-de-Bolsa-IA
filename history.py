import os
import json
import csv
from typing import Dict, Any

def save_daily_history(data: Dict[Any, Any], date_str: str) -> str:
    os.makedirs("data/history", exist_ok=True)
    json_path = f"data/history/{date_str}.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    # Actualizar history.csv plano
    csv_path = "data/history.csv"
    file_exists = os.path.exists(csv_path)
    
    picks = data.get("picks", [])
    fieldnames = [
        "date", "ticker", "company", "direction", "catalyst", 
        "premarket_price", "gap_pct", "rvol", "rr_ratio", "decision", "confidence"
    ]
    
    rows_to_write = []
    for p in picks:
        rows_to_write.append({
            "date": date_str,
            "ticker": p.get("ticker"),
            "company": p.get("company"),
            "direction": p.get("direction"),
            "catalyst": p.get("catalyst"),
            "premarket_price": p.get("premarket_price"),
            "gap_pct": p.get("gap_pct"),
            "rvol": p.get("rvol"),
            "rr_ratio": p.get("rr_ratio"),
            "decision": p.get("decision"),
            "confidence": p.get("confidence")
        })
        
    if rows_to_write:
        # Evitar duplicados simples por fecha y ticker si ya existe
        existing_rows = []
        if file_exists:
            with open(csv_path, "r", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                existing_rows = list(reader)
                
        # Filtrar existentes para esta fecha
        filtered_existing = [r for r in existing_rows if r.get("date") != date_str]
        final_rows = filtered_existing + rows_to_write
        
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)
            
    return json_path