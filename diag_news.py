"""Script de diagnóstico: muestra las noticias crudas que Yahoo devuelve."""
import json
import yfinance as yf

TICKERS = ["ALVO", "TOYO", "BBOT", "TLN"]

for ticker in TICKERS:
    print(f"\n{'='*70}\n{ticker}\n{'='*70}")
    raw = yf.Ticker(ticker).news
    print(f"Cantidad de noticias devueltas: {len(raw) if raw else 0}")
    if raw:
        print("\nPrimer elemento crudo (estructura completa):")
        print(json.dumps(raw[0], indent=2, default=str, ensure_ascii=False))