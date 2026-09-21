"""Datos de ejemplo compartidos por los tests de la fase 5 (sin red)."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from typing import Any

from report import ReportData
from utils import NY_TZ

SESSION = date(2026, 9, 21)
SNAPSHOT_TS = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)
SENT_ON_TIME = datetime(2026, 9, 21, 8, 55, tzinfo=NY_TZ)
SENT_LATE = datetime(2026, 9, 21, 9, 7, tzinfo=NY_TZ)


def make_record(ticker: str = "NVDA", prev: float = 100.0, pm: float = 103.2,
                quality: str = "ok", quality_note: str = "") -> dict[str, Any]:
    """Registro con la forma de ``technical_analysis.analyze_ticker``."""
    support = {"label": "sma20", "price": 98.2, "side": "support",
               "distance_pct": -4.84, "distance_atr": -2.38}
    resistance = {"label": "max_20d", "price": 106.0, "side": "resistance",
                  "distance_pct": 2.71, "distance_atr": 1.33}
    return {
        "ticker": ticker, "session_date": "2026-09-21",
        "premarket": {"ticker": ticker, "prev_close": prev, "premarket_price": pm,
                      "premarket_high": pm + 0.3, "premarket_low": pm - 2.2,
                      "premarket_volume": 1_250_000.0, "n_bars": 57,
                      "last_bar_age_min": 5.0, "premarket_quality": quality,
                      "quality_note": quality_note},
        "indicators": {"rsi14": 62.14, "sma20": 98.2, "sma50": 95.1, "atr14": 2.1,
                       "atr_pct": 2.1, "ema9": 99.0, "ema20": 97.0},
        "gap_pct": round((pm / prev - 1) * 100, 4), "gap_atr": 1.52,
        "rvol": {"rvol": 3.4, "numerator": 1_250_000.0, "denominator": 367_000.0,
                 "sessions_used": 20, "reason": ""},
        "pm_pct_of_adv": 12.5,
        "relative_strength": {"rs_pm_vs_spy": 2.5, "rs_pm_vs_qqq": 2.1,
                              "rs_5d_vs_spy": 1.0, "rs_5d_vs_qqq": 0.5},
        "levels": [resistance, support],
        "nearest_support": support, "nearest_resistance": resistance,
    }


def make_candidate(ticker: str = "NVDA", score: float = 82.5, **kw: Any) -> dict[str, Any]:
    return {
        "ticker": ticker, "direction": "LONG", "score": score,
        "score_detail": {"total": score, "flags": ["gap_extendido"]},
        "record": make_record(ticker, **kw),
        "news": {"catalyst_confirmed": True,
                 "catalyst_text": "Resultados (earnings): supera estimaciones",
                 "sources": ["yahoo"],
                 "catalyst": {"publisher": "Reuters", "url": "https://x.test/n1"}},
    }


def make_pick(ticker: str = "NVDA", **overrides: Any) -> dict[str, Any]:
    pick = {
        "ticker": ticker, "company": "NVIDIA Corp", "direction": "LONG",
        "catalyst": "Resultados por encima de lo esperado", "catalyst_source": "Reuters",
        "trend": "alcista", "technical_summary": "sobre SMA20 y SMA50",
        "support_level": "sma20", "resistance_level": "max_20d",
        "fade_risk": "MEDIUM", "fade_risk_explanation": "gap de 1,5 ATR con buen RVOL",
        "entry_zone_low": 103.0, "entry_zone_high": 103.6, "stop": 101.5,
        "target_1": 105.5, "target_2": 108.0, "exit_time": "10:30 ET",
        "invalidation": "pierde 103.0 con volumen", "decision": "BUY", "confidence": "HIGH",
        "reason": "catalizador claro y volumen relativo alto",
        "rr": 2.1, "max_entry_price": 104.2, "warnings": [],
    }
    pick.update(overrides)
    return pick


def make_context() -> dict[str, Any]:
    return {
        "indices": {"SPY": {"symbol": "SPY", "price": 500.0, "change_pct": 0.35,
                            "basis": "pre-market"},
                    "QQQ": {"symbol": "QQQ", "price": 430.0, "change_pct": 0.42,
                            "basis": "pre-market"}},
        "volatility": {"symbol": "^VIX", "price": 16.2, "change_pct": -2.1,
                       "basis": "último cierre (2026-09-18)"},
        "futures": {"ES=F": {"symbol": "ES=F", "price": 5000.0, "change_pct": 0.3,
                             "basis": "pre-market"}},
        "treasury_10y": {"symbol": "^TNX", "price": 4.1, "change_pct": 0.5,
                         "basis": "último cierre (2026-09-18)"},
        "sectors": {"XLK": {"symbol": "XLK", "price": 200.0, "change_pct": 1.2,
                            "basis": "pre-market"},
                    "XLE": {"symbol": "XLE", "price": 90.0, "change_pct": -0.8,
                            "basis": "pre-market"}},
        "sector_leaders": ["XLK"], "sector_laggards": ["XLE"],
        "macro": {"status": "loaded", "text": "14:00 Decisión FOMC (impacto alto)",
                  "events": [], "high_impact_today": True},
        "regime": {"risk_level": "normal", "summary": "Riesgo normal (VIX 16.2, futuros +0.30%)",
                   "reasons": []},
    }


def make_report_data(**overrides: Any) -> ReportData:
    candidates = [make_candidate("NVDA", 82.5), make_candidate("AMD", 74.0),
                  make_candidate("TSLA", 61.0)]
    gemini = {"market_view": "Sesgo alcista moderado con datos macro por la tarde.",
              "no_trade": False, "no_trade_reason": "",
              "picks": [make_pick("NVDA"), make_pick("AMD", company="AMD Inc")],
              "warnings": ["AMD: el pre-market de Gemini difería del de Python; prevalece Python"],
              "available": True}
    base = dict(session_date=SESSION, snapshot_ts=SNAPSHOT_TS, sent_at=SENT_ON_TIME,
                session_type="sesión normal", universe_count=40,
                universe_source="lista_tickers.xlsx", context=make_context(),
                candidates=candidates, gemini=gemini)
    base.update(overrides)
    return ReportData(**deepcopy(base))
