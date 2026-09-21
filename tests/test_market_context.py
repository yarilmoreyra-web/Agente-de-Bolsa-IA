"""Tests de market_context.py: cotizaciones, régimen y calendario macro (sin red)."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import config
import market_context as mc
from technical_analysis import NA
from utils import NY_TZ

SESSION = date(2026, 9, 21)
SNAPSHOT = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)


def _snapshot(price=None, prev_close=100.0, quality="ok"):
    return {
        "premarket_price": price if price is not None else NA,
        "prev_close": prev_close,
        "premarket_quality": quality,
    }


def _indicators(last_close=99.0, return_5d=1.5):
    return {"last_close": last_close, "return_5d_pct": return_5d}


# --------------------------------------------------------------------------- #
# Cotizaciones
# --------------------------------------------------------------------------- #
def test_quote_usa_el_precio_premarket_cuando_existe():
    quote = mc.build_quote("SPY", _snapshot(101.0, 100.0), _indicators())
    assert quote.source == "premarket"
    assert quote.change_pct == 1.0
    assert quote.return_5d_pct == 1.5


def test_quote_cae_al_ultimo_cierre_sin_premarket():
    quote = mc.build_quote("^VIX", _snapshot(None, 18.0, "missing"),
                           _indicators(last_close=19.8))
    assert quote.source == "last_close"
    assert quote.price == 19.8
    assert quote.change_pct == 10.0


def test_quote_sin_datos_queda_en_na():
    quote = mc.build_quote("XLK", None, {"last_close": NA, "return_5d_pct": NA})
    assert quote.source == "missing"
    assert quote.price == NA and quote.change_pct == NA


# --------------------------------------------------------------------------- #
# Régimen
# --------------------------------------------------------------------------- #
def _quotes(vix, es, nq):
    return {
        "^VIX": {"price": vix, "change_pct": 0.0},
        "ES=F": {"price": 5000, "change_pct": es},
        "NQ=F": {"price": 18000, "change_pct": nq},
    }


def test_regimen_adverso_con_vix_alto():
    regime = mc.classify_regime(_quotes(30.0, 0.1, 0.1))
    assert regime["volatility"] == "alta"
    assert regime["against_longs"] is True
    assert regime["label"] == "adverso para largos"


def test_regimen_adverso_con_futuros_negativos():
    regime = mc.classify_regime(_quotes(15.0, -0.8, -1.0))
    assert regime["tone"] == "negativo"
    assert regime["against_longs"] is True


def test_regimen_favorable():
    regime = mc.classify_regime(_quotes(13.0, 0.6, 0.9))
    assert regime["volatility"] == "baja"
    assert regime["tone"] == "positivo"
    assert regime["against_longs"] is False
    assert regime["label"] == "favorable para largos"


def test_regimen_neutral_con_volatilidad_moderada():
    regime = mc.classify_regime(_quotes(22.0, 0.1, 0.0))
    assert regime["volatility"] == "moderada"
    assert regime["tone"] == "neutral"
    assert regime["against_longs"] is False


def test_regimen_sin_datos_no_inventa():
    regime = mc.classify_regime({})
    assert regime["volatility"] == NA and regime["tone"] == NA
    assert regime["against_longs"] is False


def test_regimen_usa_indices_si_no_hay_futuros():
    quotes = {"SPY": {"change_pct": -1.2}, "QQQ": {"change_pct": -1.0},
              "^VIX": {"price": 16.0}}
    assert mc.classify_regime(quotes)["tone"] == "negativo"


# --------------------------------------------------------------------------- #
# Calendario macro
# --------------------------------------------------------------------------- #
def _write(payload) -> Path:
    tmp = Path(TemporaryDirectory().name)
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "macro_events.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_macro_sin_archivo_avisa_y_no_inventa():
    macro = mc.load_macro_events(SESSION, Path("/no/existe/macro_events.json"))
    assert macro["loaded"] is False
    assert macro["note"] == mc.MACRO_NOT_LOADED
    assert macro["today"] == []


def test_macro_con_evento_de_hoy():
    path = _write({"events": [
        {"date": "2026-09-21", "time": "08:30", "name": "IPC", "importance": "alta"},
        {"date": "2026-09-22", "time": "14:00", "name": "FOMC", "importance": "alta"},
        {"date": "2026-01-01", "time": "09:00", "name": "Viejo", "importance": "baja"},
    ]})
    macro = mc.load_macro_events(SESSION, path, lookahead_days=1)
    assert macro["loaded"] is True
    assert [e["name"] for e in macro["today"]] == ["IPC"]
    assert [e["name"] for e in macro["upcoming"]] == ["FOMC"]
    assert "1 evento" in macro["note"]


def test_macro_dia_revisado_sin_eventos():
    path = _write({"covered_dates": ["2026-09-21"], "events": []})
    macro = mc.load_macro_events(SESSION, path)
    assert macro["loaded"] is True
    assert macro["today"] == []
    assert "Sin eventos" in macro["note"]


def test_macro_ignora_fechas_invalidas():
    path = _write({"events": [{"date": "no-es-fecha", "name": "X"}]})
    macro = mc.load_macro_events(SESSION, path)
    assert macro["loaded"] is False
    assert macro["today"] == []


def test_macro_archivo_corrupto_no_rompe():
    tmp = Path(TemporaryDirectory().name)
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "macro_events.json"
    path.write_text("{no es json", encoding="utf-8")
    assert mc.load_macro_events(SESSION, path)["loaded"] is False


# --------------------------------------------------------------------------- #
# Ensamblado
# --------------------------------------------------------------------------- #
def _context() -> mc.MarketContext:
    quotes = {
        "SPY": mc.SymbolQuote("SPY", 500.0, 498.0, 0.4, "premarket", "ok", 1.2),
        "QQQ": mc.SymbolQuote("QQQ", 430.0, 428.0, 0.47, "premarket", "ok", 2.0),
        "DIA": mc.SymbolQuote("DIA", 390.0, 389.0, 0.26, "premarket", "ok", 0.5),
        "ES=F": mc.SymbolQuote("ES=F", 5010.0, 4990.0, 0.5, "premarket", "ok"),
        "NQ=F": mc.SymbolQuote("NQ=F", 18100.0, 18000.0, 0.55, "premarket", "ok"),
        "^VIX": mc.SymbolQuote("^VIX", 14.0, 14.5, -3.4, "last_close", "missing"),
        "^TNX": mc.SymbolQuote("^TNX", 4.21, 4.2, 0.24, "last_close", "missing"),
        "XLK": mc.SymbolQuote("XLK", 100.0, 99.0, 1.5, "premarket", "ok"),
        "XLE": mc.SymbolQuote("XLE", 90.0, 90.5, -0.9, "premarket", "ok"),
        "XLF": mc.SymbolQuote("XLF", 45.0, 44.9, 0.2, "premarket", "ok"),
    }
    macro = {"loaded": True, "today": [], "upcoming": [], "note": "Sin eventos macro hoy"}
    return mc.assemble_context(SESSION, SNAPSHOT, quotes, macro)


def test_assemble_ordena_sectores_y_resume():
    context = _context()
    assert context.sector_leaders[0]["symbol"] == "XLK"
    assert context.sector_laggards[0]["symbol"] == "XLE"
    assert context.regime["label"] == "favorable para largos"
    assert "SPY +0.40%" in context.summary
    assert "VIX 14.00" in context.summary
    assert context.market_against_longs is False


def test_benchmarks_para_fuerza_relativa():
    benchmarks = _context().benchmarks_for_rs()
    assert benchmarks["SPY"]["pm_change_pct"] == 0.4
    assert benchmarks["QQQ"]["return_5d_pct"] == 2.0


def test_contexto_es_serializable_a_json():
    payload = json.dumps(_context().to_dict(), ensure_ascii=False)
    assert "sector_leaders" in payload
