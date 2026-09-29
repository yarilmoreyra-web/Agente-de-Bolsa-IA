"""Tests de post_open.py: métricas en vivo, estados de las selecciones y comparaciones.

Todo con datos sintéticos (sin red). Se usa el lunes 2026-09-21 como sesión.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

import config
import post_open as po
from tests.synthetic import business_days, make_5m_bars
from utils import NY_TZ

DAY = date(2026, 9, 21)
UPDATE_TS = datetime(2026, 9, 21, 10, 0, tzinfo=NY_TZ)


def bars_from(rows, start="09:30"):
    """DataFrame 5m desde filas (open, high, low, close, volume) a partir de ``start``."""
    idx = pd.date_range(f"{DAY} {start}", periods=len(rows), freq="5min", tz=NY_TZ)
    return pd.DataFrame(rows, index=idx,
                        columns=["Open", "High", "Low", "Close", "Volume"])


def flat_bars(price=100.0, n=6, volume=1000.0):
    return bars_from([(price, price + 0.1, price - 0.1, price, volume)] * n)


def pick(**overrides):
    base = {
        "ticker": "TEST", "company": "Test Corp", "direction": "LONG",
        "entry_zone_low": 100.0, "entry_zone_high": 101.0, "max_valid_entry": 101.5,
        "stop": 98.0, "target_1": 106.0, "target_2": 108.0,
        "decision": "BUY", "confidence": "HIGH", "rr": 2.0,
    }
    base.update(overrides)
    return base


def live_for(bars, prev_close=99.0, hist=None):
    return po.compute_live_metrics("TEST", bars, hist, prev_close, UPDATE_TS)


# --------------------------------------------------------------------------- #
# Hora de la actualización
# --------------------------------------------------------------------------- #
def test_hora_por_defecto_es_las_10():
    assert po.resolve_update_ts(DAY) == UPDATE_TS
    assert po.resolve_update_ts(DAY, datetime(2026, 9, 21, 9, 47, tzinfo=NY_TZ)) == UPDATE_TS


def test_cron_retrasado_usa_la_ultima_barra_completa():
    late = datetime(2026, 9, 21, 10, 17, 40, tzinfo=NY_TZ)
    assert po.resolve_update_ts(DAY, late) == datetime(2026, 9, 21, 10, 15, tzinfo=NY_TZ)


def test_otra_fecha_usa_la_hora_base_y_nunca_pasa_del_cierre():
    assert po.resolve_update_ts(DAY, datetime(2026, 9, 22, 12, 0, tzinfo=NY_TZ)) == UPDATE_TS
    very_late = datetime(2026, 9, 21, 17, 30, tzinfo=NY_TZ)
    assert po.resolve_update_ts(DAY, very_late).time() == config.SCHEDULE.regular_close


# --------------------------------------------------------------------------- #
# Métricas en vivo
# --------------------------------------------------------------------------- #
def test_metricas_basicas_desde_la_apertura():
    bars = bars_from([
        (100.0, 100.6, 99.8, 100.5, 1000), (100.5, 101.2, 100.4, 101.0, 2000),
        (101.0, 101.5, 100.9, 101.2, 3000), (101.2, 101.4, 100.8, 100.9, 1000),
        (100.9, 101.6, 100.9, 101.5, 2000), (101.5, 102.0, 101.4, 101.8, 1000),
    ])
    live = live_for(bars, prev_close=99.0)
    assert live["quality"] == "ok" and live["n_bars"] == 6
    assert live["open_price"] == 100.0 and live["price_now"] == 101.8
    assert live["high_since_open"] == 102.0 and live["low_since_open"] == 99.8
    assert live["volume_since_open"] == 10000
    assert live["change_vs_open_pct"] == round((101.8 / 100.0 - 1) * 100, 4)
    assert live["change_vs_prev_close_pct"] == round((101.8 / 99.0 - 1) * 100, 4)
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    assert abs(live["vwap"] - (typical * bars["Volume"]).sum() / bars["Volume"].sum()) < 1e-3
    assert live["price_vs_vwap"] in ("encima", "debajo")
    assert live["last_bar_age_min"] == 5.0


def test_la_barra_de_las_10_en_curso_no_cuenta():
    # Barra 10:00 (aún incompleta) y una del pre-market: ninguna entra en la ventana.
    rows = [(100, 100.1, 99.9, 100, 1000)] * 6 + [(150, 151, 149, 150, 999_999)]
    live = live_for(bars_from(rows))
    assert live["n_bars"] == 6 and live["price_now"] == 100.0


def test_sin_barras_de_sesion_regular_queda_missing():
    premarket = make_5m_bars(DAY, "04:00", "09:30")
    for frame in (None, pd.DataFrame(), premarket):
        live = live_for(frame)
        assert live["quality"] == "missing"
        assert live["price_now"] == po.NA and live["open_rvol"] == po.NA


def test_pocas_barras_marca_sparse_y_dato_viejo_marca_stale():
    early = datetime(2026, 9, 21, 9, 40, tzinfo=NY_TZ)   # solo 2 barras, la última de hace 5 min
    sparse = po.compute_live_metrics("T", flat_bars(n=2), None, 99.0, early)
    assert sparse["quality"] == "sparse" and "pre-market" not in sparse["quality_note"]

    assert live_for(flat_bars(n=6))["quality"] == "ok"

    late = datetime(2026, 9, 21, 10, 40, tzinfo=NY_TZ)   # última barra de hace 45 min
    stale = po.compute_live_metrics("T", flat_bars(n=6), None, 99.0, late)
    assert stale["quality"] == "stale" and "pre-market" not in stale["quality_note"]


def test_rvol_de_apertura_compara_con_la_mediana_de_las_mismas_barras():
    hist = pd.concat([make_5m_bars(d, "09:30", "10:00", volume=1000.0)
                      for d in business_days("2026-09-18", 12)])
    today = flat_bars(volume=2000.0)
    live = live_for(today, hist=hist)
    assert live["open_rvol"] == 2.0


def test_rvol_sin_historico_suficiente_es_na_con_motivo():
    hist = make_5m_bars("2026-09-18", "09:30", "10:00", volume=1000.0)
    live = live_for(flat_bars(), hist=hist)
    assert live["open_rvol"] == po.NA and "sesiones" in live["open_rvol_note"]


# --------------------------------------------------------------------------- #
# Seguimiento de stop y objetivos
# --------------------------------------------------------------------------- #
def test_track_gana_lo_que_ocurre_primero():
    stop_first = bars_from([(100, 100.5, 99.5, 100, 1)] + [(100, 100.2, 97.5, 98, 1)]
                           + [(98, 107, 97, 106, 1)])
    events = po.track_events(stop_first, "LONG", 98.0, 106.0, 108.0)
    assert events["first_event"] == "stop" and events["target_1_hit"]

    target_first = bars_from([(100, 106.5, 99.5, 106, 1), (106, 106.2, 97.0, 98, 1)])
    events = po.track_events(target_first, "LONG", 98.0, 106.0, 108.0)
    assert events["first_event"] == "target_1" and events["stop_hit"]
    assert not events["target_2_hit"]


def test_track_misma_barra_cuenta_el_stop_primero():
    both = bars_from([(100, 107, 97.0, 100, 1)])
    events = po.track_events(both, "LONG", 98.0, 106.0, None)
    assert events["first_event"] == "stop" and events["ambiguous_bar"]


def test_track_short_invierte_las_condiciones():
    bars = bars_from([(100, 100.4, 99.0, 99.5, 1), (99.5, 99.9, 93.5, 94.0, 1)])
    events = po.track_events(bars, "SHORT", 102.0, 94.0, 92.0)
    assert events["first_event"] == "target_1" and not events["stop_hit"]


def test_track_sin_niveles_o_sin_barras_no_falla():
    assert po.track_events(pd.DataFrame(), "LONG", 1, 2, 3)["first_event"] is None
    assert po.track_events(flat_bars(), "LONG", None, None, None)["first_event"] is None


# --------------------------------------------------------------------------- #
# Estado de una selección
# --------------------------------------------------------------------------- #
def assess(bars, **pick_overrides):
    p = pick(**pick_overrides)
    return po.assess_pick(p, live_for(bars), bars)


def test_estado_en_zona_con_rr_y_regla_de_apertura():
    bars = flat_bars(100.5)
    result = assess(bars)
    assert result["status"] == po.STATUS_IN_ZONE
    assert result["open_rule"] == "CUMPLE"
    assert result["rr_now"] == round((106 - 100.5) / (100.5 - 98), 2)
    assert result["vwap_favorable"] is True


def test_estado_stop_tocado():
    bars = bars_from([(100, 100.2, 99.8, 100, 1)] * 3 + [(100, 100, 97.0, 97.5, 1)] * 3)
    result = assess(bars)
    assert result["status"] == po.STATUS_STOP
    assert result["events"]["stop_hit_time"].startswith("2026-09-21T09:45")


def test_estado_objetivo_alcanzado():
    bars = bars_from([(100, 100.2, 99.8, 100, 1)] * 2 + [(100, 106.4, 100, 106, 1)] * 4)
    assert assess(bars)["status"] == po.STATUS_TARGET


def test_estado_extendida_por_encima_del_maximo_de_entrada():
    bars = flat_bars(103.0)  # máximo válido 101.5
    result = assess(bars)
    assert result["status"] == po.STATUS_EXTENDED
    assert result["open_rule"] == "NO_ENTRAR"


def test_estado_en_contra_por_debajo_de_la_zona_sin_tocar_el_stop():
    bars = flat_bars(99.0)  # zona 100-101, stop 98, mínimo de barra 98.9
    assert assess(bars)["status"] == po.STATUS_AGAINST


def test_estado_sin_datos_y_sin_niveles():
    assert assess(pd.DataFrame())["status"] == po.STATUS_NO_DATA
    no_levels = assess(flat_bars(), entry_zone_low=None, entry_zone_high=None,
                       max_valid_entry=None, stop=None, target_1=None, target_2=None)
    assert no_levels["status"] == po.STATUS_NO_DATA and "niveles" in no_levels["note"]


def test_short_en_zona_y_extendida():
    short = dict(direction="SHORT", entry_zone_low=99.0, entry_zone_high=100.0,
                 max_valid_entry=98.5, stop=102.0, target_1=94.0, target_2=92.0)
    assert assess(flat_bars(99.5), **short)["status"] == po.STATUS_IN_ZONE
    assert assess(flat_bars(97.5), **short)["status"] == po.STATUS_EXTENDED
    assert assess(flat_bars(100.8), **short)["status"] == po.STATUS_AGAINST


def test_veredicto_de_python():
    def status(name, **extra):
        return {"status": name, "rr_now": 2.0, "vwap_favorable": True, **extra}

    assert po.python_verdict(status(po.STATUS_STOP), "BUY") == (po.V_INVALIDATED, "NO_TRADE")
    assert po.python_verdict(status(po.STATUS_TARGET), "BUY") == (po.V_TARGET, "NO_TRADE")
    assert po.python_verdict(status(po.STATUS_EXTENDED), "BUY") == (po.V_EXTENDED, "WAIT")
    assert po.python_verdict(status(po.STATUS_AGAINST), "BUY") == (po.V_WEAKENED, "WAIT")
    assert po.python_verdict(status(po.STATUS_IN_ZONE), "BUY") == (po.V_CONFIRMED, "BUY")
    assert po.python_verdict(status(po.STATUS_IN_ZONE), "WAIT") == (po.V_CONFIRMED, "WAIT")
    assert po.python_verdict(status(po.STATUS_IN_ZONE, vwap_favorable=False), "BUY") \
        == (po.V_WEAKENED, "WAIT")
    assert po.python_verdict(status(po.STATUS_IN_ZONE, rr_now=1.0), "BUY", 1.5) \
        == (po.V_WEAKENED, "WAIT")
    assert po.python_verdict(status(po.STATUS_NO_DATA), "BUY") == (po.V_NO_DATA, "BUY")


def test_texto_de_accion_menciona_el_hecho():
    bars = bars_from([(100, 100.2, 99.8, 100, 1)] * 3 + [(100, 100, 97.0, 97.5, 1)] * 3)
    p, live = pick(), live_for(bars)
    assessment = po.assess_pick(p, live, bars)
    text = po.default_action(p, assessment, live)
    assert "98.0" in text and "09:45" in text and "no entrar" in text.lower()


# --------------------------------------------------------------------------- #
# Mañana → estado de partida
# --------------------------------------------------------------------------- #
def make_record():
    def cand(ticker, rank, score):
        return {"ticker": ticker, "rank": rank, "score": score,
                "analysis": {"premarket": {"prev_close": 100.0, "premarket_price": 102.0},
                             "gap_pct": 2.0, "indicators": {"atr14": 3.0},
                             "rvol": {"rvol": 1.5}},
                "news": {"items": [{"headline": "Viejo titular"}]}}
    return {
        "date": "2026-09-21", "no_trade": False, "sent": True,
        "candidates": [cand("A", 1, 60), cand("B", 2, 55), cand("C", 3, 50), cand("D", 4, 40)],
        "final_selection": [pick(ticker="B"), pick(ticker="A", decision="WAIT")],
        "gemini_validated": {"available": True, "market_view": "Mercado tranquilo"},
        "market_data": {"context": {"quotes": {"SPY": {"prev_close": 500.0, "price": 502.0,
                                                       "change_pct": 0.4}}}},
        "timestamps": {"snapshot": "2026-09-21T08:45:00-04:00",
                       "sent": "2026-09-21T09:00:03-04:00"},
        "universe": {"count": 135},
    }


def test_estado_de_la_manana_y_seleccion_de_tickers():
    morning = po.load_morning_state(make_record())
    assert [p["ticker"] for p in morning["picks"]] == ["B", "A"]
    assert morning["market_view"] == "Mercado tranquilo" and morning["sent"]
    picks, others = po.select_review_tickers(morning, max_tickers=3)
    assert picks == ["B", "A"] and others == ["C"]   # las selecciones primero; tope de 3
    ref = po.morning_reference(morning, "A")
    assert ref["prev_close"] == 100.0 and ref["atr14"] == 3.0 and ref["rvol_premarket"] == 1.5


def test_registro_vacio_o_raro_no_lanza_error():
    morning = po.load_morning_state({})
    assert morning["picks"] == [] and morning["candidates"] == []
    assert po.select_review_tickers(morning) == ([], [])


# --------------------------------------------------------------------------- #
# Mercado y noticias
# --------------------------------------------------------------------------- #
def test_comparacion_de_mercado_usa_el_cierre_de_la_manana():
    context = {"quotes": {"SPY": {"prev_close": 500.0, "price": 502.0, "change_pct": 0.4}}}
    spy = bars_from([(501, 501.5, 500.5, 501, 1)] * 5 + [(501, 503.6, 500.9, 505.0, 1)])
    rows = po.compare_market(context, {"SPY": spy}, UPDATE_TS, symbols=["SPY", "QQQ"])
    assert rows[0]["morning_change_pct"] == 0.4 and rows[0]["now_change_pct"] == 1.0
    assert rows[1]["now_change_pct"] == po.NA   # QQQ sin datos: no se inventa


def test_solo_cuentan_las_noticias_nuevas_y_posteriores_al_informe():
    since = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)
    morning = {"items": [{"headline": "Viejo titular"}]}
    fresh = {"items": [
        {"headline": "viejo  TITULAR", "published_at": "2026-09-21T09:40:00-04:00"},
        {"headline": "Nuevo 1", "publisher": "A", "published_at": "2026-09-21T09:10:00-04:00"},
        {"headline": "Nuevo 2", "publisher": "B", "published_at": "2026-09-21T09:41:00-04:00",
         "is_hard_catalyst": True},
        {"headline": "Anterior al informe", "published_at": "2026-09-21T07:00:00-04:00"},
        {"headline": "Sin fecha", "published_at": "no es una fecha"},
    ]}
    result = po.new_headlines(morning, fresh, since)
    assert [n["headline"] for n in result] == ["Nuevo 2", "Nuevo 1"]  # más reciente primero
    assert result[0]["is_hard_catalyst"] is True
    assert po.new_headlines(morning, None, since) == []
    assert len(po.new_headlines({}, fresh, since, limit=1)) == 1


def test_refresh_news_apagado_o_con_fallo_devuelve_vacio(monkeypatch):
    assert po.refresh_news([], object(), UPDATE_TS) == {}
    import news
    monkeypatch.setattr(news, "build_providers", lambda env: (_ for _ in ()).throw(RuntimeError("x")))
    assert po.refresh_news(["A"], config.EnvSettings(), UPDATE_TS) == {}


# --------------------------------------------------------------------------- #
# Correcciones tras la primera ejecución con datos reales
# --------------------------------------------------------------------------- #
def test_extendida_no_manda_esperar_un_retroceso_a_la_zona():
    # Caso real (ARM): la zona 320-324 está por encima del límite válido 318.60 y el precio
    # (321.9) ya está dentro de la zona: «retroceso a la zona» no tenía sentido.
    p = pick(entry_zone_low=320.0, entry_zone_high=324.0, max_valid_entry=318.6,
             stop=306.34, target_1=336.98, target_2=346.45)
    bars = flat_bars(321.9)
    live = live_for(bars, prev_close=306.0)
    assessment = po.assess_pick(p, live, bars)
    text = po.default_action(p, assessment, live)
    assert assessment["status"] == po.STATUS_EXTENDED
    assert "retroceso hasta 318.60 o menos" in text and "a la zona" not in text
    assert "ya estaba más allá del límite" in assessment["note"]


def test_extendida_en_short_pide_retroceso_hacia_arriba():
    p = pick(direction="SHORT", entry_zone_low=99.0, entry_zone_high=100.0, max_valid_entry=98.5,
             stop=102.0, target_1=94.0, target_2=92.0)
    bars = flat_bars(97.5)
    live = live_for(bars)
    text = po.default_action(p, po.assess_pick(p, live, bars), live)
    assert "retroceso hasta 98.50 o más" in text


def test_zona_coherente_no_genera_la_nota_de_zona_inalcanzable():
    assert assess(flat_bars(100.5))["note"] == ""


def test_noticias_posteriores_a_la_hora_de_los_datos_no_cuentan():
    since = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)
    until = datetime(2026, 9, 21, 10, 0, tzinfo=NY_TZ)
    fresh = {"items": [
        {"headline": "Dentro", "published_at": "2026-09-21T09:30:00-04:00"},
        {"headline": "Del día siguiente", "published_at": "2026-09-22T19:21:00-04:00"},
    ]}
    assert [n["headline"] for n in po.new_headlines({}, fresh, since, until=until)] == ["Dentro"]
    assert len(po.new_headlines({}, fresh, since)) == 2      # sin cota, como antes
