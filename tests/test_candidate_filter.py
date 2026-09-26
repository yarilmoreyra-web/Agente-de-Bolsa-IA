"""Tests de candidate_filter.py: puertas duras, puntuación, penalizaciones y selección."""
from __future__ import annotations

import json

import candidate_filter as cf
import config
from technical_analysis import NA


# --------------------------------------------------------------------------- #
# Constructor de análisis sintéticos
# --------------------------------------------------------------------------- #
def make_analysis(
    ticker="TEST", price=104.0, prev_close=100.0, pm_high=104.5, pm_low=100.5,
    pm_volume=300_000.0, quality="ok", rvol=3.0, rvol_reason="",
    avg_volume=5_000_000.0, dollar_volume=500_000_000.0, atr=2.0,
    sma20=98.0, sma50=95.0, ema9=101.0, ema20=99.0,
    resistance_atr=2.0, rs_pm=1.5, pm_pct_of_adv=6.0,
):
    """Analítica de un ticker con la forma que devuelve ``technical_analysis``."""
    gap = round((price / prev_close - 1) * 100, 2) if prev_close else NA
    resistance = NA if resistance_atr is None else {
        "label": "max_20d", "price": round(price + resistance_atr * atr, 2),
        "side": "resistance", "distance_pct": round(resistance_atr * atr / price * 100, 2),
        "distance_atr": resistance_atr,
    }
    return {
        "ticker": ticker,
        "session_date": "2026-09-21",
        "premarket": {
            "ticker": ticker, "prev_close": prev_close, "premarket_price": price,
            "premarket_high": pm_high, "premarket_low": pm_low,
            "premarket_volume": pm_volume, "premarket_quality": quality, "n_bars": 30,
        },
        "indicators": {
            "last_close": prev_close, "rsi14": 58.0, "sma20": sma20, "sma50": sma50,
            "ema9": ema9, "ema20": ema20, "atr14": atr, "atr_pct": 2.0,
            "avg_volume_20d": avg_volume, "avg_dollar_volume_20d": dollar_volume,
            "max_20d": 110.0, "return_5d_pct": 3.0,
        },
        "gap_pct": gap,
        "gap_atr": round((price - prev_close) / atr, 2) if atr else NA,
        "premarket_range_pct": 4.0, "premarket_range_atr": 2.0,
        "rvol": {"rvol": rvol, "numerator": pm_volume, "denominator": 100_000.0,
                 "sessions_used": 20, "reason": rvol_reason},
        "pm_pct_of_adv": pm_pct_of_adv,
        "relative_strength": {"rs_pm_vs_spy": rs_pm, "rs_pm_vs_qqq": rs_pm,
                              "rs_5d_vs_spy": 1.0, "rs_5d_vs_qqq": 1.0},
        "levels": [],
        "nearest_support": {"label": "sma20", "price": sma20, "side": "support",
                            "distance_pct": -5.0, "distance_atr": -3.0},
        "nearest_resistance": resistance,
    }


CATALYST = {"catalyst_confirmed": True, "catalyst_category": "Resultados",
            "has_soft_news": False}
SOFT_NEWS = {"catalyst_confirmed": False, "catalyst_category": "", "has_soft_news": True}
NO_NEWS = {"catalyst_confirmed": False, "catalyst_category": "", "has_soft_news": False}


class FakeContext:
    def __init__(self, regime):
        self.regime = regime


GOOD_MARKET = FakeContext({"against_longs": False, "tone": "neutral", "label": "neutral"})
BAD_MARKET = FakeContext({"against_longs": True, "tone": "negativo",
                          "label": "adverso para largos"})


# --------------------------------------------------------------------------- #
# Puertas duras
# --------------------------------------------------------------------------- #
def test_candidata_valida_pasa_las_puertas():
    assert cf.apply_hard_gates(make_analysis()) == []


def test_puerta_precio_minimo():
    failures = cf.apply_hard_gates(make_analysis(price=4.0, prev_close=3.8))
    assert any("precio" in f for f in failures)


def test_puerta_volumen_medio():
    failures = cf.apply_hard_gates(make_analysis(avg_volume=300_000.0))
    assert any("volumen medio 20d" in f for f in failures)


def test_puerta_volumen_en_dolares():
    failures = cf.apply_hard_gates(make_analysis(dollar_volume=5_000_000.0))
    assert any("dólares" in f for f in failures)


def test_puerta_volumen_premarket():
    failures = cf.apply_hard_gates(make_analysis(pm_volume=1_000.0))
    assert any("volumen pre-market" in f for f in failures)


def test_puerta_gap_minimo():
    failures = cf.apply_hard_gates(make_analysis(price=100.4, prev_close=100.0))
    assert any("gap" in f for f in failures)


def test_puerta_gap_negativo_en_solo_largos():
    failures = cf.apply_hard_gates(make_analysis(price=96.0, prev_close=100.0),
                                   direction_mode="long_only")
    assert any("solo largos" in f for f in failures)


def test_gap_negativo_es_valido_en_modo_both():
    assert cf.apply_hard_gates(make_analysis(price=96.0, prev_close=100.0),
                               direction_mode="both") == []


def test_puerta_calidad_del_dato():
    failures = cf.apply_hard_gates(make_analysis(quality="missing"))
    assert any("missing" in f for f in failures)


def test_puertas_acumulan_varios_motivos():
    failures = cf.apply_hard_gates(
        make_analysis(price=3.0, prev_close=2.99, pm_volume=10.0, avg_volume=1000.0))
    assert len(failures) >= 3


# --------------------------------------------------------------------------- #
# Pre-filtro de noticias
# --------------------------------------------------------------------------- #
def test_prefiltro_ordena_por_gap_y_limita():
    analyses = {
        "AAA": make_analysis("AAA", price=102.0),
        "BBB": make_analysis("BBB", price=108.0),
        "CCC": make_analysis("CCC", price=105.0),
        "PLANO": make_analysis("PLANO", price=100.2),
        "SIN_VOL": make_analysis("SIN_VOL", price=106.0, pm_volume=100.0),
    }
    assert cf.prefilter_for_news(analyses, limit=10) == ["BBB", "CCC", "AAA"]
    assert cf.prefilter_for_news(analyses, limit=2) == ["BBB", "CCC"]


def test_prefiltro_descarta_calidad_missing():
    analyses = {"AAA": make_analysis("AAA", price=108.0, quality="missing")}
    assert cf.prefilter_for_news(analyses) == []


# --------------------------------------------------------------------------- #
# Puntuación
# --------------------------------------------------------------------------- #
def test_candidata_fuerte_puntua_alto():
    row = cf.score_candidate(make_analysis(), CATALYST, GOOD_MARKET)
    assert row.passed_gates is True
    assert float(row.score) >= 75
    assert row.penalties == []
    assert sum(c["weight"] for c in row.components.values()) == 100.0


def test_todos_los_componentes_estan_en_el_desglose():
    row = cf.score_candidate(make_analysis(), CATALYST, GOOD_MARKET)
    assert set(row.components) == {
        "rvol", "gap", "catalyst", "liquidity", "relative_strength",
        "technical", "premarket_behavior",
    }


def test_sin_catalizador_puntua_menos_y_penaliza():
    con = cf.score_candidate(make_analysis(), CATALYST, GOOD_MARKET)
    sin = cf.score_candidate(make_analysis(), NO_NEWS, GOOD_MARKET)
    assert float(sin.score) < float(con.score)
    assert any(p["flag"] == "sin_catalizador" for p in sin.penalties)


def test_noticia_blanda_puntua_entre_medias():
    blanda = cf.score_candidate(make_analysis(), SOFT_NEWS, GOOD_MARKET)
    assert 0 < blanda.components["catalyst"]["points"] < 20


def test_rvol_ausente_usa_pm_pct_of_adv_con_tope():
    row = cf.score_candidate(
        make_analysis(rvol=NA, rvol_reason="solo 4 sesiones válidas"),
        CATALYST, GOOD_MARKET)
    component = row.components["rvol"]
    assert 0 < component["points"] <= config.FILTER.weights.rvol * 0.5
    assert "pm_pct_of_adv" in component["detail"]


def test_rvol_ausente_sin_alternativa_puntua_cero():
    row = cf.score_candidate(
        make_analysis(rvol=NA, pm_pct_of_adv=NA), CATALYST, GOOD_MARKET)
    assert row.components["rvol"]["points"] == 0.0


def test_estructura_tecnica_debil_resta_puntos():
    fuerte = cf.score_candidate(make_analysis(), CATALYST, GOOD_MARKET)
    debil = cf.score_candidate(
        make_analysis(sma20=110.0, sma50=115.0, ema9=99.0, ema20=101.0),
        CATALYST, GOOD_MARKET)
    assert debil.components["technical"]["points"] < fuerte.components["technical"]["points"]


def test_comportamiento_premarket_premia_estar_cerca_del_maximo():
    cerca = cf.score_candidate(make_analysis(price=104.4, pm_high=104.5, pm_low=100.5),
                               CATALYST, GOOD_MARKET)
    lejos = cf.score_candidate(make_analysis(price=101.0, pm_high=104.5, pm_low=100.5),
                               CATALYST, GOOD_MARKET)
    assert cerca.components["premarket_behavior"]["points"] > \
        lejos.components["premarket_behavior"]["points"]


# --------------------------------------------------------------------------- #
# Penalizaciones de gap-and-fade
# --------------------------------------------------------------------------- #
def _flags(row) -> set[str]:
    return {p["flag"] for p in row.penalties}


def test_penalizacion_por_extension_excesiva():
    row = cf.score_candidate(make_analysis(price=110.0, pm_high=110.5, atr=2.0),
                             CATALYST, GOOD_MARKET)
    assert "extension_excesiva" in _flags(row)


def test_penalizacion_por_alejarse_del_maximo_premarket():
    row = cf.score_candidate(make_analysis(price=102.0, pm_high=106.0, pm_low=100.5),
                             CATALYST, GOOD_MARKET)
    assert "alejado_del_extremo" in _flags(row)


def test_penalizacion_por_resistencia_demasiado_cerca():
    row = cf.score_candidate(make_analysis(resistance_atr=0.2), CATALYST, GOOD_MARKET)
    assert "nivel_demasiado_cerca" in _flags(row)


def test_penalizacion_por_volumen_bajo_para_el_gap():
    row = cf.score_candidate(make_analysis(price=105.0, rvol=1.1), CATALYST, GOOD_MARKET)
    assert "volumen_bajo_para_el_gap" in _flags(row)


def test_penalizacion_por_mercado_en_contra():
    row = cf.score_candidate(make_analysis(), CATALYST, BAD_MARKET)
    assert "mercado_en_contra" in _flags(row)


def test_sin_contexto_no_hay_penalizacion_de_mercado():
    row = cf.score_candidate(make_analysis(), CATALYST, None)
    assert "mercado_en_contra" not in _flags(row)


def test_la_puntuacion_nunca_baja_de_cero():
    row = cf.score_candidate(
        make_analysis(price=112.0, pm_high=118.0, pm_low=100.5, rvol=1.0,
                      dollar_volume=20_000_000.0, rs_pm=-3.0, resistance_atr=0.1,
                      sma20=130.0, sma50=140.0, ema9=99.0, ema20=101.0),
        NO_NEWS, BAD_MARKET)
    assert float(row.score) >= 0.0


# --------------------------------------------------------------------------- #
# Modo "both" (cortos)
# --------------------------------------------------------------------------- #
def test_modo_both_evalua_un_gap_negativo_como_short():
    analysis = make_analysis(price=95.0, prev_close=100.0, pm_high=99.5, pm_low=94.8,
                             sma20=105.0, sma50=110.0, ema9=99.0, ema20=101.0,
                             rs_pm=-2.0, resistance_atr=None)
    row = cf.score_candidate(analysis, CATALYST, GOOD_MARKET,
                             direction=cf.resolve_direction(analysis, "both"))
    assert row.direction == cf.SHORT
    assert row.components["gap"]["points"] > 0
    assert row.components["relative_strength"]["points"] > 0
    assert row.components["technical"]["points"] > 5


# --------------------------------------------------------------------------- #
# Selección y filter_log
# --------------------------------------------------------------------------- #
def _universe():
    return {
        "STRONG": make_analysis("STRONG"),
        "MEDIO": make_analysis("MEDIO", price=101.5, rvol=1.4, rs_pm=0.1,
                               pm_volume=60_000.0),
        "BARATA": make_analysis("BARATA", price=4.5, prev_close=4.3),
        "ILIQUIDA": make_analysis("ILIQUIDA", avg_volume=100_000.0,
                                  dollar_volume=1_000_000.0),
        "PLANA": make_analysis("PLANA", price=100.3),
    }


def test_seleccion_ordena_y_marca_el_log():
    news = {"STRONG": CATALYST, "MEDIO": NO_NEWS}
    outcome = cf.filter_candidates(_universe(), news, GOOD_MARKET)
    assert outcome.candidates[0] == "STRONG"
    assert outcome.no_trade is False
    assert len(outcome.filter_log) == 5          # todos los tickers, pasen o no
    excluded = {row["ticker"]: row for row in outcome.filter_log if not row["passed_gates"]}
    assert set(excluded) == {"BARATA", "ILIQUIDA", "PLANA"}
    assert all(row["exclusion_reason"] for row in excluded.values())
    assert outcome.rows[0].rank == 1


def test_sin_candidatas_devuelve_no_operar():
    universe = {"PLANA": make_analysis("PLANA", price=100.3)}
    outcome = cf.filter_candidates(universe, {}, GOOD_MARKET)
    assert outcome.no_trade is True
    assert outcome.candidates == []
    assert "No se fuerza" in outcome.no_trade_reason


def test_puntuacion_insuficiente_queda_fuera_con_motivo():
    cfg = config.FilterConfig(min_score=95.0)
    outcome = cf.filter_candidates(_universe(), {"STRONG": CATALYST}, GOOD_MARKET, cfg)
    assert outcome.no_trade is True
    fila = next(r for r in outcome.filter_log if r["ticker"] == "STRONG")
    assert "puntuación" in fila["exclusion_reason"]


def test_se_respeta_el_maximo_de_candidatas():
    cfg = config.FilterConfig(max_candidates=2, min_score=0.0)
    universe = {f"T{i}": make_analysis(f"T{i}", price=102.0 + i) for i in range(5)}
    outcome = cf.filter_candidates(universe, {}, GOOD_MARKET, cfg)
    assert len(outcome.candidates) == 2
    sobrantes = [r for r in outcome.filter_log
                 if r["passed_gates"] and not r["selected"]]
    assert all("mejores" in r["exclusion_reason"] for r in sobrantes)


def test_un_ticker_defectuoso_no_detiene_el_filtro():
    universe = {"STRONG": make_analysis("STRONG"), "ROTO": {"ticker": "ROTO"}}
    outcome = cf.filter_candidates(universe, {"STRONG": CATALYST}, GOOD_MARKET)
    assert outcome.candidates == ["STRONG"]
    roto = next(r for r in outcome.filter_log if r["ticker"] == "ROTO")
    assert roto["passed_gates"] is False and roto["exclusion_reason"]


def test_filter_log_es_serializable_a_json():
    outcome = cf.filter_candidates(_universe(), {"STRONG": CATALYST}, GOOD_MARKET)
    assert "STRONG" in json.dumps(outcome.filter_log, ensure_ascii=False)


def test_payload_para_gemini_incluye_analisis_y_noticias():
    universe = _universe()
    outcome = cf.filter_candidates(universe, {"STRONG": CATALYST}, GOOD_MARKET)
    payload = cf.candidates_payload(outcome, universe, {"STRONG": CATALYST})
    assert payload[0]["ticker"] == "STRONG"
    assert payload[0]["news"]["catalyst_confirmed"] is True
    assert payload[0]["analysis"]["indicators"]["atr14"] == 2.0
    assert payload[0]["score_breakdown"]["rvol"]["weight"] == 25.0


def test_acepta_objetos_tickernews():
    import news as news_module

    ticker_news = news_module.TickerNews(ticker="STRONG", catalyst_confirmed=True)
    outcome = cf.filter_candidates(
        {"STRONG": make_analysis("STRONG")}, {"STRONG": ticker_news}, GOOD_MARKET)
    fila = outcome.filter_log[0]
    assert fila["metrics"]["catalyst_confirmed"] is True
