"""Tests de gemini_analyzer.py con respuestas simuladas (sin llamar a la API real)."""
from __future__ import annotations

import json
from datetime import date, datetime

import config
import gemini_analyzer as ga
from technical_analysis import NA
from utils import NY_TZ

SESSION = date(2026, 9, 21)
SNAPSHOT = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)
FAST_CFG = config.GeminiConfig(retry_base_delay=0.0)


# --------------------------------------------------------------------------- #
# Datos de Python de la candidata
# --------------------------------------------------------------------------- #
def make_analysis(ticker="TEST", price=104.0):
    return {
        "ticker": ticker,
        "premarket": {
            "prev_close": 100.0, "premarket_price": price, "premarket_high": 104.5,
            "premarket_low": 100.5, "premarket_volume": 300_000.0,
            "premarket_quality": "ok", "quality_note": "",
        },
        "gap_pct": 4.0, "gap_atr": 2.0,
        "indicators": {"atr14": 2.0, "rsi14": 58.0, "sma20": 98.0, "sma50": 95.0},
        "rvol": {"rvol": 3.0, "reason": ""},
        "pm_pct_of_adv": 6.0,
        "levels": [
            {"label": "max_20d", "price": 110.0, "side": "resistance"},
            {"label": "premarket_high", "price": 104.5, "side": "resistance"},
            {"label": "prev_close", "price": 100.0, "side": "support"},
            {"label": "sma20", "price": 98.0, "side": "support"},
        ],
    }


CANDIDATES = {"TEST": make_analysis()}


def good_pick(**overrides):
    pick = {
        "ticker": "TEST", "company": "Test Corp", "direction": "LONG",
        "catalyst": "Resultados mejores de lo esperado", "catalyst_source": "Reuters",
        "trend": "Alcista", "technical_summary": "Sobre SMA20 y SMA50",
        "support_level": "prev_close", "resistance_level": "max_20d",
        "fade_risk": "LOW", "fade_risk_explanation": "Volumen alto y catalizador claro",
        "entry_zone_low": 103.8, "entry_zone_high": 104.2,
        "stop": 101.5, "target_1": 110.0, "target_2": None,
        "exit_time": "15:30 ET", "invalidation": "Pérdida de 101,5",
        "decision": "BUY", "confidence": "HIGH", "reason": "Configuración limpia",
    }
    pick.update(overrides)
    return pick


def response(picks, **overrides):
    payload = {"market_view": "Mercado tranquilo", "no_trade": False,
               "no_trade_reason": "", "picks": picks}
    payload.update(overrides)
    return payload


class FakeClient:
    """Doble del cliente del SDK: devuelve textos fijos, uno por intento."""

    class _Response:
        def __init__(self, text):
            self.text = text

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    class _Models:
        def __init__(self, outer):
            self.outer = outer

        def generate_content(self, **kwargs):
            self.outer.calls.append(kwargs)
            item = self.outer.texts.pop(0)
            if isinstance(item, Exception):
                raise item
            return FakeClient._Response(item)

    @property
    def models(self):
        return FakeClient._Models(self)


# --------------------------------------------------------------------------- #
# Utilidades de cálculo (esto lo hace Python, no Gemini)
# --------------------------------------------------------------------------- #
def test_rr_long_y_short():
    assert ga.compute_rr(104.0, 101.5, 110.0, "LONG") == 2.4
    assert ga.compute_rr(100.0, 103.0, 91.0, "SHORT") == 3.0
    assert ga.compute_rr(104.0, 105.0, 110.0, "LONG") is None


def test_precio_maximo_de_entrada_mantiene_el_rr_minimo():
    limit = ga.max_valid_entry(98.0, 106.0, "LONG", 1.5)
    assert limit == 101.2
    assert ga.compute_rr(limit, 98.0, 106.0, "LONG") == 1.5
    assert ga.max_valid_entry(103.0, 91.0, "SHORT", 1.5) == 98.2


def test_nivel_apoyado_en_la_tabla():
    levels = make_analysis()["levels"]
    assert ga.level_supported(110.0, levels, 2.0, config.GEMINI) is True
    assert ga.level_supported(111.0, levels, 2.0, config.GEMINI) is True   # 0,5 ATR
    assert ga.level_supported(120.0, levels, 2.0, config.GEMINI) is False
    assert ga.level_supported(120.0, [], 2.0, config.GEMINI) is True       # sin tabla


# --------------------------------------------------------------------------- #
# Parseo del JSON
# --------------------------------------------------------------------------- #
def test_extract_json_admite_vallas_de_codigo():
    assert ga.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert ga.extract_json('{"a": 1}') == {"a": 1}
    assert ga.extract_json('Aquí tienes:\n{"a": 1}\nUn saludo') == {"a": 1}


def test_extract_json_corrupto_lanza_error():
    for bad in ("", "no es json", "[1, 2]"):
        try:
            ga.extract_json(bad)
        except ga.GeminiError:
            continue
        raise AssertionError(f"debería haber fallado con {bad!r}")


# --------------------------------------------------------------------------- #
# Llamada con reintentos
# --------------------------------------------------------------------------- #
def test_reintenta_y_acierta_al_segundo_intento():
    client = FakeClient([RuntimeError("429 rate limit"),
                         json.dumps(response([good_pick()]))])
    raw = ga.call_gemini({}, "clave", "modelo", "long_only", FAST_CFG, client)
    assert raw["picks"][0]["ticker"] == "TEST"
    assert len(client.calls) == 2


def test_falla_tras_agotar_los_intentos():
    client = FakeClient(["texto que no es json", "sigue sin ser json"])
    try:
        ga.call_gemini({}, "clave", "modelo", "long_only", FAST_CFG, client)
    except ga.GeminiError:
        assert len(client.calls) == 2
        return
    raise AssertionError("debería haber lanzado GeminiError")


def test_la_peticion_lleva_schema_y_temperatura_baja():
    client = FakeClient([json.dumps(response([]))])
    ga.call_gemini({}, "clave", "gemini-2.5-flash", "long_only", FAST_CFG, client)
    sent = client.calls[0]["config"]
    assert sent["response_mime_type"] == "application/json"
    assert sent["response_schema"]["properties"]["picks"]["type"] == "array"
    assert sent["temperature"] <= 0.3
    assert "no inventes" in sent["system_instruction"].lower()


def test_schema_solo_admite_long_en_modo_solo_largos():
    long_only = ga.response_schema("long_only")["properties"]["picks"]["items"]
    assert long_only["properties"]["direction"]["enum"] == ["LONG"]
    both = ga.response_schema("both")["properties"]["picks"]["items"]
    assert both["properties"]["direction"]["enum"] == ["LONG", "SHORT"]


# --------------------------------------------------------------------------- #
# Validación
# --------------------------------------------------------------------------- #
def test_respuesta_valida_produce_un_pick_con_rr_de_python():
    result = ga.validate_response(response([good_pick()]), CANDIDATES)
    assert result.available is True and result.no_trade is False
    pick = result.picks[0]
    assert pick["rr"] == 2.4
    assert pick["rr_computed_by"] == "python"
    assert pick["entry_mid"] == 104.0
    assert pick["max_valid_entry"] == 104.9
    assert pick["decision"] == "BUY"


def test_ticker_inventado_se_descarta():
    result = ga.validate_response(response([good_pick(ticker="FAKE")]), CANDIDATES)
    assert result.picks == []
    assert result.no_trade is True
    assert any("no estaba entre las candidatas" in w for w in result.warnings)


def test_niveles_incoherentes_se_descartan():
    malo = good_pick(stop=106.0, target_1=102.0)      # stop por encima de la entrada
    result = ga.validate_response(response([malo]), CANDIDATES)
    assert result.picks == []
    assert any("incoherentes" in w for w in result.warnings)


def test_stop_fuera_de_la_tabla_se_descarta():
    result = ga.validate_response(response([good_pick(stop=80.0)]), CANDIDATES)
    assert result.picks == []
    assert any("no se apoya en la tabla" in w for w in result.warnings)


def test_objetivo_2_sin_apoyo_se_ignora_sin_perder_el_pick():
    result = ga.validate_response(response([good_pick(target_2=140.0)]), CANDIDATES)
    assert len(result.picks) == 1
    assert result.picks[0]["target_2"] is None
    assert any("objetivo 2" in w for w in result.warnings)


def test_rr_insuficiente_pasa_a_wait():
    flojo = good_pick(stop=98.0, target_1=106.0, entry_zone_low=103.9,
                      entry_zone_high=104.1)
    result = ga.validate_response(response([flojo]), CANDIDATES)
    pick = result.picks[0]
    assert pick["rr"] < config.MIN_RR
    assert pick["decision"] == "WAIT"
    assert any("por debajo del mínimo" in w for w in result.warnings)


def test_rr_insuficiente_se_descarta_si_se_configura_asi():
    cfg = config.GeminiConfig(downgrade_low_rr_to_wait=False)
    flojo = good_pick(stop=98.0, target_1=106.0)
    result = ga.validate_response(response([flojo]), CANDIDATES, cfg=cfg)
    assert result.picks == []


def test_los_numeros_de_python_prevalecen_y_se_avisa():
    mentiroso = good_pick(premarket_price=95.0, gap_pct=-9.0, rvol=0.1)
    result = ga.validate_response(response([mentiroso]), CANDIDATES)
    pick = result.picks[0]
    assert pick["premarket_price"] == 104.0
    assert pick["gap_pct"] == 4.0
    assert pick["rvol"] == 3.0
    assert sum("prevalece Python" in w for w in result.warnings) == 3


def test_short_se_descarta_en_modo_solo_largos():
    corto = good_pick(direction="SHORT", stop=110.0, target_1=98.0,
                      entry_zone_low=103.8, entry_zone_high=104.2)
    assert ga.validate_response(response([corto]), CANDIDATES).picks == []
    largo = ga.validate_response(response([corto]), CANDIDATES, direction_mode="both")
    assert len(largo.picks) == 1
    assert largo.picks[0]["direction"] == "SHORT"
    assert largo.picks[0]["rr"] == ga.compute_rr(104.0, 110.0, 98.0, "SHORT")


def test_maximo_tres_picks_ordenados_por_confianza_y_rr():
    candidates = {f"T{i}": make_analysis(f"T{i}") for i in range(5)}
    picks = [
        good_pick(ticker="T0", confidence="LOW"),
        good_pick(ticker="T1", confidence="HIGH"),
        good_pick(ticker="T2", confidence="MEDIUM"),
        good_pick(ticker="T3", confidence="HIGH", target_1=109.0),
        good_pick(ticker="T4", confidence="LOW"),
    ]
    result = ga.validate_response(response(picks), candidates)
    assert [p["ticker"] for p in result.picks] == ["T1", "T3", "T2"]
    assert any("se conservan los 3 mejores" in w for w in result.warnings)


def test_pick_repetido_se_descarta():
    result = ga.validate_response(response([good_pick(), good_pick()]), CANDIDATES)
    assert len(result.picks) == 1
    assert any("repetido" in w for w in result.warnings)


def test_decision_y_confianza_invalidas_se_corrigen():
    raro = good_pick(decision="COMPRAR YA", confidence="ALTÍSIMA", fade_risk="???")
    pick = ga.validate_response(response([raro]), CANDIDATES).picks[0]
    assert pick["decision"] == "WAIT"
    assert pick["confidence"] == "LOW"
    assert pick["fade_risk"] == "HIGH"


def test_etiqueta_de_nivel_inventada_pasa_a_na():
    raro = good_pick(resistance_level="resistencia_magica")
    pick = ga.validate_response(response([raro]), CANDIDATES).picks[0]
    assert pick["resistance_level"] == NA
    assert pick["support_level"] == "prev_close"


def test_no_trade_de_gemini_se_respeta():
    result = ga.validate_response(
        response([], no_trade=True, no_trade_reason="Mercado sin dirección"), CANDIDATES)
    assert result.no_trade is True
    assert result.no_trade_reason == "Mercado sin dirección"
    assert result.available is True


def test_picks_no_es_una_lista():
    result = ga.validate_response({"market_view": "x", "picks": "nada"}, CANDIDATES)
    assert result.no_trade is True
    assert any("no traía una lista" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# analyze(): degradación con elegancia
# --------------------------------------------------------------------------- #
def _candidates_payload():
    return [{"ticker": "TEST", "rank": 1, "direction": "LONG", "score": 80.0,
             "score_breakdown": {}, "penalties": [],
             "analysis": make_analysis(), "news": {"catalyst_confirmed": True}}]


def test_analyze_sin_clave_degrada():
    result = ga.analyze(SESSION, SNAPSHOT, "sesión normal", None, _candidates_payload())
    assert result.available is False
    assert ga.UNAVAILABLE_NOTE in result.no_trade_reason
    assert result.picks == []


def test_analyze_sin_candidatas_no_llama_a_gemini():
    result = ga.analyze(SESSION, SNAPSHOT, "sesión normal", None, [],
                        api_key="clave", client=FakeClient([]))
    assert result.available is False
    assert "candidatas" in result.no_trade_reason


def test_analyze_con_json_corrupto_degrada():
    client = FakeClient(["no json", "tampoco"])
    result = ga.analyze(SESSION, SNAPSHOT, "sesión normal", None, _candidates_payload(),
                        api_key="clave", cfg=FAST_CFG, client=client)
    assert result.available is False
    assert result.picks == []
    assert result.note == ga.UNAVAILABLE_NOTE


def test_analyze_camino_feliz():
    client = FakeClient([json.dumps(response([good_pick()]))])
    result = ga.analyze(SESSION, SNAPSHOT, "sesión normal", None, _candidates_payload(),
                        api_key="clave", model="gemini-2.5-flash", cfg=FAST_CFG,
                        client=client)
    assert result.available is True
    assert result.picks[0]["rr"] == 2.4
    enviado = json.loads(client.calls[0]["contents"])
    assert enviado["candidatas"][0]["ticker"] == "TEST"
    assert enviado["candidatas"][0]["levels"][0]["label"] == "max_20d"
    assert enviado["fecha"] == "2026-09-21"


def test_payload_incluye_calidad_del_dato_y_noticias():
    payload = ga.build_payload(SESSION, SNAPSHOT, "cierre anticipado", None,
                               _candidates_payload())
    candidata = payload["candidatas"][0]
    assert candidata["premarket"]["quality"] == "ok"
    assert candidata["news"]["catalyst_confirmed"] is True
    assert payload["tipo_de_sesion"] == "cierre anticipado"


def test_resultado_es_serializable_a_json():
    result = ga.validate_response(response([good_pick()]), CANDIDATES)
    assert "TEST" in json.dumps(result.to_dict(), ensure_ascii=False)
