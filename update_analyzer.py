"""update_analyzer.py — Gemini interpreta qué ha cambiado desde el informe de las 09:00.

Misma filosofía que ``gemini_analyzer``: Python calcula, Gemini interpreta y
todo lo que devuelve pasa por validación antes de llegar al Telegram.

Lo que Gemini **puede** hacer
    * Valorar cada selección de la mañana (CONFIRMED / IMPROVED / WEAKENED /
      INVALIDATED), decidir BUY / WAIT / NO_TRADE y explicar en una frase qué
      cambió y qué hacer.
    * Señalar hasta ``UPDATE.watchlist_size`` candidatas NO seleccionadas para
      vigilar (sin niveles: no se inventan).

Lo que Gemini **no puede** hacer
    * Contradecir un hecho de Python. Si el precio tocó el stop, alcanzó el
      objetivo 1 o superó el máximo de entrada válido, el veredicto lo fija
      Python (``post_open.FORCED_VERDICTS``) y se descarta el texto de Gemini.
    * Dar BUY si el estado no es «en zona» o el R/R con el precio actual no
      alcanza el mínimo (pasa a WAIT).
    * Hablar de tickers que no estaban en el informe de la mañana.

Si la IA falla o no hay clave, ``python_only_review`` produce el mismo informe
solo con los hechos de Python (sin frases de IA).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import config
import gemini_analyzer as ga
import post_open as po

logger = logging.getLogger("trading_agent.update_analyzer")

AI_VERDICTS = (po.V_CONFIRMED, po.V_IMPROVED, po.V_WEAKENED, po.V_INVALIDATED)
DECISIONS = ga.DECISIONS
CONFIDENCES = ga.CONFIDENCES
UNAVAILABLE_NOTE = ga.UNAVAILABLE_NOTE


# --------------------------------------------------------------------------- #
# Instrucción y esquema
# --------------------------------------------------------------------------- #
def build_system_instruction() -> str:
    """Instrucción de sistema (español): Gemini compara y decide, no calcula."""
    return (
        "Eres un analista de day trading de acciones estadounidenses. Han pasado "
        "30 minutos desde la apertura del mercado y debes revisar las "
        "selecciones que un informe anterior (de las 09:00 ET, antes de la "
        "apertura) recomendó. Trabajas sobre datos ya calculados por un "
        "programa de Python.\n\n"
        "Reglas que no puedes romper:\n"
        "1. No calculas ni corriges nada. No inventes precios, volúmenes, "
        "noticias ni horas. Si falta un dato, escribe \"N/A\".\n"
        "2. El campo «estado_python» de cada selección es un HECHO: "
        "STOP_TOCADO significa que el precio ya llegó al stop; "
        "OBJETIVO_1_ALCANZADO que ya llegó al objetivo 1; EXTENDIDA que el "
        "precio superó el máximo de entrada válido; EN_CONTRA que está fuera "
        "de la zona de entrada sin tocar el stop; EN_ZONA que sigue en la "
        "zona. No lo contradigas.\n"
        "3. No propongas niveles nuevos: el stop, los objetivos y la zona de "
        "entrada son los del informe de la mañana. El R/R con el precio actual "
        "(rr_ahora) lo calcula Python.\n"
        "4. Para cada selección compara «mañana» con «ahora» y decide: "
        "verdict (CONFIRMED si la tesis sigue intacta, IMPROVED si mejora, "
        "WEAKENED si se debilita, INVALIDATED si deja de ser válida) y "
        "decision (BUY solo si aún es razonable entrar ahora; WAIT si conviene "
        "esperar; NO_TRADE si se descarta).\n"
        "5. Para valorar usa: movimiento desde la apertura, posición respecto "
        "al VWAP, volumen y RVOL de la apertura, si el precio respeta la zona "
        "de entrada y el soporte, noticias nuevas desde el informe y el "
        "comportamiento del mercado (SPY, QQQ, VIX) frente a las 08:45.\n"
        "6. Sé prudente: ante datos de calidad distinta de \"ok\" o RVOL no "
        "disponible, dilo y baja la confianza.\n"
        "7. En «watchlist» puedes citar como máximo las candidatas de la lista "
        "«otras_candidatas» que merezca la pena vigilar, con una frase de "
        "motivo. Puede quedar vacía. No propongas entrada, stop ni objetivos "
        "para ellas.\n"
        "8. Escribe todos los textos en español, claros y breves (una o dos "
        "frases por campo).\n\n"
        "Devuelve exclusivamente un JSON que cumpla el esquema indicado."
    )


def response_schema() -> dict[str, Any]:
    """Esquema JSON de la respuesta (subconjunto de OpenAPI que acepta el SDK)."""
    review = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "verdict": {"type": "string", "enum": list(AI_VERDICTS)},
            "decision": {"type": "string", "enum": list(DECISIONS)},
            "confidence": {"type": "string", "enum": list(CONFIDENCES)},
            "what_changed": {"type": "string"},
            "action": {"type": "string"},
        },
        "required": ["ticker", "verdict", "decision", "confidence",
                     "what_changed", "action"],
    }
    watch = {
        "type": "object",
        "properties": {"ticker": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["ticker", "reason"],
    }
    return {
        "type": "object",
        "properties": {
            "market_view": {"type": "string"},
            "summary": {"type": "string"},
            "reviews": {"type": "array", "items": review},
            "watchlist": {"type": "array", "items": watch},
        },
        "required": ["market_view", "summary", "reviews", "watchlist"],
    }


# --------------------------------------------------------------------------- #
# Entrada de Gemini
# --------------------------------------------------------------------------- #
def _compact_live(live: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("open_price", "price_now", "high_since_open", "low_since_open",
            "change_vs_open_pct", "change_vs_prev_close_pct", "vwap",
            "price_vs_vwap", "volume_since_open", "open_rvol", "quality",
            "quality_note")
    return {key: live.get(key, po.NA) for key in keys}


def _compact_morning(pick: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": pick.get("decision"), "confidence": pick.get("confidence"),
        "motivo": pick.get("reason"), "catalizador": pick.get("catalyst"),
        "riesgo_gap_and_fade": pick.get("fade_risk"),
        "gap_pct": reference.get("gap_pct", po.NA),
        "precio_premarket": reference.get("premarket_price", po.NA),
        "cierre_anterior": reference.get("prev_close", po.NA),
        "zona_entrada": [pick.get("entry_zone_low"), pick.get("entry_zone_high")],
        "max_entrada_valida": pick.get("max_valid_entry"),
        "stop": pick.get("stop"), "objetivo_1": pick.get("target_1"),
        "objetivo_2": pick.get("target_2"), "rr_manana": pick.get("rr"),
        "invalidacion": pick.get("invalidation"),
    }


def build_payload(
    session_date: Any, update_ts: Any, morning_view: str,
    market: Sequence[Mapping[str, Any]], picks: Sequence[Mapping[str, Any]],
    others: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """JSON que se envía a Gemini en la única llamada de la actualización.

    ``picks`` y ``others`` son las entradas que arma ``post_open.run_update``
    (con las claves ``ticker``, ``pick``, ``reference``, ``live``,
    ``assessment`` y ``news_new``).
    """
    return {
        "fecha": str(session_date),
        "hora_informe_manana_et": "09:00",
        "hora_actualizacion_et": po._hhmm(update_ts),
        "lectura_de_mercado_de_la_manana": morning_view,
        "mercado": [
            {"simbolo": row["symbol"],
             "variacion_pct_a_las_0845": row.get("morning_change_pct"),
             "variacion_pct_ahora": row.get("now_change_pct"),
             "precio_0845": row.get("morning_price"),
             "precio_ahora": row.get("now_price")}
            for row in market
        ],
        "selecciones": [
            {
                "ticker": e["ticker"], "empresa": e["pick"].get("company"),
                "direccion": e["assessment"].get("direction"),
                "manana": _compact_morning(e["pick"], e["reference"]),
                "ahora": _compact_live(e["live"]),
                "estado_python": e["assessment"].get("status"),
                "rr_ahora": e["assessment"].get("rr_now"),
                "rr_minimo": e["assessment"].get("rr_min"),
                "regla_de_apertura": e["assessment"].get("open_rule"),
                "distancia_al_stop_pct": e["assessment"].get("dist_to_stop_pct"),
                "distancia_al_objetivo_1_pct": e["assessment"].get("dist_to_target_1_pct"),
                "nota_python": e["assessment"].get("note"),
                "noticias_nuevas": [
                    {"titular": n["headline"], "medio": n["publisher"],
                     "hora": po._hhmm(n["published_at"]), "categoria": n["category"]}
                    for n in e.get("news_new", [])
                ],
            }
            for e in picks
        ],
        "otras_candidatas": [
            {
                "ticker": e["ticker"], "puntuacion_manana": e["reference"].get("score"),
                "gap_pct_manana": e["reference"].get("gap_pct", po.NA),
                "ahora": _compact_live(e["live"]),
                "noticias_nuevas": [n["headline"] for n in e.get("news_new", [])],
            }
            for e in others
        ],
    }


# --------------------------------------------------------------------------- #
# Validación de la respuesta
# --------------------------------------------------------------------------- #
def _text(value: Any) -> str:
    """Texto plano de una línea (sin saltos ni marcado que rompa el informe)."""
    if value is None:
        return ""
    flat = " ".join(str(value).replace("**", "").replace("\r", " ").split())
    return flat


def validate_response(
    raw: Mapping[str, Any], pick_tickers: Sequence[str], other_tickers: Sequence[str],
    cfg: config.UpdateConfig | None = None,
) -> dict[str, Any]:
    """Filtra la respuesta de Gemini y devuelve solo lo utilizable.

    * Descarta tickers que no estaban en la lista correspondiente y duplicados.
    * Valores fuera de los enumerados quedan en ``None`` (Python los rellena).
    * ``watchlist`` solo admite candidatas NO seleccionadas y como máximo
      ``watchlist_size``.
    Cada corrección queda como aviso en ``warnings``.
    """
    cfg = cfg or config.UPDATE
    warnings: list[str] = []
    allowed = {t.upper(): t for t in pick_tickers}
    reviews: dict[str, dict[str, Any]] = {}

    raw_reviews = raw.get("reviews") if isinstance(raw, Mapping) else None
    if not isinstance(raw_reviews, list):
        raw_reviews = []
        warnings.append("Gemini no devolvió la lista «reviews»; se usan solo los hechos de Python.")
    for item in raw_reviews:
        if not isinstance(item, Mapping):
            continue
        key = _text(item.get("ticker")).upper()
        ticker = allowed.get(key)
        if ticker is None:
            warnings.append(f"{key or '?'}: no es una selección del informe de la mañana; se ignora.")
            continue
        if ticker in reviews:
            warnings.append(f"{ticker}: valoración repetida; se usa la primera.")
            continue
        verdict, decision = item.get("verdict"), item.get("decision")
        confidence = item.get("confidence")
        if verdict not in AI_VERDICTS:
            warnings.append(f"{ticker}: verdict «{verdict}» no válido.")
            verdict = None
        if decision not in DECISIONS:
            warnings.append(f"{ticker}: decision «{decision}» no válida.")
            decision = None
        if confidence not in CONFIDENCES:
            confidence = None
        reviews[ticker] = {
            "verdict": verdict, "decision": decision, "confidence": confidence,
            "what_changed": _text(item.get("what_changed")),
            "action": _text(item.get("action")),
        }

    missing = [t for t in pick_tickers if t not in reviews]
    if missing and reviews:
        warnings.append(f"Gemini no valoró: {', '.join(missing)}; se usan solo los hechos de Python.")

    others_allowed = {t.upper(): t for t in other_tickers}
    watchlist: list[dict[str, str]] = []
    raw_watch = raw.get("watchlist") if isinstance(raw, Mapping) else None
    for item in raw_watch if isinstance(raw_watch, list) else []:
        if not isinstance(item, Mapping):
            continue
        ticker = others_allowed.get(_text(item.get("ticker")).upper())
        if ticker is None or any(w["ticker"] == ticker for w in watchlist):
            continue
        reason = _text(item.get("reason"))
        if reason:
            watchlist.append({"ticker": ticker, "reason": reason})
    if len(watchlist) > cfg.watchlist_size:
        watchlist = watchlist[: cfg.watchlist_size]

    return {
        "reviews": reviews, "watchlist": watchlist, "warnings": warnings,
        "market_view": _text(raw.get("market_view")) if isinstance(raw, Mapping) else "",
        "summary": _text(raw.get("summary")) if isinstance(raw, Mapping) else "",
    }


# --------------------------------------------------------------------------- #
# Fusión: hechos de Python + interpretación de Gemini
# --------------------------------------------------------------------------- #
def merge_pick(entry: Mapping[str, Any], review: Mapping[str, Any] | None,
               min_rr: float | None = None) -> dict[str, Any]:
    """Veredicto final de una selección: Python impone los hechos, Gemini el resto.

    Devuelve ``verdict``, ``decision``, ``confidence``, ``what_changed``,
    ``action``, ``source`` (``"gemini"`` o ``"python"``) y ``warnings``.
    """
    min_rr = config.TRADE.min_rr if min_rr is None else min_rr
    pick, live, assessment = entry["pick"], entry["live"], entry["assessment"]
    ticker = entry["ticker"]
    status = assessment.get("status")
    morning_decision = str(pick.get("decision") or "WAIT")
    py_verdict, py_decision = po.python_verdict(assessment, morning_decision, min_rr)
    warnings: list[str] = []

    base = {
        "verdict": py_verdict, "decision": py_decision,
        "confidence": pick.get("confidence") or "N/A",
        "what_changed": "", "action": po.default_action(pick, assessment, live),
        "source": "python", "warnings": warnings,
    }

    # Estados que Python impone: se descarta el texto de Gemini para que no contradiga.
    if status in po.FORCED_VERDICTS or status == po.STATUS_NO_DATA:
        if review and review.get("verdict") and review["verdict"] != py_verdict \
                and status != po.STATUS_NO_DATA:
            warnings.append(
                f"{ticker}: Gemini dijo {review['verdict']}/{review.get('decision')} pero el "
                f"estado es {status}; prevalece Python.")
        base["confidence"] = "N/A"
        if review and review.get("what_changed") and status != po.STATUS_NO_DATA:
            base["what_changed"] = ""   # el texto de la IA podría contradecir el hecho
        return base

    if not review or not review.get("verdict") or not review.get("decision"):
        return base

    verdict, decision = review["verdict"], review["decision"]
    # Coherencia interna de la respuesta.
    if verdict == po.V_INVALIDATED and decision != "NO_TRADE":
        warnings.append(f"{ticker}: INVALIDATED con decisión {decision}; pasa a NO_TRADE.")
        decision = "NO_TRADE"
    elif decision == "NO_TRADE" and verdict != po.V_INVALIDATED:
        warnings.append(f"{ticker}: NO_TRADE con veredicto {verdict}; pasa a INVALIDATED.")
        verdict = po.V_INVALIDATED

    # BUY solo si está en zona y el R/R actual llega al mínimo.
    if decision == "BUY":
        rr_now = po._num(assessment.get("rr_now"))
        if status != po.STATUS_IN_ZONE:
            warnings.append(f"{ticker}: BUY con estado {status}; pasa a WAIT.")
            decision = "WAIT"
        elif rr_now is None or rr_now < min_rr:
            warnings.append(
                f"{ticker}: BUY pero el R/R actual ({rr_now}) es menor que {min_rr}; pasa a WAIT.")
            decision = "WAIT"
        if decision == "WAIT" and verdict in (po.V_CONFIRMED, po.V_IMPROVED):
            verdict = po.V_WEAKENED

    base.update({
        "verdict": verdict, "decision": decision,
        "confidence": review.get("confidence") or pick.get("confidence") or "N/A",
        "what_changed": review.get("what_changed") or "",
        "action": review.get("action") or base["action"],
        "source": "gemini",
    })
    return base


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #
def python_only_review(reason: str) -> dict[str, Any]:
    """Resultado «sin IA»: no hay valoraciones y el informe usa solo hechos de Python."""
    return {
        "available": False, "error": reason, "model": "", "reviews": {},
        "watchlist": [], "warnings": [], "market_view": "", "summary": "",
        "raw_response": {}, "note": UNAVAILABLE_NOTE,
    }


def analyze(
    payload: Mapping[str, Any], pick_tickers: Sequence[str], other_tickers: Sequence[str],
    api_key: str = "", model: str = "", cfg: config.GeminiConfig | None = None,
    update_cfg: config.UpdateConfig | None = None, client: Any = None,
) -> dict[str, Any]:
    """Una llamada a Gemini y su validación. Nunca lanza excepciones."""
    model = model or config.DEFAULT_GEMINI_MODEL
    if not pick_tickers:
        return python_only_review("no había selecciones que revisar")
    if not api_key and client is None:
        return python_only_review("falta GEMINI_API_KEY")
    try:
        raw = ga.call_gemini(
            payload, api_key, model, cfg=cfg or config.GEMINI, client=client,
            system_instruction=build_system_instruction(), schema=response_schema())
    except ga.GeminiError as exc:
        return python_only_review(str(exc))
    except Exception as exc:  # noqa: BLE001 - nada puede tumbar la actualización
        return python_only_review(f"{type(exc).__name__}: {exc}")

    checked = validate_response(raw, pick_tickers, other_tickers, update_cfg)
    checked.update({"available": True, "error": "", "model": model,
                    "raw_response": raw, "note": ""})
    for warning in checked["warnings"]:
        logger.warning("Validación (actualización): %s", warning)
    logger.info("Gemini (actualización): %d valoraciones, %d en vigilancia.",
                len(checked["reviews"]), len(checked["watchlist"]))
    return checked
