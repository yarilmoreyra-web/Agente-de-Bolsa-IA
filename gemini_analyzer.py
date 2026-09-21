"""gemini_analyzer.py — Una llamada a Gemini con salida estructurada y validación.

Reparto de responsabilidades (regla global 3)
---------------------------------------------
* **Python calcula**: precios, gap, volumen, RVOL, indicadores, niveles, R/R y
  precio máximo de entrada válido.
* **Gemini interpreta**: qué significa el catalizador, cómo está la estructura,
  qué riesgo de gap-and-fade hay y si merece la pena operar. Elige niveles *de
  la tabla entregada*; no inventa ninguno.

Todo lo que devuelve Gemini pasa por ``validate_response`` antes de llegar al
informe. Lo que no cuadra se descarta o se corrige, y cada corrección queda como
aviso en ``warnings`` (y, más tarde, en el histórico).

SDK
---
Se usa ``from google import genai`` (paquete ``google-genai``), **no**
``google-generativeai``. El ``response_schema`` se declara como diccionario
(subconjunto de OpenAPI que acepta el SDK) en vez de con Pydantic, para no
añadir dependencias y para poder probar el módulo sin la librería instalada.

Degradación
-----------
Si la llamada falla, si el JSON es inválido tras el reintento o si no hay clave,
se devuelve un ``GeminiResult`` con ``available = False``. El informe se genera
entonces solo con datos de Python, con la nota «Análisis IA no disponible» y sin
ninguna selección BUY.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import config
from technical_analysis import NA, is_num

logger = logging.getLogger("trading_agent.gemini_analyzer")

UNAVAILABLE_NOTE = "Análisis IA no disponible"
LONG, SHORT = "LONG", "SHORT"
DECISIONS = ("BUY", "WAIT", "NO_TRADE")
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")
_CONFIDENCE_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
FADE_RISKS = ("LOW", "MEDIUM", "HIGH")


class GeminiError(RuntimeError):
    """Fallo al llamar a Gemini o al interpretar su respuesta."""


@dataclass
class GeminiResult:
    """Resultado ya validado, listo para el informe y para el histórico."""

    available: bool = False
    model: str = ""
    market_view: str = ""
    no_trade: bool = True
    no_trade_reason: str = ""
    picks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: Any = None
    error: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unavailable(reason: str, model: str = "") -> GeminiResult:
    """Resultado degradado: sin IA, sin picks y con el motivo a la vista."""
    logger.warning("%s: %s", UNAVAILABLE_NOTE, reason)
    return GeminiResult(
        available=False, model=model, no_trade=True,
        no_trade_reason=f"{UNAVAILABLE_NOTE} ({reason}).",
        error=reason, note=UNAVAILABLE_NOTE,
    )


# --------------------------------------------------------------------------- #
# Instrucción de sistema y esquema de respuesta
# --------------------------------------------------------------------------- #
def build_system_instruction(direction_mode: str = "long_only") -> str:
    """Instrucción de sistema en español (Gemini interpreta, nunca calcula)."""
    direction_rule = (
        "Solo puedes proponer operaciones LONG (compras). El campo direction "
        "siempre vale \"LONG\"."
        if direction_mode != "both" else
        "Puedes proponer LONG o SHORT. Para SHORT invierte los niveles: el stop "
        "queda por encima de la entrada y los objetivos por debajo."
    )
    return (
        "Eres un analista de day trading de acciones estadounidenses. Trabajas "
        "sobre datos ya calculados por un programa de Python y tu única tarea es "
        "INTERPRETARLOS y decidir.\n\n"
        "Reglas que no puedes romper:\n"
        "1. No calculas ni corriges nada. No inventes precios, volúmenes, "
        "indicadores, noticias, catalizadores ni fechas. Si un dato falta, "
        "escribe \"N/A\".\n"
        "2. Los soportes, resistencias, stops y objetivos deben salir de la "
        "tabla de niveles («levels») de cada candidata, o derivarse de ella "
        "usando el ATR como referencia (por ejemplo, un stop a 1 ATR por debajo "
        "de un soporte de la tabla). Nunca propongas un nivel que no guarde "
        "relación con esa tabla.\n"
        "3. No calcules la relación riesgo/beneficio: la calcula Python. "
        "Limítate a proponer entrada, stop y objetivos coherentes.\n"
        f"4. {direction_rule}\n"
        "5. Evalúa siempre: calidad del catalizador (y si está confirmado), "
        "momentum pre-market, volumen y RVOL, estructura técnica, fuerza "
        "relativa, contexto de mercado y riesgo de gap-and-fade (tamaño del gap "
        "respecto al ATR, distancia a la siguiente resistencia, comportamiento "
        "frente al máximo pre-market, liquidez y tono del mercado).\n"
        "6. Selecciona COMO MÁXIMO 3 candidatas, y solo las que de verdad "
        "merezcan la pena. Responder que no hay operación (no_trade = true) es "
        "una respuesta válida y preferible a forzar una mala.\n"
        "7. Si la calidad del dato pre-market no es \"ok\", sé más prudente y "
        "dilo en el campo reason.\n"
        "8. Escribe todos los textos en español, claros y breves.\n\n"
        "Devuelve exclusivamente un JSON que cumpla el esquema indicado."
    )


def response_schema(direction_mode: str = "long_only") -> dict[str, Any]:
    """Esquema JSON de la respuesta (formato que acepta ``response_schema``)."""
    directions = ["LONG"] if direction_mode != "both" else [LONG, SHORT]
    pick = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "company": {"type": "string"},
            "direction": {"type": "string", "enum": directions},
            "catalyst": {"type": "string"},
            "catalyst_source": {"type": "string"},
            "trend": {"type": "string"},
            "technical_summary": {"type": "string"},
            "support_level": {"type": "string"},
            "resistance_level": {"type": "string"},
            "fade_risk": {"type": "string", "enum": list(FADE_RISKS)},
            "fade_risk_explanation": {"type": "string"},
            "entry_zone_low": {"type": "number", "nullable": True},
            "entry_zone_high": {"type": "number", "nullable": True},
            "stop": {"type": "number", "nullable": True},
            "target_1": {"type": "number", "nullable": True},
            "target_2": {"type": "number", "nullable": True},
            "exit_time": {"type": "string"},
            "invalidation": {"type": "string"},
            "decision": {"type": "string", "enum": list(DECISIONS)},
            "confidence": {"type": "string", "enum": list(CONFIDENCES)},
            "reason": {"type": "string"},
        },
        "required": [
            "ticker", "company", "direction", "catalyst", "catalyst_source",
            "trend", "technical_summary", "support_level", "resistance_level",
            "fade_risk", "fade_risk_explanation", "entry_zone_low",
            "entry_zone_high", "stop", "target_1", "target_2", "exit_time",
            "invalidation", "decision", "confidence", "reason",
        ],
    }
    return {
        "type": "object",
        "properties": {
            "market_view": {"type": "string"},
            "no_trade": {"type": "boolean"},
            "no_trade_reason": {"type": "string"},
            "picks": {"type": "array", "items": pick},
        },
        "required": ["market_view", "no_trade", "no_trade_reason", "picks"],
    }


# --------------------------------------------------------------------------- #
# Entrada de Gemini
# --------------------------------------------------------------------------- #
def _compact_news(ticker_news: Mapping[str, Any] | None, limit: int) -> dict[str, Any]:
    """Noticias resumidas: titular, medio, hora, URL y categoría."""
    ticker_news = ticker_news or {}
    items = []
    for item in list(ticker_news.get("items", []))[:limit]:
        items.append({
            "headline": item.get("headline", ""),
            "publisher": item.get("publisher", ""),
            "published_at": item.get("published_at", ""),
            "url": item.get("url", ""),
            "category": item.get("category_label", item.get("category", "")),
            "is_hard_catalyst": item.get("is_hard_catalyst", False),
        })
    return {
        "catalyst_confirmed": bool(ticker_news.get("catalyst_confirmed")),
        "catalyst_text": ticker_news.get("catalyst_text", ""),
        "catalyst_source": ticker_news.get("catalyst_source", ""),
        "items": items,
    }


def _compact_candidate(entry: Mapping[str, Any], cfg: config.GeminiConfig) -> dict[str, Any]:
    """Una candidata en el formato que recibe Gemini (solo lo necesario)."""
    analysis = entry.get("analysis", {})
    premarket = analysis.get("premarket", {})
    return {
        "ticker": entry.get("ticker"),
        "direction_sugerida": entry.get("direction", LONG),
        "score": entry.get("score"),
        "score_breakdown": {
            name: {"points": comp.get("points"), "detail": comp.get("detail")}
            for name, comp in (entry.get("score_breakdown") or {}).items()
        },
        "fade_flags": [
            {"flag": p.get("flag"), "detail": p.get("detail")}
            for p in (entry.get("penalties") or [])
        ],
        "premarket": {
            "prev_close": premarket.get("prev_close", NA),
            "price": premarket.get("premarket_price", NA),
            "high": premarket.get("premarket_high", NA),
            "low": premarket.get("premarket_low", NA),
            "volume": premarket.get("premarket_volume", NA),
            "quality": premarket.get("premarket_quality", "missing"),
            "quality_note": premarket.get("quality_note", ""),
            "last_bar_time": premarket.get("last_bar_time", NA),
        },
        "gap_pct": analysis.get("gap_pct", NA),
        "gap_atr": analysis.get("gap_atr", NA),
        "premarket_range_pct": analysis.get("premarket_range_pct", NA),
        "rvol": analysis.get("rvol", {}),
        "pm_pct_of_adv": analysis.get("pm_pct_of_adv", NA),
        "indicators": analysis.get("indicators", {}),
        "relative_strength": analysis.get("relative_strength", {}),
        "levels": analysis.get("levels", []),
        "nearest_support": analysis.get("nearest_support", NA),
        "nearest_resistance": analysis.get("nearest_resistance", NA),
        "news": _compact_news(entry.get("news"), cfg.max_news_per_candidate),
    }


def build_payload(
    session_date: date, snapshot_ts: datetime, session_type: str,
    market_context: Any, candidates: Sequence[Mapping[str, Any]],
    cfg: config.GeminiConfig | None = None,
) -> dict[str, Any]:
    """JSON estructurado que se envía a Gemini en la única llamada del día."""
    cfg = cfg or config.GEMINI
    context = market_context.to_dict() if hasattr(market_context, "to_dict") \
        else dict(market_context or {})
    return {
        "fecha": session_date.isoformat(),
        "hora_snapshot_et": snapshot_ts.strftime("%H:%M"),
        "tipo_de_sesion": session_type,
        "contexto_mercado": {
            "resumen": context.get("summary", ""),
            "regimen": context.get("regime", {}),
            "indices": context.get("indices", {}),
            "futuros": context.get("futures", {}),
            "volatilidad": context.get("volatility", {}),
            "tipos": context.get("rates", {}),
            "sectores_lideres": context.get("sector_leaders", []),
            "sectores_rezagados": context.get("sector_laggards", []),
            "macro": context.get("macro", {}),
        },
        "candidatas": [
            _compact_candidate(entry, cfg)
            for entry in list(candidates)[: cfg.max_candidates_sent]
        ],
    }


# --------------------------------------------------------------------------- #
# Llamada
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict[str, Any]:
    """Convierte la respuesta en un diccionario, tolerando ```json ... ```."""
    if not isinstance(text, str) or not text.strip():
        raise GeminiError("respuesta vacía")
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise GeminiError(f"JSON inválido: {exc}") from exc
        try:
            payload = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc2:
            raise GeminiError(f"JSON inválido: {exc2}") from exc2
    if not isinstance(payload, dict):
        raise GeminiError("el JSON de la respuesta no es un objeto")
    return payload


def _response_text(response: Any) -> str:
    """Texto de la respuesta del SDK, con varias rutas por si cambia el formato."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    for candidate in getattr(response, "candidates", None) or []:
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        joined = "".join(getattr(part, "text", "") or "" for part in parts)
        if joined.strip():
            return joined
    raise GeminiError("la respuesta no contiene texto")


def call_gemini(
    payload: Mapping[str, Any], api_key: str, model: str,
    direction_mode: str = "long_only", cfg: config.GeminiConfig | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Una sola llamada (con reintentos) y el JSON crudo ya parseado.

    ``client`` permite inyectar un doble de prueba; en producción se crea con
    ``genai.Client(api_key=...)``. Lanza ``GeminiError`` si no se consigue un
    JSON válido tras ``cfg.attempts`` intentos.
    """
    import time as time_module  # noqa: PLC0415

    cfg = cfg or config.GEMINI
    if client is None:
        try:
            from google import genai  # noqa: PLC0415 - import diferido
        except ImportError as exc:
            raise GeminiError(f"falta el paquete google-genai: {exc}") from exc
        client = genai.Client(api_key=api_key)

    request = {
        "model": model,
        "contents": json.dumps(payload, ensure_ascii=False),
        "config": {
            "system_instruction": build_system_instruction(direction_mode),
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": response_schema(direction_mode),
        },
    }

    delay = cfg.retry_base_delay
    last_error: Exception | None = None
    for attempt in range(1, max(1, cfg.attempts) + 1):
        try:
            response = client.models.generate_content(**request)
            return extract_json(_response_text(response))
        except Exception as exc:  # noqa: BLE001 - 429, 5xx, JSON inválido, red...
            last_error = exc
            if attempt >= max(1, cfg.attempts):
                break
            logger.warning(
                "Gemini falló (intento %d/%d): %s. Reintento en %.1fs",
                attempt, cfg.attempts, exc, delay,
            )
            time_module.sleep(delay)
            delay *= 2
    raise GeminiError(str(last_error))


# --------------------------------------------------------------------------- #
# Validación en Python
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    return float(value) if is_num(value) else None


def compute_rr(entry: float, stop: float, target: float, direction: str) -> float | None:
    """R/R = beneficio / riesgo, siempre calculado en Python."""
    risk = (entry - stop) if direction == LONG else (stop - entry)
    reward = (target - entry) if direction == LONG else (entry - target)
    if risk <= 0 or reward <= 0:
        return None
    return round(reward / risk, 2)


def max_valid_entry(stop: float, target: float, direction: str,
                    min_rr: float) -> float | None:
    """Precio de entrada límite con el que el R/R sigue siendo ≥ ``min_rr``.

    Para LONG es un máximo: por encima, «NO ENTRAR». Para SHORT es un mínimo.
    Se despeja de ``(target - E) / (E - stop) = min_rr``.
    """
    if min_rr <= -1:
        return None
    price = (target + min_rr * stop) / (1.0 + min_rr)
    if direction == LONG and not (stop < price < target):
        return None
    if direction == SHORT and not (target < price < stop):
        return None
    return round(price, 2)


def level_supported(
    price: float, levels: Sequence[Mapping[str, Any]], atr: float | None,
    cfg: config.GeminiConfig,
) -> bool:
    """True si ``price`` coincide con un nivel de la tabla o está en tolerancia.

    Tolerancia: ``level_tolerance_atr`` × ATR; sin ATR, ``level_tolerance_pct``.
    Sin tabla de niveles no se puede comprobar, así que se acepta (el aviso lo
    pone quien llama).
    """
    prices = [float(row["price"]) for row in levels if is_num(row.get("price"))]
    if not prices:
        return True
    if atr and atr > 0:
        tolerance = cfg.level_tolerance_atr * atr
    else:
        tolerance = abs(price) * cfg.level_tolerance_pct / 100.0
    return any(abs(price - level) <= tolerance for level in prices)


def _overwrite_market_numbers(
    pick: dict[str, Any], analysis: Mapping[str, Any], cfg: config.GeminiConfig,
    warnings: list[str],
) -> None:
    """Sobrescribe los números de mercado con los de Python (Python prevalece)."""
    premarket = analysis.get("premarket", {})
    rvol = (analysis.get("rvol", {}) or {}).get("rvol", NA)
    python_values = {
        "prev_close": premarket.get("prev_close", NA),
        "premarket_price": premarket.get("premarket_price", NA),
        "premarket_high": premarket.get("premarket_high", NA),
        "premarket_low": premarket.get("premarket_low", NA),
        "premarket_volume": premarket.get("premarket_volume", NA),
        "premarket_quality": premarket.get("premarket_quality", "missing"),
        "gap_pct": analysis.get("gap_pct", NA),
        "gap_atr": analysis.get("gap_atr", NA),
        "rvol": rvol,
        "pm_pct_of_adv": analysis.get("pm_pct_of_adv", NA),
        "atr14": analysis.get("indicators", {}).get("atr14", NA),
        "rsi14": analysis.get("indicators", {}).get("rsi14", NA),
        "sma20": analysis.get("indicators", {}).get("sma20", NA),
        "sma50": analysis.get("indicators", {}).get("sma50", NA),
    }
    ticker = pick.get("ticker", "?")
    for key, value in python_values.items():
        claimed = pick.get(key)
        if claimed is not None and is_num(claimed) and is_num(value):
            reference = abs(float(value)) or 1.0
            if abs(float(claimed) - float(value)) / reference * 100 > cfg.numeric_mismatch_pct:
                warnings.append(
                    f"{ticker}: Gemini dio {key}={claimed} y Python {value}; "
                    "prevalece Python."
                )
        pick[key] = value


def _validate_levels(
    pick: dict[str, Any], analysis: Mapping[str, Any], direction: str,
    cfg: config.GeminiConfig, min_rr: float, warnings: list[str],
) -> bool:
    """Comprueba entrada, stop y objetivos. False = el pick se descarta."""
    ticker = pick.get("ticker", "?")
    low, high = _num(pick.get("entry_zone_low")), _num(pick.get("entry_zone_high"))
    stop, target_1 = _num(pick.get("stop")), _num(pick.get("target_1"))
    target_2 = _num(pick.get("target_2"))

    if low is None or high is None or stop is None or target_1 is None:
        warnings.append(f"{ticker}: descartado, faltan entrada, stop u objetivo 1.")
        return False
    if low > high:
        low, high = high, low
        warnings.append(f"{ticker}: rango de entrada invertido; se ha reordenado.")
    entry = round((low + high) / 2, 4)

    ordered = (stop < entry < target_1) if direction == LONG \
        else (target_1 < entry < stop)
    if not ordered:
        warnings.append(
            f"{ticker}: descartado, niveles incoherentes para {direction} "
            f"(stop {stop}, entrada {entry}, objetivo 1 {target_1})."
        )
        return False
    if target_2 is not None:
        further = target_2 > target_1 if direction == LONG else target_2 < target_1
        if not further:
            warnings.append(
                f"{ticker}: objetivo 2 ({target_2}) no va más allá del objetivo 1; "
                "se ignora.")
            target_2 = None

    levels = analysis.get("levels", []) or []
    atr = _num(analysis.get("indicators", {}).get("atr14"))
    if not levels:
        warnings.append(f"{ticker}: sin tabla de niveles; no se pudo verificar el stop.")
    for label, value in (("stop", stop), ("objetivo 1", target_1)):
        if not level_supported(value, levels, atr, cfg):
            warnings.append(
                f"{ticker}: descartado, {label} ({value}) no se apoya en la tabla "
                f"de niveles (tolerancia {cfg.level_tolerance_atr} ATR).")
            return False
    if target_2 is not None and not level_supported(target_2, levels, atr, cfg):
        warnings.append(
            f"{ticker}: objetivo 2 ({target_2}) no se apoya en la tabla de niveles; "
            "se ignora.")
        target_2 = None

    labels = {str(row.get("label")) for row in levels}
    for key in ("support_level", "resistance_level"):
        value = pick.get(key)
        if isinstance(value, str) and value and value != NA and labels \
                and value not in labels:
            warnings.append(
                f"{ticker}: {key}='{value}' no es una etiqueta de la tabla; se pone N/A.")
            pick[key] = NA

    rr = compute_rr(entry, stop, target_1, direction)
    if rr is None:
        warnings.append(f"{ticker}: descartado, R/R no calculable.")
        return False

    pick["entry_zone_low"], pick["entry_zone_high"] = round(low, 2), round(high, 2)
    pick["entry_mid"] = entry
    pick["stop"], pick["target_1"], pick["target_2"] = stop, target_1, target_2
    pick["rr"] = rr
    pick["rr_target_2"] = compute_rr(entry, stop, target_2, direction) \
        if target_2 is not None else NA
    pick["max_valid_entry"] = max_valid_entry(stop, target_1, direction, min_rr) or NA
    pick["rr_computed_by"] = "python"

    if rr < min_rr:
        if cfg.downgrade_low_rr_to_wait:
            pick["decision"] = "WAIT"
            pick["rr_warning"] = (
                f"R/R {rr} por debajo del mínimo {min_rr}; el pick pasa a WAIT.")
            warnings.append(f"{pick['ticker']}: {pick['rr_warning']}")
        else:
            warnings.append(f"{ticker}: descartado, R/R {rr} < {min_rr}.")
            return False
    return True


def validate_response(
    raw: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    direction_mode: str = "long_only",
    cfg: config.GeminiConfig | None = None,
    min_rr: float | None = None,
    model: str = "",
) -> GeminiResult:
    """Valida la respuesta de Gemini contra los datos de Python.

    Comprobaciones: ticker dentro de las candidatas enviadas, dirección válida,
    números de mercado sobrescritos con los de Python, coherencia de stop y
    objetivos, niveles apoyados en la tabla, R/R calculado en Python frente a
    ``MIN_RR``, decisión y confianza dentro de los valores permitidos, y un
    máximo de ``cfg.max_picks`` picks ordenados por confianza y R/R.

    ``candidates`` es ``{ticker: analysis}`` (el análisis de Python de cada
    candidata enviada).
    """
    cfg = cfg or config.GEMINI
    min_rr = min_rr if min_rr is not None else getattr(config, "MIN_RR", 1.5)
    warnings: list[str] = []

    result = GeminiResult(
        available=True, model=model, raw_response=dict(raw),
        market_view=str(raw.get("market_view", "") or ""),
        no_trade=bool(raw.get("no_trade", False)),
        no_trade_reason=str(raw.get("no_trade_reason", "") or ""),
    )

    picks_raw = raw.get("picks")
    if not isinstance(picks_raw, list):
        warnings.append("La respuesta no traía una lista 'picks'; se asume NO OPERAR.")
        picks_raw = []

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in picks_raw:
        if not isinstance(item, Mapping):
            warnings.append("Pick ignorado: no es un objeto.")
            continue
        pick = dict(item)
        ticker = str(pick.get("ticker", "")).strip().upper()
        pick["ticker"] = ticker

        if ticker not in candidates:
            warnings.append(
                f"{ticker or '(sin ticker)'}: descartado, no estaba entre las "
                "candidatas enviadas.")
            continue
        if ticker in seen:
            warnings.append(f"{ticker}: descartado, repetido en la respuesta.")
            continue

        direction = str(pick.get("direction", LONG)).strip().upper() or LONG
        if direction not in (LONG, SHORT):
            warnings.append(f"{ticker}: dirección '{direction}' no válida; se usa LONG.")
            direction = LONG
        if direction_mode != "both" and direction != LONG:
            warnings.append(
                f"{ticker}: descartado, propone {direction} y el modo es solo largos.")
            continue
        pick["direction"] = direction

        analysis = candidates[ticker]
        _overwrite_market_numbers(pick, analysis, cfg, warnings)
        if not _validate_levels(pick, analysis, direction, cfg, min_rr, warnings):
            continue

        decision = str(pick.get("decision", "")).strip().upper()
        if decision not in DECISIONS:
            warnings.append(f"{ticker}: decisión '{decision}' no válida; se usa WAIT.")
            decision = "WAIT"
        pick["decision"] = decision

        confidence = str(pick.get("confidence", "")).strip().upper()
        if confidence not in CONFIDENCES:
            warnings.append(f"{ticker}: confianza '{confidence}' no válida; se usa LOW.")
            confidence = "LOW"
        pick["confidence"] = confidence

        fade = str(pick.get("fade_risk", "")).strip().upper()
        pick["fade_risk"] = fade if fade in FADE_RISKS else "HIGH"

        seen.add(ticker)
        validated.append(pick)

    validated.sort(
        key=lambda p: (_CONFIDENCE_ORDER.get(p["confidence"], 0), float(p.get("rr") or 0)),
        reverse=True,
    )
    if len(validated) > cfg.max_picks:
        warnings.append(
            f"Gemini propuso {len(validated)} picks; se conservan los "
            f"{cfg.max_picks} mejores.")
        validated = validated[: cfg.max_picks]

    result.picks = validated
    result.warnings = warnings
    if not validated:
        result.no_trade = True
        if not result.no_trade_reason:
            result.no_trade_reason = (
                "Ningún pick de la IA superó la validación de Python."
                if picks_raw else
                "La IA no encontró ninguna configuración con suficiente calidad.")
    else:
        result.no_trade = False

    logger.info(
        "Gemini validado: %d picks aceptados, %d avisos.", len(validated), len(warnings))
    for warning in warnings:
        logger.warning("Validación: %s", warning)
    return result


# --------------------------------------------------------------------------- #
# Punto de entrada del módulo
# --------------------------------------------------------------------------- #
def analyze(
    session_date: date, snapshot_ts: datetime, session_type: str,
    market_context: Any, candidates: Sequence[Mapping[str, Any]],
    api_key: str = "", model: str = "", direction_mode: str | None = None,
    cfg: config.GeminiConfig | None = None, client: Any = None,
) -> GeminiResult:
    """Llama a Gemini una sola vez y devuelve el resultado ya validado.

    Nunca lanza excepciones: cualquier problema se convierte en un resultado
    degradado (``available = False``) para que el informe salga igualmente.
    """
    cfg = cfg or config.GEMINI
    model = model or getattr(config, "GEMINI_MODEL", "")
    direction_mode = direction_mode or getattr(config, "DIRECTION_MODE", "long_only")

    if not candidates:
        return unavailable("no había candidatas que analizar", model)
    if not api_key and client is None:
        return unavailable("falta GEMINI_API_KEY", model)

    payload = build_payload(session_date, snapshot_ts, session_type,
                            market_context, candidates, cfg)
    analyses = {
        str(entry.get("ticker")): entry.get("analysis", {})
        for entry in list(candidates)[: cfg.max_candidates_sent]
    }

    try:
        raw = call_gemini(payload, api_key, model, direction_mode, cfg, client)
    except GeminiError as exc:
        return unavailable(str(exc), model)
    except Exception as exc:  # noqa: BLE001 - nada puede tumbar el informe
        return unavailable(f"{type(exc).__name__}: {exc}", model)

    return validate_response(raw, analyses, direction_mode, cfg, model=model)


__all__ = [
    "UNAVAILABLE_NOTE", "GeminiError", "GeminiResult", "unavailable",
    "build_system_instruction", "response_schema", "build_payload", "extract_json",
    "call_gemini", "compute_rr", "max_valid_entry", "level_supported",
    "validate_response", "analyze",
]
