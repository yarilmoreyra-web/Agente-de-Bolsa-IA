"""Informe diario: texto en Markdown y su conversión a HTML para Telegram.

``build_report`` recibe un :class:`ReportData` (ver ``sample_data.py`` y
``main.run_pipeline``) y no asume que todos los campos estén siempre
presentes: cuando falta un dato se muestra "N/A" en vez de inventarlo.
"""
from __future__ import annotations

import html
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import config
import utils

logger = logging.getLogger("trading_agent.report")

MEDALS = ("🥇", "🥈", "🥉")
DISCLAIMER = "Herramienta informativa, no es asesoramiento financiero."
LATE_MARK = "⏱ Informe tardío"

_WEEKDAYS_ES = (
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
)
_MONTHS_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)

_QUALITY_LABELS = {
    "stale": "dato desactualizado",
    "sparse": "volumen parcial",
    "partial": "volumen parcial",
}

_LEVEL_LABELS = {
    "sma20": "SMA20", "sma50": "SMA50", "ema9": "EMA9", "ema20": "EMA20",
    "max_20d": "máx. 20 sesiones", "min_20d": "mín. 20 sesiones",
    "prev_close": "cierre anterior",
}


@dataclass
class ReportData:
    """Todo lo que necesita el informe de un día."""

    session_date: Any
    snapshot_ts: Any
    sent_at: Optional[datetime] = None
    session_type: str = "sesión normal"
    universe_size: int = 0
    universe_source: str = "N/A"
    universe_source_label: str = "N/A"
    context: Optional[Dict[str, Any]] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    gemini: Optional[Dict[str, Any]] = field(default_factory=dict)
    failures: Dict[str, Any] = field(default_factory=dict)
    filter_no_trade_reason: Optional[str] = None
    disclaimer: str = DISCLAIMER
    no_trade_reason: Optional[str] = None
    filter_log: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Formato de números y fechas
# --------------------------------------------------------------------------- #
def fmt_num(value: Any, decimals: int = 2) -> str:
    """Número con separador de miles, o "N/A" si falta o no es numérico."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "N/A"
    return f"{number:,.{decimals}f}"


def fmt_pct(value: Any, decimals: int = 2, signed: bool = True) -> str:
    """Porcentaje con signo (por defecto), p.ej. "+3.20%"."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{decimals}f}%"


def fmt_pp(value: Any, decimals: int = 2) -> str:
    """Puntos porcentuales con signo, p.ej. "-0.50 pp"."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "N/A"
    return f"{number:+.{decimals}f} pp"


def fmt_rvol(value: Any) -> str:
    """RVOL con dos decimales y sufijo 'x', p.ej. "3.40x"."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "N/A"
    return f"{number:.2f}x"


def fmt_volume(value: Any) -> str:
    """Volumen abreviado: 1.25M, 850K o el número tal cual si es pequeño."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "N/A"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.0f}K"
    return f"{number:.0f}"


def spanish_date(value: Any) -> str:
    """Fecha en español con día de la semana, p.ej. "lunes 21 de septiembre de 2026"."""
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, str):
        value = datetime.strptime(value, "%Y-%m-%d").date()
    weekday = _WEEKDAYS_ES[value.weekday()]
    month = _MONTHS_ES[value.month - 1]
    return f"{weekday} {value.day} de {month} de {value.year}"


def is_late(sent_at: Any, target_time: Any = None) -> bool:
    """True si ``sent_at`` (datetime) es posterior a la hora de informe configurada."""
    if sent_at is None:
        return False
    target = target_time or config.SCHEDULE.report_time
    time_part = sent_at.time() if isinstance(sent_at, datetime) else sent_at
    return time_part > target


def safe_text(text: Any) -> str:
    """Aplana texto libre: sin saltos de línea ni marcado que rompa el informe."""
    if text is None:
        return "N/A"
    flat = str(text).replace("\r", "").replace("\n", " ")
    flat = flat.replace("**", "")
    flat = flat.replace("](", ") (")
    flat = " ".join(flat.split())
    return flat or "N/A"


# --------------------------------------------------------------------------- #
# Accesores defensivos sobre una candidata / pick
# --------------------------------------------------------------------------- #
def _analysis_of(candidate: Dict[str, Any]) -> Dict[str, Any]:
    record = candidate.get("record")
    if isinstance(record, dict):
        return record
    analysis = candidate.get("analysis")
    return analysis if isinstance(analysis, dict) else {}


def _score_of(candidate: Dict[str, Any]) -> Any:
    score = candidate.get("score")
    if isinstance(score, (int, float)):
        return score
    detail = candidate.get("score_detail") or candidate.get("score_breakdown")
    if isinstance(detail, dict) and isinstance(detail.get("total"), (int, float)):
        return detail["total"]
    return None


def _find_candidate(candidates: Sequence[Dict[str, Any]], ticker: str) -> Optional[Dict[str, Any]]:
    for candidate in candidates or []:
        if isinstance(candidate, dict) and candidate.get("ticker") == ticker:
            return candidate
    return None


def _level_label(label: Any) -> str:
    if not label:
        return "N/A"
    return _LEVEL_LABELS.get(label, str(label))


def _source_display(source: Any) -> str:
    if not source:
        return "N/A"
    text = str(source)
    if text.startswith("http://") or text.startswith("https://"):
        return f"[fuente]({text})"
    return text


# --------------------------------------------------------------------------- #
# Diagnóstico del filtro: por qué cada ticker quedó o no quedó dentro
# --------------------------------------------------------------------------- #
def _row_weakness(row: Dict[str, Any]) -> str:
    """Explica en una frase el motivo principal de la puntuación de una fila
    que sí pasó las puertas duras: la penalización más fuerte y/o el
    componente más flojo del desglose."""
    parts: List[str] = []
    penalties = row.get("penalties") or []
    if penalties:
        worst = max(penalties, key=lambda p: p.get("points", 0))
        parts.append(f"penalización: {worst.get('detail', 'N/A')} (-{fmt_num(worst.get('points'), 1)} pts)")
    components = row.get("components") or {}
    if components:
        weakest_key, weakest = min(
            components.items(), key=lambda kv: kv[1].get("ratio", 1.0)
        )
        if weakest.get("ratio", 1.0) < 0.5:
            parts.append(f"punto débil: {weakest.get('detail', weakest_key)}")
    return "; ".join(parts) if parts else "sin banderas relevantes"


def _diagnostic_block(filter_log: List[Dict[str, Any]], limit: int = 6) -> str:
    """Bloque de transparencia: qué tickers se analizaron, su puntuación y el
    motivo exacto de inclusión/exclusión. Se muestra siempre (haya o no
    candidatas) para que el informe nunca sea solo un veredicto sin detalle.
    """
    rows = filter_log or []
    if not rows:
        return ""
    passed = [r for r in rows if r.get("passed_gates")]
    failed = [r for r in rows if not r.get("passed_gates")]
    passed_sorted = sorted(
        passed, key=lambda r: float(r.get("score") or 0.0), reverse=True
    )

    lines: List[str] = ["🔍 **DIAGNÓSTICO DEL FILTRO**"]
    if passed_sorted:
        lines.append(f"Puntuadas ({len(passed_sorted)} de {len(rows)} analizadas), de mayor a menor:")
        for row in passed_sorted[:limit]:
            ticker = row.get("ticker", "N/A")
            score = fmt_num(row.get("score"), 1)
            tag = " ✅ seleccionada" if row.get("selected") else ""
            reason = row.get("exclusion_reason") or _row_weakness(row)
            lines.append(f"• {ticker} — {score} pts{tag}. {reason}")
        if len(passed_sorted) > limit:
            lines.append(f"… y {len(passed_sorted) - limit} más puntuadas por debajo de estas.")
    remaining_slots = max(0, limit - len(passed_sorted[:limit]))
    if failed and (not passed_sorted or remaining_slots > 0):
        show = failed[:max(remaining_slots, 3)]
        lines.append("Descartadas en puertas duras (no llegaron a puntuarse):")
        for row in show:
            lines.append(f"• {row.get('ticker', 'N/A')}: {row.get('exclusion_reason', 'N/A')}")
        if len(failed) > len(show):
            lines.append(f"… y {len(failed) - len(show)} más descartadas en puertas duras.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Bloque de una candidata seleccionada (pick)
# --------------------------------------------------------------------------- #
def _format_pick_block(position: int, pick: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    medal = MEDALS[position] if position < len(MEDALS) else "•"
    ticker = pick.get("ticker", "N/A")
    direction = (pick.get("direction") or "LONG").upper()
    company = pick.get("company") or "N/A"

    catalyst = safe_text(pick.get("catalyst")) if pick.get("catalyst") else "Sin catalizador confirmado"
    source = _source_display(pick.get("catalyst_source"))

    premarket = analysis.get("premarket") if isinstance(analysis.get("premarket"), dict) else {}
    prev_close = fmt_num(premarket.get("prev_close"))
    pm_price = fmt_num(premarket.get("premarket_price"))
    gap = fmt_pct(analysis.get("gap_pct"))

    volume = fmt_volume(premarket.get("premarket_volume"))
    rvol_info = analysis.get("rvol")
    rvol_value = rvol_info.get("rvol") if isinstance(rvol_info, dict) else rvol_info
    rvol = fmt_rvol(rvol_value)

    indicators = analysis.get("indicators") if isinstance(analysis.get("indicators"), dict) else {}
    rsi = fmt_num(indicators.get("rsi14"), 1)
    sma20 = fmt_num(indicators.get("sma20"))
    sma50 = fmt_num(indicators.get("sma50"))
    atr = fmt_num(indicators.get("atr14"))

    support = analysis.get("nearest_support") if isinstance(analysis.get("nearest_support"), dict) else {}
    resistance = analysis.get("nearest_resistance") if isinstance(analysis.get("nearest_resistance"), dict) else {}
    support_txt = f"{_level_label(support.get('label'))} ({fmt_num(support.get('price'))})" if support else "N/A"
    resistance_txt = (
        f"{_level_label(resistance.get('label'))} ({fmt_num(resistance.get('price'))})"
        if resistance else "N/A"
    )

    rs_info = analysis.get("relative_strength") if isinstance(analysis.get("relative_strength"), dict) else {}
    rs_line = f"vs SPY {fmt_pp(rs_info.get('rs_pm_vs_spy'))} · vs QQQ {fmt_pp(rs_info.get('rs_pm_vs_qqq'))}"

    fade_risk = pick.get("fade_risk") or "N/A"
    fade_explanation = pick.get("fade_risk_explanation")
    fade_line = fade_risk + (f" — {safe_text(fade_explanation)}" if fade_explanation else "")

    entry_low = fmt_num(pick.get("entry_zone_low"))
    entry_high = fmt_num(pick.get("entry_zone_high"))
    stop = fmt_num(pick.get("stop"))
    target1 = fmt_num(pick.get("target_1"))
    target2 = fmt_num(pick.get("target_2"))
    rr = fmt_num(pick.get("rr"))
    max_entry = fmt_num(pick.get("max_entry_price"))
    exit_time = pick.get("exit_time") or "N/A"
    invalidation = safe_text(pick.get("invalidation")) if pick.get("invalidation") else "N/A"

    decision = pick.get("decision", "N/A")
    confidence = pick.get("confidence", "N/A")
    reason = safe_text(pick.get("reason")) if pick.get("reason") else "N/A"

    entry_label = "máximo" if direction != "SHORT" else "mínimo"
    entry_side = "por encima" if direction != "SHORT" else "por debajo"

    quality = premarket.get("premarket_quality")
    quality_note = premarket.get("quality_note")
    quality_flag = " ⚠" if quality and quality != "ok" else ""

    lines = [
        f"{medal} **{ticker}**{quality_flag} ({direction}) — {company}",
        f"Catalizador: {catalyst} / {source}",
        f"**Cierre anterior / Pre-market / Gap:** {prev_close} / {pm_price} / {gap}",
        f"**Volumen / RVOL:** {volume} / {rvol}",
        f"**RSI / SMA20 / SMA50 / ATR:** {rsi} / {sma20} / {sma50} / {atr}",
        f"**Soporte / Resistencia:** {support_txt} / {resistance_txt}",
        rs_line,
        f"**Riesgo gap & fade:** {fade_line}",
        f"**Entrada prevista:** {entry_low} – {entry_high}",
        f"**Stop / Objetivo 1 / Objetivo 2 / R/R (Python):** {stop} / {target1} / {target2} / {rr}",
        f"**Precio {entry_label} de entrada válido:** {max_entry} "
        f"({entry_side}: NO ENTRAR)",
        f"**Hora objetivo / Invalidación:** {exit_time} / {invalidation}",
        f"**Decisión / Razón:** {decision} (confianza {confidence}) — {reason}",
    ]
    if quality and quality != "ok" and quality_note:
        label = _QUALITY_LABELS.get(quality, quality)
        lines.append(f"⚠ Calidad del dato pre-market: {label} ({quality_note})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Informe completo en Markdown
# --------------------------------------------------------------------------- #
def build_report(data: "ReportData | Dict[str, Any]") -> str:
    """Construye el informe completo en Markdown a partir de ``ReportData``.

    También acepta el dict que produce ``main.run_pipeline`` directamente
    (se adapta internamente), para no obligar a cada llamador a construir
    un ``ReportData`` a mano.
    """
    if not isinstance(data, ReportData):
        data = _report_data_from_pipeline(data)
    candidates = data.candidates or []
    context = data.context
    gemini = data.gemini if isinstance(data.gemini, dict) else None
    ia_available = gemini is not None and gemini.get("available", True) is not False

    snapshot_ts = data.snapshot_ts
    snapshot_hhmm = (
        snapshot_ts.strftime("%H:%M") if isinstance(snapshot_ts, datetime)
        else str(snapshot_ts)[11:16]
    )
    envio = ""
    if data.sent_at is not None:
        late = is_late(data.sent_at)
        envio = f" · Envío {data.sent_at.strftime('%H:%M')} ET"
        if late:
            envio += f" · {LATE_MARK}"

    if context:
        regime = context.get("regime") or {}
        context_line = regime.get("summary") or context.get("summary") or "N/A"
    else:
        context_line = "Contexto de mercado no disponible (N/A)"

    candidates_line = ""
    quality_lines: List[str] = []
    if candidates:
        parts = []
        for candidate in candidates:
            ticker = candidate.get("ticker", "N/A")
            score = _score_of(candidate)
            premarket = _analysis_of(candidate).get("premarket", {})
            quality = premarket.get("premarket_quality") if isinstance(premarket, dict) else None
            flag = " ⚠" if quality and quality != "ok" else ""
            parts.append(f"{ticker} {fmt_num(score, 1)}{flag}")
            if quality and quality != "ok":
                note = premarket.get("quality_note", "")
                label = _QUALITY_LABELS.get(quality, quality)
                quality_lines.append(f"⚠ Calidad del dato pre-market: {label} ({note})")
        candidates_line = ", ".join(parts)

    header_lines = [
        f"📅 **{spanish_date(data.session_date)}** ({data.session_type})",
        f"🕒 Snapshot {snapshot_hhmm} ET{envio}",
        "📊 **CONTEXTO DE MERCADO**",
        context_line,
        f"📋 Universo: {data.universe_size} tickers",
        f"🔎 **CANDIDATAS ANALIZADAS** ({len(candidates)}, con puntuación)",
    ]
    if candidates_line:
        header_lines.append(candidates_line)
    header_lines.extend(quality_lines)
    blocks: List[str] = ["\n".join(header_lines)]

    # -- Determinar si el informe es "NO OPERAR" y por qué -------------------
    no_trade_reason: Optional[str] = None
    if gemini is not None and gemini.get("no_trade"):
        no_trade_reason = gemini.get("no_trade_reason") or data.no_trade_reason or "sin motivo indicado"
    elif not candidates:
        no_trade_reason = data.filter_no_trade_reason or "Ninguna candidata superó el filtro"
    elif not ia_available:
        no_trade_reason = "Análisis IA no disponible: no se emite selección BUY."

    if not ia_available:
        aviso = (
            f"⚠️ Análisis IA no disponible: {(gemini.get('error') or 'N/A') if gemini else 'N/A'}. "
            "El informe usa solo datos de Python."
        )
        if candidates:
            nombres = ", ".join(f"**{c.get('ticker', 'N/A')}**" for c in candidates)
            aviso += f"\nCandidatas sin validar por IA: {nombres}"
        blocks.append(aviso)

    diagnostic = _diagnostic_block(data.filter_log)
    if diagnostic:
        blocks.append(diagnostic)

    rendered_picks: List[Dict[str, Any]] = []
    if no_trade_reason is not None:
        blocks.append(f"🚫 **NO OPERAR**\n• {no_trade_reason}")
    else:
        raw_picks = (gemini.get("picks") or []) if gemini else []
        for pick in raw_picks:
            if not isinstance(pick, dict):
                continue
            ticker = pick.get("ticker")
            if not ticker or _find_candidate(candidates, ticker) is None:
                continue
            rendered_picks.append(pick)
            if len(rendered_picks) == len(MEDALS):
                break

        for position, pick in enumerate(rendered_picks):
            analysis = _analysis_of(_find_candidate(candidates, pick.get("ticker")) or {})
            blocks.append(_format_pick_block(position, pick, analysis))

        if rendered_picks:
            if not any((p.get("decision") or "").upper() == "BUY" for p in rendered_picks):
                blocks.append("Ninguna candidata está en BUY.")
            blocks.append(
                "🔔 Regla de apertura\n"
                "La entrada prevista se basa en el pre-market, no es una entrada "
                "confirmada; la entrada confirmada depende del precio real de apertura.\n"
                "Regla de apertura: si el precio de apertura supera el precio máximo de "
                "entrada válido, o invalida el R/R: NO ENTRAR."
            )

    # -- Riesgos ---------------------------------------------------------
    risk_lines: List[str] = []
    if gemini:
        for warning in gemini.get("warnings") or []:
            risk_lines.append(f"- {warning}")
    for pick in rendered_picks:
        for warning in pick.get("warnings") or []:
            risk_lines.append(f"⚠ Aviso: {warning}")
        if (pick.get("fade_risk") or "").upper() == "HIGH":
            risk_lines.append(f"- {pick.get('ticker')}: riesgo alto de gap-and-fade")
    if context:
        regime = context.get("regime") or {}
        if regime.get("risk_level") not in (None, "normal"):
            risk_lines.append(f"- {regime.get('summary', 'N/A')}")
    for candidate in candidates:
        premarket = _analysis_of(candidate).get("premarket", {})
        quality = premarket.get("premarket_quality") if isinstance(premarket, dict) else None
        if quality and quality != "ok":
            label = _QUALITY_LABELS.get(quality, quality)
            risk_lines.append(f"- {candidate.get('ticker')}: {label}")
    failures = data.failures or {}
    failed_tickers = sorted({
        t for group in ("daily", "intraday_history", "intraday_today", "analysis")
        for t in (failures.get(group) or {})
    })
    if failed_tickers:
        risk_lines.append(f"- Sin datos completos para: {', '.join(failed_tickers)}")

    if risk_lines:
        blocks.append("⚠️ **RIESGOS**\n" + "\n".join(risk_lines))

    blocks.append(data.disclaimer or DISCLAIMER)
    return "\n\n".join(blocks)


# Alias de compatibilidad: código antiguo que esperaba build_markdown.
build_markdown = build_report


def save_report(markdown_text: str, session_date: date, base_dir: Optional[Path] = None) -> Path:
    """Guarda el informe Markdown en ``<base_dir>/AAAA-MM-DD.md``."""
    base = Path(base_dir) if base_dir is not None else config.PATHS.reports_dir
    path = base / f"{session_date.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text, encoding="utf-8")
    logger.info("Informe guardado en %s", path)
    return path


# Alias de compatibilidad.
def report_path(session_date: date) -> Path:
    return config.PATHS.reports_dir / f"{session_date.isoformat()}.md"


def save_markdown(markdown_text: str, session_date: date) -> Path:
    return save_report(markdown_text, session_date)


# --------------------------------------------------------------------------- #
# Conversión a HTML seguro para Telegram
# --------------------------------------------------------------------------- #
def _markdown_line_to_html(line: str) -> str:
    """Convierte una línea Markdown (enlaces, código, negrita, cursiva) a HTML escapado."""
    escaped = html.escape(line, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)", r'<a href="\2">\1</a>', escaped
    )
    parts = escaped.split("**")
    if len(parts) > 1 and len(parts) % 2 == 1:
        rebuilt = ""
        for index, part in enumerate(parts):
            rebuilt += f"<b>{part}</b>" if index % 2 == 1 else part
        escaped = rebuilt
    return escaped


def markdown_to_telegram_html(markdown_text: str) -> str:
    """Convierte el Markdown del informe a HTML válido para ``parse_mode=HTML``."""
    return "\n".join(
        _markdown_line_to_html(line) for line in markdown_text.split("\n")
    )


def build_telegram_messages(result: Dict[str, Any]) -> List[str]:
    """Construye los mensajes HTML listos para enviar por Telegram a partir de ``result``."""
    import telegram as telegram_module  # noqa: PLC0415 - evita import circular al cargar

    data = result if isinstance(result, ReportData) else _report_data_from_pipeline(result)
    markdown_text = build_report(data)
    html_text = markdown_to_telegram_html(markdown_text)
    return telegram_module.split_message(html_text)


def _report_data_from_pipeline(result: Dict[str, Any]) -> ReportData:
    """Adapta el dict que produce ``main.run_pipeline`` a ``ReportData`` (compatibilidad)."""
    return ReportData(
        session_date=result.get("session_date"),
        snapshot_ts=result.get("snapshot_ts"),
        session_type=result.get("session_type", "sesión normal"),
        universe_size=result.get("universe_size", 0),
        context=result.get("market_context"),
        candidates=result.get("candidates_payload", []),
        gemini=result.get("gemini"),
        failures=result.get("failures", {}),
        filter_no_trade_reason=result.get("no_trade_reason"),
        filter_log=result.get("filter_log", []),
        no_trade_reason=result.get("no_trade_reason"),
    )
