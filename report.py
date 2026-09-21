"""report.py — Informe diario en Markdown canónico y conversión a HTML de Telegram.

El Markdown (``data/reports/YYYY-MM-DD.md``) es la versión canónica. Se separa en
bloques con una línea en blanco (cabecera, contexto, candidatas, una por pick, riesgos)
para que ``telegram.py`` pueda dividirlo sin cortar una candidata por la mitad.

Reglas que cumple el informe
----------------------------
* Ningún dato se inventa: lo que falta se muestra como ``N/A``.
* Todas las cifras de mercado salen de Python (registro de ``analyze_ticker``); Gemini
  solo aporta interpretación. R/R y precio máximo de entrada los calcula Python.
* «Entrada prevista» (rango basado en el pre-market) no es «entrada confirmada»: esta
  depende del precio real de apertura. El pre-market nunca se presenta como precio de
  ejecución garantizado.
* Se muestra la calidad del dato pre-market cuando no es ``ok``.
* No se fuerzan tres candidatas; ``NO OPERAR`` es una salida válida.
* Termina con el aviso de que no es asesoramiento financiero.

Entradas (``ReportData``)
-------------------------
``candidates``: candidatas ordenadas por puntuación (ver contrato en ``history.py``).
``gemini``: resultado ya validado por Python: ``market_view``, ``no_trade``,
``no_trade_reason``, ``picks`` (con ``rr`` y ``max_entry_price``), ``warnings`` y
``available`` (False o ``gemini=None`` => «Análisis IA no disponible»).
``context``: ``MarketContext.to_dict()`` (índices, VIX, futuros, sectores, macro, régimen).
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import config
from history import (
    analysis_record, candidate_score, find_level, pick_field, write_text_atomic,
)
from technical_analysis import NA, is_num

logger = logging.getLogger("trading_agent.report")

DISCLAIMER = "Herramienta informativa, no es asesoramiento financiero."
AI_UNAVAILABLE_NOTE = "Análisis IA no disponible"
OPENING_RULE = ("Regla de apertura: si el precio de apertura supera el precio máximo de "
                "entrada válido, o invalida el R/R: NO ENTRAR.")
ENTRY_NOTE = ("La entrada prevista se basa en el pre-market; la entrada confirmada depende "
              "del precio real de apertura (09:30 ET).")
LATE_MARK = "⏱ Informe tardío"
MEDALS = ("🥇", "🥈", "🥉")

WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre")
LEVEL_LABELS_ES = {
    "prev_close": "cierre anterior", "max_5d": "máx. 5 sesiones", "min_5d": "mín. 5 sesiones",
    "max_20d": "máx. 20 sesiones", "min_20d": "mín. 20 sesiones",
    "high_52w": "máx. 52 semanas", "low_52w": "mín. 52 semanas", "sma20": "SMA20",
    "sma50": "SMA50", "ema9": "EMA9", "ema20": "EMA20", "premarket_high": "máx. pre-market",
    "premarket_low": "mín. pre-market",
}
QUALITY_LABELS_ES = {"stale": "dato desactualizado", "sparse": "volumen parcial",
                     "missing": "sin datos pre-market"}


# --------------------------------------------------------------------------- #
# Datos de entrada
# --------------------------------------------------------------------------- #
@dataclass
class ReportData:
    """Todo lo que necesita el informe (lo ensambla ``main.py``)."""

    session_date: date
    snapshot_ts: datetime
    sent_at: datetime | None = None
    session_type: str = "sesión normal"
    universe_count: int | None = None
    universe_source: str = ""
    context: Mapping[str, Any] | None = None
    candidates: Sequence[Mapping[str, Any]] = field(default_factory=list)
    gemini: Mapping[str, Any] | None = None
    filter_no_trade_reason: str = ""
    extra_risks: Sequence[str] = field(default_factory=list)

    @property
    def ai_available(self) -> bool:
        """True si hay respuesta de Gemini validada y utilizable."""
        return bool(self.gemini) and self.gemini.get("available", True) is not False


# --------------------------------------------------------------------------- #
# Formato de números y textos
# --------------------------------------------------------------------------- #
def fmt_num(value: Any, digits: int = 2) -> str:
    """Número con separador de miles y decimales fijos, o ``N/A``."""
    return f"{float(value):,.{digits}f}" if is_num(value) else NA


def fmt_pct(value: Any, digits: int = 2, signed: bool = True) -> str:
    """Porcentaje con signo (``+3.20%``), o ``N/A``."""
    if not is_num(value):
        return NA
    return f"{float(value):+.{digits}f}%" if signed else f"{float(value):.{digits}f}%"


def fmt_volume(value: Any) -> str:
    """Volumen compacto (``1.25M``, ``850K``), o ``N/A``."""
    if not is_num(value):
        return NA
    v = float(value)
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return f"{v:,.0f}"


def fmt_rvol(value: Any) -> str:
    """RVOL como múltiplo (``3.40x``), o ``N/A``."""
    return f"{float(value):.2f}x" if is_num(value) else NA


def fmt_pp(value: Any) -> str:
    """Diferencia en puntos porcentuales (``+2.50 pp``), o ``N/A``."""
    return f"{float(value):+.2f} pp" if is_num(value) else NA


def safe_text(value: Any, default: str = NA) -> str:
    """Texto externo (Gemini, noticias) en una sola línea y sin sintaxis Markdown propia."""
    if value is None:
        return default
    text = " ".join(str(value).split())
    text = text.replace("**", "").replace("`", "'").replace("[", "(").replace("]", ")")
    return text or default


def spanish_date(day: date) -> str:
    """``lunes 21 de septiembre de 2026``."""
    return f"{WEEKDAYS[day.weekday()]} {day.day} de {MONTHS[day.month - 1]} de {day.year}"


def is_late(sent_at: datetime | None, schedule: config.ScheduleConfig | None = None) -> bool:
    """True si el envío ocurre después de ``SCHEDULE.report_time`` (hora de Nueva York)."""
    schedule = schedule or config.SCHEDULE
    return sent_at is not None and sent_at.timetz().replace(tzinfo=None) > schedule.report_time


def quality_note(premarket: Mapping[str, Any]) -> str | None:
    """Texto de calidad del dato pre-market, o None si es ``ok`` o no consta."""
    quality = premarket.get("premarket_quality")
    if quality in (None, NA, "ok"):
        return None
    label = QUALITY_LABELS_ES.get(str(quality), str(quality))
    note = safe_text(premarket.get("quality_note"), "")
    return f"{label} ({note})" if note else label


def _level_text(record: Mapping[str, Any], label: Any, fallback: Any = None) -> str:
    """``SMA20 (98.20)`` a partir de una etiqueta de la tabla de niveles."""
    row = find_level(record, label)
    if row is None and isinstance(fallback, Mapping) and fallback.get("label"):
        row = dict(fallback)
    if row is None:
        return NA
    name = LEVEL_LABELS_ES.get(row.get("label"), str(row.get("label")))
    return f"{name} ({fmt_num(row.get('price'))})"


# --------------------------------------------------------------------------- #
# Bloques del informe
# --------------------------------------------------------------------------- #
def _header_block(data: ReportData) -> str:
    lines = [f"📅 **{spanish_date(data.session_date)}** ({safe_text(data.session_type)})"]
    schedule = f"🕒 Snapshot {data.snapshot_ts:%H:%M} ET"
    if data.sent_at is not None:
        schedule += f" · Envío {data.sent_at:%H:%M} ET"
        if is_late(data.sent_at):
            schedule += f" · {LATE_MARK}"
    lines.append(schedule)
    return "\n".join(lines)


def _move_text(name: str, entry: Any, show_level: bool = False) -> str:
    """``SPY +0.35%`` (añade ``(último cierre)`` si no es pre-market)."""
    if not isinstance(entry, Mapping):
        return f"{name} {NA}"
    level = f"{fmt_num(entry.get('price'))} " if show_level else ""
    text = f"{name} {level}{fmt_pct(entry.get('change_pct'))}"
    basis = str(entry.get("basis", ""))
    if basis.startswith("último cierre"):
        text += " (último cierre)"
    elif basis not in ("pre-market", NA, ""):
        text += f" ({safe_text(basis)})"
    return text


def _context_block(data: ReportData) -> str:
    lines = ["📊 **CONTEXTO DE MERCADO**"]
    ctx = data.context
    if not isinstance(ctx, Mapping) or not ctx:
        lines.append("• Contexto de mercado no disponible (N/A)")
    else:
        indices = ctx.get("indices") or {}
        lines.append("• Índices: " + (" · ".join(_move_text(s, e) for s, e in indices.items())
                                     or NA))
        vol = ctx.get("volatility")
        lines.append("• Volatilidad: " + (_move_text(str(vol.get("symbol", "VIX")).lstrip("^"),
                                                       vol, show_level=True)
                                         if isinstance(vol, Mapping) and vol else NA))
        futures = ctx.get("futures") or {}
        lines.append("• Futuros: " + (" · ".join(_move_text(s, e) for s, e in futures.items())
                                      or NA))
        ten = ctx.get("treasury_10y")
        lines.append("• Bono 10 años: " + (_move_text(str(ten.get("symbol", "10Y")).lstrip("^"),
                                                        ten, show_level=True)
                                          if isinstance(ten, Mapping) and ten else NA))
        sectors = ctx.get("sectors") or {}
        leaders, laggards = ctx.get("sector_leaders") or [], ctx.get("sector_laggards") or []
        if leaders or laggards:
            lead = " · ".join(_move_text(s, sectors.get(s)) for s in leaders) or NA
            lag = " · ".join(_move_text(s, sectors.get(s)) for s in laggards) or NA
            lines.append(f"• Sectores líderes: {lead} | rezagados: {lag}")
        macro = ctx.get("macro") or {}
        lines.append("• Macro/Fed: " + safe_text(macro.get("text") if isinstance(macro, Mapping)
                                                   else None))
        regime = ctx.get("regime") or {}
        lines.append("• Régimen: " + safe_text(regime.get("summary") if isinstance(regime, Mapping)
                                                else None))
    if data.ai_available and data.gemini.get("market_view"):
        lines.append("• Visión de mercado (IA): " + safe_text(data.gemini.get("market_view")))
    return "\n".join(lines)


def _candidates_block(data: ReportData) -> str:
    lines = []
    source = f" (lista leída de {safe_text(data.universe_source)})" if data.universe_source else ""
    universe = data.universe_count if data.universe_count is not None else NA
    lines.append(f"📋 Universo: {universe} tickers{source}")
    cands = list(data.candidates)
    lines.append(f"🔎 **CANDIDATAS ANALIZADAS** ({len(cands)}, con puntuación)")
    if not cands:
        lines.append("• Ninguna candidata superó el filtro")
        return "\n".join(lines)
    parts, flagged = [], False
    for cand in cands:
        score = candidate_score(cand)
        premarket = analysis_record(cand).get("premarket", {}) or {}
        mark = " ⚠" if quality_note(premarket) else ""
        flagged = flagged or bool(mark)
        parts.append(f"{safe_text(cand.get('ticker'))} {fmt_num(score, 1)}{mark}")
    lines.append("• " + " · ".join(parts))
    if flagged:
        lines.append("⚠ = calidad del dato pre-market no óptima (detalle en cada candidata)")
    return "\n".join(lines)


def _metrics_lines(record: Mapping[str, Any]) -> list[str]:
    """Líneas de datos calculados por Python (comunes a picks y candidatas sin IA)."""
    premarket = record.get("premarket", {}) or {}
    ind = record.get("indicators", {}) or {}
    rvol = (record.get("rvol") or {}).get("rvol", NA)
    lines = [
        f"• **Cierre anterior / Pre-market / Gap:** {fmt_num(premarket.get('prev_close'))} / "
        f"{fmt_num(premarket.get('premarket_price'))} / {fmt_pct(record.get('gap_pct'))}",
        f"• **Volumen / RVOL:** {fmt_volume(premarket.get('premarket_volume'))} / "
        f"{fmt_rvol(rvol)} (pre-market = {fmt_pct(record.get('pm_pct_of_adv'), 1, False)} "
        f"del volumen medio diario)",
        f"• **RSI / SMA20 / SMA50 / ATR:** {fmt_num(ind.get('rsi14'), 1)} / "
        f"{fmt_num(ind.get('sma20'))} / {fmt_num(ind.get('sma50'))} / {fmt_num(ind.get('atr14'))}",
    ]
    return lines


def _rs_line(record: Mapping[str, Any]) -> str:
    rs = record.get("relative_strength", {}) or {}
    return (f"• **Fuerza relativa:** pre-market vs SPY {fmt_pp(rs.get('rs_pm_vs_spy'))} · "
            f"vs QQQ {fmt_pp(rs.get('rs_pm_vs_qqq'))}; 5 sesiones vs SPY "
            f"{fmt_pp(rs.get('rs_5d_vs_spy'))} · vs QQQ {fmt_pp(rs.get('rs_5d_vs_qqq'))}")


def _flags_text(candidate: Mapping[str, Any]) -> str:
    flags = (candidate.get("score_detail") or {}).get("flags") or []
    return ", ".join(str(f).replace("_", " ") for f in flags) if flags else "ninguna"


def _catalyst_lines(pick: Mapping[str, Any] | None, candidate: Mapping[str, Any]) -> str:
    news = candidate.get("news") or {}
    catalyst = (pick or {}).get("catalyst") or news.get("catalyst_text")
    source = (pick or {}).get("catalyst_source")
    if not source and isinstance(news.get("catalyst"), Mapping):
        source = news["catalyst"].get("publisher") or news["catalyst"].get("url")
    text = safe_text(catalyst)
    src = safe_text(source)
    if isinstance(source, str) and re.fullmatch(r"https?://[^\s)\"<>]+", source.strip()):
        src = f"[fuente]({source.strip()})"
    return f"• **Catalizador / Fuente:** {text} / {src}"


def _pick_block(rank: int, pick: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    record = analysis_record(candidate)
    direction = str(pick.get("direction") or candidate.get("direction") or NA)
    medal = MEDALS[rank] if rank < len(MEDALS) else "▫️"
    name = safe_text(pick.get("company"), "")
    title = f"{medal} **{safe_text(pick.get('ticker'))}**" + (f" — {name}" if name else "")
    score = candidate_score(candidate)
    title += f" · {direction} · Puntuación {fmt_num(score, 1)}"

    lines = [title, _catalyst_lines(pick, candidate), *_metrics_lines(record)]
    lines.append(
        f"• **Soporte / Resistencia:** "
        f"{_level_text(record, pick.get('support_level'), record.get('nearest_support'))} / "
        f"{_level_text(record, pick.get('resistance_level'), record.get('nearest_resistance'))}")
    lines.append(_rs_line(record))
    lines.append(f"• **Riesgo gap & fade:** {safe_text(pick.get('fade_risk'))} — "
                 f"{safe_text(pick.get('fade_risk_explanation'))} (banderas: "
                 f"{_flags_text(candidate)})")
    lines.append(f"• **Entrada prevista:** {fmt_num(pick.get('entry_zone_low'))} – "
                 f"{fmt_num(pick.get('entry_zone_high'))} (rango basado en pre-market y niveles)")
    lines.append(f"• **Stop / Objetivo 1 / Objetivo 2 / R/R (Python):** "
                 f"{fmt_num(pick.get('stop'))} / {fmt_num(pick.get('target_1'))} / "
                 f"{fmt_num(pick.get('target_2'))} / {fmt_num(pick_field(pick, 'rr'))}")
    limit = fmt_num(pick_field(pick, "max_entry_price"))
    if direction == "SHORT":
        lines.append(f"• **Precio mínimo de entrada válido:** {limit} (por debajo: NO ENTRAR)")
    else:
        lines.append(f"• **Precio máximo de entrada válido:** {limit} (por encima: NO ENTRAR)")
    lines.append(f"• **Hora objetivo / Invalidación:** {safe_text(pick.get('exit_time'))} / "
                 f"{safe_text(pick.get('invalidation'))}")
    lines.append(f"• **Decisión / Razón:** {safe_text(pick.get('decision'))} "
                 f"(confianza {safe_text(pick.get('confidence'))}) — {safe_text(pick.get('reason'))}")
    note = quality_note(record.get("premarket", {}) or {})
    if note:
        lines.append(f"⚠ Calidad del dato pre-market: {note}")
    for warning in (pick.get("warnings") or [])[:3]:
        lines.append(f"⚠ Aviso: {safe_text(warning)}")
    return "\n".join(lines)


def _candidate_block(rank: int, candidate: Mapping[str, Any]) -> str:
    """Bloque solo con datos de Python (sin IA): sin entrada, stop ni objetivos."""
    record = analysis_record(candidate)
    medal = MEDALS[rank] if rank < len(MEDALS) else "▫️"
    direction = str(candidate.get("direction") or NA)
    lines = [f"{medal} **{safe_text(candidate.get('ticker'))}** · {direction} · Puntuación "
             f"{fmt_num(candidate_score(candidate), 1)} (solo datos de Python)",
             _catalyst_lines(None, candidate), *_metrics_lines(record)]
    lines.append(f"• **Soporte / Resistencia:** "
                 f"{_level_text(record, None, record.get('nearest_support'))} / "
                 f"{_level_text(record, None, record.get('nearest_resistance'))}")
    lines.append(_rs_line(record))
    lines.append(f"• **Riesgo gap & fade (banderas):** {_flags_text(candidate)}")
    note = quality_note(record.get("premarket", {}) or {})
    if note:
        lines.append(f"⚠ Calidad del dato pre-market: {note}")
    return "\n".join(lines)


def _risk_lines(data: ReportData, picks: Sequence[Mapping[str, Any]]) -> list[str]:
    risks: list[str] = []
    ctx = data.context if isinstance(data.context, Mapping) else {}
    regime = ctx.get("regime") or {}
    if isinstance(regime, Mapping) and regime.get("risk_level") == "alto":
        risks.append(safe_text(regime.get("summary")))
    macro = ctx.get("macro") or {}
    if isinstance(macro, Mapping) and macro.get("high_impact_today"):
        risks.append("Evento macro de impacto alto hoy: " + safe_text(macro.get("text")))
    for pick in picks:
        if str(pick.get("fade_risk")).upper() == "HIGH":
            risks.append(f"{safe_text(pick.get('ticker'))}: riesgo alto de gap-and-fade")
    for cand in data.candidates:
        note = quality_note(analysis_record(cand).get("premarket", {}) or {})
        if note:
            risks.append(f"{safe_text(cand.get('ticker'))}: {note}")
    if data.ai_available:
        for warning in (data.gemini.get("warnings") or [])[:5]:
            risks.append("Validación: " + safe_text(warning))
        if picks and not any(str(p.get("decision")).upper() == "BUY" for p in picks):
            risks.append("Ninguna candidata está en BUY: esperar confirmación tras la apertura")
    risks.extend(safe_text(r) for r in data.extra_risks)
    risks.append("El pre-market tiene poca liquidez: la apertura puede diferir de forma "
                 "relevante y los gaps pueden revertirse (gap-and-fade)")
    return list(dict.fromkeys(risks))


def _no_trade_reason(data: ReportData, picks: Sequence[Mapping[str, Any]]) -> str | None:
    """Motivo de NO OPERAR, o None si hay selección utilizable."""
    if not data.candidates and data.filter_no_trade_reason:
        return safe_text(data.filter_no_trade_reason)
    if not data.candidates:
        return "Ninguna candidata superó el filtro."
    if not data.ai_available:
        return f"{AI_UNAVAILABLE_NOTE}: no se emite selección BUY."
    gem = data.gemini or {}
    if gem.get("no_trade"):
        return safe_text(gem.get("no_trade_reason"), "Gemini no recomienda operar hoy.")
    if not picks:
        return "Ninguna candidata superó la validación (ver avisos)."
    return None


def build_report(data: ReportData) -> str:
    """Genera el informe completo en Markdown canónico."""
    by_ticker = {c.get("ticker"): c for c in data.candidates if isinstance(c, Mapping)}
    picks: list[Mapping[str, Any]] = []
    if data.ai_available:
        picks = [p for p in (data.gemini.get("picks") or [])
                 if isinstance(p, Mapping) and p.get("ticker") in by_ticker][: config.TRADE.max_picks]

    blocks = [_header_block(data), _context_block(data), _candidates_block(data)]
    if not data.ai_available:
        blocks.append(f"🤖 **{AI_UNAVAILABLE_NOTE}.** El informe usa solo datos calculados por "
                      f"Python y no incluye selección BUY.")
        for rank, cand in enumerate(list(data.candidates)[: config.TRADE.max_picks]):
            blocks.append(_candidate_block(rank, cand))
    else:
        for rank, pick in enumerate(picks):
            blocks.append(_pick_block(rank, pick, by_ticker[pick["ticker"]]))
    if picks:
        blocks.append(f"🔔 {OPENING_RULE}\nℹ️ {ENTRY_NOTE}")

    blocks.append("⚠️ **RIESGOS**\n" + "\n".join(f"• {r}" for r in _risk_lines(data, picks)))
    reason = _no_trade_reason(data, picks)
    if reason:
        blocks.append(f"🚫 **NO OPERAR**\n• {reason}")
    blocks.append(f"ℹ️ {DISCLAIMER}")
    return "\n\n".join(blocks) + "\n"


def save_report(markdown: str, target: date, reports_dir: Path | None = None) -> Path:
    """Guarda el informe canónico en ``data/reports/YYYY-MM-DD.md``."""
    path = (reports_dir or config.PATHS.reports_dir) / f"{target.isoformat()}.md"
    write_text_atomic(path, markdown)
    logger.info("Informe guardado en %s", path)
    return path


# --------------------------------------------------------------------------- #
# Markdown -> HTML seguro para Telegram
# --------------------------------------------------------------------------- #
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s\"<>]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`\n]+)`")


def markdown_to_telegram_html(markdown: str) -> str:
    """Convierte el Markdown del informe a HTML admitido por Telegram.

    Primero se escapa **todo** el texto con ``html.escape`` y solo después se generan
    las etiquetas ``<b>``, ``<code>`` y ``<a href>`` a partir de ``**negrita**``,
    `` `código` `` y ``[texto](https://url)``. Así, ningún contenido externo puede
    inyectar etiquetas. Solo se aceptan enlaces http(s). La conversión es línea a
    línea: una etiqueta nunca abarca varias líneas.
    """
    out_lines = []
    for line in markdown.splitlines():
        escaped = html.escape(line, quote=False)
        escaped = _LINK_RE.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', escaped)
        escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
        escaped = _CODE_RE.sub(r"<code>\1</code>", escaped)
        out_lines.append(escaped)
    return "\n".join(out_lines)
