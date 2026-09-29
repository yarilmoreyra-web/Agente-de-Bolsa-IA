"""update_report.py — Texto del Telegram de la actualización post-apertura.

Recibe el diccionario que produce ``update_agent.run_update`` y devuelve
Markdown con el mismo estilo que ``report.py`` (``**negrita**``, emojis), de
modo que ``report.markdown_to_telegram_html`` y ``telegram.send_report`` sirven
tal cual. Cuando falta un dato se muestra ``N/A``; nunca se inventa.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import config
import post_open as po
from report import (
    DISCLAIMER, MEDALS, fmt_num, fmt_pct, fmt_rvol, fmt_volume, safe_text,
    spanish_date,
)

logger = logging.getLogger("trading_agent.update_report")

VERDICT_LABELS = {
    po.V_CONFIRMED: ("✅", "Confirmada"),
    po.V_IMPROVED: ("🔼", "Mejorada"),
    po.V_WEAKENED: ("⚠️", "Debilitada"),
    po.V_INVALIDATED: ("❌", "Invalidada"),
    po.V_EXTENDED: ("⏫", "Extendida: no perseguir"),
    po.V_TARGET: ("🎯", "Objetivo 1 alcanzado"),
    po.V_NO_DATA: ("❓", "Sin datos"),
}
# Orden y etiqueta (plural) del resumen de cambios.
_COUNT_LABELS = (
    (po.V_CONFIRMED, "✅ {n} confirmada(s)"),
    (po.V_IMPROVED, "🔼 {n} mejorada(s)"),
    (po.V_WEAKENED, "⚠️ {n} debilitada(s)"),
    (po.V_INVALIDATED, "❌ {n} invalidada(s)"),
    (po.V_EXTENDED, "⏫ {n} extendida(s)"),
    (po.V_TARGET, "🎯 {n} con objetivo 1 alcanzado"),
    (po.V_NO_DATA, "❓ {n} sin datos"),
)
_SYMBOL_NAMES = {"^VIX": "VIX"}
_QUALITY_LABELS = {"stale": "dato desactualizado", "sparse": "pocas barras",
                   "missing": "sin datos"}


def _hhmm(value: Any) -> str:
    return po._hhmm(value)


def _same_day_hhmm(value: Any, session_date: Any) -> str | None:
    """HH:MM de ``value`` solo si es del mismo día que la sesión."""
    text = str(value or "")
    return _hhmm(text) if text[:10] == str(session_date) and len(text) >= 16 else None


def _delta(morning: Any, now: Any, pct: bool) -> str:
    """«antes → ahora» con el formato adecuado (porcentaje con signo o nivel)."""
    fmt = fmt_pct if pct else fmt_num
    return f"{fmt(morning)} → {fmt(now)}"


def _market_line(rows: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for row in rows:
        symbol = row.get("symbol", "N/A")
        name = _SYMBOL_NAMES.get(symbol, symbol)
        if symbol == "^VIX":
            parts.append(f"{name} {_delta(row.get('morning_price'), row.get('now_price'), False)}")
        else:
            parts.append(
                f"{name} {_delta(row.get('morning_change_pct'), row.get('now_change_pct'), True)}")
    return " | ".join(parts) if parts else "Mercado no disponible (N/A)"


def _decision_text(decision: Any, confidence: Any) -> str:
    conf = confidence if confidence not in (None, "", "N/A") else None
    return f"{decision or 'N/A'}" + (f" ({conf})" if conf else "")


def _side_words(direction: str) -> str:
    return "mínimo" if direction == po.SHORT else "máximo"


def _pick_block(position: int, entry: Mapping[str, Any]) -> str:
    pick, live, assessment = entry["pick"], entry["live"], entry["assessment"]
    ticker = entry["ticker"]
    direction = assessment.get("direction") or po.LONG
    emoji, label = VERDICT_LABELS.get(entry.get("verdict"), ("•", str(entry.get("verdict"))))
    medal = MEDALS[position] if position < len(MEDALS) else "•"
    status = assessment.get("status")

    before = _decision_text(pick.get("decision"), pick.get("confidence"))
    after = _decision_text(entry.get("decision"), entry.get("confidence"))
    rr_now = assessment.get("rr_now")
    max_valid = pick.get("max_valid_entry")

    lines = [
        f"{medal} **{ticker}** ({direction}) — {pick.get('company') or 'N/A'}",
        f"🔁 Informe 09:00: {before} → Ahora: **{after}** {emoji} {label}",
        (f"📈 Apertura {fmt_num(live.get('open_price'))} · Ahora {fmt_num(live.get('price_now'))} "
         f"· vs apertura {fmt_pct(live.get('change_vs_open_pct'))} "
         f"· vs cierre ant. {fmt_pct(live.get('change_vs_prev_close_pct'))}"),
        (f"📊 VWAP {fmt_num(live.get('vwap'))} (precio {live.get('price_vs_vwap', 'N/A')}) "
         f"· Volumen 30 min {fmt_volume(live.get('volume_since_open'))} "
         f"· RVOL apertura {fmt_rvol(live.get('open_rvol'))}"),
        (f"🎯 Niveles del informe: entrada {fmt_num(pick.get('entry_zone_low'))}–"
         f"{fmt_num(pick.get('entry_zone_high'))} · stop {fmt_num(pick.get('stop'))} · "
         f"O1 {fmt_num(pick.get('target_1'))} · O2 {fmt_num(pick.get('target_2'))}"),
        (f"⚙️ Estado (Python): {po.STATUS_LABELS.get(status, status)} · R/R con precio actual "
         f"{fmt_num(rr_now)} (mañana {fmt_num(pick.get('rr'))}) · "
         f"{_side_words(direction)} de entrada válido {fmt_num(max_valid)}"),
    ]
    rule = assessment.get("open_rule")
    if rule == "CUMPLE":
        lines.append(f"🔔 Regla de apertura: abrió a {fmt_num(live.get('open_price'))}, "
                     f"dentro del límite ({fmt_num(max_valid)}) → cumplía")
    elif rule == "NO_ENTRAR":
        lines.append(f"🔔 Regla de apertura: abrió a {fmt_num(live.get('open_price'))}, "
                     f"fuera del límite ({fmt_num(max_valid)}) → NO ENTRAR desde la apertura")
    if assessment.get("note"):
        lines.append(f"ℹ️ {safe_text(assessment['note'])}")
    if entry.get("what_changed"):
        lines.append(f"🧠 Qué cambió: {safe_text(entry['what_changed'])}")
    if entry.get("action"):
        lines.append(f"➡️ Qué hacer: {safe_text(entry['action'])}")
    for news in entry.get("news_new", []):
        hard = " 🔴" if news.get("is_hard_catalyst") else ""
        lines.append(f"📰 Nueva ({_hhmm(news.get('published_at'))}){hard}: "
                     f"{safe_text(news.get('headline'))} — {news.get('publisher') or 'N/A'}")
    quality = live.get("quality")
    if quality and quality != "ok":
        lines.append(f"⚠ Calidad del dato: {_QUALITY_LABELS.get(quality, quality)} "
                     f"({live.get('quality_note') or 'N/A'})")
    return "\n".join(lines)


def _other_line(entry: Mapping[str, Any]) -> str:
    live = entry["live"]
    change = po._num(live.get("change_vs_open_pct"))
    hot = " 🔥" if change is not None and abs(change) >= config.UPDATE.mover_pct else ""
    if live.get("quality") == "missing":
        return f"• {entry['ticker']}: sin datos de la sesión regular"
    return (f"• {entry['ticker']}{hot}: {fmt_pct(live.get('change_vs_open_pct'))} vs apertura "
            f"({fmt_pct(live.get('change_vs_prev_close_pct'))} vs cierre ant.) · "
            f"RVOL {fmt_rvol(live.get('open_rvol'))} · VWAP: {live.get('price_vs_vwap', 'N/A')}"
            + (f" · 📰 {len(entry['news_new'])} nueva(s)" if entry.get("news_new") else ""))


def build_update_report(result: Mapping[str, Any]) -> str:
    """Construye el Markdown de la actualización a partir del resultado del pipeline."""
    session_date = result.get("session_date")
    morning = result.get("morning") or {}
    picks = result.get("picks") or []
    others = result.get("others") or []
    gemini = result.get("gemini") or {}
    ia_available = bool(gemini.get("available"))

    morning_hhmm = _same_day_hhmm(morning.get("sent_at"), session_date) \
        or config.SCHEDULE.report_time.strftime("%H:%M")
    sent_at = result.get("sent_at")
    envio = f" · Envío {sent_at.strftime('%H:%M')} ET" if isinstance(sent_at, datetime) else ""

    blocks: list[str] = [
        "\n".join([
            f"🔄 **ACTUALIZACIÓN POST-APERTURA** — {spanish_date(session_date)} "
            f"({result.get('session_type', 'sesión normal')})",
            f"🕒 Datos hasta las {_hhmm(result.get('update_ts'))} ET · "
            f"Informe previo {morning_hhmm} ET{envio}",
        ]),
    ]

    # -- Mercado: mañana vs. ahora ----------------------------------------
    market_lines = [f"📊 **MERCADO: {_hhmm(morning.get('snapshot_ts'))} → "
                    f"{_hhmm(result.get('update_ts'))}**",
                    _market_line(result.get("market") or [])]
    if gemini.get("market_view"):
        market_lines.append(f"🧠 {safe_text(gemini['market_view'])}")
    blocks.append("\n".join(market_lines))

    if picks and not ia_available:
        blocks.append(
            f"⚠️ {gemini.get('note') or 'Análisis IA no disponible'}: "
            f"{safe_text(gemini.get('error') or 'N/A')}. "
            "La comparación usa solo datos de Python.")

    # -- Sin selecciones en la mañana -------------------------------------
    if not picks:
        reason = safe_text(morning.get("no_trade_reason")) if morning.get("no_trade_reason") \
            else "sin selecciones"
        blocks.append(
            "🚫 **El informe de las 09:00 fue NO OPERAR**\n"
            f"• {reason}\n"
            "Esta actualización no genera nuevas señales de entrada: solo muestra "
            "cómo se han movido las candidatas analizadas.")
    else:
        # -- Resumen de cambios -------------------------------------------
        counts = result.get("counts") or {}
        parts = [text.format(n=counts[key]) for key, text in _COUNT_LABELS if counts.get(key)]
        summary_lines = ["📌 **CAMBIOS RESPECTO AL INFORME DE LAS 09:00**",
                         " · ".join(parts) if parts else "Sin datos para comparar."]
        if gemini.get("summary"):
            summary_lines.append(safe_text(gemini["summary"]))
        blocks.append("\n".join(summary_lines))

        for position, entry in enumerate(picks):
            blocks.append(_pick_block(position, entry))

        blocks.append(
            "🔔 Recordatorio\nLos niveles (entrada, stop y objetivos) son los del informe "
            "de las 09:00; la actualización solo recalcula el R/R con el precio actual. "
            "Si el precio ya salió de la zona, no persigas la entrada.")

    # -- Otras candidatas --------------------------------------------------
    if others:
        blocks.append("\n".join(
            [f"🔎 **OTRAS CANDIDATAS DE LAS 09:00** ({len(others)}, sin selección)"]
            + [_other_line(entry) for entry in others]))

    watchlist = result.get("watchlist") or []
    if watchlist:
        blocks.append("\n".join(
            ["👀 **PARA VIGILAR** (sin niveles: no es una señal de entrada)"]
            + [f"• {w['ticker']} — {safe_text(w['reason'])}" for w in watchlist]))

    # -- Riesgos y avisos --------------------------------------------------
    risks: list[str] = [f"- {w}" for w in gemini.get("warnings") or []]
    for entry in picks:
        risks.extend(f"- {w}" for w in entry.get("warnings") or [])
    failed = sorted((result.get("failures") or {}).get("intraday_today", {}))
    if failed:
        risks.append(f"- Sin datos de la sesión regular para: {', '.join(failed)}")
    if risks:
        blocks.append("⚠️ **AVISOS**\n" + "\n".join(risks))

    blocks.append(DISCLAIMER)
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Guardado
# --------------------------------------------------------------------------- #
def report_path(session_date: date, base_dir: Path | None = None) -> Path:
    """Ruta del informe de actualización: ``AAAA-MM-DD_actualizacion.md``."""
    base = Path(base_dir) if base_dir is not None else config.PATHS.reports_dir
    return base / f"{session_date.isoformat()}_actualizacion.md"


def save_update_report(markdown_text: str, session_date: date,
                       base_dir: Path | None = None) -> Path:
    """Guarda el Markdown junto al informe de la mañana (sin pisarlo)."""
    path = report_path(session_date, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text, encoding="utf-8")
    logger.info("Informe de actualización guardado en %s", path)
    return path
