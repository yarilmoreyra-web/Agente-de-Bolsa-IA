"""market_context.py — Contexto de mercado del día (índices, sectores, macro, régimen).

Qué mide
--------
* Índices (SPY, QQQ, DIA), futuros (ES=F, NQ=F), volatilidad (^VIX), tipo a 10
  años (^TNX) y los once ETFs sectoriales SPDR.
* Para cada símbolo se usa la misma maquinaria que para las acciones
  (``market_data.build_snapshots``): variación pre-market frente al cierre
  anterior. Si no hay barras pre-market (típico en ^VIX o en símbolos poco
  líquidos), se cae al último dato diario disponible y se indica en ``source``.
  Lo que no se puede calcular es ``"N/A"``: no se estima nada.
* Eventos macro y de la Fed: se leen de ``data/macro_events.json``, que mantiene
  el usuario a mano. Si el archivo no cubre la fecha, el informe dice
  «Calendario macro no cargado para hoy». Nunca se inventan fechas.
* El régimen de riesgo lo decide Python con reglas explícitas (ver
  ``classify_regime``), no Gemini.

Nota sobre ^TNX: Yahoo publica el rendimiento del bono a 10 años ya en
porcentaje (por ejemplo 4.25 = 4,25 %). Se muestra tal cual, sin convertir.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import config
import utils
from technical_analysis import NA, compute_daily_indicators, gap_pct, is_num

logger = logging.getLogger("trading_agent.market_context")

MACRO_NOT_LOADED = "Calendario macro no cargado para hoy"


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@dataclass
class SymbolQuote:
    """Estado de un símbolo de referencia a la hora del snapshot."""

    symbol: str
    price: float | str = NA
    prev_close: float | str = NA
    change_pct: float | str = NA
    source: str = "missing"        # premarket | last_close | missing
    quality: str = "missing"
    return_5d_pct: float | str = NA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketContext:
    """Contexto completo del día, serializable a JSON."""

    session_date: str
    snapshot_ts: str
    quotes: dict[str, dict[str, Any]] = field(default_factory=dict)
    indices: dict[str, Any] = field(default_factory=dict)
    futures: dict[str, Any] = field(default_factory=dict)
    volatility: dict[str, Any] = field(default_factory=dict)
    rates: dict[str, Any] = field(default_factory=dict)
    sectors: dict[str, Any] = field(default_factory=dict)
    sector_leaders: list[dict[str, Any]] = field(default_factory=list)
    sector_laggards: list[dict[str, Any]] = field(default_factory=list)
    regime: dict[str, Any] = field(default_factory=dict)
    macro: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    failures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def market_against_longs(self) -> bool:
        """True si el mercado está en contra de las compras (lo usa el filtro)."""
        return bool(self.regime.get("against_longs", False))

    def benchmarks_for_rs(self) -> dict[str, dict[str, Any]]:
        """Datos de SPY/QQQ en el formato que espera ``relative_strength``."""
        out: dict[str, dict[str, Any]] = {}
        for symbol in ("SPY", "QQQ"):
            quote = self.quotes.get(symbol)
            if quote:
                out[symbol] = {
                    "pm_change_pct": quote.get("change_pct", NA),
                    "return_5d_pct": quote.get("return_5d_pct", NA),
                }
        return out


# --------------------------------------------------------------------------- #
# Construcción de cotizaciones
# --------------------------------------------------------------------------- #
def build_quote(
    symbol: str, snapshot: Mapping[str, Any] | None, indicators: Mapping[str, Any],
) -> SymbolQuote:
    """Combina snapshot pre-market e indicadores diarios en un ``SymbolQuote``.

    Prioriza el precio pre-market. Si no hay, usa el último cierre diario y marca
    ``source = "last_close"`` (la variación es entonces la de la sesión anterior).
    """
    snapshot = snapshot or {}
    quote = SymbolQuote(symbol=symbol, return_5d_pct=indicators.get("return_5d_pct", NA))
    quote.quality = str(snapshot.get("premarket_quality", "missing"))
    prev_close = snapshot.get("prev_close", NA)

    pm_price = snapshot.get("premarket_price", NA)
    if is_num(pm_price) and is_num(prev_close):
        quote.price, quote.prev_close = pm_price, prev_close
        quote.change_pct = gap_pct(pm_price, prev_close)
        quote.source = "premarket"
        return quote

    last_close = indicators.get("last_close", NA)
    if is_num(last_close):
        quote.price = last_close
        quote.prev_close = prev_close if is_num(prev_close) else NA
        quote.change_pct = gap_pct(last_close, quote.prev_close)
        quote.source = "last_close"
    return quote


def _sorted_by_change(quotes: Mapping[str, Mapping[str, Any]],
                      symbols: Sequence[str]) -> list[dict[str, Any]]:
    """Símbolos con variación numérica, de mayor a menor."""
    rows = [
        {"symbol": s, "change_pct": quotes[s]["change_pct"]}
        for s in symbols
        if s in quotes and is_num(quotes[s].get("change_pct"))
    ]
    return sorted(rows, key=lambda r: float(r["change_pct"]), reverse=True)


# --------------------------------------------------------------------------- #
# Calendario macro
# --------------------------------------------------------------------------- #
def load_macro_events(
    session_date: date, path: Path | None = None, lookahead_days: int = 1,
) -> dict[str, Any]:
    """Lee ``data/macro_events.json`` y devuelve los eventos de la fecha.

    Formato esperado::

        {"events": [{"date": "2026-09-21", "time": "08:30",
                     "name": "IPC de agosto", "importance": "alta",
                     "notes": "Publicación antes de la apertura"}]}

    Si el archivo no existe, está vacío o no cubre la fecha, se devuelve
    ``loaded = False`` y la nota «Calendario macro no cargado para hoy».
    No se inventa ningún evento.
    """
    path = path or getattr(config, "MACRO_EVENTS_FILE",
                           config.PATHS.history_dir.parent / "macro_events.json")
    raw = utils.read_json(Path(path), default=None)

    result: dict[str, Any] = {
        "loaded": False, "today": [], "upcoming": [], "note": MACRO_NOT_LOADED,
        "file": str(path),
    }
    if raw is None:
        logger.info("Calendario macro no encontrado en %s.", path)
        return result

    events = raw.get("events") if isinstance(raw, Mapping) else raw
    if not isinstance(events, list):
        logger.warning("El calendario macro %s no tiene una lista 'events'.", path)
        return result

    horizon = session_date + timedelta(days=max(0, lookahead_days))
    today, upcoming = [], []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        try:
            event_date = date.fromisoformat(str(event.get("date", "")))
        except ValueError:
            logger.warning("Evento macro con fecha inválida: %r", event.get("date"))
            continue
        clean = {
            "date": event_date.isoformat(),
            "time": str(event.get("time", "")) or NA,
            "name": str(event.get("name", "")) or NA,
            "importance": str(event.get("importance", "")) or NA,
            "notes": str(event.get("notes", "")),
        }
        if event_date == session_date:
            today.append(clean)
        elif session_date < event_date <= horizon:
            upcoming.append(clean)

    result["today"] = sorted(today, key=lambda e: e["time"])
    result["upcoming"] = sorted(upcoming, key=lambda e: (e["date"], e["time"]))
    covered = any(
        str(e.get("date", "")) == session_date.isoformat()
        for e in events if isinstance(e, Mapping)
    )
    file_covers_date = covered or bool(upcoming) or _file_declares_date(raw, session_date)
    result["loaded"] = file_covers_date
    if file_covers_date:
        result["note"] = (
            f"{len(today)} evento(s) macro hoy" if today
            else "Sin eventos macro relevantes hoy"
        )
    return result


def _file_declares_date(raw: Any, session_date: date) -> bool:
    """True si el archivo declara explícitamente que cubre la fecha.

    Permite marcar días sin eventos con ``{"covered_dates": ["2026-09-21"]}``,
    para distinguir «no hay nada hoy» de «no he cargado el calendario».
    """
    if not isinstance(raw, Mapping):
        return False
    covered = raw.get("covered_dates")
    if isinstance(covered, list):
        return session_date.isoformat() in {str(d) for d in covered}
    return False


# --------------------------------------------------------------------------- #
# Régimen de mercado
# --------------------------------------------------------------------------- #
def classify_regime(
    quotes: Mapping[str, Mapping[str, Any]],
    cfg: config.MarketContextConfig | None = None,
) -> dict[str, Any]:
    """Clasifica el régimen con reglas simples y explícitas.

    * Volatilidad: ``^VIX`` por encima de ``vix_high`` = alta; por encima de
      ``vix_elevated`` = moderada; por debajo = baja.
    * Tono: media de la variación de los futuros (o de los índices si no hay
      futuros) comparada con ``futures_positive_pct`` / ``futures_negative_pct``.
    * ``against_longs`` es True si la volatilidad es alta **o** el tono es
      negativo. El filtro lo usa para penalizar.
    """
    cfg = cfg or config.MARKET_CONTEXT

    vix_quote = quotes.get("^VIX", {})
    vix_level = vix_quote.get("price", NA)
    if is_num(vix_level):
        level = float(vix_level)
        volatility = ("alta" if level >= cfg.vix_high
                      else "moderada" if level >= cfg.vix_elevated else "baja")
    else:
        volatility = NA

    changes = [
        float(quotes[s]["change_pct"]) for s in cfg.futures
        if s in quotes and is_num(quotes[s].get("change_pct"))
    ]
    if not changes:
        changes = [
            float(quotes[s]["change_pct"]) for s in cfg.indices
            if s in quotes and is_num(quotes[s].get("change_pct"))
        ]
    if changes:
        average = sum(changes) / len(changes)
        tone = ("positivo" if average >= cfg.futures_positive_pct
                else "negativo" if average <= cfg.futures_negative_pct else "neutral")
    else:
        average, tone = None, NA

    against = volatility == "alta" or tone == "negativo"
    if volatility == "baja" and tone == "positivo":
        label = "favorable para largos"
    elif against:
        label = "adverso para largos"
    else:
        label = "neutral"

    return {
        "volatility": volatility,
        "vix_level": vix_level,
        "tone": tone,
        "futures_avg_pct": round(average, 3) if average is not None else NA,
        "label": label,
        "against_longs": against,
    }


def build_summary(context: MarketContext,
                  cfg: config.MarketContextConfig | None = None) -> str:
    """Resumen en español, de una o dos líneas, para el informe."""
    cfg = cfg or config.MARKET_CONTEXT
    parts: list[str] = []
    for symbol in cfg.indices:
        quote = context.quotes.get(symbol, {})
        change = quote.get("change_pct", NA)
        parts.append(f"{symbol} {change:+.2f}%" if is_num(change) else f"{symbol} N/A")

    vix = context.regime.get("vix_level", NA)
    vix_text = f"VIX {float(vix):.2f} ({context.regime.get('volatility', NA)})" \
        if is_num(vix) else "VIX N/A"
    tnx = context.quotes.get("^TNX", {}).get("price", NA)
    tnx_text = f"10 años {float(tnx):.2f}%" if is_num(tnx) else "10 años N/A"

    leaders = ", ".join(
        f"{row['symbol']} {float(row['change_pct']):+.2f}%"
        for row in context.sector_leaders[:2]
    ) or "N/A"

    return (
        f"{' | '.join(parts)} | {vix_text} | {tnx_text}. "
        f"Régimen: {context.regime.get('label', NA)} "
        f"(tono {context.regime.get('tone', NA)}). "
        f"Sectores líderes: {leaders}. {context.macro.get('note', MACRO_NOT_LOADED)}."
    )


# --------------------------------------------------------------------------- #
# Ensamblado
# --------------------------------------------------------------------------- #
def assemble_context(
    session_date: date, snapshot_ts: datetime,
    quotes: Mapping[str, SymbolQuote], macro: Mapping[str, Any],
    failures: Mapping[str, str] | None = None,
    cfg: config.MarketContextConfig | None = None,
) -> MarketContext:
    """Monta el ``MarketContext`` a partir de cotizaciones ya calculadas.

    Función pura: no descarga nada, así los tests no necesitan red.
    """
    cfg = cfg or config.MARKET_CONTEXT
    flat = {symbol: quote.to_dict() for symbol, quote in quotes.items()}

    def subset(symbols: Sequence[str]) -> dict[str, Any]:
        return {s: flat[s] for s in symbols if s in flat}

    context = MarketContext(
        session_date=session_date.isoformat(),
        snapshot_ts=snapshot_ts.isoformat(),
        quotes=flat,
        indices=subset(cfg.indices),
        futures=subset(cfg.futures),
        volatility=subset(cfg.volatility),
        rates=subset(cfg.rates),
        sectors=subset(cfg.sectors),
        macro=dict(macro),
        failures=dict(failures or {}),
    )
    ranking = _sorted_by_change(flat, cfg.sectors)
    context.sector_leaders = ranking[:3]
    context.sector_laggards = ranking[-3:][::-1]
    context.regime = classify_regime(flat, cfg)
    context.summary = build_summary(context, cfg)
    return context


def collect_market_context(
    session_date: date, snapshot_ts: datetime,
    cfg: config.MarketContextConfig | None = None,
    market_cfg: config.MarketDataConfig | None = None,
    macro_path: Path | None = None,
    bars_out: dict[str, Any] | None = None,
) -> MarketContext:
    """Descarga los símbolos de referencia y monta el contexto de mercado.

    Reutiliza ``market_data`` (mismos reintentos, mismos lotes, misma zona
    horaria). Un símbolo que falle queda en ``failures`` con ``"N/A"`` en sus
    campos; el resto del contexto se construye igualmente.

    ``bars_out``, si se pasa, se rellena con las barras 5m de hoy de cada
    símbolo, para que quien llama pueda guardar las de SPY/QQQ sin repetir la
    descarga (las necesita el backtest de la Fase 5).
    """
    import market_data  # noqa: PLC0415 - import diferido (evita ciclos y red en tests)

    cfg = cfg or config.MARKET_CONTEXT
    market_cfg = market_cfg or config.MARKET_DATA
    symbols = list(cfg.all_symbols)

    daily = market_data.download_daily(symbols, market_cfg)
    intraday = market_data.download_intraday_today(symbols, session_date, market_cfg)
    snapshots = market_data.build_snapshots(
        symbols, daily.data, intraday.data, snapshot_ts, market_cfg)
    if bars_out is not None:
        bars_out.update(intraday.data)

    quotes: dict[str, SymbolQuote] = {}
    failures: dict[str, str] = dict(daily.failed)
    for symbol in symbols:
        try:
            snapshot = snapshots.get(symbol)
            indicators = compute_daily_indicators(daily.data.get(symbol), session_date)
            quote = build_quote(
                symbol, snapshot.to_dict() if snapshot else None, indicators)
            if quote.source == "missing":
                failures.setdefault(symbol, "sin precio utilizable")
            quotes[symbol] = quote
        except Exception as exc:  # noqa: BLE001 - un símbolo no puede tumbar el contexto
            failures[symbol] = f"{type(exc).__name__}: {exc}"
            logger.error("%s: no se pudo calcular el contexto (%s)", symbol, exc)
            quotes[symbol] = SymbolQuote(symbol=symbol)

    macro = load_macro_events(session_date, macro_path, cfg.macro_lookahead_days)
    context = assemble_context(session_date, snapshot_ts, quotes, macro, failures, cfg)
    logger.info("Contexto de mercado: %s", context.summary)
    return context


__all__ = [
    "MACRO_NOT_LOADED", "SymbolQuote", "MarketContext", "build_quote",
    "load_macro_events", "classify_regime", "build_summary", "assemble_context",
    "collect_market_context",
]
