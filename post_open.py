"""post_open.py — Actualización post-apertura (segunda pasada del día).

Qué hace
--------
30 minutos después de la apertura (10:00 ET, ``config.UPDATE.snapshot_time``)
el agente vuelve a analizar las candidatas del informe de las 09:00, las
compara con lo que dijo entonces y envía un segundo Telegram.

Reparto de trabajo (igual que en el informe de la mañana)
---------------------------------------------------------
* **Python calcula**: precio de apertura y actual, máximo/mínimo desde la
  apertura, VWAP, volumen y RVOL de los primeros 30 minutos, si el precio tocó
  el stop o el objetivo (y cuál de los dos primero), R/R con el precio actual y
  el *estado* de cada selección. Son hechos, no opiniones.
* **Gemini interpreta** (``update_analyzer``): qué ha cambiado y qué hacer. Si
  su respuesta contradice un hecho de Python, manda Python.

Qué NO hace
-----------
* No inventa niveles nuevos: stop, objetivos y zona de entrada son los del
  informe de las 09:00. Solo se recalcula el R/R con el precio actual.
* No busca tickers nuevos fuera de las candidatas de la mañana.
* No sobrescribe el histórico de la mañana (``data/history``): la actualización
  se guarda aparte en ``data/updates/AAAA-MM-DD.json``.

Módulos relacionados: ``update_analyzer.py`` (Gemini) y ``update_report.py``
(texto del Telegram). El flujo completo está en ``run_update_cli`` (se lanza
con ``python main.py --update``).
"""
from __future__ import annotations

import dataclasses
import logging
import time as time_module
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

import config
import history as history_module
import market_data
import technical_analysis as ta
import utils
from gemini_analyzer import compute_rr

logger = logging.getLogger("trading_agent.post_open")

NA = ta.NA
LONG, SHORT = "LONG", "SHORT"

# --- Estados de una selección (los decide Python, son hechos) ---------------
STATUS_STOP = "STOP_TOCADO"
STATUS_TARGET = "OBJETIVO_1_ALCANZADO"
STATUS_EXTENDED = "EXTENDIDA"
STATUS_IN_ZONE = "EN_ZONA"
STATUS_AGAINST = "EN_CONTRA"
STATUS_NO_DATA = "SIN_DATOS"

STATUS_LABELS = {
    STATUS_STOP: "Stop tocado",
    STATUS_TARGET: "Objetivo 1 alcanzado",
    STATUS_EXTENDED: "Extendida (por encima del máximo de entrada válido)",
    STATUS_IN_ZONE: "En zona de entrada",
    STATUS_AGAINST: "En contra (fuera de la zona, sin tocar el stop)",
    STATUS_NO_DATA: "Sin datos de la sesión regular",
}

# --- Veredictos (comparación con el informe de las 09:00) -------------------
V_CONFIRMED = "CONFIRMED"
V_IMPROVED = "IMPROVED"
V_WEAKENED = "WEAKENED"
V_INVALIDATED = "INVALIDATED"
V_EXTENDED = "EXTENDED"
V_TARGET = "TARGET_REACHED"
V_NO_DATA = "NO_DATA"

# Estados que Python impone: Gemini no puede cambiarlos.
FORCED_VERDICTS = {
    STATUS_STOP: (V_INVALIDATED, "NO_TRADE"),
    STATUS_TARGET: (V_TARGET, "NO_TRADE"),
    STATUS_EXTENDED: (V_EXTENDED, "WAIT"),
}


# --------------------------------------------------------------------------- #
# Utilidades pequeñas
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    """``float`` si ``value`` es un número finito; ``None`` en otro caso."""
    return float(value) if ta.is_num(value) else None


def _p(value: Any) -> str:
    """Precio con 2 decimales para los textos («N/A» si no es un número)."""
    number = _num(value)
    return f"{number:.2f}" if number is not None else "N/A"


def _hhmm(value: Any) -> str:
    """HH:MM de un datetime o de un texto ISO ('2026-09-25T08:45:00-04:00')."""
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    text = str(value or "")
    return text[11:16] if len(text) >= 16 else "N/A"


def _is_short(direction: Any) -> bool:
    return str(direction or LONG).upper() == SHORT


def wait_until(target: time, session_date: date,
               sleep: Callable[[float], Any] = time_module.sleep) -> None:
    """Espera hasta ``target`` (hora de Nueva York) si es hoy y aún no ha llegado."""
    now = utils.now_ny()
    if now.date() != session_date or now.time() >= target:
        return
    objetivo = datetime.combine(session_date, target, tzinfo=utils.NY_TZ)
    seconds = (objetivo - now).total_seconds()
    logger.info("Esperando %.0f s hasta las %s ET.", seconds, utils.format_hhmm(target))
    print(f"Esperando hasta las {utils.format_hhmm(target)} ET ({seconds:.0f} s)...")
    while seconds > 0:
        sleep(min(seconds, 15.0))
        seconds = (objetivo - utils.now_ny()).total_seconds()


def resolve_update_ts(
    session_date: date, now: datetime | None = None,
    update_cfg: config.UpdateConfig | None = None,
    market_cfg: config.MarketDataConfig | None = None,
    schedule: config.ScheduleConfig | None = None,
) -> datetime:
    """Instante de los datos de la actualización, en hora de Nueva York.

    Por defecto es ``UPDATE.snapshot_time``. Si ``now`` es de ese día y ya pasó
    esa hora (el cron se retrasó), se usa ``now`` redondeado hacia abajo al
    tamaño de barra (solo barras completas), sin pasar del cierre.
    """
    update_cfg = update_cfg or config.UPDATE
    market_cfg = market_cfg or config.MARKET_DATA
    schedule = schedule or config.SCHEDULE
    base = datetime.combine(session_date, update_cfg.snapshot_time, tzinfo=utils.NY_TZ)
    if now is None:
        return base
    now_ny = utils.to_ny(now) if now.tzinfo else now.replace(tzinfo=utils.NY_TZ)
    if now_ny.date() != session_date:
        return base
    step = market_data.interval_minutes(market_cfg.intraday_interval)
    floored = now_ny.replace(minute=(now_ny.minute // step) * step, second=0, microsecond=0)
    close = datetime.combine(session_date, schedule.regular_close, tzinfo=utils.NY_TZ)
    return min(max(base, floored), close)


# --------------------------------------------------------------------------- #
# Lo que dijo el informe de la mañana
# --------------------------------------------------------------------------- #
def load_morning_state(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extrae del histórico de la mañana lo que necesita la actualización.

    ``record`` es el JSON de ``data/history/AAAA-MM-DD.json``. Nunca lanza: los
    campos ausentes quedan vacíos.
    """
    gemini = record.get("gemini_validated") or {}
    stamps = record.get("timestamps") or {}
    picks = [p for p in (record.get("final_selection") or [])
             if isinstance(p, dict) and p.get("ticker")]
    candidates = [c for c in (record.get("candidates") or [])
                  if isinstance(c, dict) and c.get("ticker")]
    context = (record.get("market_data") or {}).get("context") or {}
    return {
        "picks": picks,
        "candidates": candidates,
        "context": context if isinstance(context, dict) else {},
        "market_view": str(gemini.get("market_view") or ""),
        "gemini_available": bool(gemini.get("available")),
        "no_trade": bool(record.get("no_trade")),
        "no_trade_reason": str(record.get("no_trade_reason") or ""),
        "snapshot_ts": stamps.get("snapshot"),
        "sent_at": stamps.get("sent"),
        "sent": bool(record.get("sent")),
        "universe_size": (record.get("universe") or {}).get("count", 0),
    }


def _candidate_by_ticker(morning: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["ticker"]: c for c in morning.get("candidates", [])}


def select_review_tickers(morning: Mapping[str, Any],
                          max_tickers: int | None = None) -> tuple[list[str], list[str]]:
    """Tickers a reanalizar: primero las selecciones y luego el resto de candidatas.

    Devuelve ``(picks, others)``: ``picks`` son las seleccionadas por el informe
    de las 09:00 y ``others`` las demás candidatas, en su orden original.
    """
    max_tickers = max_tickers or config.UPDATE.max_tickers
    picks = list(dict.fromkeys(p["ticker"] for p in morning.get("picks", [])))[:max_tickers]
    others: list[str] = []
    for candidate in morning.get("candidates", []):
        ticker = candidate["ticker"]
        if ticker not in picks and ticker not in others:
            others.append(ticker)
    room = max(0, max_tickers - len(picks))
    return picks, others[:room]


def morning_reference(morning: Mapping[str, Any], ticker: str) -> dict[str, Any]:
    """Cierre anterior, ATR y demás datos de la mañana de un ticker (sin descargar nada)."""
    candidate = _candidate_by_ticker(morning).get(ticker, {})
    analysis = history_module.analysis_record(candidate)
    premarket = analysis.get("premarket") if isinstance(analysis.get("premarket"), dict) else {}
    indicators = analysis.get("indicators") if isinstance(analysis.get("indicators"), dict) else {}
    rvol = analysis.get("rvol")
    return {
        "prev_close": _num(premarket.get("prev_close")),
        "premarket_price": _num(premarket.get("premarket_price")),
        "gap_pct": _num(analysis.get("gap_pct")),
        "atr14": _num(indicators.get("atr14")),
        "rvol_premarket": _num(rvol.get("rvol") if isinstance(rvol, dict) else rvol),
        "score": history_module.candidate_score(candidate),
        "rank": candidate.get("rank"),
        "news": candidate.get("news") if isinstance(candidate.get("news"), dict) else {},
    }


# --------------------------------------------------------------------------- #
# Métricas en vivo (sesión regular hasta la hora de la actualización)
# --------------------------------------------------------------------------- #
def regular_bars(today: pd.DataFrame | None, snapshot_ts: datetime,
                 schedule: config.ScheduleConfig | None = None) -> pd.DataFrame:
    """Barras de la sesión regular con inicio en [09:30, hora de la actualización)."""
    schedule = schedule or config.SCHEDULE
    bars = ta.window_bars(today, snapshot_ts.date(), schedule.regular_open,
                          snapshot_ts.time())
    if len(bars):
        bars = bars.dropna(subset=["Close"])
    return bars


def _empty_live(ticker: str, reason: str) -> dict[str, Any]:
    return {
        "ticker": ticker, "quality": "missing", "quality_note": reason, "n_bars": 0,
        "open_price": NA, "open_bar_time": NA, "price_now": NA,
        "high_since_open": NA, "low_since_open": NA, "volume_since_open": NA,
        "vwap": NA, "price_vs_vwap": NA, "change_vs_prev_close_pct": NA,
        "change_vs_open_pct": NA, "open_rvol": NA, "open_rvol_note": reason,
        "last_bar_time": NA, "last_bar_age_min": NA,
    }


def compute_live_metrics(
    ticker: str, today: pd.DataFrame | None, hist: pd.DataFrame | None,
    prev_close: float | None, snapshot_ts: datetime,
    market_cfg: config.MarketDataConfig | None = None,
    schedule: config.ScheduleConfig | None = None,
    rvol_cfg: config.RvolConfig | None = None,
) -> dict[str, Any]:
    """Métricas de un ticker desde la apertura hasta ``snapshot_ts``.

    Sin barras de la sesión regular devuelve todo ``"N/A"`` y calidad
    ``missing`` (no se estima nada).
    """
    market_cfg = market_cfg or config.MARKET_DATA
    schedule = schedule or config.SCHEDULE
    bars = regular_bars(today, snapshot_ts, schedule)
    if len(bars) == 0:
        return _empty_live(ticker, "sin barras de la sesión regular")

    last_time = bars.index[-1]
    age = round((snapshot_ts - last_time.to_pydatetime()).total_seconds() / 60.0, 1)
    quality, note = market_data.classify_premarket_quality(len(bars), age, market_cfg)
    note = note.replace("pre-market", "de la sesión regular")

    first = bars.iloc[0]
    open_price = _num(first.get("Open")) if "Open" in bars.columns else None
    if open_price is None:
        open_price = _num(first.get("Close"))
    price_now = _num(bars["Close"].iloc[-1])
    high = _num(bars["High"].max()) if "High" in bars.columns else None
    low = _num(bars["Low"].min()) if "Low" in bars.columns else None

    volume = bars["Volume"].astype(float).fillna(0.0) if "Volume" in bars.columns else None
    volume_total = float(volume.sum()) if volume is not None else None
    vwap = None
    if volume is not None and volume_total and volume_total > 0 \
            and {"High", "Low", "Close"} <= set(bars.columns):
        typical = (bars["High"].astype(float) + bars["Low"].astype(float)
                   + bars["Close"].astype(float)) / 3.0
        vwap = float((typical * volume).sum() / volume_total)

    price_vs_vwap = NA
    if price_now is not None and vwap is not None:
        price_vs_vwap = "encima" if price_now >= vwap else "debajo"

    # RVOL de los primeros minutos: mismo cálculo que el RVOL pre-market, pero
    # con la ventana [09:30, hora de la actualización) en vez de [04:00, 09:30).
    open_schedule = dataclasses.replace(
        schedule, premarket_start=schedule.regular_open, regular_open=time(23, 59))
    rvol = ta.compute_rvol(hist, today, snapshot_ts, rvol_cfg, open_schedule).to_dict()

    return {
        "ticker": ticker,
        "quality": quality,
        "quality_note": note,
        "n_bars": int(len(bars)),
        "open_price": ta._f(open_price),
        "open_bar_time": first_bar_time(bars),
        "price_now": ta._f(price_now),
        "high_since_open": ta._f(high),
        "low_since_open": ta._f(low),
        "volume_since_open": ta._f(volume_total, 0),
        "vwap": ta._f(vwap),
        "price_vs_vwap": price_vs_vwap,
        "change_vs_prev_close_pct": ta.gap_pct(price_now, prev_close),
        "change_vs_open_pct": ta.gap_pct(price_now, open_price),
        "open_rvol": rvol.get("rvol", NA),
        "open_rvol_note": str(rvol.get("reason") or "").replace(
            "pre-market", "de la apertura"),
        "last_bar_time": last_time.isoformat(),
        "last_bar_age_min": age,
    }


def first_bar_time(bars: pd.DataFrame) -> str:
    """Hora de inicio de la primera barra de la sesión regular (normalmente 09:30)."""
    return bars.index[0].isoformat() if len(bars) else NA


# --------------------------------------------------------------------------- #
# Seguimiento de stop y objetivos con las barras (¿qué ocurrió primero?)
# --------------------------------------------------------------------------- #
def track_events(bars: pd.DataFrame, direction: str, stop: float | None,
                 target_1: float | None, target_2: float | None) -> dict[str, Any]:
    """Recorre las barras en orden y anota cuándo se tocó el stop y los objetivos.

    Con barras de 5 minutos no se sabe qué ocurrió primero dentro de una misma
    barra: si en una barra se toca el stop y el objetivo 1 a la vez, se
    considera que se tocó **primero el stop** (lectura conservadora) y se marca
    ``ambiguous_bar``.
    """
    short = _is_short(direction)
    events: dict[str, Any] = {
        "stop_hit": False, "stop_hit_time": None,
        "target_1_hit": False, "target_1_hit_time": None,
        "target_2_hit": False, "target_2_hit_time": None,
        "first_event": None, "ambiguous_bar": False,
    }
    if bars is None or len(bars) == 0 or not {"High", "Low"} <= set(bars.columns):
        return events

    for stamp, row in bars.iterrows():
        high, low = _num(row["High"]), _num(row["Low"])
        if high is None or low is None:
            continue
        hit_stop = stop is not None and ((high >= stop) if short else (low <= stop))
        hit_t1 = target_1 is not None and ((low <= target_1) if short else (high >= target_1))
        hit_t2 = target_2 is not None and ((low <= target_2) if short else (high >= target_2))
        when = stamp.isoformat()

        if hit_stop and not events["stop_hit"]:
            events["stop_hit"], events["stop_hit_time"] = True, when
            if events["first_event"] is None:
                events["first_event"] = "stop"
                events["ambiguous_bar"] = bool(hit_t1 and not events["target_1_hit"])
        if hit_t1 and not events["target_1_hit"]:
            events["target_1_hit"], events["target_1_hit_time"] = True, when
            if events["first_event"] is None:
                events["first_event"] = "target_1"
        if hit_t2 and not events["target_2_hit"]:
            events["target_2_hit"], events["target_2_hit_time"] = True, when
    return events


# --------------------------------------------------------------------------- #
# Estado de una selección de la mañana (hechos)
# --------------------------------------------------------------------------- #
def assess_pick(
    pick: Mapping[str, Any], live: Mapping[str, Any], bars: pd.DataFrame | None,
    min_rr: float | None = None,
) -> dict[str, Any]:
    """Compara una selección de las 09:00 con lo que ha pasado desde la apertura.

    Devuelve un diccionario con el estado (``status``), el R/R con el precio
    actual, la regla de apertura y los eventos de stop/objetivo. Todo lo
    calcula Python a partir de los niveles del propio informe de la mañana.
    """
    min_rr = config.TRADE.min_rr if min_rr is None else min_rr
    direction = str(pick.get("direction") or LONG).upper()
    short = _is_short(direction)
    sign = -1.0 if short else 1.0

    stop = _num(pick.get("stop"))
    target_1, target_2 = _num(pick.get("target_1")), _num(pick.get("target_2"))
    entry_low, entry_high = _num(pick.get("entry_zone_low")), _num(pick.get("entry_zone_high"))
    max_valid = _num(pick.get("max_valid_entry"))
    price = _num(live.get("price_now"))
    open_price = _num(live.get("open_price"))

    out: dict[str, Any] = {
        "direction": direction, "status": STATUS_NO_DATA, "rr_now": None,
        "rr_min": min_rr, "open_rule": NA, "events": {}, "note": "",
        "dist_to_stop_pct": None, "dist_to_target_1_pct": None,
        "vwap_favorable": None,
    }

    if price is None or live.get("quality") == "missing":
        out["note"] = "sin datos de la sesión regular"
        return out
    if all(v is None for v in (stop, target_1, entry_low, entry_high, max_valid)):
        out["note"] = "el informe de la mañana no trae niveles para este valor"
        return out

    events = track_events(bars if bars is not None else pd.DataFrame(),
                          direction, stop, target_1, target_2)
    out["events"] = events

    # Regla de apertura del informe: ¿el precio de apertura respetaba el máximo
    # (o mínimo, en SHORT) de entrada válido?
    if open_price is not None and max_valid is not None:
        respected = (open_price >= max_valid) if short else (open_price <= max_valid)
        out["open_rule"] = "CUMPLE" if respected else "NO_ENTRAR"

    if stop is not None and price:
        out["dist_to_stop_pct"] = round(sign * (price - stop) / price * 100.0, 2)
    if target_1 is not None and price:
        out["dist_to_target_1_pct"] = round(sign * (target_1 - price) / price * 100.0, 2)
    if stop is not None and target_1 is not None:
        out["rr_now"] = compute_rr(price, stop, target_1, direction)

    vwap = _num(live.get("vwap"))
    if vwap is not None:
        out["vwap_favorable"] = (price <= vwap) if short else (price >= vwap)

    # Estado: los eventos de stop/objetivo mandan; después la posición del precio.
    if events["first_event"] == "stop":
        out["status"] = STATUS_STOP
        if events["ambiguous_bar"]:
            out["note"] = "stop y objetivo 1 en la misma barra de 5 min: se cuenta el stop"
        return out
    if events["first_event"] == "target_1":
        out["status"] = STATUS_TARGET
        if events["stop_hit"]:
            out["note"] = "después de alcanzar el objetivo 1, el precio llegó también al stop"
        return out

    if short:
        upper = -entry_low if (max_valid is None and entry_low is not None) else (
            -max_valid if max_valid is not None else None)
        lower = -entry_high if entry_high is not None else None
    else:
        upper = max_valid if max_valid is not None else entry_high
        lower = entry_low
    if upper is not None and lower is not None and upper < lower and not out["note"]:
        out["note"] = ("la zona de entrada del informe de las 09:00 ya estaba más allá del "
                       "límite de entrada válido: solo era ejecutable con un retroceso")
    p = sign * price
    if upper is not None and p > upper:
        out["status"] = STATUS_EXTENDED
    elif lower is not None and p < lower:
        out["status"] = STATUS_AGAINST
    else:
        out["status"] = STATUS_IN_ZONE
    return out


def python_verdict(assessment: Mapping[str, Any], morning_decision: str,
                   min_rr: float | None = None) -> tuple[str, str]:
    """Veredicto y decisión que salen solo de los hechos de Python.

    Se usa (a) como respaldo cuando Gemini no está disponible y (b) para fijar
    los estados que Gemini no puede cambiar (stop tocado, objetivo alcanzado,
    extendida).
    """
    min_rr = config.TRADE.min_rr if min_rr is None else min_rr
    status = assessment.get("status")
    if status in FORCED_VERDICTS:
        return FORCED_VERDICTS[status]
    if status == STATUS_NO_DATA:
        return V_NO_DATA, morning_decision
    if status == STATUS_AGAINST:
        return V_WEAKENED, "WAIT"
    rr_now = _num(assessment.get("rr_now"))
    if assessment.get("vwap_favorable") is False or rr_now is None or rr_now < min_rr:
        return V_WEAKENED, "WAIT"
    return V_CONFIRMED, morning_decision


def default_action(pick: Mapping[str, Any], assessment: Mapping[str, Any],
                   live: Mapping[str, Any]) -> str:
    """Texto de «qué hacer» generado por Python a partir del estado (sin IA)."""
    status = assessment.get("status")
    short = _is_short(assessment.get("direction"))
    events = assessment.get("events") or {}
    stop, target_1 = _p(pick.get("stop")), _p(pick.get("target_1"))
    price = _p(live.get("price_now"))
    lo, hi = _p(pick.get("entry_zone_low")), _p(pick.get("entry_zone_high"))
    max_valid = _p(pick.get("max_valid_entry"))
    above, below = ("por debajo", "por encima") if short else ("por encima", "por debajo")

    if status == STATUS_STOP:
        return (f"El precio tocó el stop ({stop}) a las {_hhmm(events.get('stop_hit_time'))}. "
                "Operación invalidada: no entrar.")
    if status == STATUS_TARGET:
        return (f"El precio alcanzó el objetivo 1 ({target_1}) a las "
                f"{_hhmm(events.get('target_1_hit_time'))}. No perseguir la entrada; "
                "si ya estás dentro, valora asegurar ganancias.")
    if status == STATUS_EXTENDED:
        back = "o más" if short else "o menos"
        return (f"El precio ({price}) ya está {above} del límite de entrada válido "
                f"({max_valid}). No perseguir: esperar un retroceso hasta {max_valid} "
                f"{back} (donde el R/R vuelve a compensar) o descartar.")
    if status == STATUS_AGAINST:
        return (f"El precio ({price}) está {below} de la zona de entrada ({lo}–{hi}) "
                f"sin haber tocado el stop ({stop}). Esperar a que recupere la zona; "
                "no entrar mientras siga así.")
    if status == STATUS_NO_DATA:
        return "Sin datos de la sesión regular para este valor: no se puede actualizar."
    if assessment.get("vwap_favorable") is False:
        return "Sigue en zona pero con el precio en el lado desfavorable del VWAP: esperar confirmación."
    rr_now, min_rr = assessment.get("rr_now"), assessment.get("rr_min")
    if rr_now is None or (min_rr is not None and rr_now < min_rr):
        return "Sigue en zona pero el R/R con el precio actual ya no alcanza el mínimo: esperar."
    return "Sigue en zona y las reglas del informe de las 09:00 siguen siendo válidas."


# --------------------------------------------------------------------------- #
# Mercado: mañana vs. ahora
# --------------------------------------------------------------------------- #
def compare_market(
    morning_context: Mapping[str, Any], bars: Mapping[str, pd.DataFrame],
    snapshot_ts: datetime, symbols: Sequence[str] | None = None,
    schedule: config.ScheduleConfig | None = None,
) -> list[dict[str, Any]]:
    """Variación de SPY, QQQ, VIX... a las 08:45 (informe) y ahora.

    El cierre anterior sale de las cotizaciones guardadas por el informe de la
    mañana, así que no hace falta descargar datos diarios.
    """
    symbols = symbols or config.UPDATE.benchmarks
    quotes = morning_context.get("quotes") if isinstance(morning_context, Mapping) else {}
    quotes = quotes if isinstance(quotes, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        quote = quotes.get(symbol) if isinstance(quotes.get(symbol), Mapping) else {}
        prev_close = _num(quote.get("prev_close"))
        frame = regular_bars(bars.get(symbol), snapshot_ts, schedule)
        price_now = _num(frame["Close"].iloc[-1]) if len(frame) else None
        rows.append({
            "symbol": symbol,
            "prev_close": prev_close if prev_close is not None else NA,
            "morning_price": quote.get("price", NA),
            "morning_change_pct": quote.get("change_pct", NA),
            "now_price": ta._f(price_now),
            "now_change_pct": ta.gap_pct(price_now, prev_close),
        })
    return rows


# --------------------------------------------------------------------------- #
# Noticias nuevas desde el informe
# --------------------------------------------------------------------------- #
def _norm_headline(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def new_headlines(
    morning_news: Mapping[str, Any] | None, fresh_news: Mapping[str, Any] | None,
    since: datetime | None, limit: int = 3, until: datetime | None = None,
) -> list[dict[str, Any]]:
    """Titulares que no estaban en el informe de la mañana y salieron después de él.

    ``since`` es la hora del informe y ``until`` el límite superior (hora de los
    datos de la actualización): sin él, al repetir una fecha pasada con
    ``--date`` se colarían titulares de días posteriores.

    ``morning_news`` y ``fresh_news`` son ``TickerNews.to_dict()``. Si un
    titular no trae fecha interpretable se descarta: sin fecha no se puede
    afirmar que sea posterior al informe (regla global 2: no inventar).
    """
    seen = {_norm_headline(i.get("headline"))
            for i in (morning_news or {}).get("items", []) if isinstance(i, Mapping)}
    fresh: list[dict[str, Any]] = []
    for item in (fresh_news or {}).get("items", []):
        if not isinstance(item, Mapping):
            continue
        headline = item.get("headline")
        if not headline or _norm_headline(headline) in seen:
            continue
        try:
            published = datetime.fromisoformat(str(item.get("published_at")))
        except ValueError:
            continue
        if since is not None and published < since:
            continue
        if until is not None and published > until:
            continue
        fresh.append({
            "headline": str(headline), "publisher": item.get("publisher", ""),
            "published_at": published.isoformat(), "url": item.get("url", ""),
            "category": item.get("category_label") or item.get("category", ""),
            "is_hard_catalyst": bool(item.get("is_hard_catalyst")),
        })
    fresh.sort(key=lambda i: i["published_at"], reverse=True)
    return fresh[:limit]


def refresh_news(tickers: Sequence[str], env: Any, reference: datetime) -> dict[str, dict[str, Any]]:
    """Vuelve a pedir noticias (solo Yahoo) de ``tickers``; ``{}`` si falla.

    Se quita la clave de Alpha Vantage a propósito: su cuota gratuita es muy
    baja y ya se usó en el informe de las 09:00.
    """
    if not tickers or not config.UPDATE.refresh_news:
        return {}
    try:
        import news as news_module  # noqa: PLC0415 - import diferido
        yahoo_only = dataclasses.replace(env, alphavantage_api_key="") \
            if dataclasses.is_dataclass(env) else env
        providers = news_module.build_providers(yahoo_only)
        fresh = news_module.gather_news(list(tickers), providers, reference=reference)
        return {t: item.to_dict() for t, item in fresh.items()}
    except Exception as exc:  # noqa: BLE001 - las noticias nunca deben tumbar la actualización
        logger.warning("No se pudieron refrescar las noticias: %s", exc)
        return {}
