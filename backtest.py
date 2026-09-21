"""backtest.py — Validación posterior sencilla: ¿qué hizo cada pick durante la sesión?

Uso: ``python main.py --backtest-date YYYY-MM-DD`` (``main.py`` llama a
``run_backtest_command``).

Qué hace
--------
1. Lee ``data/history/YYYY-MM-DD.json`` y toma los picks de ``final_selection``.
2. Obtiene las barras de 5 minutos de la **sesión regular** (09:30-16:00 ET):
   primero las guardadas en ``data/intraday/``; si no existen (o solo cubren el
   pre-market), intenta yfinance, que sirve barras de 5 min de unos 60 días, y lo indica
   en ``data_source`` y en ``notes``. Fuera de ese rango: ``"N/A"``.
3. Para cada pick calcula apertura, máximo, mínimo y cierre; si la apertura estaba
   dentro del rango de entrada y dentro del límite de entrada válido (por debajo del
   precio máximo en largos; por encima del mínimo en cortos); excursión máxima
   favorable (MFE) y adversa (MAE) medidas desde la apertura; y si tocó objetivo 1,
   objetivo 2 y stop.
4. Guarda ``data/backtests/YYYY-MM-DD.json`` y actualiza ``data/backtest_summary.csv``
   (reemplaza las filas de esa fecha; nunca duplica).

Reglas de simulación (deliberadamente simples)
----------------------------------------------
* Se supone entrada a precio de apertura y se recorre la sesión completa.
* LONG: toca objetivo si ``High >= objetivo``; toca stop si ``Low <= stop``. SHORT al revés.
* Si stop y objetivo 1 caen en la **misma barra** no se sabe qué ocurrió antes: se marca
  ``ambiguous = True`` y se trata de forma conservadora, como stop.
* ``outcome``: ``objetivo_2``, ``objetivo_1``, ``stop`` o ``sin_resolucion`` (ni objetivo
  ni stop). Los toques sueltos (``touched_*``) se informan aunque ocurrieran después
  del primer evento.
* ``entry_valid`` sigue la regla del informe: apertura dentro del límite de entrada
  válido (o, si no hay límite, dentro del rango de entrada). Los resultados se calculan
  igualmente si no era válida, para poder analizarlo.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import config
import history
import market_data as md
from technical_analysis import NA, is_num, window_bars
from utils import now_ny, write_json_atomic

logger = logging.getLogger("trading_agent.backtest")

# Límite del proveedor (yfinance): barras de 5 min de aproximadamente los últimos 60 días.
YF_INTRADAY_PERIOD = "60d"
YF_MAX_AGE_DAYS = 59

SOURCE_SAVED = "barras guardadas (data/intraday)"
SOURCE_YFINANCE = "yfinance 5m"

BACKTEST_COLUMNS = [
    "date", "ticker", "direction", "decision", "data_source", "entry_valid",
    "open_in_entry_range", "open_within_entry_limit", "open", "high", "low", "close",
    "mfe_pct", "mae_pct", "touched_target_1", "touched_target_2", "touched_stop", "outcome",
    "ambiguous", "rr_planned", "entry_low", "entry_high", "stop", "target_1", "target_2",
    "max_entry_price", "notes",
]


# --------------------------------------------------------------------------- #
# Barras de la sesión
# --------------------------------------------------------------------------- #
def _regular_session(df: pd.DataFrame | None, target: date,
                     schedule: config.ScheduleConfig | None = None) -> pd.DataFrame:
    schedule = schedule or config.SCHEDULE
    bars = window_bars(df, target, schedule.regular_open, schedule.regular_close)
    return bars.dropna(subset=["Open", "High", "Low", "Close"]).sort_index() if len(bars) else bars


def load_session_bars(
    ticker: str, target: date, intraday_dir: Path | None = None,
    today: date | None = None, download: bool = True,
) -> tuple[pd.DataFrame, str, str]:
    """Barras 5m de la sesión regular de ``target``: ``(barras, fuente, nota)``.

    ``fuente`` es ``SOURCE_SAVED``, ``SOURCE_YFINANCE`` o ``"N/A"`` (barras vacías).
    """
    saved = md.load_intraday_bars(ticker, target, intraday_dir)
    session = _regular_session(saved, target)
    if len(session):
        return session, SOURCE_SAVED, ""

    reason = ("las barras guardadas solo cubren el pre-market" if len(saved)
              else "no hay barras guardadas")
    if not download:
        return pd.DataFrame(), NA, f"{reason} y la descarga está desactivada"
    age_days = ((today or now_ny().date()) - target).days
    if age_days < 0:
        return pd.DataFrame(), NA, "la fecha es futura: no hay sesión que evaluar"
    if age_days > YF_MAX_AGE_DAYS:
        return (pd.DataFrame(), NA,
                f"{reason} y la fecha queda fuera del rango de yfinance (~60 días)")

    result = md.download_frames(
        [ticker], period=YF_INTRADAY_PERIOD, interval=config.MARKET_DATA.intraday_interval,
        prepost=False, label="intradía de backtest")
    session = _regular_session(result.data.get(ticker), target)
    if len(session):
        return session, SOURCE_YFINANCE, f"{reason}; se usaron barras de yfinance"
    return pd.DataFrame(), NA, f"{reason} y yfinance no devolvió barras de esa fecha"


# --------------------------------------------------------------------------- #
# Simulación de un pick
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    return float(value) if is_num(value) else None


def _first_true(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def _round(value: float | None, digits: int) -> float | str:
    return round(float(value), digits) if value is not None else NA


def simulate_pick(pick: Mapping[str, Any], bars: pd.DataFrame) -> dict[str, Any]:
    """Evalúa un pick contra las barras de la sesión (ver reglas en el docstring del módulo)."""
    direction = str(pick.get("direction") or "").upper()
    entry_low, entry_high = _num(pick.get("entry_zone_low")), _num(pick.get("entry_zone_high"))
    stop, t1, t2 = _num(pick.get("stop")), _num(pick.get("target_1")), _num(pick.get("target_2"))
    limit = _num(history.pick_field(pick, "max_entry_price"))
    rr = _num(history.pick_field(pick, "rr"))
    notes: list[str] = []

    result: dict[str, Any] = {
        "ticker": pick.get("ticker"), "direction": direction or NA,
        "decision": pick.get("decision", NA),
        "rr_planned": _round(rr, 2), "entry_low": _round(entry_low, 4),
        "entry_high": _round(entry_high, 4), "stop": _round(stop, 4),
        "target_1": _round(t1, 4), "target_2": _round(t2, 4),
        "max_entry_price": _round(limit, 4),
        "open": NA, "high": NA, "low": NA, "close": NA,
        "open_in_entry_range": NA, "open_within_entry_limit": NA, "entry_valid": NA,
        "mfe_pct": NA, "mae_pct": NA,
        "touched_target_1": NA, "touched_target_2": NA, "touched_stop": NA,
        "outcome": NA, "ambiguous": False, "notes": notes,
    }
    if direction not in ("LONG", "SHORT"):
        notes.append("dirección del pick no válida")
        return result
    if bars is None or len(bars) == 0:
        notes.append("sin barras de la sesión regular")
        return result

    bars = bars.sort_index()
    highs, lows = bars["High"].to_numpy(float), bars["Low"].to_numpy(float)
    open_price, close_price = float(bars["Open"].iloc[0]), float(bars["Close"].iloc[-1])
    high, low = float(highs.max()), float(lows.min())
    result.update(open=_round(open_price, 4), high=_round(high, 4), low=_round(low, 4),
                  close=_round(close_price, 4))

    is_long = direction == "LONG"
    if entry_low is not None and entry_high is not None:
        result["open_in_entry_range"] = bool(entry_low <= open_price <= entry_high)
    if limit is not None:
        result["open_within_entry_limit"] = bool(open_price <= limit if is_long
                                                 else open_price >= limit)
    within_limit = result["open_within_entry_limit"]
    result["entry_valid"] = (within_limit if isinstance(within_limit, bool)
                             else result["open_in_entry_range"])
    if result["entry_valid"] is False:
        notes.append("la apertura incumplía la regla de entrada (NO ENTRAR); resultados "
                     "calculados solo a efectos de análisis")

    if open_price > 0:
        favorable = (high - open_price) if is_long else (open_price - low)
        adverse = (low - open_price) if is_long else (open_price - high)
        result["mfe_pct"] = round(favorable / open_price * 100, 3)
        result["mae_pct"] = round(adverse / open_price * 100, 3)

    def hit(level: float | None, is_stop: bool) -> np.ndarray:
        if level is None:
            return np.zeros(len(bars), dtype=bool)
        below = (is_long and is_stop) or (not is_long and not is_stop)
        return (lows <= level) if below else (highs >= level)

    stop_i = _first_true(hit(stop, True))
    t1_i = _first_true(hit(t1, False))
    t2_i = _first_true(hit(t2, False))
    for name, level in (("stop", stop), ("objetivo 1", t1), ("objetivo 2", t2)):
        if level is None:
            notes.append(f"{name} N/A: no se evalúa")
    result["touched_stop"] = stop is not None and stop_i is not None
    result["touched_target_1"] = t1 is not None and t1_i is not None
    result["touched_target_2"] = t2 is not None and t2_i is not None

    ambiguous = False
    if stop_i is not None and (t1_i is None or stop_i <= t1_i):
        outcome = "stop"
        ambiguous = t1_i is not None and stop_i == t1_i
    elif t1_i is not None:
        outcome = "objetivo_1"
        if t2_i is not None and (stop_i is None or t2_i < stop_i):
            outcome = "objetivo_2"
        ambiguous = t2_i is not None and stop_i is not None and t2_i == stop_i
    else:
        outcome = "sin_resolucion"
    if ambiguous:
        notes.append("stop y objetivo en la misma barra: ambiguo, tratado como stop"
                     if outcome == "stop" else
                     "objetivo 2 y stop en la misma barra: ambiguo, no se cuenta el objetivo 2")
    result["outcome"], result["ambiguous"] = outcome, ambiguous
    return result


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
def _summary(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    def count(key: str, value: Any) -> int:
        return sum(1 for r in results if r.get(key) == value)

    return {
        "picks": len(results),
        "entry_valid": count("entry_valid", True),
        "touched_target_1": count("touched_target_1", True),
        "touched_target_2": count("touched_target_2", True),
        "touched_stop": count("touched_stop", True),
        "ambiguous": count("ambiguous", True),
        "outcomes": {o: count("outcome", o)
                     for o in ("objetivo_2", "objetivo_1", "stop", "sin_resolucion", NA)
                     if count("outcome", o)},
    }


def run_backtest(
    target: date, history_dir: Path | None = None, intraday_dir: Path | None = None,
    backtests_dir: Path | None = None, summary_csv: Path | None = None,
    today: date | None = None, download: bool = True,
) -> dict[str, Any]:
    """Ejecuta el backtest de una fecha y guarda JSON + CSV. Devuelve el resultado.

    Si no existe el histórico devuelve ``{"ok": False, "error": ...}`` sin escribir nada.
    Cada pick se procesa de forma aislada.
    """
    data = history.load_history(target, history_dir)
    if data is None:
        error = f"No hay histórico para {target.isoformat()}: no se puede hacer el backtest"
        logger.error(error)
        return {"ok": False, "date": target.isoformat(), "error": error, "results": []}

    picks = [p for p in (data.get("final_selection") or []) if isinstance(p, Mapping)]
    results: list[dict[str, Any]] = []
    for pick in picks:
        ticker = str(pick.get("ticker"))
        try:
            bars, source, note = load_session_bars(ticker, target, intraday_dir, today, download)
            outcome = simulate_pick(pick, bars)
            outcome["data_source"] = source
            if note:
                outcome["notes"].insert(0, note)
            results.append(outcome)
        except Exception as exc:  # noqa: BLE001 - aislamiento por ticker
            logger.error("%s: no se pudo hacer el backtest (%s)", ticker, exc)
            results.append({"ticker": ticker, "direction": pick.get("direction", NA),
                            "decision": pick.get("decision", NA), "data_source": NA,
                            "outcome": NA, "ambiguous": False,
                            "notes": [f"error: {type(exc).__name__}: {exc}"]})

    payload = {
        "ok": True, "date": target.isoformat(), "generated_at": now_ny().isoformat(),
        "history_sent": bool(data.get("sent")), "summary": _summary(results),
        "results": results,
    }
    out_dir = backtests_dir or config.PATHS.backtests_dir
    write_json_atomic(out_dir / f"{target.isoformat()}.json", payload)

    rows = [{**r, "date": target.isoformat(),
             "notes": " | ".join(str(n) for n in r.get("notes", []))} for r in results]
    history.upsert_csv_rows(summary_csv or config.PATHS.backtest_summary_csv, rows,
                            BACKTEST_COLUMNS, replace_dates=[target.isoformat()])
    logger.info("Backtest de %s: %s", target, payload["summary"])
    return payload


def format_summary(payload: Mapping[str, Any]) -> str:
    """Resumen legible del backtest para la consola."""
    if not payload.get("ok"):
        return str(payload.get("error", "Backtest no realizado"))
    lines = [f"Backtest {payload['date']}: {payload['summary']['picks']} pick(s)"]
    for r in payload["results"]:
        lines.append(
            f"  {r.get('ticker')} {r.get('direction')}: resultado={r.get('outcome')}"
            f"{' (ambiguo)' if r.get('ambiguous') else ''}, entrada válida={r.get('entry_valid')}, "
            f"MFE={r.get('mfe_pct')}%, MAE={r.get('mae_pct')}%, fuente={r.get('data_source')}")
    return "\n".join(lines)


def run_backtest_command(date_text: str) -> int:
    """Punto de entrada para ``--backtest-date``. Devuelve el código de salida (0 = ok)."""
    try:
        target = datetime.strptime(date_text.strip(), "%Y-%m-%d").date()
    except ValueError:
        logger.error("Fecha de backtest inválida: %r (usa YYYY-MM-DD)", date_text)
        return 2
    payload = run_backtest(target)
    print(format_summary(payload))
    return 0 if payload.get("ok") else 1
