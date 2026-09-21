"""technical_analysis.py — Indicadores técnicos, RVOL, niveles y fuerza relativa.

Solo pandas/numpy (sin TA-Lib). Python calcula; Gemini interpreta (regla global 3).

Convenciones
------------
* Un dato que no se puede calcular se devuelve como la cadena ``"N/A"`` (constante
  ``NA``). Nunca se rellena ni se inventa. Usa ``is_num`` para distinguir.
* Los indicadores diarios se calculan solo con **sesiones completadas**: se excluye
  la fila de hoy, aunque yfinance la devuelva incompleta.
* Los periodos de los indicadores (RSI 14, ATR 14, SMA 20/50, EMA 9/20) forman parte
  de la definición de cada indicador y por eso son constantes de este módulo; los
  umbrales y ventanas operativas (RVOL, horarios, calidad del dato) viven en ``config.py``.

Definiciones
------------
RSI(14) y ATR(14): suavizado de Wilder con semilla SMA. La primera media es la media
    simple de los primeros ``period`` valores; después
    ``media = (media_previa * (period - 1) + valor) / period``.
    El primer True Range usa solo High - Low (no hay cierre previo).
EMA: factor ``2 / (n + 1)`` con semilla SMA de las primeras ``n`` barras.
Gap %: ``(precio pre-market / cierre anterior - 1) * 100``.

RVOL (relative volume pre-market)
---------------------------------
* Numerador: volumen pre-market acumulado hoy desde las 04:00 ET hasta la hora del
  snapshot (barras cuyo inicio es < hora del snapshot).
* Denominador: **mediana** del volumen acumulado en esa misma ventana horaria durante
  las últimas ``config.RVOL.lookback_sessions`` (20) sesiones previas presentes en
  los datos.
* Una sesión es válida si tiene al menos una barra con volumen en la ventana. Hacen
  falta ``config.RVOL.min_sessions`` (10) sesiones válidas; si no, ``rvol = "N/A"``
  con el motivo en ``reason``.
* ``RVOL = numerador / denominador``. Si la mediana es 0, ``rvol = "N/A"``.
* ``pm_pct_of_adv`` (volumen pre-market / volumen medio diario de 20 sesiones, en %)
  es una métrica distinta y **nunca se llama RVOL**.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("trading_agent.technical_analysis")

NA = "N/A"

RSI_PERIOD = 14
ATR_PERIOD = 14
SMA_FAST, SMA_SLOW = 20, 50
EMA_FAST, EMA_SLOW = 9, 20
SHORT_LOOKBACK, LONG_LOOKBACK = 5, 20
VOLUME_AVG_SESSIONS = 20
WEEKS_52 = 52
RS_BENCHMARKS = ("SPY", "QQQ")   # índices frente a los que se mide la fuerza relativa

LEVEL_LABELS = (
    "prev_close", "max_5d", "min_5d", "max_20d", "min_20d", "high_52w", "low_52w",
    "sma20", "sma50", "ema9", "ema20", "premarket_high", "premarket_low",
)
_LEVELS_FROM_INDICATORS = (
    "max_5d", "min_5d", "max_20d", "min_20d", "high_52w", "low_52w",
    "sma20", "sma50", "ema9", "ema20",
)


# --------------------------------------------------------------------------- #
# Utilidades básicas
# --------------------------------------------------------------------------- #
def is_num(value: Any) -> bool:
    """True si ``value`` es un número finito (excluye bool, NaN, inf y "N/A")."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return False
    return math.isfinite(float(value))


def _f(value: Any, ndigits: int | None = 4) -> float | str:
    """Convierte a ``float`` nativo (apto para JSON) o devuelve ``"N/A"``."""
    if not is_num(value):
        return NA
    number = float(value)
    return round(number, ndigits) if ndigits is not None else number


def _last(series: pd.Series) -> float | None:
    """Último valor no nulo de una serie, o None."""
    valid = series.dropna()
    return float(valid.iloc[-1]) if len(valid) else None


def _dates(index: pd.Index) -> np.ndarray:
    """Array de objetos ``date`` (fecha local del índice)."""
    return np.asarray(pd.DatetimeIndex(index).date)


def _dates_before(index: pd.Index, day: date) -> np.ndarray:
    """Máscara booleana: filas cuya fecha es estrictamente anterior a ``day``."""
    return np.fromiter((d < day for d in _dates(index)), dtype=bool, count=len(index))


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Columna como float; serie de NaN si no existe."""
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def window_bars(df: pd.DataFrame | None, day: date, start: time, end: time) -> pd.DataFrame:
    """Barras de ``day`` cuya hora de inicio cumple ``start <= hora < end``.

    Se usa la hora local del índice (debe estar en America/New_York).
    """
    if df is None:
        return pd.DataFrame()
    if len(df) == 0:
        return df.iloc[0:0]
    idx = pd.DatetimeIndex(df.index)
    minutes = np.asarray(idx.hour) * 60 + np.asarray(idx.minute)
    lo = start.hour * 60 + start.minute
    hi = end.hour * 60 + end.minute
    mask = (_dates(idx) == day) & (minutes >= lo) & (minutes < hi)
    return df[mask]


# --------------------------------------------------------------------------- #
# Indicadores (series completas)
# --------------------------------------------------------------------------- #
def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Suavizado de Wilder con semilla SMA. NaN hasta tener ``period`` valores."""
    out = np.full(len(values), np.nan)
    if period < 1 or len(values) < period:
        return out
    out[period - 1] = values[:period].mean()
    for i in range(period, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def sma(series: pd.Series, period: int) -> pd.Series:
    """Media móvil simple; NaN hasta tener ``period`` observaciones."""
    return series.astype(float).rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA con factor 2/(n+1) y semilla SMA de las primeras ``period`` barras."""
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan)
    if period >= 1 and len(values) >= period:
        alpha = 2.0 / (period + 1)
        out[period - 1] = values[:period].mean()
        for i in range(period, len(values)):
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI de Wilder. Primer valor válido en la posición ``period``.

    Casos límite: sin pérdidas y con ganancias => 100; sin movimiento => 50.
    """
    close = close.astype(float)
    delta = close.diff().to_numpy()
    gains = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0))
    losses = np.where(np.isnan(delta), np.nan, np.maximum(-delta, 0.0))
    out = np.full(len(close), np.nan)
    if len(close) > period:
        avg_gain = _wilder_smooth(gains[1:], period)
        avg_loss = _wilder_smooth(losses[1:], period)
        for i in range(len(avg_gain)):
            g, l = avg_gain[i], avg_loss[i]
            if math.isnan(g) or math.isnan(l):
                continue
            if l == 0:
                out[i + 1] = 50.0 if g == 0 else 100.0
            else:
                out[i + 1] = 100.0 - 100.0 / (1.0 + g / l)
    return pd.Series(out, index=close.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range; el primero es High - Low (sin cierre previo)."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    tr = ranges.max(axis=1, skipna=True)
    tr.iloc[0] = (high - low).iloc[0] if len(tr) else np.nan
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = ATR_PERIOD) -> pd.Series:
    """ATR de Wilder (semilla = media de los primeros ``period`` TR)."""
    tr = true_range(high.astype(float), low.astype(float), close.astype(float))
    return pd.Series(_wilder_smooth(tr.to_numpy(), period), index=close.index)


# --------------------------------------------------------------------------- #
# Indicadores diarios "de hoy"
# --------------------------------------------------------------------------- #
def _completed_sessions(daily: pd.DataFrame | None, session_date: date) -> pd.DataFrame:
    """Sesiones diarias completadas (fecha < ``session_date``) con cierre válido."""
    if daily is None or len(daily) == 0 or "Close" not in daily.columns:
        return pd.DataFrame()
    df = daily[_dates_before(daily.index, session_date)]
    return df.dropna(subset=["Close"])


def compute_daily_indicators(daily: pd.DataFrame | None, session_date: date) -> dict[str, Any]:
    """Indicadores y estadísticas diarias a partir de sesiones completadas.

    Devuelve siempre las mismas claves; lo que no se pueda calcular es ``"N/A"``.
    Máximos/mínimos de 5 y 20 sesiones excluyen hoy (usa High/Low). El rango de
    52 semanas cubre las sesiones dentro de las 52 semanas previas a ``session_date``.
    """
    keys = (
        "last_close", "last_close_date", "rsi14", "sma20", "sma50", "ema9", "ema20",
        "atr14", "atr_pct", "avg_volume_20d", "avg_dollar_volume_20d",
        "max_5d", "min_5d", "max_20d", "min_20d", "high_52w", "low_52w",
        "return_5d_pct",
    )
    df = _completed_sessions(daily, session_date)
    n = len(df)
    out: dict[str, Any] = {k: NA for k in keys}
    out["sessions_available"] = n
    if n == 0:
        return out

    close, high, low, volume = (_col(df, c) for c in ("Close", "High", "Low", "Volume"))
    last_close = float(close.iloc[-1])
    atr = _last(atr_wilder(high, low, close))

    out["last_close"] = _f(last_close)
    out["last_close_date"] = df.index[-1].date().isoformat()
    out["rsi14"] = _f(_last(rsi_wilder(close)))
    out["sma20"] = _f(_last(sma(close, SMA_FAST)))
    out["sma50"] = _f(_last(sma(close, SMA_SLOW)))
    out["ema9"] = _f(_last(ema(close, EMA_FAST)))
    out["ema20"] = _f(_last(ema(close, EMA_SLOW)))
    out["atr14"] = _f(atr)
    if atr is not None and last_close > 0:
        out["atr_pct"] = _f(atr / last_close * 100)

    if n >= VOLUME_AVG_SESSIONS:
        out["avg_volume_20d"] = _f(volume.tail(VOLUME_AVG_SESSIONS).mean(), 2)
        out["avg_dollar_volume_20d"] = _f((close * volume).tail(VOLUME_AVG_SESSIONS).mean(), 2)
    if n >= SHORT_LOOKBACK:
        out["max_5d"] = _f(high.tail(SHORT_LOOKBACK).max())
        out["min_5d"] = _f(low.tail(SHORT_LOOKBACK).min())
    if n >= LONG_LOOKBACK:
        out["max_20d"] = _f(high.tail(LONG_LOOKBACK).max())
        out["min_20d"] = _f(low.tail(LONG_LOOKBACK).min())
    if n >= SHORT_LOOKBACK + 1:
        out["return_5d_pct"] = _f((close.iloc[-1] / close.iloc[-1 - SHORT_LOOKBACK] - 1) * 100)

    start_52w = session_date - timedelta(weeks=WEEKS_52)
    in_52w = np.fromiter((d >= start_52w for d in _dates(df.index)), dtype=bool, count=n)
    if in_52w.any():
        out["high_52w"] = _f(high[in_52w].max())
        out["low_52w"] = _f(low[in_52w].min())
    return out


# --------------------------------------------------------------------------- #
# Gap, rango pre-market, volumen
# --------------------------------------------------------------------------- #
def gap_pct(premarket_price: Any, prev_close: Any) -> float | str:
    """Gap % = (precio pre-market / cierre anterior - 1) * 100."""
    if not (is_num(premarket_price) and is_num(prev_close)) or float(prev_close) == 0:
        return NA
    return _f((float(premarket_price) / float(prev_close) - 1) * 100)


def gap_in_atr(premarket_price: Any, prev_close: Any, atr: Any) -> float | str:
    """Gap expresado en múltiplos de ATR: (pre-market - cierre anterior) / ATR."""
    if not (is_num(premarket_price) and is_num(prev_close) and is_num(atr)) or float(atr) <= 0:
        return NA
    return _f((float(premarket_price) - float(prev_close)) / float(atr))


def premarket_range(pm_high: Any, pm_low: Any, prev_close: Any, atr: Any) -> dict[str, Any]:
    """Rango pre-market: en % del cierre anterior y en múltiplos de ATR."""
    out: dict[str, Any] = {"premarket_range_pct": NA, "premarket_range_atr": NA}
    if not (is_num(pm_high) and is_num(pm_low)):
        return out
    span = float(pm_high) - float(pm_low)
    if is_num(prev_close) and float(prev_close) > 0:
        out["premarket_range_pct"] = _f(span / float(prev_close) * 100)
    if is_num(atr) and float(atr) > 0:
        out["premarket_range_atr"] = _f(span / float(atr))
    return out


def pm_pct_of_adv(premarket_volume: Any, avg_volume_20d: Any) -> float | str:
    """Volumen pre-market / volumen medio diario (20 sesiones), en %.

    NO es RVOL: compara con un día completo, no con la misma ventana horaria.
    """
    if not (is_num(premarket_volume) and is_num(avg_volume_20d)) or float(avg_volume_20d) <= 0:
        return NA
    return _f(float(premarket_volume) / float(avg_volume_20d) * 100)


# --------------------------------------------------------------------------- #
# RVOL
# --------------------------------------------------------------------------- #
@dataclass
class RvolResult:
    """Resultado transparente del RVOL (ver definición en el docstring del módulo)."""

    rvol: float | str
    numerator: float | str
    denominator: float | str
    sessions_used: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _window_volume(df: pd.DataFrame | None, day: date, start: time, end: time) -> pd.Series:
    """Volumen (sin NaN) de las barras de ``day`` en la ventana [start, end)."""
    win = window_bars(df, day, start, end)
    if len(win) == 0 or "Volume" not in win.columns:
        return pd.Series(dtype=float)
    return win["Volume"].astype(float).dropna()


def compute_rvol(
    intraday_hist: pd.DataFrame | None,
    intraday_today: pd.DataFrame | None,
    snapshot_ts: datetime,
    rvol_cfg: config.RvolConfig | None = None,
    schedule: config.ScheduleConfig | None = None,
) -> RvolResult:
    """RVOL pre-market: volumen acumulado hoy / mediana histórica de la misma ventana.

    Args:
        intraday_hist: barras 5m (prepost) de sesiones previas, índice en NY.
        intraday_today: barras 5m de hoy hasta el snapshot, índice en NY.
        snapshot_ts: instante del snapshot (tz NY). Define el fin de la ventana
            (limitado a la apertura, 09:30) y la fecha de "hoy".
        rvol_cfg: ``config.RVOL`` (``lookback_sessions``, ``min_sessions``).
        schedule: ``config.SCHEDULE`` (``premarket_start``, ``regular_open``).
    """
    rvol_cfg = rvol_cfg or config.RVOL
    schedule = schedule or config.SCHEDULE
    day = snapshot_ts.date()
    end = min(snapshot_ts.time(), schedule.regular_open)
    start = schedule.premarket_start

    today_vol = _window_volume(intraday_today, day, start, end)
    if today_vol.empty:
        return RvolResult(NA, NA, NA, 0, "sin barras pre-market de hoy en la ventana")
    numerator = float(today_vol.sum())

    if intraday_hist is None or len(intraday_hist) == 0:
        return RvolResult(NA, _f(numerator, 2), NA, 0, "sin histórico intradía")

    all_days = sorted({d for d in _dates(intraday_hist.index) if d < day})
    recent_days = all_days[-rvol_cfg.lookback_sessions:]
    cumulative: list[float] = []
    for d in recent_days:
        vol = _window_volume(intraday_hist, d, start, end)
        if not vol.empty:
            cumulative.append(float(vol.sum()))

    sessions = len(cumulative)
    if sessions < rvol_cfg.min_sessions:
        return RvolResult(
            NA, _f(numerator, 2), NA, sessions,
            f"solo {sessions} sesiones válidas (mínimo {rvol_cfg.min_sessions})",
        )
    denominator = float(np.median(cumulative))
    if denominator <= 0:
        return RvolResult(NA, _f(numerator, 2), 0.0, sessions,
                          "mediana histórica de volumen igual a 0")
    return RvolResult(_f(numerator / denominator), _f(numerator, 2),
                      _f(denominator, 2), sessions, "")


# --------------------------------------------------------------------------- #
# Tabla de niveles
# --------------------------------------------------------------------------- #
def build_levels_table(
    price: Any, atr: Any, indicators: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Tabla de niveles con etiqueta y distancia al precio pre-market.

    Cada fila: ``label``, ``price``, ``side`` (``resistance`` si está por encima,
    ``support`` si está por debajo, ``at_price`` si coincide), ``distance_pct`` y
    ``distance_atr`` (con signo; positivo = por encima del precio).
    Los soportes y resistencias del informe salen **solo** de esta lista. Los niveles
    sin dato se omiten. Devuelve lista vacía si no hay precio pre-market.
    """
    if not is_num(price) or float(price) <= 0:
        return []
    ref = float(price)
    raw: dict[str, Any] = {
        "prev_close": snapshot.get("prev_close"),
        "premarket_high": snapshot.get("premarket_high"),
        "premarket_low": snapshot.get("premarket_low"),
    }
    for key in _LEVELS_FROM_INDICATORS:
        raw[key] = indicators.get(key)

    use_atr = is_num(atr) and float(atr) > 0
    rows: list[dict[str, Any]] = []
    for label in LEVEL_LABELS:
        value = raw.get(label)
        if not is_num(value):
            continue
        diff = float(value) - ref
        side = "resistance" if diff > 0 else "support" if diff < 0 else "at_price"
        rows.append({
            "label": label,
            "price": _f(value),
            "side": side,
            "distance_pct": _f(diff / ref * 100),
            "distance_atr": _f(diff / float(atr)) if use_atr else NA,
        })
    rows.sort(key=lambda r: r["price"], reverse=True)
    return rows


def nearest_level(levels: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any] | str:
    """Nivel más cercano al precio del lado indicado (``support`` o ``resistance``)."""
    candidates = [r for r in levels if r.get("side") == side]
    if not candidates:
        return NA
    return dict(min(candidates, key=lambda r: abs(r["distance_pct"])))


# --------------------------------------------------------------------------- #
# Fuerza relativa
# --------------------------------------------------------------------------- #
def benchmark_metrics(
    daily: pd.DataFrame | None, snapshot: Mapping[str, Any], session_date: date
) -> dict[str, Any]:
    """Variación pre-market y retorno a 5 sesiones de un índice (SPY, QQQ...)."""
    indicators = compute_daily_indicators(daily, session_date)
    return {
        "pm_change_pct": gap_pct(snapshot.get("premarket_price"), snapshot.get("prev_close")),
        "return_5d_pct": indicators["return_5d_pct"],
    }


def relative_strength(
    pm_change_pct: Any,
    return_5d_pct: Any,
    benchmarks: Mapping[str, Mapping[str, Any]] | None,
    names: Sequence[str] = RS_BENCHMARKS,
) -> dict[str, Any]:
    """Fuerza relativa frente a cada índice, en puntos porcentuales.

    * ``rs_pm_vs_<idx>``: variación pre-market del ticker menos la del índice.
    * ``rs_5d_vs_<idx>``: retorno de 5 sesiones del ticker menos el del índice.
    ``"N/A"`` si falta cualquiera de los dos datos.
    """
    benchmarks = benchmarks or {}
    out: dict[str, Any] = {}
    for name in names:
        metrics = benchmarks.get(name, {})
        key = name.lower()
        for label, own, idx_val in (
            ("pm", pm_change_pct, metrics.get("pm_change_pct")),
            ("5d", return_5d_pct, metrics.get("return_5d_pct")),
        ):
            out[f"rs_{label}_vs_{key}"] = (
                _f(float(own) - float(idx_val)) if is_num(own) and is_num(idx_val) else NA
            )
    return out


# --------------------------------------------------------------------------- #
# Análisis completo de un ticker
# --------------------------------------------------------------------------- #
def analyze_ticker(
    ticker: str,
    daily: pd.DataFrame | None,
    intraday_hist: pd.DataFrame | None,
    intraday_today: pd.DataFrame | None,
    snapshot: Mapping[str, Any],
    snapshot_ts: datetime,
    benchmarks: Mapping[str, Mapping[str, Any]] | None = None,
    rvol_cfg: config.RvolConfig | None = None,
    schedule: config.ScheduleConfig | None = None,
) -> dict[str, Any]:
    """Combina snapshot pre-market e indicadores en un registro serializable a JSON.

    ``snapshot`` es ``PremarketSnapshot.to_dict()`` (ver ``market_data.py``).
    """
    session_date = snapshot_ts.date()
    indicators = compute_daily_indicators(daily, session_date)

    pm_price = snapshot.get("premarket_price")
    prev_close = snapshot.get("prev_close")
    atr = indicators["atr14"]
    gap = gap_pct(pm_price, prev_close)
    levels = build_levels_table(pm_price, atr, indicators, snapshot)

    return {
        "ticker": ticker,
        "session_date": session_date.isoformat(),
        "premarket": dict(snapshot),
        "indicators": indicators,
        "gap_pct": gap,
        "gap_atr": gap_in_atr(pm_price, prev_close, atr),
        **premarket_range(snapshot.get("premarket_high"), snapshot.get("premarket_low"),
                          prev_close, atr),
        "rvol": compute_rvol(intraday_hist, intraday_today, snapshot_ts, rvol_cfg,
                             schedule).to_dict(),
        "pm_pct_of_adv": pm_pct_of_adv(snapshot.get("premarket_volume"),
                                       indicators["avg_volume_20d"]),
        "relative_strength": relative_strength(gap, indicators["return_5d_pct"],
                                               benchmarks),
        "levels": levels,
        "nearest_support": nearest_level(levels, "support"),
        "nearest_resistance": nearest_level(levels, "resistance"),
    }


def analyze_many(
    tickers: Sequence[str],
    daily: Mapping[str, pd.DataFrame],
    intraday_hist: Mapping[str, pd.DataFrame],
    intraday_today: Mapping[str, pd.DataFrame],
    snapshots: Mapping[str, Mapping[str, Any]],
    snapshot_ts: datetime,
    benchmarks: Mapping[str, Mapping[str, Any]] | None = None,
    rvol_cfg: config.RvolConfig | None = None,
    schedule: config.ScheduleConfig | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Analiza cada ticker de forma aislada (regla global 5).

    Un fallo en un ticker se registra como ``ERROR <ticker>: ...`` y no detiene al resto.
    Devuelve ``(resultados, fallos)``.
    """
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for ticker in tickers:
        try:
            if ticker not in snapshots:
                raise KeyError("sin snapshot pre-market")
            results[ticker] = analyze_ticker(
                ticker, daily.get(ticker), intraday_hist.get(ticker),
                intraday_today.get(ticker), snapshots[ticker], snapshot_ts, benchmarks,
                rvol_cfg, schedule,
            )
        except Exception as exc:  # noqa: BLE001 - aislamiento deliberado por ticker
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            logger.error("%s: no se pudo analizar (%s)", ticker, failures[ticker])
    return results, failures
