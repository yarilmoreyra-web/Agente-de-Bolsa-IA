"""market_data.py — Descarga de datos de mercado y snapshot pre-market (yfinance).

Usa la configuración y las utilidades ya existentes: ``config.MARKET_DATA``,
``config.SCHEDULE``, ``config.PATHS``, ``utils.retry``, ``utils.NY_TZ``,
``utils.to_ny`` y ``utils.to_yahoo_symbol``.

Limitaciones de yfinance (léelas antes de fiarte del dato)
----------------------------------------------------------
* No es una API oficial: Yahoo puede cambiar formatos o limitar peticiones sin aviso.
* Latencia: los datos pre-market pueden llegar con retraso o incompletos; por eso
  cada snapshot lleva ``premarket_quality`` (``ok`` | ``stale`` | ``sparse`` |
  ``missing``) y nunca se rellenan huecos.
* Granularidad: las barras de 5 minutos solo cubren unos 60 días y los valores
  ilíquidos pueden tener barras sueltas en el pre-market.
* Las IPs de GitHub Actions pueden ser bloqueadas o limitadas por Yahoo.

Convenciones
------------
* Todos los índices se convierten a ``America/New_York`` (``to_ny_index``).
* Los resultados usan como clave el ticker del universo (p. ej. ``BRK.B``); a Yahoo
  se le envía su símbolo (``BRK-B``) y se traduce de vuelta.
* Precios diarios sin ajuste por dividendos (``auto_adjust=False``, columna ``Close``),
  para que sean comparables con los precios pre-market. Sí están ajustados por splits.
* Las barras de yfinance se etiquetan con su **hora de inicio**. Una barra ``08:40``
  cubre 08:40-08:45. Un snapshot a las 08:45 usa las barras con inicio < 08:45, y la
  antigüedad del dato se mide desde el inicio de la última barra.
* Pre-market = ``SCHEDULE.premarket_start`` (04:00, incl.) a ``SCHEDULE.regular_open``
  (09:30, excl.).
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pandas as pd

import config
from technical_analysis import NA, _dates, _dates_before, window_bars
from utils import NY_TZ, retry, to_ny, to_yahoo_symbol

logger = logging.getLogger("trading_agent.market_data")

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
_YF_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


class DownloadError(RuntimeError):
    """Fallo de descarga (respuesta vacía o excepción de yfinance)."""


@dataclass
class BatchResult:
    """Resultado de una descarga por lotes: datos por ticker y motivos de fallo."""

    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)


@dataclass
class PremarketSnapshot:
    """Foto del pre-market de un ticker a la hora del snapshot.

    Los campos sin dato valen ``"N/A"``. ``last_bar_age_min`` se mide entre el inicio
    de la última barra y ``snapshot_ts``.
    """

    ticker: str
    session_date: str
    snapshot_ts: str
    prev_close: float | str = NA
    premarket_price: float | str = NA
    premarket_high: float | str = NA
    premarket_low: float | str = NA
    premarket_volume: float | str = NA
    last_bar_time: str = NA
    last_bar_age_min: float | str = NA
    n_bars: int = 0
    premarket_quality: str = "missing"
    quality_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _get_yf():
    """Importa yfinance de forma diferida (permite tests sin la librería ni red)."""
    import yfinance as yf  # noqa: PLC0415

    return yf


def _as_ny(moment: datetime) -> datetime:
    """Devuelve ``moment`` en Nueva York (si es naive, se asume que ya está en NY)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=NY_TZ)
    return to_ny(moment)


def interval_minutes(interval: str) -> int:
    """Minutos de un intervalo de yfinance (``"5m"`` -> 5)."""
    match = re.fullmatch(r"(\d+)m", interval.strip())
    if not match:
        raise ValueError(f"Intervalo intradía no soportado: {interval!r}")
    return int(match.group(1))


def chunked(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Divide ``items`` en lotes de como máximo ``size`` elementos."""
    if size < 1:
        raise ValueError("size debe ser >= 1")
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


def to_ny_index(df: pd.DataFrame | None) -> pd.DataFrame:
    """Convierte el índice a America/New_York.

    Índices con zona horaria se convierten; los naive (p. ej. fechas diarias) se
    interpretan como hora de Nueva York. Las horas ambiguas o inexistentes se descartan.
    """
    if df is None:
        return pd.DataFrame()
    if len(df) == 0:
        return df.copy()
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ, ambiguous="NaT", nonexistent="NaT")
    else:
        idx = idx.tz_convert(NY_TZ)
    out.index = idx
    return out[~out.index.isna()]


def _normalize_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    """Deja solo columnas OHLCV, sin filas sin cierre ni duplicados, en hora NY."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    cols = [c for c in OHLCV if c in df.columns]
    if "Close" not in cols:
        return pd.DataFrame()
    out = df[cols].dropna(subset=["Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return to_ny_index(out)


def _split_download(raw: pd.DataFrame, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Separa la salida de ``yf.download`` en un DataFrame por símbolo de Yahoo.

    Soporta columnas MultiIndex ``(símbolo, campo)`` o ``(campo, símbolo)`` y el
    formato plano de un solo símbolo.
    """
    frames: dict[str, pd.DataFrame] = {}
    if raw is None or len(raw) == 0:
        return frames
    if isinstance(raw.columns, pd.MultiIndex):
        if set(raw.columns.get_level_values(0)) <= _YF_FIELDS:
            raw = raw.swaplevel(axis=1)
        level0 = set(raw.columns.get_level_values(0))
        for symbol in symbols:
            if symbol in level0:
                frames[symbol] = _normalize_frame(raw[symbol])
    elif len(symbols) == 1:
        frames[symbols[0]] = _normalize_frame(raw)
    return {s: f for s, f in frames.items() if len(f)}


# --------------------------------------------------------------------------- #
# Descargas por lotes
# --------------------------------------------------------------------------- #
def download_frames(
    tickers: Sequence[str], *, period: str, interval: str, prepost: bool, label: str,
    market_cfg: config.MarketDataConfig | None = None,
) -> BatchResult:
    """Descarga ``tickers`` por lotes con ``yf.download`` (una llamada por lote).

    Cada llamada se reintenta con ``utils.retry`` (``max_retries``,
    ``retry_base_delay``, backoff exponencial). Un lote que falla tras los reintentos
    deja a sus tickers en ``failed`` y el resto continúa. Cada fallo se registra como
    ``<TICKER>: no se pudieron obtener datos <label> (<motivo>)``.
    """
    market_cfg = market_cfg or config.MARKET_DATA
    unique = list(dict.fromkeys(tickers))
    result = BatchResult()

    for batch in chunked(unique, market_cfg.batch_size):
        groups: dict[str, list[str]] = {}
        for ticker in batch:
            groups.setdefault(to_yahoo_symbol(ticker), []).append(ticker)
        symbols = list(groups)

        @retry(attempts=market_cfg.max_retries, base_delay=market_cfg.retry_base_delay,
               exceptions=(Exception,))
        def _download(symbols: list[str] = symbols) -> pd.DataFrame:
            raw = _get_yf().download(
                tickers=symbols, period=period, interval=interval, prepost=prepost,
                group_by="ticker", auto_adjust=False,
                threads=market_cfg.download_threads, progress=False,
            )
            if raw is None or len(raw) == 0:
                raise DownloadError("respuesta vacía de yfinance")
            return raw

        try:
            frames = _split_download(_download(), symbols)
            reason = "sin datos en la respuesta"
        except Exception as exc:  # noqa: BLE001 - un lote caído no debe parar el resto
            frames, reason = {}, f"lote fallido: {exc}"

        for symbol, originals in groups.items():
            for ticker in originals:
                if symbol in frames:
                    result.data[ticker] = frames[symbol]
                else:
                    result.failed[ticker] = reason
                    logger.error("%s: no se pudieron obtener datos %s (%s)",
                                 ticker, label, reason)
    return result


def download_daily(tickers: Sequence[str],
                   market_cfg: config.MarketDataConfig | None = None) -> BatchResult:
    """1 año de OHLCV diario (indicadores, máximos y mínimos)."""
    market_cfg = market_cfg or config.MARKET_DATA
    return download_frames(tickers, period=market_cfg.daily_period, interval="1d",
                           prepost=False, label="diarios", market_cfg=market_cfg)


def download_intraday_history(
    tickers: Sequence[str], market_cfg: config.MarketDataConfig | None = None,
) -> BatchResult:
    """Barras 5m con pre/post-market de las últimas ~4 semanas (base del RVOL).

    Se descarga temprano (antes del snapshot); las barras de hoy que incluya se
    ignoran en el cálculo del RVOL.
    """
    market_cfg = market_cfg or config.MARKET_DATA
    return download_frames(tickers, period=market_cfg.intraday_baseline_period,
                           interval=market_cfg.intraday_interval, prepost=True,
                           label="intradía históricos", market_cfg=market_cfg)


def download_intraday_today(
    tickers: Sequence[str], session_date: date,
    market_cfg: config.MarketDataConfig | None = None,
) -> BatchResult:
    """Barras 5m de ``session_date`` con pre-market (para el snapshot).

    Se piden unos días (``intraday_today_period``) y se filtra el día de hoy. Un ticker
    sin ninguna barra de hoy queda en ``failed``; ``extract_premarket_snapshot`` lo
    tratará como ``missing``.
    """
    market_cfg = market_cfg or config.MARKET_DATA
    raw = download_frames(tickers, period=market_cfg.intraday_today_period,
                          interval=market_cfg.intraday_interval, prepost=True,
                          label="intradía de hoy", market_cfg=market_cfg)
    result = BatchResult(failed=dict(raw.failed))
    for ticker, df in raw.data.items():
        today = df[_dates(df.index) == session_date]
        if len(today):
            result.data[ticker] = today
        else:
            result.failed[ticker] = "sin barras de hoy"
            logger.warning("%s: sin barras pre-market de %s", ticker, session_date)
    return result


# --------------------------------------------------------------------------- #
# Snapshot pre-market
# --------------------------------------------------------------------------- #
def resolve_snapshot_ts(
    session_date: date, now: datetime | None = None,
    market_cfg: config.MarketDataConfig | None = None,
    schedule: config.ScheduleConfig | None = None,
) -> datetime:
    """Instante del snapshot en NY.

    Por defecto es ``SCHEDULE.snapshot_time`` del día. Si ``now`` es de ese mismo día
    y ya pasó esa hora, se usa ``now`` redondeado hacia abajo al tamaño de barra
    (solo barras completas), sin superar la apertura.
    """
    market_cfg = market_cfg or config.MARKET_DATA
    schedule = schedule or config.SCHEDULE
    base = datetime.combine(session_date, schedule.snapshot_time, tzinfo=NY_TZ)
    if now is None:
        return base
    now_ny = _as_ny(now)
    if now_ny.date() != session_date:
        return base
    step = interval_minutes(market_cfg.intraday_interval)
    floored = now_ny.replace(minute=(now_ny.minute // step) * step, second=0, microsecond=0)
    cap = datetime.combine(session_date, schedule.regular_open, tzinfo=NY_TZ)
    return min(max(base, floored), cap)


def classify_premarket_quality(
    n_bars: int, age_minutes: float | None,
    market_cfg: config.MarketDataConfig | None = None,
) -> tuple[str, str]:
    """Clasifica la calidad del dato: ``missing`` > ``stale`` > ``sparse`` > ``ok``.

    Usa ``stale_minutes`` y ``min_premarket_bars`` de ``config.MARKET_DATA``.
    """
    market_cfg = market_cfg or config.MARKET_DATA
    if n_bars <= 0 or age_minutes is None:
        return "missing", "sin barras pre-market"
    if age_minutes > market_cfg.stale_minutes:
        return "stale", (f"última barra hace {age_minutes:.0f} min "
                         f"(máx. {market_cfg.stale_minutes})")
    if n_bars < market_cfg.min_premarket_bars:
        return "sparse", f"solo {n_bars} barras (mínimo {market_cfg.min_premarket_bars})"
    return "ok", ""


def _previous_close(daily: pd.DataFrame | None, session_date: date) -> float | str:
    """Último cierre diario de una sesión anterior a ``session_date``."""
    if daily is None or len(daily) == 0 or "Close" not in daily.columns:
        return NA
    prior = daily[_dates_before(daily.index, session_date)].dropna(subset=["Close"])
    return round(float(prior["Close"].iloc[-1]), 4) if len(prior) else NA


def extract_premarket_snapshot(
    ticker: str, daily: pd.DataFrame | None, intraday_today: pd.DataFrame | None,
    snapshot_ts: datetime, market_cfg: config.MarketDataConfig | None = None,
    schedule: config.ScheduleConfig | None = None,
) -> PremarketSnapshot:
    """Construye el snapshot pre-market de un ticker.

    Usa las barras de hoy con inicio en [premarket_start, min(snapshot_ts, regular_open)).
    Precio pre-market = cierre de la última barra. Sin barras => calidad ``missing`` y
    campos ``"N/A"`` (no se rellena ni se estima nada).
    """
    market_cfg = market_cfg or config.MARKET_DATA
    schedule = schedule or config.SCHEDULE
    snapshot_ts = _as_ny(snapshot_ts)
    session_date = snapshot_ts.date()
    snap = PremarketSnapshot(
        ticker=ticker, session_date=session_date.isoformat(),
        snapshot_ts=snapshot_ts.isoformat(),
        prev_close=_previous_close(daily, session_date),
    )

    end = min(snapshot_ts.time(), schedule.regular_open)
    bars = window_bars(intraday_today, session_date, schedule.premarket_start, end)
    if len(bars):
        bars = bars.dropna(subset=["Close"])

    if len(bars) == 0:
        snap.premarket_quality, snap.quality_note = classify_premarket_quality(
            0, None, market_cfg)
        return snap

    last_time = bars.index[-1]
    age = round((snapshot_ts - last_time.to_pydatetime()).total_seconds() / 60.0, 1)
    volume = bars["Volume"].astype(float) if "Volume" in bars.columns else None

    snap.premarket_price = round(float(bars["Close"].iloc[-1]), 4)
    if "High" in bars.columns and bars["High"].notna().any():
        snap.premarket_high = round(float(bars["High"].max()), 4)
    if "Low" in bars.columns and bars["Low"].notna().any():
        snap.premarket_low = round(float(bars["Low"].min()), 4)
    if volume is not None and volume.notna().any():
        snap.premarket_volume = float(volume.sum())
    snap.last_bar_time = last_time.isoformat()
    snap.last_bar_age_min = age
    snap.n_bars = int(len(bars))
    snap.premarket_quality, snap.quality_note = classify_premarket_quality(
        snap.n_bars, age, market_cfg)
    return snap


def build_snapshots(
    tickers: Sequence[str], daily: Mapping[str, pd.DataFrame],
    intraday_today: Mapping[str, pd.DataFrame], snapshot_ts: datetime,
    market_cfg: config.MarketDataConfig | None = None,
    schedule: config.ScheduleConfig | None = None,
) -> dict[str, PremarketSnapshot]:
    """Snapshots de varios tickers, cada uno aislado de los demás (regla global 5)."""
    snapshots: dict[str, PremarketSnapshot] = {}
    for ticker in tickers:
        try:
            snapshots[ticker] = extract_premarket_snapshot(
                ticker, daily.get(ticker), intraday_today.get(ticker), snapshot_ts,
                market_cfg, schedule)
        except Exception as exc:  # noqa: BLE001 - aislamiento deliberado por ticker
            logger.error("%s: no se pudo construir el snapshot pre-market (%s)", ticker, exc)
    return snapshots


# --------------------------------------------------------------------------- #
# Persistencia de barras 5m (para el backtest)
# --------------------------------------------------------------------------- #
def _safe_name(ticker: str) -> str:
    """Nombre de archivo seguro para un ticker (``^VIX`` -> ``_VIX``)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)


def _intraday_folder(session_date: date, base_dir: str | Path | None) -> Path:
    base = Path(base_dir) if base_dir else config.PATHS.intraday_dir
    return base / session_date.isoformat()


def save_intraday_bars(
    bars: Mapping[str, pd.DataFrame], session_date: date,
    base_dir: str | Path | None = None,
) -> list[Path]:
    """Guarda las barras 5m en ``data/intraday/YYYY-MM-DD/<TICKER>.csv.gz``.

    Pensado para las candidatas finalistas y SPY/QQQ. La carpeta base es
    ``config.PATHS.intraday_dir`` salvo que se indique ``base_dir``. Un fallo de
    escritura en un ticker se registra y no detiene al resto.
    """
    folder = _intraday_folder(session_date, base_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ticker, df in bars.items():
        if df is None or len(df) == 0:
            logger.warning("%s: sin barras que guardar en %s", ticker, folder)
            continue
        path = folder / f"{_safe_name(ticker)}.csv.gz"
        try:
            df.to_csv(path, compression="gzip", index_label="datetime")
            written.append(path)
        except OSError as exc:
            logger.error("%s: no se pudieron guardar las barras intradía (%s)", ticker, exc)
    return written


def load_intraday_bars(ticker: str, session_date: date,
                       base_dir: str | Path | None = None) -> pd.DataFrame:
    """Lee las barras guardadas por ``save_intraday_bars`` (DataFrame vacío si no hay)."""
    path = _intraday_folder(session_date, base_dir) / f"{_safe_name(ticker)}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return to_ny_index(df)
