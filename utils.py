"""Utilidades comunes del agente.

Contiene:
- logging (consola + ``logs/agent.log`` con rotación);
- helpers de hora en Nueva York y guarda horaria;
- reintentos con backoff exponencial;
- lectura/escritura segura de JSON;
- lectura del universo de tickers desde Excel (PASO 0);
- calendario bursátil US (fines de semana, festivos, cierres anticipados);
- comprobación de idempotencia (¿ya se envió el informe de hoy?).
"""
from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
import tempfile
import time as time_module
from dataclasses import dataclass, field
from datetime import date, datetime, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
from zoneinfo import ZoneInfo

import config

logger = logging.getLogger("trading_agent.utils")

NY_TZ = ZoneInfo(config.TIMEZONE)
F = TypeVar("F", bound=Callable[..., Any])

_HANDLER_FLAG = "_trading_agent_handler"


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(
    log_file: Path,
    level: str = "INFO",
    max_bytes: int = 1_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Configura el logging a consola y a archivo (con rotación).

    Es seguro llamarla varias veces: elimina antes los manejadores que puso
    ella misma, así no se duplican los mensajes.
    """
    shutdown_logging()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_FLAG, True)
        root.addHandler(handler)

    # Librerías muy "habladoras": solo mostramos avisos y errores.
    for noisy in ("urllib3", "yfinance", "peewee", "httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("trading_agent")


def shutdown_logging() -> None:
    """Quita y cierra los manejadores de logging creados por ``setup_logging``."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            root.removeHandler(handler)
            handler.close()


# --------------------------------------------------------------------------
# Hora en Nueva York
# --------------------------------------------------------------------------
def now_ny() -> datetime:
    """Devuelve la fecha y hora actuales en Nueva York."""
    return datetime.now(NY_TZ)


def to_ny(moment: datetime) -> datetime:
    """Convierte un datetime con zona horaria a hora de Nueva York."""
    if moment.tzinfo is None:
        raise ValueError("Se necesita un datetime con zona horaria (tz-aware).")
    return moment.astimezone(NY_TZ)


def is_within_window(moment: datetime, start: time, end: time) -> bool:
    """True si la hora local de Nueva York de ``moment`` está en [start, end]."""
    local_time = to_ny(moment).time()
    return start <= local_time <= end


def format_hhmm(value: time) -> str:
    """Formatea una hora como HH:MM."""
    return value.strftime("%H:%M")


# --------------------------------------------------------------------------
# Reintentos
# --------------------------------------------------------------------------
def retry(
    attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorador: reintenta una función con espera exponencial.

    Si tras ``attempts`` intentos sigue fallando, relanza la última excepción.
    """
    total_attempts = max(1, attempts)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, total_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == total_attempts:
                        logger.error(
                            "%s falló tras %d intentos: %s",
                            func.__name__, total_attempts, exc,
                        )
                        raise
                    logger.warning(
                        "%s falló (intento %d/%d): %s. Reintento en %.1fs",
                        func.__name__, attempt, total_attempts, exc, delay,
                    )
                    time_module.sleep(delay)
                    delay *= backoff
            raise RuntimeError("retry: no se ejecutó ningún intento")

        return wrapper  # type: ignore[return-value]

    return decorator


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------
def sanitize_for_json(obj: Any) -> Any:
    """Convierte recursivamente un objeto a algo que JSON acepte.

    - NaN e infinitos pasan a ``None``.
    - Fechas y horas pasan a texto ISO.
    - Tipos de numpy pasan a tipos básicos de Python.
    """
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return sanitize_for_json(obj.item())
        except (ValueError, TypeError):
            return str(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def write_json_atomic(path: Path, data: Any) -> None:
    """Escribe JSON de forma segura (archivo temporal + reemplazo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(sanitize_for_json(data), handle, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path, default: Any = None) -> Any:
    """Lee un JSON. Si no existe o está corrupto, devuelve ``default``."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("No se pudo leer %s: %s", path, exc)
        return default


def mask_secret(value: str) -> str:
    """Oculta un secreto para poder mostrarlo en logs."""
    if not value:
        return "(vacío)"
    if len(value) <= 8:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


# --------------------------------------------------------------------------
# Universo de tickers (PASO 0)
# --------------------------------------------------------------------------
class UniverseError(Exception):
    """Error al leer la lista de tickers."""


@dataclass(frozen=True)
class UniverseResult:
    """Resultado de leer la lista de tickers."""

    tickers: List[str]
    source_file: Path
    raw_rows: int
    empty_dropped: int
    duplicates_dropped: int
    invalid_dropped: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Número de tickers válidos."""
        return len(self.tickers)


_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def _is_valid_ticker(value: str) -> bool:
    """Un ticker válido: 1-10 caracteres, con al menos una letra."""
    return bool(_TICKER_PATTERN.match(value)) and any(c.isalpha() for c in value)


def to_yahoo_symbol(ticker: str) -> str:
    """Convierte un ticker al formato de Yahoo Finance (BRK.B -> BRK-B)."""
    return ticker.replace(".", "-")


def _find_ticker_column(
    sheet: pd.DataFrame, header_name: str, scan_rows: int
) -> Optional[Tuple[int, int]]:
    """Busca la celda de cabecera ``Ticker`` en las primeras filas de una hoja."""
    target = header_name.strip().lower()
    row_limit = min(scan_rows, len(sheet))
    for row_idx in range(row_limit):
        for col_idx in range(sheet.shape[1]):
            value = sheet.iat[row_idx, col_idx]
            if isinstance(value, str) and value.strip().lower() == target:
                return row_idx, col_idx
    return None


def load_universe(path: Path) -> UniverseResult:
    """Lee el archivo Excel y devuelve los tickers válidos.

    Pasos:
    1. Lee el Excel (todas las hojas) desde ``path``.
    2. Busca la cabecera ``Ticker`` (sin distinguir mayúsculas ni espacios).
    3. Elimina vacíos, duplicados y valores no válidos.
    4. Normaliza a mayúsculas (y quita un ``$`` inicial si lo hay).

    Lanza ``UniverseError`` si el archivo o la columna no existen, o si no
    queda ningún ticker válido.
    """
    if not path.is_file():
        raise UniverseError(f"No se encuentra el archivo de tickers: {path}")

    try:
        sheets: Dict[str, pd.DataFrame] = pd.read_excel(
            path, sheet_name=None, header=None, dtype=object, engine="openpyxl"
        )
    except Exception as exc:  # archivo corrupto, formato no válido, etc.
        raise UniverseError(f"No se pudo leer {path.name}: {exc}") from exc

    location: Optional[Tuple[pd.DataFrame, int, int]] = None
    for sheet in sheets.values():
        found = _find_ticker_column(
            sheet, config.UNIVERSE.ticker_column, config.UNIVERSE.header_scan_rows
        )
        if found is not None:
            location = (sheet, found[0], found[1])
            break
    if location is None:
        raise UniverseError(
            f"No se encontró la columna '{config.UNIVERSE.ticker_column}' "
            f"en {path.name}. Escribe 'Ticker' en la primera celda de la columna."
        )

    sheet, header_row, column = location
    raw_values = sheet.iloc[header_row + 1:, column].tolist()

    tickers: List[str] = []
    seen = set()
    empty_dropped = 0
    duplicates_dropped = 0
    invalid: List[str] = []

    for value in raw_values:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            empty_dropped += 1
            continue
        text = str(value).strip().upper().lstrip("$").strip()
        if not text:
            empty_dropped += 1
            continue
        if not _is_valid_ticker(text):
            invalid.append(text)
            continue
        if text in seen:
            duplicates_dropped += 1
            continue
        seen.add(text)
        tickers.append(text)

    if not tickers:
        raise UniverseError(f"No hay tickers válidos en {path.name}.")

    if invalid:
        logger.warning("Valores no válidos ignorados en la lista: %s", invalid)

    return UniverseResult(
        tickers=tickers,
        source_file=path,
        raw_rows=len(raw_values),
        empty_dropped=empty_dropped,
        duplicates_dropped=duplicates_dropped,
        invalid_dropped=invalid,
    )


# --------------------------------------------------------------------------
# Calendario bursátil US
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionInfo:
    """Información de la sesión bursátil de una fecha."""

    date: date
    is_session: bool
    reason: str
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    is_early_close: bool = False


@functools.lru_cache(maxsize=1)
def _nyse_calendar() -> Any:
    """Devuelve (y cachea) el calendario NYSE de pandas_market_calendars."""
    import pandas_market_calendars as mcal  # import diferido

    return mcal.get_calendar("NYSE")


def get_session_info(target: date) -> SessionInfo:
    """Indica si ``target`` es sesión bursátil normal, festivo o cierre anticipado.

    Detecta fines de semana, festivos de NYSE y sesiones con cierre anticipado.
    Si el calendario no está disponible (por ejemplo, la librería no está
    instalada), se registra un aviso y se asume sesión normal en días laborables.
    """
    if target.weekday() >= 5:
        return SessionInfo(target, False, "fin de semana")

    try:
        schedule = _nyse_calendar().schedule(
            start_date=target.isoformat(), end_date=target.isoformat()
        )
    except Exception as exc:
        logger.warning(
            "Calendario NYSE no disponible (%s). Se asume sesión normal.", exc
        )
        return SessionInfo(
            target, True, "calendario no disponible; se asume sesión normal"
        )

    if schedule.empty:
        return SessionInfo(target, False, "festivo bursátil (NYSE cerrado)")

    row = schedule.iloc[0]
    open_ny = row["market_open"].tz_convert(config.TIMEZONE).to_pydatetime()
    close_ny = row["market_close"].tz_convert(config.TIMEZONE).to_pydatetime()
    early = close_ny.time().replace(tzinfo=None) < config.SCHEDULE.regular_close
    reason = "cierre anticipado" if early else "sesión normal"
    return SessionInfo(target, True, reason, open_ny, close_ny, early)


# --------------------------------------------------------------------------
# Idempotencia
# --------------------------------------------------------------------------
def already_sent(history_dir: Path, target: date) -> bool:
    """True si el histórico de ``target`` existe y está marcado como enviado."""
    data = read_json(history_dir / f"{target.isoformat()}.json")
    return isinstance(data, dict) and data.get("sent") is True
