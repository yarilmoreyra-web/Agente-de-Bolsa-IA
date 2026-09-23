"""Histórico de ejecuciones: un JSON enriquecido por día y un CSV acumulado.

El JSON diario (``data/history/AAAA-MM-DD.json``) guarda todo lo que produjo
la sesión (universo, contexto de mercado, candidatas con sus indicadores y
catalizadores, log del filtro, respuesta cruda y validada de Gemini,
selección final, timestamps y fuentes) más el resultado del envío a
Telegram. El campo ``sent`` es el que usa ``utils.already_sent`` para no
repetir un informe ya entregado.

El CSV (``data/history.csv``) tiene una fila por candidata (ninguna si no
hubo candidatas ese día) y sirve para revisar el histórico rápidamente sin
abrir cada JSON.

Guardar el histórico y marcar el envío son dos pasos separados a propósito
(``save_history`` y luego ``update_send_status``): así el histórico queda
guardado aunque el envío a Telegram falle o el proceso se interrumpa antes.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import config
import utils

logger = logging.getLogger("trading_agent.history")

# Columnas del CSV acumulado: una fila por candidata y día.
CSV_COLUMNS = [
    "date", "ticker", "rank", "score", "selected", "decision", "rr",
    "max_entry_price", "stop", "gap_pct", "rvol", "premarket_quality",
    "fade_flags", "catalyst_confirmed", "rs_pm_vs_spy", "sent",
]

# Alias conocidos para acceder a los campos de un "pick" con distintos
# nombres, según de dónde venga (Gemini, Python, versiones antiguas).
_PICK_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "rr": ("rr", "risk_reward"),
    "max_entry_price": ("max_entry_price", "max_entry"),
}


def history_path(session_date: date, history_dir: Optional[Path] = None) -> Path:
    """Ruta del histórico JSON de una fecha."""
    base = Path(history_dir) if history_dir is not None else config.PATHS.history_dir
    return base / f"{session_date.isoformat()}.json"


@dataclass
class HistoryRecord:
    """Lo que se guarda en el histórico de un día.

    ``candidates`` es la lista de candidatas tal y como las produce
    ``candidate_filter.candidates_payload`` (cada una con su análisis técnico
    y sus noticias). ``final_selection`` es la lista de picks ya validados
    (los que de verdad entran en el informe).
    """

    date: date
    universe: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    filter_log: List[Dict[str, Any]] = field(default_factory=list)
    gemini_raw: Dict[str, Any] = field(default_factory=dict)
    gemini_validated: Dict[str, Any] = field(default_factory=dict)
    final_selection: List[Dict[str, Any]] = field(default_factory=list)
    timestamps: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, Any] = field(default_factory=dict)
    no_trade: bool = False
    no_trade_reason: str = ""


# --------------------------------------------------------------------------- #
# Accesores defensivos (una candidata puede venir de distintos productores)
# --------------------------------------------------------------------------- #
def analysis_record(candidate: Any) -> Any:
    """Devuelve el registro analítico (indicadores, pre-market, niveles...) de una candidata.

    Acepta la candidata completa (con clave ``record`` o, en el payload real
    de ``candidate_filter``, ``analysis``) o directamente el propio registro
    analítico, que se devuelve tal cual. Cualquier otra cosa da ``{}``.
    """
    if not isinstance(candidate, dict):
        return {}
    for key in ("record", "analysis"):
        value = candidate.get(key)
        if isinstance(value, dict):
            return value
    return candidate


def candidate_score(candidate: Any) -> Any:
    """Puntuación de una candidata (``score``, o el total de su desglose); "N/A" si no hay."""
    if not isinstance(candidate, dict):
        return "N/A"
    score = candidate.get("score")
    if isinstance(score, (int, float)):
        return score
    for key in ("score_detail", "score_breakdown"):
        detail = candidate.get(key)
        if isinstance(detail, dict) and isinstance(detail.get("total"), (int, float)):
            return detail["total"]
    return "N/A"


def find_level(record: Any, level_name: str) -> Any:
    """Busca en ``record['levels']`` el nivel técnico con esa etiqueta (soporte/resistencia)."""
    if not isinstance(record, dict):
        return None
    levels = record.get("levels")
    if isinstance(levels, dict):
        return levels.get(level_name)
    if isinstance(levels, (list, tuple)):
        for level in levels:
            if isinstance(level, dict) and level.get("label") == level_name:
                return level
    return None


def pick_field(pick: Any, key: str, default: Any = "N/A") -> Any:
    """Busca ``key`` en un pick probando alias conocidos (p.ej. ``rr``/``risk_reward``)."""
    if not isinstance(pick, dict):
        return default
    for alias in _PICK_FIELD_ALIASES.get(key, (key,)):
        if alias in pick and pick[alias] is not None:
            return pick[alias]
    return default


def to_plain(obj: Any) -> Any:
    """Convierte dataclasses, objetos con ``to_dict()`` y estructuras anidadas a tipos nativos.

    Necesario antes de guardar en JSON: las candidatas o las noticias pueden
    llegar como dataclasses (p.ej. ``TickerNews``) en lugar de dicts.
    """
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_plain(to_dict())
    if hasattr(obj, "__dataclass_fields__"):
        return to_plain(asdict(obj))
    if isinstance(obj, dict):
        return {key: to_plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(item) for item in obj]
    return obj


# --------------------------------------------------------------------------- #
# Guardado del histórico (JSON + CSV)
# --------------------------------------------------------------------------- #
def _derive_market_data(
    candidates: List[Dict[str, Any]], context: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """A partir de las candidatas, separa pre-market/indicadores/catalizadores por ticker."""
    premarket: Dict[str, Any] = {}
    indicators: Dict[str, Any] = {}
    catalysts: Dict[str, Any] = {}
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        ticker = candidate.get("ticker")
        if not ticker:
            continue
        record = analysis_record(candidate)
        premarket[ticker] = record.get("premarket", {}) if isinstance(record, dict) else {}
        indicators[ticker] = record.get("indicators", {}) if isinstance(record, dict) else {}
        catalysts[ticker] = candidate.get("news", {})
    return {"premarket": premarket, "context": context}, indicators, catalysts


def _next_run_count(path: Path) -> int:
    """Cuántas veces se ha guardado ya el histórico de esta fecha (para incrementarlo)."""
    previous = utils.read_json(path)
    if isinstance(previous, dict):
        try:
            return int(previous.get("run_count", 0)) + 1
        except (TypeError, ValueError):
            return 1
    return 1


def save_history(
    record: HistoryRecord,
    history_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> Path:
    """Guarda (o sobrescribe) el histórico JSON del día y actualiza el CSV acumulado.

    Es seguro llamarla varias veces para la misma fecha (por ejemplo al
    reintentar con ``--force``): el JSON se sobrescribe incrementando
    ``run_count``, y en el CSV se reemplazan solo las filas de esa fecha.
    ``sent`` empieza siempre en ``False``; se marca aparte con
    ``update_send_status`` una vez se sabe si el envío a Telegram funcionó.
    """
    history_dir = Path(history_dir) if history_dir is not None else config.PATHS.history_dir
    csv_path = Path(csv_path) if csv_path is not None else config.PATHS.history_csv

    session_date = record.date
    if isinstance(session_date, datetime):
        session_date = session_date.date()

    path = history_path(session_date, history_dir)
    run_count = _next_run_count(path)
    market_data, indicators, catalysts = _derive_market_data(record.candidates, record.context)

    payload: Dict[str, Any] = {
        "date": session_date,
        "universe": record.universe,
        "market_data": market_data,
        "indicators": indicators,
        "catalysts": catalysts,
        "filter_log": record.filter_log,
        "candidates": record.candidates,
        "gemini_raw": record.gemini_raw,
        "gemini_validated": record.gemini_validated,
        "final_selection": record.final_selection,
        "timestamps": record.timestamps,
        "sources": record.sources,
        "no_trade": record.no_trade,
        "no_trade_reason": record.no_trade_reason,
        "sent": False,
        "send_status": None,
        "run_count": run_count,
    }
    payload = to_plain(payload)
    utils.write_json_atomic(path, payload)
    logger.info("Histórico guardado en %s (run_count=%d).", path, run_count)

    rows = _rows_for_date(record.candidates, record.final_selection, session_date, sent=False)
    _rewrite_csv(csv_path, session_date, rows)
    return path


def load_history(session_date: date, history_dir: Optional[Path] = None) -> Any:
    """Lee el histórico JSON de una fecha (o ``None`` si no existe o está corrupto)."""
    base = Path(history_dir) if history_dir is not None else config.PATHS.history_dir
    return utils.read_json(history_path(session_date, base))


def update_send_status(
    session_date: date,
    sent: bool,
    status: Optional[Dict[str, Any]] = None,
    history_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> bool:
    """Marca en el histórico (JSON y CSV) el resultado del envío a Telegram.

    Devuelve ``True`` si había histórico de esa fecha y se actualizó;
    ``False`` (con un aviso en el log) si no existía nada que actualizar.
    """
    history_dir = Path(history_dir) if history_dir is not None else config.PATHS.history_dir
    csv_path = Path(csv_path) if csv_path is not None else config.PATHS.history_csv

    path = history_path(session_date, history_dir)
    data = utils.read_json(path)
    if not isinstance(data, dict):
        logger.warning(
            "No se pudo actualizar el estado de envío: no hay histórico para %s.",
            session_date,
        )
        return False

    data["sent"] = bool(sent)
    data["send_status"] = status if status is not None else data.get("send_status")
    timestamps = data.get("timestamps")
    if not isinstance(timestamps, dict):
        timestamps = {}
    if sent:
        timestamps["sent"] = utils.now_ny().isoformat()
    data["timestamps"] = timestamps

    utils.write_json_atomic(path, data)
    logger.info("Estado de envío actualizado para %s: sent=%s.", session_date, sent)

    _update_csv_sent_column(csv_path, session_date, sent)
    return True


# --------------------------------------------------------------------------- #
# CSV acumulado
# --------------------------------------------------------------------------- #
def _fmt_csv(value: Any) -> str:
    """Formatea un valor para una celda de CSV: bools en minúsculas, ``N/A`` si falta."""
    if value is None or value == "N/A":
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _score_sort_key(candidate: Mapping[str, Any]) -> float:
    score = candidate_score(candidate)
    return score if isinstance(score, (int, float)) else float("-inf")


def _rows_for_date(
    candidates: List[Dict[str, Any]],
    final_selection: List[Dict[str, Any]],
    session_date: date,
    sent: bool,
) -> List[Dict[str, str]]:
    """Una fila de CSV por candidata, ordenadas por puntuación descendente."""
    picks_by_ticker = {
        pick.get("ticker"): pick
        for pick in (final_selection or [])
        if isinstance(pick, dict) and pick.get("ticker")
    }
    ordered = sorted(
        (c for c in (candidates or []) if isinstance(c, dict) and c.get("ticker")),
        key=_score_sort_key,
        reverse=True,
    )

    rows: List[Dict[str, str]] = []
    for rank, candidate in enumerate(ordered, start=1):
        ticker = candidate["ticker"]
        record = analysis_record(candidate)
        pick = picks_by_ticker.get(ticker)
        selected = ticker in picks_by_ticker

        premarket = record.get("premarket", {}) if isinstance(record, dict) else {}
        rvol_info = record.get("rvol") if isinstance(record, dict) else None
        rvol_value = rvol_info.get("rvol") if isinstance(rvol_info, dict) else rvol_info
        rs_info = record.get("relative_strength") if isinstance(record, dict) else None
        rs_value = rs_info.get("rs_pm_vs_spy") if isinstance(rs_info, dict) else None

        flags: List[str] = []
        for key in ("score_detail", "score_breakdown"):
            detail = candidate.get(key)
            if isinstance(detail, dict) and isinstance(detail.get("flags"), (list, tuple)):
                flags = [str(flag) for flag in detail["flags"]]
                break

        news = to_plain(candidate.get("news")) or {}
        news = news if isinstance(news, dict) else {}

        rows.append({
            "date": session_date.isoformat(),
            "ticker": ticker,
            "rank": str(rank),
            "score": _fmt_csv(candidate_score(candidate)),
            "selected": _fmt_csv(selected),
            "decision": _fmt_csv(pick_field(pick, "decision")),
            "rr": _fmt_csv(pick_field(pick, "rr")),
            "max_entry_price": _fmt_csv(pick_field(pick, "max_entry_price")),
            "stop": _fmt_csv(pick_field(pick, "stop")),
            "gap_pct": _fmt_csv(record.get("gap_pct")) if isinstance(record, dict) else "N/A",
            "rvol": _fmt_csv(rvol_value),
            "premarket_quality": _fmt_csv(premarket.get("premarket_quality")),
            "fade_flags": ";".join(flags) if flags else "N/A",
            "catalyst_confirmed": _fmt_csv(bool(news.get("catalyst_confirmed"))),
            "rs_pm_vs_spy": _fmt_csv(rs_value),
            "sent": _fmt_csv(sent),
        })
    return rows


def upsert_csv_rows(
    csv_path: Path,
    rows: List[Dict[str, Any]],
    columns: List[str],
    replace_dates: Optional[List[str]] = None,
    date_field: str = "date",
) -> None:
    """Reescribe un CSV reemplazando las filas cuyo ``date_field`` esté en
    ``replace_dates`` por ``rows`` (nunca duplica). Añade columnas de
    ``columns`` que falten y conserva cualquier columna extra ya presente en
    el archivo. Con ``replace_dates=None`` se reemplaza por la fecha de cada
    fila nueva individualmente. Usada por ``history.py`` y ``backtest.py``.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if replace_dates is not None:
        replace_set = set(replace_dates)
    else:
        replace_set = {row.get(date_field) for row in rows if row.get(date_field)}

    existing_fieldnames: List[str] = []
    existing_rows: List[Dict[str, Any]] = []
    if csv_path.is_file():
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = [row for row in reader if row.get(date_field) not in replace_set]

    fieldnames = list(existing_fieldnames)
    for column in columns:
        if column not in fieldnames:
            fieldnames.append(column)

    all_rows = existing_rows + list(rows)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    logger.info("CSV actualizado: %s", csv_path)


def _rewrite_csv(csv_path: Path, session_date: date, new_rows: List[Dict[str, str]]) -> None:
    """Reescribe el CSV de historia reemplazando solo las filas de ``session_date``."""
    upsert_csv_rows(csv_path, new_rows, CSV_COLUMNS, replace_dates=[session_date.isoformat()])


def _update_csv_sent_column(csv_path: Path, session_date: date, sent: bool) -> None:
    """Actualiza la columna ``sent`` de todas las filas de ``session_date`` ya escritas."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return
    target = session_date.isoformat()
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    changed = False
    for row in rows:
        if row.get("date") == target:
            row["sent"] = "true" if sent else "false"
            changed = True
    if not changed:
        return

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
