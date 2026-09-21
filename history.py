"""history.py — Histórico diario (JSON) y tabla plana (CSV) de candidatas.

* ``data/history/YYYY-MM-DD.json``: todo lo necesario para auditar el día (universo,
  datos de mercado, indicadores, catalizadores, ``filter_log``, candidatas, respuesta
  cruda y validada de Gemini, selección final, marcas de tiempo, fuentes y estado de
  envío). La clave ``sent`` está en el nivel superior porque ``utils.already_sent`` la
  lee para la idempotencia.
* ``data/history.csv``: una fila por candidata y día, con columnas planas. Reejecutar la
  misma fecha **reemplaza** las filas de ese día (nunca las duplica).

Este módulo también contiene los accesores que comparten ``report.py`` y
``backtest.py`` para leer candidatas y picks sin depender de una forma exacta.

Contrato de datos que se espera (claves que se leen)
----------------------------------------------------
Candidata (``candidates[i]``): ``ticker``, ``score`` (o ``score_detail.total``),
``direction``, ``record`` (registro de ``technical_analysis.analyze_ticker``),
``news`` (con ``catalyst_confirmed`` y ``catalyst_text``) y ``score_detail.flags``.
Pick validado (``final_selection[i]``): los campos del esquema de Gemini más los que
calcula Python: ``rr`` (R/R) y ``max_entry_price`` (precio máximo de entrada válido).
"""
from __future__ import annotations

import csv
import dataclasses
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import config
from technical_analysis import NA, is_num
from utils import now_ny, read_json, write_json_atomic

logger = logging.getLogger("trading_agent.history")

# Nombres alternativos aceptados para los campos que calcula Python en cada pick.
PICK_ALIASES: dict[str, tuple[str, ...]] = {
    "rr": ("rr", "risk_reward", "r_r", "rr_ratio"),
    "max_entry_price": ("max_entry_price", "max_entry", "max_valid_entry",
                        "precio_maximo_entrada"),
}

CSV_COLUMNS = [
    "date", "ticker", "rank", "direction", "score", "selected", "decision", "confidence",
    "catalyst_confirmed", "catalyst", "premarket_quality", "prev_close", "premarket_price",
    "gap_pct", "premarket_volume", "rvol", "pm_pct_of_adv", "rsi14", "sma20", "sma50",
    "atr14", "atr_pct", "rs_pm_vs_spy", "rs_pm_vs_qqq", "fade_flags", "fade_risk",
    "entry_low", "entry_high", "stop", "target_1", "target_2", "rr", "max_entry_price",
    "sent",
]


# --------------------------------------------------------------------------- #
# Accesores compartidos
# --------------------------------------------------------------------------- #
def to_plain(obj: Any) -> Any:
    """Convierte dataclasses y objetos con ``to_dict`` en diccionarios y listas."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_plain(obj.to_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_plain(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]
    return obj


def analysis_record(candidate: Any) -> Mapping[str, Any]:
    """Registro de ``analyze_ticker`` de una candidata (``candidate["record"]``)."""
    if not isinstance(candidate, Mapping):
        return {}
    record = candidate.get("record")
    if isinstance(record, Mapping):
        return record
    if "premarket" in candidate or "indicators" in candidate:
        return candidate
    return {}


def candidate_score(candidate: Mapping[str, Any]) -> float | str:
    """Puntuación de una candidata (``score`` o ``score_detail.total``)."""
    for value in (candidate.get("score"), (candidate.get("score_detail") or {}).get("total")):
        if is_num(value):
            return float(value)
    return NA


def pick_field(pick: Mapping[str, Any], name: str) -> Any:
    """Valor de un campo de un pick, aceptando los nombres alternativos de ``PICK_ALIASES``."""
    for key in PICK_ALIASES.get(name, (name,)):
        if key in pick and pick[key] is not None:
            return pick[key]
    return NA


def find_level(record: Mapping[str, Any], label: Any) -> dict[str, Any] | None:
    """Fila de la tabla de niveles con esa etiqueta (None si no existe)."""
    if not isinstance(label, str):
        return None
    for row in record.get("levels", []) or []:
        if isinstance(row, Mapping) and row.get("label") == label:
            return dict(row)
    return None


def _get(mapping: Any, *path: str) -> Any:
    """Acceso anidado tolerante: devuelve ``"N/A"`` si falta cualquier eslabón."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current or current[key] is None:
            return NA
        current = current[key]
    return current


# --------------------------------------------------------------------------- #
# Escritura atómica de texto y CSV
# --------------------------------------------------------------------------- #
def write_text_atomic(path: Path, text: str) -> None:
    """Escribe un archivo de texto de forma segura (temporal + reemplazo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _csv_cell(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return NA if value != value or value in (float("inf"), float("-inf")) else value
    return value


def upsert_csv_rows(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str],
    date_key: str = "date", replace_dates: Sequence[str] = (),
) -> int:
    """Añade filas a un CSV reemplazando las de las mismas fechas (sin duplicar).

    Se reemplazan las filas existentes de las fechas de ``rows`` y de ``replace_dates``
    (útil cuando una reejecución ya no produce ninguna fila para esa fecha). Conserva las
    filas de otras fechas y las columnas extra que ya tuviera el archivo. Devuelve el
    número total de filas escritas.
    """
    existing: list[dict[str, str]] = []
    extra_columns: list[str] = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                extra_columns = [c for c in (reader.fieldnames or []) if c not in columns]
                existing = [dict(row) for row in reader]
        except (OSError, csv.Error) as exc:
            logger.warning("No se pudo leer %s (%s); se recreará", path, exc)

    new_dates = {str(row.get(date_key)) for row in rows} | {str(d) for d in replace_dates}
    kept = [row for row in existing if row.get(date_key) not in new_dates]
    fieldnames = [*columns, *extra_columns]

    import io  # noqa: PLC0415 - solo aquí

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in kept:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    for row in rows:
        writer.writerow({k: _csv_cell(row.get(k, NA)) for k in fieldnames})
    write_text_atomic(path, buffer.getvalue())
    return len(kept) + len(rows)


# --------------------------------------------------------------------------- #
# Registro del día
# --------------------------------------------------------------------------- #
@dataclass
class HistoryRecord:
    """Contenido del histórico de un día (ver el docstring del módulo)."""

    date: date
    session_type: str = "sesión normal"
    universe: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] | None = None
    candidates: Sequence[Mapping[str, Any]] = field(default_factory=list)
    filter_log: Sequence[Mapping[str, Any]] = field(default_factory=list)
    gemini_raw: Any = None
    gemini_validated: Mapping[str, Any] | None = None
    final_selection: Sequence[Mapping[str, Any]] = field(default_factory=list)
    no_trade: bool = False
    no_trade_reason: str = ""
    timestamps: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, Any] = field(default_factory=dict)
    sent: bool = False
    send_status: Mapping[str, Any] = field(default_factory=dict)
    report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Diccionario con las claves del histórico (indicadores y catalizadores derivados)."""
        candidates = to_plain(list(self.candidates))
        indicators: dict[str, Any] = {}
        premarket: dict[str, Any] = {}
        catalysts: dict[str, Any] = {}
        for cand in candidates:
            ticker = cand.get("ticker")
            if not ticker:
                continue
            record = analysis_record(cand)
            indicators[ticker] = record.get("indicators", {})
            premarket[ticker] = record.get("premarket", {})
            news = cand.get("news") or {}
            catalysts[ticker] = {
                "catalyst_confirmed": news.get("catalyst_confirmed", NA),
                "catalyst_text": news.get("catalyst_text", NA),
                "catalyst": news.get("catalyst"),
                "sources": news.get("sources", []),
            }
        return {
            "date": self.date.isoformat(),
            "session_type": self.session_type,
            "universe": to_plain(self.universe),
            "market_data": {"context": to_plain(self.context), "premarket": premarket},
            "indicators": indicators,
            "catalysts": catalysts,
            "filter_log": to_plain(list(self.filter_log)),
            "candidates": candidates,
            "gemini_raw": to_plain(self.gemini_raw),
            "gemini_validated": to_plain(self.gemini_validated),
            "final_selection": to_plain(list(self.final_selection)),
            "no_trade": self.no_trade,
            "no_trade_reason": self.no_trade_reason,
            "timestamps": to_plain(self.timestamps),
            "sources": to_plain(self.sources),
            "sent": bool(self.sent),
            "send_status": to_plain(self.send_status),
            "report_path": self.report_path,
            "run_count": 1,
        }


def history_path(target: date, history_dir: Path | None = None) -> Path:
    """Ruta del JSON de histórico de una fecha."""
    return (history_dir or config.PATHS.history_dir) / f"{target.isoformat()}.json"


def load_history(target: date, history_dir: Path | None = None) -> dict[str, Any] | None:
    """Lee el histórico de una fecha (None si no existe o está corrupto)."""
    data = read_json(history_path(target, history_dir), default=None)
    return data if isinstance(data, dict) else None


def candidate_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Filas planas del CSV (una por candidata) a partir del dict del histórico."""
    picks = {p.get("ticker"): p for p in record.get("final_selection", []) or []
             if isinstance(p, Mapping)}
    rows: list[dict[str, Any]] = []
    for index, cand in enumerate(record.get("candidates", []) or [], start=1):
        if not isinstance(cand, Mapping) or not cand.get("ticker"):
            continue
        ticker = cand["ticker"]
        rec = analysis_record(cand)
        pick = picks.get(ticker, {})
        news = cand.get("news") or {}
        flags = _get(cand, "score_detail", "flags")
        rows.append({
            "date": record.get("date"), "ticker": ticker, "rank": index,
            "direction": cand.get("direction") or pick.get("direction") or NA,
            "score": candidate_score(cand),
            "selected": ticker in picks,
            "decision": pick.get("decision", NA), "confidence": pick.get("confidence", NA),
            "catalyst_confirmed": news.get("catalyst_confirmed", NA),
            "catalyst": news.get("catalyst_text", NA),
            "premarket_quality": _get(rec, "premarket", "premarket_quality"),
            "prev_close": _get(rec, "premarket", "prev_close"),
            "premarket_price": _get(rec, "premarket", "premarket_price"),
            "gap_pct": _get(rec, "gap_pct"),
            "premarket_volume": _get(rec, "premarket", "premarket_volume"),
            "rvol": _get(rec, "rvol", "rvol"),
            "pm_pct_of_adv": _get(rec, "pm_pct_of_adv"),
            "rsi14": _get(rec, "indicators", "rsi14"),
            "sma20": _get(rec, "indicators", "sma20"),
            "sma50": _get(rec, "indicators", "sma50"),
            "atr14": _get(rec, "indicators", "atr14"),
            "atr_pct": _get(rec, "indicators", "atr_pct"),
            "rs_pm_vs_spy": _get(rec, "relative_strength", "rs_pm_vs_spy"),
            "rs_pm_vs_qqq": _get(rec, "relative_strength", "rs_pm_vs_qqq"),
            "fade_flags": "|".join(flags) if isinstance(flags, list) else NA,
            "fade_risk": pick.get("fade_risk", NA),
            "entry_low": pick.get("entry_zone_low", NA),
            "entry_high": pick.get("entry_zone_high", NA),
            "stop": pick.get("stop", NA),
            "target_1": pick.get("target_1", NA), "target_2": pick.get("target_2", NA),
            "rr": pick_field(pick, "rr"),
            "max_entry_price": pick_field(pick, "max_entry_price"),
            "sent": bool(record.get("sent")),
        })
    return rows


def save_history(
    record: HistoryRecord | Mapping[str, Any],
    history_dir: Path | None = None, csv_path: Path | None = None,
) -> Path:
    """Guarda el JSON del día y actualiza ``history.csv`` sin duplicar filas.

    Si ya existía un histórico de esa fecha (reejecución) se sobrescribe y se
    incrementa ``run_count``. Devuelve la ruta del JSON.
    """
    data = record.to_dict() if isinstance(record, HistoryRecord) else dict(to_plain(record))
    target = date.fromisoformat(str(data["date"]))
    path = history_path(target, history_dir)

    previous = read_json(path, default=None)
    if isinstance(previous, dict):
        data["run_count"] = int(previous.get("run_count", 1)) + 1
    data.setdefault("run_count", 1)
    data.setdefault("saved_at", now_ny().isoformat())

    write_json_atomic(path, data)
    total = upsert_csv_rows(csv_path or config.PATHS.history_csv, candidate_rows(data),
                            CSV_COLUMNS, replace_dates=[str(data["date"])])
    logger.info("Histórico guardado en %s (%d filas en el CSV)", path, total)
    return path


def update_send_status(
    target: date, sent: bool, detail: Mapping[str, Any] | None = None,
    history_dir: Path | None = None, csv_path: Path | None = None,
) -> bool:
    """Actualiza ``sent`` (y el detalle del envío) de un histórico ya guardado.

    Devuelve False si no existe el histórico de esa fecha.
    """
    data = load_history(target, history_dir)
    if data is None:
        logger.warning("No hay histórico de %s para actualizar el estado de envío", target)
        return False
    data["sent"] = bool(sent)
    data["send_status"] = to_plain(detail or {})
    timestamps = dict(data.get("timestamps") or {})
    if sent:
        timestamps["sent"] = now_ny().isoformat()
    data["timestamps"] = timestamps
    write_json_atomic(history_path(target, history_dir), data)
    upsert_csv_rows(csv_path or config.PATHS.history_csv, candidate_rows(data), CSV_COLUMNS,
                    replace_dates=[str(data["date"])])
    return True
