"""diagnostico.py — Por qué salieron N candidatas (o ninguna) en una sesión.

Herramienta de diagnóstico, NO forma parte del pipeline de producción
(``main.py`` no la importa ni la usa). Responde una pregunta muy concreta:
"¿de verdad no había movimiento pre-market hoy, o el filtro se comió algo que
debería haber pasado, o la hora a la que se tomó el snapshot ya no era
pre-market de verdad?"

Qué hace
--------
1. Descarga una sola vez los datos diarios, el histórico intradía (para RVOL)
   y las barras de 5 minutos de todo el día de la fecha indicada.
2. Con esas mismas barras, reconstruye el snapshot pre-market en **varias
   horas distintas** (por defecto, la hora prevista de snapshot y el cierre
   de la ventana pre-market, justo antes de la apertura). Esto es posible sin
   volver a pedir nada a Yahoo porque ``extract_premarket_snapshot`` solo
   filtra por hora de corte las barras que ya se descargaron.
3. Para cada hora, calcula el gap, el RVOL, la calidad del dato y el motivo
   exacto de exclusión de las puertas duras (``candidate_filter.apply_hard_gates``)
   de cada ticker, y muestra el ranking por |gap%|, pase o no el filtro.

Qué NO hace (a propósito, para que sea rápido)
-----------------------------------------------
No busca noticias ni llama a Gemini: la puntuación final depende del
catalizador confirmado, que aquí no se comprueba. Este script solo evalúa las
puertas duras (precio, volumen, liquidez, calidad del dato, gap mínimo), que
es donde se decide si una candidata llega siquiera a pedir noticias.

Ejemplos de uso
---------------
    python diagnostico.py
    python diagnostico.py --date 2026-09-22 --top 20
    python diagnostico.py --times 08:45,09:15,09:29
    python diagnostico.py --all --universe otra_lista.xlsx
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Sequence

import candidate_filter
import config
import market_data
import technical_analysis as ta
import utils

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s: %(message)s")


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"fecha inválida '{value}': usa AAAA-MM-DD (ej. 2026-09-22)") from exc


def _parse_time(value: str) -> time:
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"hora inválida '{value}': usa HH:MM en hora de Nueva York (ej. 08:45)") from exc


def _parse_times_list(value: str) -> list[time]:
    return [_parse_time(part) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diagnostico.py",
        description="Ranking de gap pre-market y motivo de exclusión del filtro, "
                    "comparando distintas horas de snapshot.",
    )
    parser.add_argument("--date", type=_parse_date, default=None,
                        help="Fecha a diagnosticar (AAAA-MM-DD). Por defecto, hoy en NY.")
    parser.add_argument("--universe", type=Path, default=None,
                        help="Excel de tickers. Por defecto, el de config.PATHS.universe_file.")
    parser.add_argument("--times", type=_parse_times_list, default=None,
                        help="Horas ET a comparar, separadas por comas (ej. 08:45,09:29). "
                             "Por defecto: la hora de snapshot configurada y el cierre de "
                             "la ventana pre-market.")
    parser.add_argument("--top", type=int, default=15,
                        help="Cuántos tickers mostrar por hora (por defecto 15).")
    parser.add_argument("--all", action="store_true",
                        help="Muestra todos los tickers con gap calculable, no solo el top.")
    return parser


def default_times(session_date: date) -> list[time]:
    """La hora de snapshot configurada y el cierre de la ventana pre-market.

    Comparar solo estas dos suele bastar: cualquier hora posterior a la
    apertura da el mismo resultado que justo antes de abrir, porque
    ``resolve_snapshot_ts``/``extract_premarket_snapshot`` recortan siempre al
    cierre de la ventana pre-market (``SCHEDULE.regular_open``).
    """
    schedule = config.SCHEDULE
    edge = (datetime.combine(session_date, schedule.regular_open) - timedelta(minutes=1)).time()
    times = sorted({schedule.snapshot_time, edge})
    return times


def _fmt(value: Any, decimals: int = 2) -> str:
    if not ta.is_num(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def diagnose_at(
    tickers: Sequence[str],
    daily: dict[str, Any],
    intraday_hist: dict[str, Any],
    intraday_today: dict[str, Any],
    session_date: date,
    snapshot_time: time,
) -> dict[str, Any]:
    """Reconstruye el snapshot y el análisis a una hora concreta, sin red."""
    snapshot_ts = datetime.combine(session_date, snapshot_time, tzinfo=utils.NY_TZ)
    snapshots = market_data.build_snapshots(tickers, daily, intraday_today, snapshot_ts)
    analyses, failures = ta.analyze_many(
        tickers, daily, intraday_hist, intraday_today,
        {t: s.to_dict() for t, s in snapshots.items()}, snapshot_ts,
    )

    quality_counts: dict[str, int] = {}
    for snapshot in snapshots.values():
        quality_counts[snapshot.premarket_quality] = \
            quality_counts.get(snapshot.premarket_quality, 0) + 1

    rows = []
    for ticker, analysis in analyses.items():
        gap = analysis.get("gap_pct")
        if not ta.is_num(gap):
            continue
        gate_failures = candidate_filter.apply_hard_gates(analysis)
        rows.append({
            "ticker": ticker,
            "gap_pct": float(gap),
            "quality": analysis["premarket"].get("premarket_quality", "missing"),
            "pm_volume": analysis["premarket"].get("premarket_volume"),
            "rvol": (analysis.get("rvol", {}) or {}).get("rvol"),
            "passed": not gate_failures,
            "reason": "; ".join(gate_failures) if gate_failures else "pasa las puertas duras",
        })
    rows.sort(key=lambda r: abs(r["gap_pct"]), reverse=True)

    return {
        "snapshot_ts": snapshot_ts,
        "quality_counts": quality_counts,
        "rows": rows,
        "no_gap_count": len(analyses) - len(rows),
        "analysis_failures": failures,
    }


def print_report(result: dict[str, Any], top: int, show_all: bool) -> None:
    snapshot_ts = result["snapshot_ts"]
    rows = result["rows"]
    passed = sum(1 for r in rows if r["passed"])

    print(f"\n{'=' * 78}")
    print(f"Snapshot reconstruido a las {snapshot_ts.strftime('%H:%M')} ET "
          f"({snapshot_ts.date()})")
    print(f"Calidad del dato: {result['quality_counts']}")
    print(f"Tickers con gap calculable: {len(rows)} de "
          f"{len(rows) + result['no_gap_count']} analizados | "
          f"pasan las puertas duras: {passed}")
    if result["analysis_failures"]:
        print(f"Tickers con error al analizar: {len(result['analysis_failures'])} "
              f"(ver detalle con --all o en el log)")
    print("-" * 78)

    shown = rows if show_all else rows[:top]
    if not shown:
        print("(ningún ticker tiene un gap calculable a esta hora)")
    else:
        header = f"{'Ticker':<8} {'Gap%':>7} {'Calidad':<8} {'Vol. PM':>10} " \
                 f"{'RVOL':>6} {'Pasa':<5} Motivo"
        print(header)
        print("-" * len(header))
        for row in shown:
            print(
                f"{row['ticker']:<8} {row['gap_pct']:>+7.2f} {row['quality']:<8} "
                f"{_fmt(row['pm_volume'], 0):>10} {_fmt(row['rvol']):>6} "
                f"{'sí' if row['passed'] else 'no':<5} {row['reason']}"
            )
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_date = args.date or utils.now_ny().date()
    times = args.times or default_times(session_date)
    universe_path = args.universe or config.PATHS.universe_file

    try:
        universe = utils.load_universe(universe_path)
    except utils.UniverseError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Diagnóstico de {session_date} | {universe.count} tickers | "
          f"horas a comparar: {', '.join(t.strftime('%H:%M') for t in times)} ET")
    print(f"(reconstruido en {utils.now_ny().strftime('%Y-%m-%d %H:%M')} ET; "
          "las barras de 5 minutos de Yahoo se conservan varias semanas, así "
          "que puede diagnosticar días recientes aunque el mercado ya haya cerrado)")

    print("\nDescargando datos diarios...")
    daily = market_data.download_daily(universe.tickers)
    print(f"  {len(daily.data)} con datos, {len(daily.failed)} sin ellos.")

    print("Descargando histórico intradía (para RVOL)...")
    intraday_hist = market_data.download_intraday_history(universe.tickers)
    print(f"  {len(intraday_hist.data)} con histórico intradía.")

    print(f"Descargando barras de 5 minutos de {session_date} (una sola vez)...")
    today = market_data.download_intraday_today(universe.tickers, session_date)
    print(f"  {len(today.data)} con barras de ese día, {len(today.failed)} sin ellas.")
    if not today.data:
        print("\nNingún ticker tiene barras de ese día: revisa la fecha (¿fue "
              "sesión bursátil?) o si sigue dentro del histórico de 5 minutos "
              "que conserva Yahoo (unas semanas).")
        return 1

    for snapshot_time in times:
        result = diagnose_at(
            universe.tickers, daily.data, intraday_hist.data, today.data,
            session_date, snapshot_time,
        )
        print_report(result, args.top, args.all)

    if len(times) > 1:
        print("\nSi el número de candidatas que pasan las puertas duras cambia "
              "mucho entre las horas comparadas, el momento del snapshot SÍ "
              "está afectando al resultado (el pre-market se \"asienta\" o se "
              "diluye a medida que se acerca la apertura). Si el número es "
              "parecido en ambas horas, el filtro está siendo consistente y "
              "un día con pocas o ninguna candidata es, sencillamente, un día "
              "tranquilo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
