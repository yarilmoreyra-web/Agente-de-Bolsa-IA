"""Punto de entrada del agente de análisis pre-market.

Ejemplos de uso:

    python main.py --dry-run --no-gemini --no-telegram --force
    python main.py --date 2026-09-21
    python main.py --backtest-date 2026-09-21

Cronograma que sigue ``run_pipeline`` (hora de Nueva York):

    08:35-08:45  universo, datos diarios, barras 5m históricas y contexto de mercado
    08:45        snapshot pre-market (``SCHEDULE.snapshot_time``)
    08:45-08:57  indicadores, pre-filtro, noticias, filtro, Gemini y validación
    09:00        envío del informe (Fase 5)

Códigos de salida: 0 = correcto (incluye "hoy no toca ejecutar"),
1 = error de configuración o de datos, 2 = función aún no disponible.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time as time_module
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Optional, Sequence

import candidate_filter
import config
import gemini_analyzer
import market_context as market_context_module
import market_data
import news as news_module
import technical_analysis
import utils

logger = logging.getLogger("trading_agent.main")

BENCHMARKS = ("SPY", "QQQ")
DISCLAIMER = "Herramienta informativa, no es asesoramiento financiero."


def _parse_date(value: str) -> date:
    """Convierte 'AAAA-MM-DD' en fecha; falla con un mensaje claro."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"fecha inválida '{value}': usa el formato AAAA-MM-DD "
            "(por ejemplo 2026-09-21)"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Agente de análisis bursátil pre-market (no ejecuta operaciones).",
    )
    parser.add_argument(
        "--date", type=_parse_date, default=None,
        help="Fecha de la sesión (AAAA-MM-DD). Por defecto, hoy en Nueva York.",
    )
    parser.add_argument(
        "--backtest-date", type=_parse_date, default=None,
        help="Evalúa a posteriori el informe de esa fecha (Fase 5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Genera el informe sin enviar a Telegram ni escribir histórico.",
    )
    parser.add_argument(
        "--no-gemini", action="store_true",
        help="No llama a Gemini: solo análisis de Python.",
    )
    parser.add_argument(
        "--no-telegram", action="store_true", help="No envía nada a Telegram.",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="No espera hasta la hora del snapshot ni hasta la del informe.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignora el calendario, la ventana horaria y la comprobación de "
             "'ya enviado'. Útil para pruebas manuales.",
    )
    parser.add_argument(
        "--universe", type=Path, default=None,
        help="Ruta alternativa al Excel de tickers (por defecto lista_tickers.xlsx).",
    )
    return parser


def _configure_console_encoding() -> None:
    """Pide a la consola que use UTF-8 (evita errores con tildes y emojis en Windows)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _describe_mode(args: argparse.Namespace) -> str:
    """Resume en una línea los flags activos."""
    active = [
        name for name, on in (
            ("dry-run", args.dry_run),
            ("no-gemini", args.no_gemini),
            ("no-telegram", args.no_telegram),
            ("no-wait", args.no_wait),
            ("force", args.force),
        ) if on
    ]
    return ", ".join(active) if active else "normal"


def wait_until(target: time, session_date: date, sleep: Any = time_module.sleep) -> None:
    """Espera hasta ``target`` (hora de Nueva York) si esa hora es hoy y está por llegar.

    No espera si la fecha analizada no es la de hoy o si la hora ya pasó.
    """
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


# --------------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------------- #
def run_pipeline(
    tickers: Sequence[str],
    session: utils.SessionInfo,
    *,
    use_gemini: bool = True,
    wait_for_snapshot: bool = True,
    env: Any = None,
    save_bars: bool = True,
) -> dict[str, Any]:
    """Ejecuta el análisis completo de un día y devuelve todo lo calculado.

    Pasos: datos diarios y barras 5m históricas → contexto de mercado →
    snapshot pre-market → indicadores → pre-filtro → noticias → filtro →
    Gemini → validación. El diccionario resultante es la materia prima del
    informe, del histórico (Fase 5) y del backtest.

    Ningún fallo aislado detiene el proceso: los tickers que no se puedan
    descargar o analizar quedan registrados en ``failures``.
    """
    env = env if env is not None else config.load_env_settings()
    session_date = session.date
    session_type = "cierre anticipado" if session.is_early_close else "sesión normal"
    started = utils.now_ny()

    # 1) Datos diarios y barras 5m históricas (base del RVOL) -----------------
    print(f"[1/7] Descargando datos diarios de {len(tickers)} tickers...")
    daily = market_data.download_daily(tickers)
    print(f"      {len(daily.data)} con datos diarios, {len(daily.failed)} sin ellos.")

    print("[2/7] Descargando barras de 5 minutos históricas (base del RVOL)...")
    intraday_hist = market_data.download_intraday_history(tickers)
    print(f"      {len(intraday_hist.data)} tickers con histórico intradía.")

    # 2) Contexto de mercado ---------------------------------------------------
    snapshot_ts = market_data.resolve_snapshot_ts(session_date, utils.now_ny())
    print("[3/7] Contexto de mercado (índices, futuros, VIX, sectores, macro)...")
    benchmark_bars: dict[str, Any] = {}
    context = market_context_module.collect_market_context(
        session_date, snapshot_ts, bars_out=benchmark_bars)
    print(f"      {context.summary}")

    # 3) Snapshot pre-market a la hora prevista --------------------------------
    if wait_for_snapshot:
        wait_until(config.SCHEDULE.snapshot_time, session_date)
    snapshot_ts = market_data.resolve_snapshot_ts(session_date, utils.now_ny())
    print(f"[4/7] Snapshot pre-market a las {snapshot_ts.strftime('%H:%M')} ET...")
    intraday_today = market_data.download_intraday_today(tickers, session_date)
    snapshots = market_data.build_snapshots(
        tickers, daily.data, intraday_today.data, snapshot_ts)
    quality_counts: dict[str, int] = {}
    for snapshot in snapshots.values():
        quality_counts[snapshot.premarket_quality] = \
            quality_counts.get(snapshot.premarket_quality, 0) + 1
    print(f"      Calidad del dato: {quality_counts}")

    # 4) Indicadores y métricas ------------------------------------------------
    print("[5/7] Calculando indicadores, RVOL, niveles y fuerza relativa...")
    analyses, failures = technical_analysis.analyze_many(
        tickers, daily.data, intraday_hist.data, intraday_today.data,
        {t: s.to_dict() for t, s in snapshots.items()}, snapshot_ts,
        context.benchmarks_for_rs(),
    )
    print(f"      {len(analyses)} analizados, {len(failures)} con error.")

    # 5) Noticias solo de los tickers que valen la pena ------------------------
    news_tickers = candidate_filter.prefilter_for_news(analyses)
    print(f"[6/7] Buscando noticias de {len(news_tickers)} tickers...")
    news = news_module.gather_news(
        news_tickers, news_module.build_providers(env)) if news_tickers else {}
    confirmed = sum(1 for item in news.values() if item.catalyst_confirmed)
    print(f"      {confirmed} con catalizador confirmado.")

    # 6) Filtro y selección de candidatas --------------------------------------
    outcome = candidate_filter.filter_candidates(analyses, news, context)
    print(f"[7/7] Candidatas: {', '.join(outcome.candidates) or 'ninguna'}")

    # ⚙️ Segunda fuente de noticias (Alpha Vantage), solo para las finalistas.
    if outcome.candidates and getattr(env, "alphavantage_enabled", False):
        for provider in news_module.build_providers(env):
            if provider.name == "alphavantage":
                news = news_module.enrich_with_alphavantage(
                    outcome.candidates, news, provider)
                outcome = candidate_filter.filter_candidates(analyses, news, context)

    candidates = candidate_filter.candidates_payload(outcome, analyses, news)

    # 7) Gemini ----------------------------------------------------------------
    if outcome.no_trade:
        gemini = gemini_analyzer.unavailable("no hay candidatas: NO OPERAR")
        gemini.no_trade_reason = outcome.no_trade_reason
        print("      Sin candidatas: no se llama a Gemini. Informe = NO OPERAR.")
    elif not use_gemini:
        gemini = gemini_analyzer.unavailable("desactivado con --no-gemini")
        print("      Gemini desactivado (--no-gemini).")
    else:
        print(f"      Llamando a Gemini ({config.DEFAULT_GEMINI_MODEL})...")
        gemini = gemini_analyzer.analyze(
            session_date, snapshot_ts, session_type, context, candidates,
            api_key=getattr(env, "gemini_api_key", ""),
        )
        estado = "OK" if gemini.available else f"degradado ({gemini.error})"
        print(f"      Gemini: {estado}, {len(gemini.picks)} picks validados.")

    # 8) Barras 5m de finalistas y de los índices (las necesita el backtest) ----
    saved: list[Path] = []
    if save_bars:
        to_save = {t: intraday_today.data[t] for t in outcome.candidates
                   if t in intraday_today.data}
        to_save.update({s: benchmark_bars[s] for s in BENCHMARKS if s in benchmark_bars})
        saved = market_data.save_intraday_bars(to_save, session_date)

    return {
        "session_date": session_date.isoformat(),
        "session_type": session_type,
        "snapshot_ts": snapshot_ts.isoformat(),
        "started_at": started.isoformat(),
        "finished_at": utils.now_ny().isoformat(),
        "universe_size": len(tickers),
        "market_context": context.to_dict(),
        "analyses": analyses,
        "snapshots": {t: s.to_dict() for t, s in snapshots.items()},
        "news": {t: item.to_dict() for t, item in news.items()},
        "filter_log": outcome.filter_log,
        "candidates": outcome.candidates,
        "candidates_payload": candidates,
        "gemini": gemini.to_dict(),
        "no_trade": gemini.no_trade,
        "no_trade_reason": gemini.no_trade_reason,
        "failures": {
            "daily": daily.failed,
            "intraday_history": intraday_hist.failed,
            "intraday_today": intraday_today.failed,
            "analysis": failures,
            "market_context": context.failures,
        },
        "intraday_files": [str(path) for path in saved],
        "disclaimer": DISCLAIMER,
    }


def print_provisional_summary(result: dict[str, Any]) -> None:
    """Resumen por consola mientras el informe formal llega en la Fase 5."""
    print("\n" + "=" * 70)
    print(f"RESUMEN PROVISIONAL — {result['session_date']} ({result['session_type']})")
    print(f"Snapshot: {result['snapshot_ts'][11:16]} ET | "
          f"Universo: {result['universe_size']} tickers")
    print(f"Contexto: {result['market_context'].get('summary', 'N/A')}")
    print("-" * 70)

    gemini = result["gemini"]
    if not gemini.get("available"):
        print(f"{gemini_analyzer.UNAVAILABLE_NOTE}: {gemini.get('error', 'N/A')}")
    if result["no_trade"] or not gemini.get("picks"):
        print(f"🚫 NO OPERAR — {result['no_trade_reason']}")
    for position, pick in enumerate(gemini.get("picks", []), start=1):
        print(f"\n{position}. {pick['ticker']} ({pick['direction']}) — "
              f"{pick.get('company', 'N/A')}")
        print(f"   Catalizador: {pick.get('catalyst', 'N/A')} "
              f"[{pick.get('catalyst_source', 'N/A')}]")
        print(f"   Cierre anterior {pick.get('prev_close')} | "
              f"Pre-market {pick.get('premarket_price')} | "
              f"Gap {pick.get('gap_pct')}% | RVOL {pick.get('rvol')}")
        print(f"   Entrada {pick.get('entry_zone_low')}-{pick.get('entry_zone_high')} | "
              f"Stop {pick.get('stop')} | O1 {pick.get('target_1')} | "
              f"O2 {pick.get('target_2')} | R/R {pick.get('rr')}")
        print(f"   Precio máximo de entrada válido: {pick.get('max_valid_entry')} "
              "(por encima: NO ENTRAR)")
        print(f"   Decisión: {pick.get('decision')} ({pick.get('confidence')}) — "
              f"{pick.get('reason', '')}")

    if gemini.get("warnings"):
        print("\nAvisos de validación:")
        for warning in gemini["warnings"]:
            print(f"  · {warning}")

    print("-" * 70)
    print(DISCLAIMER)
    print("El informe con formato, el envío por Telegram y el histórico "
          "se añaden en la Fase 5.")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ejecuta el agente. Devuelve el código de salida."""
    args = build_parser().parse_args(argv)
    _configure_console_encoding()

    env = config.load_env_settings()
    utils.setup_logging(
        config.PATHS.log_file,
        config.LOGGING.level,
        config.LOGGING.max_bytes,
        config.LOGGING.backup_count,
    )

    try:
        config.validate_config()
    except ValueError as exc:
        logger.error("Configuración inválida: %s", exc)
        print(f"ERROR de configuración: {exc}")
        return 1

    if args.backtest_date is not None:
        print(
            f"El backtest de {args.backtest_date} aún no está disponible: "
            "se implementa en la Fase 5."
        )
        return 2

    now = utils.now_ny()
    today = now.date()
    target = args.date or today
    schedule = config.SCHEDULE

    print("=== Agente de análisis pre-market ===")
    print(
        f"Fecha objetivo: {target} | Hora actual en Nueva York: "
        f"{now.strftime('%H:%M')} | Modo: {_describe_mode(args)}"
    )
    logger.info("Inicio. fecha=%s modo=%s", target, _describe_mode(args))
    logger.info(
        "Claves detectadas -> Gemini: %s | Telegram: %s | Alpha Vantage: %s",
        "sí" if env.gemini_enabled else "no",
        "sí" if env.telegram_enabled else "no",
        "sí" if env.alphavantage_enabled else "no",
    )

    # 1) Calendario bursátil -------------------------------------------------
    session = utils.get_session_info(target)
    if not session.is_session:
        if not args.force:
            print(
                f"Sin sesión bursátil el {target}: {session.reason}. "
                "No se genera informe. (Usa --force para probar igualmente.)"
            )
            logger.info("Sin sesión el %s: %s", target, session.reason)
            return 0
        print(f"AVISO: el {target} no es sesión ({session.reason}); sigo por --force.")
    elif session.is_early_close and session.close_time is not None:
        print(
            f"AVISO: cierre anticipado hoy, el mercado cierra a las "
            f"{session.close_time.strftime('%H:%M')} ET."
        )

    # 2) Guarda horaria (descarta el cron de verano/invierno que no toca) ----
    guard_applies = target == today and not (args.force or args.dry_run)
    if guard_applies and not utils.is_within_window(
        now, schedule.run_window_start, schedule.run_window_end
    ):
        print(
            f"Fuera de la ventana de ejecución "
            f"({utils.format_hhmm(schedule.run_window_start)}-"
            f"{utils.format_hhmm(schedule.run_window_end)} ET); hora actual: "
            f"{now.strftime('%H:%M')} ET. Es normal si este es el cron del "
            "horario contrario. Usa --force o --dry-run para ejecutar igualmente."
        )
        logger.info("Ejecución descartada por la guarda horaria.")
        return 0

    # 3) Idempotencia: no repetir el informe de un día ya enviado ------------
    if not (args.force or args.dry_run) and utils.already_sent(
        config.PATHS.history_dir, target
    ):
        print(f"El informe del {target} ya fue enviado. Nada que hacer.")
        logger.info("Informe del %s ya enviado; se omite.", target)
        return 0

    # 4) Universo (PASO 0) -----------------------------------------------------
    universe_path = args.universe or config.PATHS.universe_file
    try:
        universe = utils.load_universe(universe_path)
    except utils.UniverseError as exc:
        logger.error("Universo: %s", exc)
        print(f"ERROR: {exc}")
        return 1

    print(f"Lista leída del archivo: {universe.count} tickers")
    logger.info(
        "Universo: %d tickers (vacíos=%d, duplicados=%d, inválidos=%d)",
        universe.count,
        universe.empty_dropped,
        universe.duplicates_dropped,
        len(universe.invalid_dropped),
    )
    if universe.duplicates_dropped or universe.invalid_dropped:
        print(
            f"  (duplicados eliminados: {universe.duplicates_dropped}; "
            f"valores no válidos ignorados: {len(universe.invalid_dropped)})"
        )

    print(
        f"Snapshot pre-market previsto: {utils.format_hhmm(schedule.snapshot_time)} ET | "
        f"Informe previsto: {utils.format_hhmm(schedule.report_time)} ET"
    )

    # 5) Análisis completo ----------------------------------------------------
    session_for_run = session if session.is_session else utils.SessionInfo(
        target, True, "sesión forzada con --force")
    try:
        result = run_pipeline(
            universe.tickers,
            session_for_run,
            use_gemini=not args.no_gemini and env.gemini_enabled,
            wait_for_snapshot=not args.no_wait,
            env=env,
            save_bars=not args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - el fallo debe verse en el log y en la salida
        logger.exception("El análisis falló: %s", exc)
        print(f"ERROR durante el análisis: {type(exc).__name__}: {exc}")
        return 1

    if not args.no_gemini and not env.gemini_enabled:
        print("AVISO: no hay GEMINI_API_KEY; el informe sale solo con datos de Python.")

    print_provisional_summary(result)
    logger.info(
        "Fin. candidatas=%s picks=%d",
        result["candidates"], len(result["gemini"].get("picks", [])),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
