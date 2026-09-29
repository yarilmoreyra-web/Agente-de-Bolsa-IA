"""update_agent.py — Orquestador de la actualización post-apertura (10:00 ET).

Se lanza con ``python main.py --update`` (ver ``main.py``). Cronograma (hora de
Nueva York):

    09:45        arranca el workflow (puede retrasarse hasta ``run_window_end``)
    09:45-10:00  espera hasta las 10:00 (``UPDATE.snapshot_time``)
    10:00        descarga las barras de 5 min de la sesión regular
    10:00-10:03  métricas, estado de cada selección, noticias nuevas, Gemini
    ~10:03       informe y envío por Telegram; se guarda en ``data/updates``

Requisito: existe el histórico de la mañana (``data/history/AAAA-MM-DD.json``).
La actualización NO lo modifica: se guarda aparte, con su propio marcador de
«ya enviado», así que la ejecución de la mañana y la de las 10:00 son
independientes.

Códigos de salida: 0 = correcto (incluye «hoy no toca»), 1 = error o sin datos.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import requests

import config
import history as history_module
import market_data
import post_open as po
import report as report_module
import telegram as telegram_module
import update_analyzer
import update_report
import utils

logger = logging.getLogger("trading_agent.update_agent")


# --------------------------------------------------------------------------- #
# Estado en disco (aparte del histórico de la mañana)
# --------------------------------------------------------------------------- #
def update_path(session_date: date, updates_dir: Path | None = None) -> Path:
    """Ruta del JSON de la actualización de una fecha."""
    base = Path(updates_dir) if updates_dir is not None else config.PATHS.updates_dir
    return base / f"{session_date.isoformat()}.json"


def update_already_sent(session_date: date, updates_dir: Path | None = None) -> bool:
    """True si la actualización de ``session_date`` ya se envió por Telegram."""
    data = utils.read_json(update_path(session_date, updates_dir))
    return isinstance(data, dict) and data.get("sent") is True


def save_update(result: Mapping[str, Any], markdown: str, sent: bool,
                status: Mapping[str, Any] | None, updates_dir: Path | None = None,
                new_run: bool = True) -> Path:
    """Guarda (o sobrescribe) ``data/updates/AAAA-MM-DD.json``.

    Seguro de llamar varias veces para la misma fecha (p. ej. al repetir con
    ``--force``). ``new_run=True`` incrementa ``run_count``; se pasa ``False``
    en la segunda escritura de una misma ejecución (la que anota el resultado
    del envío) para no contarla dos veces.
    """
    session_date = date.fromisoformat(str(result["session_date"]))
    path = update_path(session_date, updates_dir)
    previous = utils.read_json(path)
    run_count = int(previous.get("run_count", 0)) if isinstance(previous, dict) else 0
    if new_run or run_count == 0:
        run_count += 1
    stamps = {
        "update_ts": result.get("update_ts"), "started": result.get("started_at"),
        "finished": result.get("finished_at"),
        "sent": utils.now_ny().isoformat() if sent else None,
    }
    utils.write_json_atomic(path, {
        "date": session_date, "sent": bool(sent), "send_status": dict(status) if status else None,
        "run_count": run_count, "timestamps": stamps, "report_markdown": markdown,
        "result": dict(result),
    })
    logger.info("Actualización guardada en %s (run_count=%d, sent=%s).", path, run_count, sent)
    return path


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _parse_iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def run_update(
    morning: Mapping[str, Any], session: utils.SessionInfo, *,
    use_gemini: bool = True, wait_for_snapshot: bool = True, env: Any = None,
    client: Any = None, sleep: Any = None,
) -> dict[str, Any]:
    """Reanaliza las candidatas de la mañana y las compara con el informe de las 09:00.

    Ningún fallo aislado detiene el proceso: un ticker sin datos queda con
    calidad ``missing`` y el informe lo dice.
    """
    env = env if env is not None else config.load_env_settings()
    session_date = session.date
    started = utils.now_ny()
    update_cfg = config.UPDATE
    pick_tickers, other_tickers = po.select_review_tickers(morning)
    tickers = pick_tickers + other_tickers
    print(f"[1/5] Selecciones de la mañana: {', '.join(pick_tickers) or 'ninguna'} | "
          f"otras candidatas: {len(other_tickers)}")

    # 1) Esperar a la hora de la actualización --------------------------------
    if wait_for_snapshot:
        wait_kwargs = {"sleep": sleep} if sleep is not None else {}
        po.wait_until(update_cfg.snapshot_time, session_date, **wait_kwargs)
    update_ts = po.resolve_update_ts(session_date, utils.now_ny())

    # 2) Datos de la sesión regular ------------------------------------------
    print(f"[2/5] Descargando barras de 5 min de hoy hasta las {update_ts.strftime('%H:%M')} ET...")
    symbols = list(dict.fromkeys(tickers + list(update_cfg.benchmarks)))
    today = market_data.download_intraday_today(symbols, session_date)
    hist = market_data.download_intraday_history(tickers) if tickers else \
        market_data.BatchResult()

    # 3) Métricas y estado de cada selección ----------------------------------
    print("[3/5] Calculando métricas y estado de cada selección...")
    picks_by_ticker = {p["ticker"]: p for p in morning.get("picks", [])}
    entries: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        reference = po.morning_reference(morning, ticker)
        try:
            live = po.compute_live_metrics(
                ticker, today.data.get(ticker), hist.data.get(ticker),
                reference["prev_close"], update_ts)
        except Exception as exc:  # noqa: BLE001 - aislamiento por ticker
            logger.error("%s: no se pudieron calcular las métricas (%s)", ticker, exc)
            live = po._empty_live(ticker, f"error de cálculo: {exc}")
        entry: dict[str, Any] = {
            "ticker": ticker, "reference": {k: v for k, v in reference.items() if k != "news"},
            "live": live, "news_new": [], "_morning_news": reference["news"],
        }
        if ticker in picks_by_ticker:
            entry["pick"] = picks_by_ticker[ticker]
            entry["assessment"] = po.assess_pick(
                picks_by_ticker[ticker], live,
                po.regular_bars(today.data.get(ticker), update_ts), config.TRADE.min_rr)
        entries[ticker] = entry

    quality_counts: dict[str, int] = {}
    for entry in entries.values():
        quality = entry["live"].get("quality", "missing")
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    print(f"      Calidad del dato: {quality_counts}")
    no_data = bool(tickers) and quality_counts.get("missing", 0) == len(tickers)

    market = po.compare_market(morning.get("context") or {}, today.data, update_ts)

    # 4) Noticias nuevas desde el informe -------------------------------------
    print("[4/5] Buscando noticias publicadas después del informe...")
    since = _parse_iso(morning.get("snapshot_ts"))
    now = utils.now_ny()
    # En directo cuentan hasta este mismo instante; al repetir una fecha pasada,
    # solo las publicadas hasta la hora de los datos (nunca las de días posteriores).
    until = now if now.date() == session_date else update_ts
    fresh = {} if no_data else po.refresh_news(tickers, env, reference=now)
    for ticker, entry in entries.items():
        entry["news_new"] = po.new_headlines(
            entry.pop("_morning_news"), fresh.get(ticker), since, until=until)
    with_news = sum(1 for e in entries.values() if e["news_new"])
    print(f"      {with_news} ticker(s) con noticias nuevas.")

    pick_entries = [entries[t] for t in pick_tickers]
    other_entries = [entries[t] for t in other_tickers]

    # 5) Gemini + fusión con los hechos de Python -----------------------------
    if no_data:
        review = update_analyzer.python_only_review("sin datos de la sesión regular")
        print("[5/5] Sin datos de la sesión regular: no se llama a Gemini.")
    elif not pick_entries:
        review = update_analyzer.python_only_review("no había selecciones que revisar")
        print("[5/5] El informe de la mañana no tenía selecciones: no se llama a Gemini.")
    elif not use_gemini:
        review = update_analyzer.python_only_review("desactivado con --no-gemini")
        print("[5/5] Gemini desactivado (--no-gemini).")
    else:
        print(f"[5/5] Llamando a Gemini ({getattr(env, 'gemini_model', '') or config.DEFAULT_GEMINI_MODEL})...")
        payload = update_analyzer.build_payload(
            session_date, update_ts, morning.get("market_view", ""), market,
            pick_entries, other_entries)
        review = update_analyzer.analyze(
            payload, pick_tickers, other_tickers,
            api_key=getattr(env, "gemini_api_key", ""),
            model=getattr(env, "gemini_model", "") or config.DEFAULT_GEMINI_MODEL,
            client=client)
        state = "OK" if review["available"] else f"degradado ({review['error']})"
        print(f"      Gemini: {state}, {len(review['reviews'])} valoraciones.")

    counts: dict[str, int] = {}
    for entry in pick_entries:
        entry.update(update_analyzer.merge_pick(
            entry, review["reviews"].get(entry["ticker"]), config.TRADE.min_rr))
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1

    return {
        "session_date": session_date.isoformat(),
        "session_type": "cierre anticipado" if session.is_early_close else "sesión normal",
        "update_ts": update_ts.isoformat(),
        "started_at": started.isoformat(),
        "finished_at": utils.now_ny().isoformat(),
        "morning": {k: morning.get(k) for k in (
            "snapshot_ts", "sent_at", "sent", "no_trade", "no_trade_reason",
            "market_view", "gemini_available")},
        "market": market,
        "picks": pick_entries,
        "others": other_entries,
        "watchlist": review["watchlist"],
        "gemini": {k: review.get(k) for k in (
            "available", "model", "error", "note", "warnings", "market_view",
            "summary", "raw_response")},
        "counts": counts,
        "quality_counts": quality_counts,
        "no_data": no_data,
        "failures": {"intraday_today": today.failed, "intraday_history": hist.failed},
        "disclaimer": report_module.DISCLAIMER,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def print_summary(result: Mapping[str, Any]) -> None:
    """Resumen breve por consola (el texto completo es el del Telegram)."""
    print("\n" + "=" * 70)
    print(f"ACTUALIZACIÓN — {result['session_date']} | datos hasta las "
          f"{po._hhmm(result['update_ts'])} ET")
    for row in result.get("market", []):
        print(f"  {row['symbol']}: {row.get('morning_change_pct')} → {row.get('now_change_pct')}")
    for entry in result.get("picks", []):
        pick = entry["pick"]
        print(f"  {entry['ticker']}: {pick.get('decision')} → {entry.get('decision')} "
              f"[{entry.get('verdict')}] estado={entry['assessment'].get('status')}")
    print("=" * 70)


def _telegram_notice(text: str, env: Any) -> None:
    """Envía un aviso corto (p. ej. «sin datos»). Los fallos solo se registran."""
    try:
        telegram_module.send_report(
            report_module.markdown_to_telegram_html(text), env, http_post=requests.post)
    except Exception as exc:  # noqa: BLE001 - el aviso nunca debe romper el flujo
        logger.error("No se pudo enviar el aviso a Telegram: %s", exc)


def run_update_cli(args: argparse.Namespace, env: Any) -> int:
    """Flujo completo de ``python main.py --update``. Devuelve el código de salida."""
    now = utils.now_ny()
    today = now.date()
    target = args.date or today
    upd = config.UPDATE
    force = bool(args.force)
    dry_run = bool(args.dry_run)

    print("=== Actualización post-apertura ===")
    print(f"Fecha objetivo: {target} | Hora actual en Nueva York: {now.strftime('%H:%M')} | "
          f"Datos previstos: {utils.format_hhmm(upd.snapshot_time)} ET")
    logger.info("Actualización. fecha=%s force=%s dry_run=%s", target, force, dry_run)

    # 1) Calendario -----------------------------------------------------------
    session = utils.get_session_info(target)
    if not session.is_session:
        if not force:
            print(f"Sin sesión bursátil el {target}: {session.reason}. "
                  "No hay actualización. (Usa --force para probar igualmente.)")
            return 0
        print(f"AVISO: el {target} no es sesión ({session.reason}); sigo por --force.")
        session = utils.SessionInfo(target, True, "sesión forzada con --force")

    # 2) Guarda horaria -------------------------------------------------------
    if target == today and not (force or dry_run) and not utils.is_within_window(
            now, upd.run_window_start, upd.run_window_end):
        print(f"Fuera de la ventana de la actualización "
              f"({utils.format_hhmm(upd.run_window_start)}-{utils.format_hhmm(upd.run_window_end)} ET); "
              f"hora actual: {now.strftime('%H:%M')} ET. Es normal si este es el cron del horario "
              "contrario. Usa --force o --dry-run para ejecutar igualmente.")
        logger.info("Actualización descartada por la guarda horaria.")
        return 0

    # 3) Idempotencia ---------------------------------------------------------
    if not (force or dry_run) and update_already_sent(target):
        print(f"La actualización del {target} ya fue enviada. Nada que hacer.")
        return 0

    # 4) Informe de la mañana (lo que se va a comparar) -----------------------
    record = history_module.load_history(target)
    if not isinstance(record, dict):
        print(f"No hay informe de la mañana del {target} (data/history/{target}.json): "
              "no hay nada con qué comparar.")
        logger.warning("Actualización omitida: no existe el histórico de %s.", target)
        return 0
    morning = po.load_morning_state(record)
    if not morning["candidates"] and not morning["picks"]:
        print("El informe de la mañana no tenía candidatas: no hay nada que actualizar.")
        return 0

    # 5) Análisis -------------------------------------------------------------
    try:
        result = run_update(
            morning, session,
            use_gemini=not args.no_gemini and env.gemini_enabled,
            wait_for_snapshot=not args.no_wait, env=env)
    except Exception as exc:  # noqa: BLE001 - el fallo debe verse en el log y en la salida
        logger.exception("La actualización falló: %s", exc)
        print(f"ERROR durante la actualización: {type(exc).__name__}: {exc}")
        return 1

    if not args.no_gemini and not env.gemini_enabled:
        print("AVISO: no hay GEMINI_API_KEY; la actualización sale solo con datos de Python.")

    if result["no_data"]:
        message = (f"⚠️ Actualización de las {po._hhmm(result['update_ts'])} ET no disponible: "
                   "no hay datos de la sesión regular (¿el mercado no ha abierto o Yahoo va con retraso?).")
        print(message)
        if not (dry_run or args.no_telegram) and env.telegram_enabled:
            _telegram_notice(message, env)
        return 1

    result["sent_at"] = utils.now_ny()
    print_summary(result)
    markdown = update_report.build_update_report(result)

    if dry_run:
        print("\n" + "=" * 70)
        print("ACTUALIZACIÓN (--dry-run: no se envía a Telegram ni se guarda)")
        print("=" * 70)
        print(markdown)
        return 0

    update_report.save_update_report(markdown, target)
    save_update(result, markdown, sent=False, status=None)

    sent, error, attempted = False, "", False
    if args.no_telegram:
        print("Telegram desactivado (--no-telegram): la actualización no se envía.")
        error = "envío desactivado con --no-telegram"
    elif not env.telegram_enabled:
        print("AVISO: faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID; no se puede enviar.")
        error = "faltan credenciales de Telegram"
    else:
        attempted = True
        send_result = telegram_module.send_report(
            report_module.markdown_to_telegram_html(markdown), env, http_post=requests.post)
        if send_result.ok:
            sent = True
            print(f"Actualización enviada a Telegram ({send_result.parts_sent} mensaje(s)).")
            logger.info("Telegram: actualización enviada (%d mensajes).", send_result.parts_sent)
        else:
            error = send_result.error
            print(f"ERROR al enviar a Telegram: {send_result.error}")
            logger.error("Telegram: fallo en el envío de la actualización: %s", send_result.error)

    save_update(result, markdown, sent=sent, status={"error": error} if error else {"ok": True},
                new_run=False)
    return 0 if sent or not attempted else 1
