"""Tests de la actualización post-apertura: Gemini, informe, pipeline, CLI y workflow.

Sin red: las descargas de ``market_data`` y las llamadas a Gemini/Telegram se
sustituyen por datos sintéticos y dobles de prueba.
"""
from __future__ import annotations

import dataclasses
import json
import re
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import config
import main
import market_data
import post_open as po
import report as report_module
import telegram as telegram_module
import update_agent
import update_analyzer as ua
import update_report
import utils
from html_utils import is_balanced
from tests.synthetic import business_days, make_5m_bars
from utils import NY_TZ

DAY = "2026-09-21"
ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "update_agent.yml"
MORNING_WORKFLOW = ROOT / ".github" / "workflows" / "daily_agent.yml"


# --------------------------------------------------------------------------- #
# Escenario sintético
# --------------------------------------------------------------------------- #
def pick(ticker, decision, confidence, **levels):
    base = {"ticker": ticker, "company": f"{ticker} Corp", "direction": "LONG",
            "decision": decision, "confidence": confidence, "catalyst": "Resultados",
            "reason": "Motivo de la mañana", "fade_risk": "LOW", "rr": 1.8}
    base.update(levels)
    return base


def candidate(ticker, rank, prev_close, headline="Viejo titular"):
    return {"ticker": ticker, "rank": rank, "score": 70 - rank,
            "analysis": {"premarket": {"prev_close": prev_close, "premarket_price": prev_close * 1.02},
                         "gap_pct": 2.0, "indicators": {"atr14": 2.0}, "rvol": {"rvol": 1.5}},
            "news": {"items": [{"headline": headline}]}}


def morning_record(with_picks=True):
    picks = [
        pick("P", "BUY", "HIGH", entry_zone_low=101.0, entry_zone_high=102.0,
             max_valid_entry=101.8, stop=99.0, target_1=106.0, target_2=108.0),
        pick("Q", "WAIT", "MEDIUM", entry_zone_low=49.8, entry_zone_high=50.5,
             max_valid_entry=50.4, stop=49.0, target_1=53.0, target_2=55.0, rr=2.0),
    ] if with_picks else []
    return {
        "date": DAY, "sent": True, "no_trade": not with_picks,
        "no_trade_reason": "" if with_picks else "Ninguna candidata convence",
        "candidates": [candidate("P", 1, 100.0), candidate("Q", 2, 50.0),
                       candidate("R", 3, 29.0), candidate("S", 4, 20.0)],
        "final_selection": picks,
        "gemini_validated": {"available": True, "market_view": "Mercado neutral"},
        "market_data": {"context": {"quotes": {
            "SPY": {"prev_close": 500.0, "price": 501.5, "change_pct": 0.3},
            "QQQ": {"prev_close": 400.0, "price": 401.6, "change_pct": 0.4},
            "^VIX": {"prev_close": 15.0, "price": 15.2, "change_pct": 1.3}}}},
        "timestamps": {"snapshot": f"{DAY}T08:45:00-04:00", "sent": f"{DAY}T09:00:04-04:00"},
        "universe": {"count": 135},
    }


def install_market(monkeypatch, *, empty=False, fresh_news=True):
    """Datos de la sesión: P en zona, Q extendida, R normal, S sin datos."""
    frames = {} if empty else {
        "P": make_5m_bars(DAY, "09:30", "10:00", price=101.4, slope=0.02),
        "Q": make_5m_bars(DAY, "09:30", "10:00", price=50.5, slope=0.3),
        "R": make_5m_bars(DAY, "09:30", "10:00", price=30.0, slope=0.1),
        "SPY": make_5m_bars(DAY, "09:30", "10:00", price=502.0, slope=0.1),
        "QQQ": make_5m_bars(DAY, "09:30", "10:00", price=401.0, slope=0.1),
        "^VIX": make_5m_bars(DAY, "09:30", "10:00", price=15.5, slope=0.01),
    }
    hist = pd.concat([make_5m_bars(d, "09:30", "10:00", volume=1000.0)
                      for d in business_days("2026-09-18", 12)])

    def fake_today(tickers, session_date, *a, **k):
        return market_data.BatchResult(
            data={t: frames[t] for t in tickers if t in frames},
            failed={t: "sin barras de hoy" for t in tickers if t not in frames})

    monkeypatch.setattr(market_data, "download_intraday_today", fake_today)
    monkeypatch.setattr(market_data, "download_intraday_history",
                        lambda tickers, *a, **k: market_data.BatchResult(
                            data={} if empty else {"P": hist}))
    news = {"P": {"items": [{
        "headline": "Titular nuevo", "publisher": "Reuters", "category_label": "Resultados",
        "published_at": f"{DAY}T09:41:00-04:00", "is_hard_catalyst": True}]}} if fresh_news else {}
    monkeypatch.setattr(po, "refresh_news", lambda tickers, env, reference: news)


class FakeClient:
    """Doble del cliente del SDK de Gemini: devuelve textos fijos, uno por intento."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.requests = []
        outer = self

        class _Models:
            def generate_content(self, **request):
                outer.requests.append(request)
                item = outer.texts.pop(0)
                if isinstance(item, Exception):
                    raise item
                return SimpleNamespace(text=item)

        self.models = _Models()


def gemini_reply(**overrides):
    payload = {
        "market_view": "Mercado plano desde las 08:45",
        "summary": "P sigue válida; Q se ha disparado.",
        "reviews": [
            {"ticker": "P", "verdict": "CONFIRMED", "decision": "BUY", "confidence": "HIGH",
             "what_changed": "Mantiene la zona con volumen", "action": "Mantener el plan"},
            {"ticker": "Q", "verdict": "IMPROVED", "decision": "BUY", "confidence": "HIGH",
             "what_changed": "Sube con fuerza", "action": "Entrar ya"},
            {"ticker": "ZZZ", "verdict": "CONFIRMED", "decision": "BUY", "confidence": "LOW",
             "what_changed": "x", "action": "y"},
        ],
        "watchlist": [{"ticker": "R", "reason": "Sube con volumen"},
                      {"ticker": "P", "reason": "ya es una selección"}],
    }
    payload.update(overrides)
    return json.dumps(payload)


SESSION = utils.SessionInfo(date(2026, 9, 21), True, "sesión normal")
FAST_GEMINI = config.GeminiConfig(retry_base_delay=0.0)


def run(monkeypatch, *, record=None, client=None, use_gemini=True, **market):
    install_market(monkeypatch, **market)
    morning = po.load_morning_state(record or morning_record())
    env = config.EnvSettings(gemini_api_key="clave" if client or use_gemini else "")
    return update_agent.run_update(
        morning, SESSION, use_gemini=use_gemini, wait_for_snapshot=False, env=env,
        client=client)


# --------------------------------------------------------------------------- #
# update_analyzer
# --------------------------------------------------------------------------- #
def test_esquema_e_instruccion_de_la_actualizacion():
    schema = ua.response_schema()
    review = schema["properties"]["reviews"]["items"]
    assert review["properties"]["verdict"]["enum"] == list(ua.AI_VERDICTS)
    assert set(schema["required"]) == {"market_view", "summary", "reviews", "watchlist"}
    text = ua.build_system_instruction()
    assert "STOP_TOCADO" in text and "No propongas niveles nuevos" in text


def test_validacion_descarta_lo_que_no_es_de_la_manana():
    checked = ua.validate_response(
        json.loads(gemini_reply()), ["P", "Q"], ["R", "S"])
    assert set(checked["reviews"]) == {"P", "Q"}
    assert [w["ticker"] for w in checked["watchlist"]] == ["R"]   # P ya es selección
    assert any("ZZZ" in w for w in checked["warnings"])


def test_validacion_corrige_enumerados_repetidos_y_tope_de_vigilancia():
    raw = {"reviews": [
        {"ticker": "p", "verdict": "GENIAL", "decision": "COMPRA", "confidence": "??",
         "what_changed": "a  **b**", "action": ""},
        {"ticker": "P", "verdict": "CONFIRMED", "decision": "BUY", "confidence": "HIGH"},
    ], "watchlist": [{"ticker": t, "reason": "r"} for t in ("R", "S", "T", "U")]}
    cfg = dataclasses.replace(config.UPDATE, watchlist_size=2)
    checked = ua.validate_response(raw, ["P"], ["R", "S", "T", "U"], cfg)
    review = checked["reviews"]["P"]
    assert review["verdict"] is None and review["decision"] is None and review["confidence"] is None
    assert review["what_changed"] == "a b"
    assert len(checked["watchlist"]) == 2
    assert any("repetida" in w for w in checked["warnings"])


def test_respuesta_sin_lista_de_reviews_no_rompe():
    checked = ua.validate_response({"picks": []}, ["P"], [])
    assert checked["reviews"] == {} and checked["warnings"]
    assert ua.validate_response("no es un dict", ["P"], [])["reviews"] == {}


def entry_for(status, price_bars, **pick_over):
    levels = dict(entry_zone_low=100.0, entry_zone_high=101.0, max_valid_entry=101.5,
                  stop=98.0, target_1=106.0, target_2=108.0)
    levels.update(pick_over)
    p = pick("T", "BUY", "HIGH", **levels)
    idx = pd.date_range(f"{DAY} 09:30", periods=len(price_bars), freq="5min", tz=NY_TZ)
    bars = pd.DataFrame([(c, c + 0.1, c - 0.1, c, 1000.0) for c in price_bars], index=idx,
                        columns=["Open", "High", "Low", "Close", "Volume"])
    ts = datetime(2026, 9, 21, 10, 0, tzinfo=NY_TZ)
    live = po.compute_live_metrics("T", bars, None, 99.0, ts)
    return {"ticker": "T", "pick": p, "live": live, "assessment": po.assess_pick(p, live, bars)}


def test_python_impone_los_hechos_a_gemini():
    entry = entry_for(None, [100.0] * 3 + [97.0] * 3)          # stop tocado
    assert entry["assessment"]["status"] == po.STATUS_STOP
    review = {"verdict": "CONFIRMED", "decision": "BUY", "confidence": "HIGH",
              "what_changed": "Todo bien", "action": "Comprar"}
    merged = ua.merge_pick(entry, review)
    assert (merged["verdict"], merged["decision"], merged["source"]) == (
        po.V_INVALIDATED, "NO_TRADE", "python")
    assert merged["what_changed"] == "" and "stop" in merged["action"].lower()
    assert any("prevalece Python" in w for w in merged["warnings"])


def test_buy_solo_si_esta_en_zona_y_el_rr_alcanza():
    review = {"verdict": "CONFIRMED", "decision": "BUY", "confidence": "HIGH",
              "what_changed": "Bien", "action": "Entrar"}
    ok = ua.merge_pick(entry_for(None, [100.5] * 6), review)
    assert (ok["verdict"], ok["decision"], ok["source"]) == (po.V_CONFIRMED, "BUY", "gemini")

    against = ua.merge_pick(entry_for(None, [99.2] * 6), review)      # bajo la zona
    assert against["decision"] == "WAIT" and against["verdict"] == po.V_WEAKENED

    low_rr = ua.merge_pick(entry_for(None, [100.5] * 6, target_1=101.5), review)  # R/R < 1.5
    assert low_rr["decision"] == "WAIT" and any("R/R" in w for w in low_rr["warnings"])


def test_coherencia_entre_veredicto_y_decision():
    entry = entry_for(None, [100.5] * 6)
    a = ua.merge_pick(entry, {"verdict": "INVALIDATED", "decision": "BUY", "confidence": "LOW",
                              "what_changed": "", "action": ""})
    assert (a["verdict"], a["decision"]) == (po.V_INVALIDATED, "NO_TRADE")
    b = ua.merge_pick(entry, {"verdict": "CONFIRMED", "decision": "NO_TRADE", "confidence": "LOW",
                              "what_changed": "", "action": ""})
    assert (b["verdict"], b["decision"]) == (po.V_INVALIDATED, "NO_TRADE")


def test_sin_valoracion_de_gemini_manda_python():
    merged = ua.merge_pick(entry_for(None, [100.5] * 6), None)
    assert merged["source"] == "python" and merged["verdict"] == po.V_CONFIRMED
    assert merged["decision"] == "BUY" and merged["confidence"] == "HIGH"


def test_analyze_usa_su_propio_prompt_y_esquema():
    client = FakeClient([gemini_reply()])
    result = ua.analyze({"fecha": DAY}, ["P", "Q"], ["R"], api_key="k", model="m",
                        cfg=FAST_GEMINI, client=client)
    request = client.requests[0]["config"]
    assert result["available"] and set(result["reviews"]) == {"P", "Q"}
    assert request["response_schema"] == ua.response_schema()
    assert request["system_instruction"] == ua.build_system_instruction()
    assert request["temperature"] == config.GEMINI.temperature


def test_analyze_degrada_sin_clave_sin_selecciones_o_con_json_roto():
    assert ua.analyze({}, ["P"], [], api_key="")["available"] is False
    assert "selecciones" in ua.analyze({}, [], ["R"], api_key="k")["error"]
    broken = ua.analyze({}, ["P"], [], api_key="k", cfg=FAST_GEMINI,
                        client=FakeClient(["no json"] * 5))
    assert broken["available"] is False and broken["reviews"] == {}


def test_call_gemini_de_la_manana_sigue_usando_sus_valores_por_defecto():
    import gemini_analyzer as ga
    client = FakeClient([json.dumps({"market_view": "x", "no_trade": True,
                                     "no_trade_reason": "y", "picks": []})])
    ga.call_gemini({}, "k", "m", cfg=FAST_GEMINI, client=client)
    request = client.requests[0]["config"]
    assert request["system_instruction"] == ga.build_system_instruction("long_only")
    assert request["response_schema"] == ga.response_schema("long_only")


# --------------------------------------------------------------------------- #
# Pipeline completo
# --------------------------------------------------------------------------- #
def test_pipeline_compara_la_manana_con_ahora(monkeypatch):
    result = run(monkeypatch, client=FakeClient([gemini_reply()]))
    picks = {e["ticker"]: e for e in result["picks"]}
    assert set(picks) == {"P", "Q"} and [e["ticker"] for e in result["others"]] == ["R", "S"]

    p, q = picks["P"], picks["Q"]
    assert p["assessment"]["status"] == po.STATUS_IN_ZONE
    assert (p["verdict"], p["decision"], p["source"]) == (po.V_CONFIRMED, "BUY", "gemini")
    assert p["live"]["open_rvol"] == 1.0 and p["assessment"]["open_rule"] == "CUMPLE"
    assert [n["headline"] for n in p["news_new"]] == ["Titular nuevo"]

    assert q["assessment"]["status"] == po.STATUS_EXTENDED     # Gemini dijo IMPROVED/BUY
    assert (q["verdict"], q["decision"], q["source"]) == (po.V_EXTENDED, "WAIT", "python")

    assert result["counts"] == {po.V_CONFIRMED: 1, po.V_EXTENDED: 1}
    assert [w["ticker"] for w in result["watchlist"]] == ["R"]
    assert result["others"][1]["live"]["quality"] == "missing"          # S sin datos
    assert result["update_ts"].startswith(f"{DAY}T10:00")
    market = {row["symbol"]: row for row in result["market"]}
    assert market["SPY"]["morning_change_pct"] == 0.3 and market["SPY"]["now_change_pct"] > 0
    json.dumps(utils.sanitize_for_json(result))                          # serializable


def test_pipeline_sin_gemini_usa_solo_hechos_de_python(monkeypatch):
    result = run(monkeypatch, use_gemini=False)
    assert result["gemini"]["available"] is False
    picks = {e["ticker"]: e for e in result["picks"]}
    assert (picks["P"]["verdict"], picks["P"]["source"]) == (po.V_CONFIRMED, "python")
    assert picks["P"]["decision"] == "BUY" and picks["Q"]["verdict"] == po.V_EXTENDED


def test_pipeline_sin_datos_no_llama_a_gemini(monkeypatch):
    client = FakeClient([])
    result = run(monkeypatch, client=client, empty=True)
    assert result["no_data"] is True and client.requests == []
    assert all(e["verdict"] == po.V_NO_DATA for e in result["picks"])


def test_pipeline_sin_selecciones_no_llama_a_gemini(monkeypatch):
    client = FakeClient([])
    result = run(monkeypatch, client=client, record=morning_record(with_picks=False))
    assert result["picks"] == [] and client.requests == []
    assert [e["ticker"] for e in result["others"]] == ["P", "Q", "R", "S"]


def test_un_ticker_que_falla_no_detiene_al_resto(monkeypatch):
    real = po.compute_live_metrics

    def flaky(ticker, *args, **kwargs):
        if ticker == "R":
            raise ValueError("datos corruptos")
        return real(ticker, *args, **kwargs)

    monkeypatch.setattr(po, "compute_live_metrics", flaky)
    result = run(monkeypatch, use_gemini=False)
    others = {e["ticker"]: e for e in result["others"]}
    assert others["R"]["live"]["quality"] == "missing"
    assert "datos corruptos" in others["R"]["live"]["quality_note"]
    assert result["picks"][0]["live"]["quality"] == "ok"


# --------------------------------------------------------------------------- #
# Informe / Telegram
# --------------------------------------------------------------------------- #
def test_informe_muestra_la_comparacion_con_la_manana(monkeypatch):
    result = run(monkeypatch, client=FakeClient([gemini_reply()]))
    result["sent_at"] = datetime(2026, 9, 21, 10, 3, tzinfo=NY_TZ)
    text = update_report.build_update_report(result)

    assert "ACTUALIZACIÓN POST-APERTURA" in text and "lunes 21 de septiembre de 2026" in text
    assert "Datos hasta las 10:00 ET" in text and "Informe previo 09:00 ET" in text
    assert "Envío 10:03 ET" in text
    assert "MERCADO: 08:45 → 10:00" in text and "SPY +0.30% →" in text and "VIX 15.20 →" in text
    assert "Informe 09:00: BUY (HIGH) → Ahora: **BUY (HIGH)** ✅ Confirmada" in text
    assert "Informe 09:00: WAIT (MEDIUM) → Ahora: **WAIT** ⏫ Extendida" in text
    assert "Estado (Python): En zona de entrada" in text
    assert "Regla de apertura: abrió a" in text and "cumplía" in text
    assert "Nueva (09:41) 🔴: Titular nuevo — Reuters" in text
    assert "1 confirmada(s)" in text and "1 extendida(s)" in text
    assert "OTRAS CANDIDATAS DE LAS 09:00" in text and "S: sin datos de la sesión regular" in text
    assert "PARA VIGILAR" in text and "R — Sube con volumen" in text
    assert "ZZZ" in text            # el aviso de validación queda a la vista
    assert text.rstrip().endswith("no es asesoramiento financiero.")


def test_informe_convertido_a_html_es_valido_y_cabe_en_telegram(monkeypatch):
    result = run(monkeypatch, client=FakeClient([gemini_reply()]))
    html_text = report_module.markdown_to_telegram_html(update_report.build_update_report(result))
    assert is_balanced(html_text)
    parts = telegram_module.split_message(html_text)
    assert parts and all(len(p) <= config.TELEGRAM.max_message_chars + 200 for p in parts)


def test_informe_sin_ia_lo_avisa(monkeypatch):
    text = update_report.build_update_report(run(monkeypatch, use_gemini=False))
    assert "Análisis IA no disponible" in text and "solo datos de Python" in text


def test_informe_si_la_manana_fue_no_operar(monkeypatch):
    result = run(monkeypatch, record=morning_record(with_picks=False))
    text = update_report.build_update_report(result)
    assert "El informe de las 09:00 fue NO OPERAR" in text and "Ninguna candidata convence" in text
    assert "no genera nuevas señales" in text and "Análisis IA no disponible" not in text


def test_informe_usa_la_hora_configurada_si_el_envio_previo_es_de_otro_dia(monkeypatch):
    result = run(monkeypatch, use_gemini=False)
    result["morning"]["sent_at"] = "2026-09-22T11:38:50-04:00"    # reejecución de otro día
    assert "Informe previo 09:00 ET" in update_report.build_update_report(result)


def test_informe_se_guarda_sin_pisar_el_de_la_manana(tmp_path):
    path = update_report.save_update_report("hola", date(2026, 9, 21), tmp_path)
    assert path.name == "2026-09-21_actualizacion.md" and path.read_text(encoding="utf-8") == "hola"


# --------------------------------------------------------------------------- #
# Estado en disco
# --------------------------------------------------------------------------- #
def test_guardado_marca_de_enviado_y_run_count(monkeypatch, tmp_path):
    result = run(monkeypatch, use_gemini=False)
    d = date(2026, 9, 21)
    assert update_agent.update_already_sent(d, tmp_path) is False
    update_agent.save_update(result, "md", sent=False, status=None, updates_dir=tmp_path)
    assert update_agent.update_already_sent(d, tmp_path) is False
    update_agent.save_update(result, "md", sent=True, status={"ok": True}, updates_dir=tmp_path,
                             new_run=False)
    data = json.loads(update_agent.update_path(d, tmp_path).read_text(encoding="utf-8"))
    assert data["sent"] is True and data["run_count"] == 1 and data["timestamps"]["sent"]
    assert data["result"]["picks"][0]["ticker"] == "P"
    update_agent.save_update(result, "md", sent=False, status=None, updates_dir=tmp_path)
    assert json.loads(update_agent.update_path(d, tmp_path).read_text(encoding="utf-8"))["run_count"] == 2


# --------------------------------------------------------------------------- #
# CLI (python main.py --update)
# --------------------------------------------------------------------------- #
def fijar_hora(monkeypatch, hour, minute, day=21):
    fake_now = datetime(2026, 9, day, hour, minute, tzinfo=NY_TZ)
    monkeypatch.setattr(utils, "now_ny", lambda: fake_now)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Rutas aisladas, histórico de la mañana escrito y Telegram simulado."""
    monkeypatch.setattr(config, "PATHS", dataclasses.replace(
        config.PATHS, updates_dir=tmp_path / "updates", reports_dir=tmp_path / "reports"))
    utils.write_json_atomic(config.PATHS.history_dir / f"{DAY}.json", morning_record())
    sent_messages = []

    def fake_send(text, env, http_post=None, telegram_cfg=None):
        sent_messages.append(text)
        return SimpleNamespace(ok=True, parts_sent=1, error="")

    monkeypatch.setattr(telegram_module, "send_report", fake_send)
    install_market(monkeypatch)
    return SimpleNamespace(messages=sent_messages,
                           env=config.EnvSettings(telegram_bot_token="t", telegram_chat_id="c"))


def args_for(*argv):
    return main.build_parser().parse_args(list(argv))


def test_cli_envia_guarda_y_no_repite(cli_env, monkeypatch, capsys):
    fijar_hora(monkeypatch, 10, 0)
    code = update_agent.run_update_cli(args_for("--update", "--no-wait", "--no-gemini"), cli_env.env)
    out = capsys.readouterr().out
    assert code == 0 and "Actualización enviada a Telegram" in out
    assert len(cli_env.messages) == 1 and "ACTUALIZACIÓN POST-APERTURA" in cli_env.messages[0]
    assert update_agent.update_already_sent(date(2026, 9, 21))
    assert (config.PATHS.reports_dir / f"{DAY}_actualizacion.md").is_file()
    assert (config.PATHS.history_dir / f"{DAY}.json").is_file()       # el de la mañana intacto
    morning_after = json.loads((config.PATHS.history_dir / f"{DAY}.json").read_text(encoding="utf-8"))
    assert morning_after["sent"] is True and "run_count" not in morning_after

    code = update_agent.run_update_cli(args_for("--update", "--no-wait", "--no-gemini"), cli_env.env)
    assert code == 0 and "ya fue enviada" in capsys.readouterr().out
    assert len(cli_env.messages) == 1


def test_cli_fuera_de_la_ventana_no_hace_nada(cli_env, monkeypatch, capsys):
    fijar_hora(monkeypatch, 12, 0)
    code = update_agent.run_update_cli(args_for("--update", "--no-gemini"), cli_env.env)
    assert code == 0 and "Fuera de la ventana de la actualización" in capsys.readouterr().out
    assert cli_env.messages == []
    fijar_hora(monkeypatch, 10, 45)      # el cron se retrasó demasiado (fin de ventana 10:30)
    assert update_agent.run_update_cli(args_for("--update", "--no-gemini"), cli_env.env) == 0
    assert cli_env.messages == []


def test_cli_sin_informe_de_la_manana_no_hay_con_que_comparar(cli_env, monkeypatch, capsys):
    (config.PATHS.history_dir / f"{DAY}.json").unlink()
    fijar_hora(monkeypatch, 10, 0)
    code = update_agent.run_update_cli(args_for("--update", "--no-wait"), cli_env.env)
    assert code == 0 and "No hay informe de la mañana" in capsys.readouterr().out
    assert cli_env.messages == []


def test_cli_dry_run_no_envia_ni_guarda(cli_env, monkeypatch, capsys):
    fijar_hora(monkeypatch, 12, 0)       # el dry-run ignora la ventana horaria
    code = update_agent.run_update_cli(
        args_for("--update", "--dry-run", "--no-wait", "--no-gemini", "--date", DAY), cli_env.env)
    out = capsys.readouterr().out
    assert code == 0 and "--dry-run: no se envía" in out and "MERCADO: 08:45" in out
    assert cli_env.messages == [] and not config.PATHS.updates_dir.exists()


def test_cli_sin_datos_avisa_y_falla_visiblemente(cli_env, monkeypatch, capsys):
    install_market(monkeypatch, empty=True)
    fijar_hora(monkeypatch, 10, 0)
    code = update_agent.run_update_cli(args_for("--update", "--no-wait", "--no-gemini"), cli_env.env)
    assert code == 1 and "no hay datos de la sesión regular" in capsys.readouterr().out
    assert len(cli_env.messages) == 1 and "no disponible" in cli_env.messages[0]
    assert not update_agent.update_already_sent(date(2026, 9, 21))   # se podrá reintentar


def test_cli_si_telegram_falla_devuelve_error_y_no_marca_enviado(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(telegram_module, "send_report",
                        lambda *a, **k: SimpleNamespace(ok=False, parts_sent=0, error="429"))
    fijar_hora(monkeypatch, 10, 0)
    code = update_agent.run_update_cli(args_for("--update", "--no-wait", "--no-gemini"), cli_env.env)
    assert code == 1 and "ERROR al enviar a Telegram" in capsys.readouterr().out
    assert not update_agent.update_already_sent(date(2026, 9, 21))


def test_cli_fin_de_semana_sin_force_no_hace_nada(cli_env, capsys):
    code = update_agent.run_update_cli(args_for("--update", "--date", "2026-09-19"), cli_env.env)
    assert code == 0 and "Sin sesión bursátil" in capsys.readouterr().out


def test_main_enruta_update_sin_ejecutar_el_flujo_de_la_manana(monkeypatch, tmp_path):
    seen = {}

    def fake(args, env):
        seen["update"] = args.update
        return 7

    monkeypatch.setattr(update_agent, "run_update_cli", fake)
    monkeypatch.setattr(main, "run_pipeline", lambda *a, **k: pytest.fail("no debe correr"))
    assert main.main(["--update", "--dry-run"]) == 7 and seen["update"] is True


# --------------------------------------------------------------------------- #
# Workflow y configuración
# --------------------------------------------------------------------------- #
def workflow_text():
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_de_actualizacion_tiene_dos_crons_y_disparo_manual():
    text = workflow_text()
    crons = re.findall(r'cron:\s*"([^"]+)"', text)
    assert len(crons) == 2 and all(c.endswith("* * 1-5") for c in crons)
    assert {c.split()[1] for c in crons} == {"13", "14"}        # 09:45 ET en verano / invierno
    assert {c.split()[0] for c in crons} == {"45"}
    assert "workflow_dispatch:" in text


def test_los_cron_de_la_manana_y_las_10_no_se_pisan():
    morning = re.findall(r'cron:\s*"([^"]+)"', MORNING_WORKFLOW.read_text(encoding="utf-8"))
    update = re.findall(r'cron:\s*"([^"]+)"', workflow_text())
    assert not set(morning) & set(update)


def test_workflow_comparte_concurrencia_y_permisos_con_el_de_la_manana():
    text, morning = workflow_text(), MORNING_WORKFLOW.read_text(encoding="utf-8")
    group = re.search(r"group:\s*(\S+)", text).group(1)
    assert group == re.search(r"group:\s*(\S+)", morning).group(1)
    assert "cancel-in-progress: false" in text and "contents: write" in text
    assert "timeout-minutes:" in text


def test_workflow_lanza_update_y_sus_flags_existen_en_main():
    text = workflow_text()
    assert "args=(--update)" in text
    flags = set(re.findall(r"args\+=\((--[a-z-]+)", text)) | {"--update"}
    known = {opt for a in main.build_parser()._actions for opt in a.option_strings}
    assert {"--date", "--dry-run", "--no-gemini", "--force", "--update"} <= flags
    assert flags <= known


def test_workflow_pasa_secretos_y_no_pega_entradas_en_el_script():
    text = workflow_text()
    for name in ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        assert f"{name}: ${{{{ secrets.{name} }}}}" in text
    assert "GEMINI_MODEL: ${{ vars.GEMINI_MODEL }}" in text
    for block in re.findall(r"run: \|\n((?:[ ]{10,}.*\n)+)", text):
        assert "${{ inputs." not in block


def test_la_puerta_horaria_del_workflow_coincide_con_la_configuracion():
    upd = config.UPDATE
    start = upd.run_window_start.hour * 100 + upd.run_window_start.minute
    end = upd.run_window_end.hour * 100 + upd.run_window_end.minute
    assert f'-lt {start} ' in workflow_text() and f'-gt {end} ' in workflow_text()


def test_configuracion_de_la_actualizacion_es_coherente():
    config.validate_config()
    assert config.UPDATE.snapshot_time == datetime(2026, 1, 1, 10, 0).time()
    # 30 minutos después de la apertura
    open_min = config.SCHEDULE.regular_open.hour * 60 + config.SCHEDULE.regular_open.minute
    snap_min = config.UPDATE.snapshot_time.hour * 60 + config.UPDATE.snapshot_time.minute
    assert snap_min - open_min == 30


def test_configuracion_incoherente_se_rechaza(monkeypatch):
    from datetime import time
    bad = dataclasses.replace(config.UPDATE, run_window_start=time(9, 0))      # antes de la apertura
    monkeypatch.setattr(config, "UPDATE", bad)
    with pytest.raises(ValueError):
        config.validate_config()
    bad = dataclasses.replace(config.UPDATE, run_window_start=time(9, 35), snapshot_time=time(17, 0),
                              run_window_end=time(17, 30))
    monkeypatch.setattr(config, "UPDATE", bad)
    with pytest.raises(ValueError):
        config.validate_config()
