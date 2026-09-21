"""Test de extremo a extremo del orquestador (``main.run_pipeline``) sin red.

Las descargas de ``market_data`` se sustituyen por datos sintéticos, de modo que
se ejercita el encadenado real:

datos diarios → contexto → snapshot → indicadores → pre-filtro →
noticias → filtro → Gemini → validación.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import candidate_filter
import config
import gemini_analyzer as ga
import main
import market_data
import news as news_module
import technical_analysis
import utils
from utils import NY_TZ


SESSION = date(2026, 9, 21)
SNAPSHOT = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)
SESSION_INFO = utils.SessionInfo(SESSION, True, "sesión normal")

UNIVERSE = ["FUERTE", "PLANA", "BARATA", "ROTA"]

BASE_PRICE = {
    "FUERTE": 100.0,
    "PLANA": 80.0,
    "BARATA": 3.0,
    "ROTA": 50.0,
    "SPY": 500.0,
    "QQQ": 400.0,
    "DIA": 390.0,
    "ES=F": 5000.0,
    "NQ=F": 18000.0,
    "^VIX": 15.0,
    "^TNX": 4.2,
}

GAP = {
    "FUERTE": 0.04,
    "PLANA": 0.001,
    "BARATA": 0.05,
    "ROTA": 0.03,
}


# --------------------------------------------------------------------------- #
# Datos sintéticos
# --------------------------------------------------------------------------- #

def _sessions(n: int) -> list[date]:
    days, cursor = [], SESSION - timedelta(days=1)

    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)

    return sorted(days)


def daily_frame(base: float, sessions: int = 60) -> pd.DataFrame:
    """Serie diaria con dos picos para generar niveles históricos."""
    days = _sessions(sessions)

    close = np.linspace(base * 0.9, base, sessions)
    high = close * 1.005

    # Máximo de 52 semanas.
    high[-25] = base * 1.10

    # Máximo de 20 sesiones.
    high[-10] = base * 1.08

    index = pd.DatetimeIndex(
        [datetime.combine(d, datetime.min.time()) for d in days]
    )

    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": high,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full(sessions, 5_000_000.0),
        },
        index=index.tz_localize(NY_TZ),
    )


def _bars(
    day: date,
    start_hour: int,
    end_minutes: int,
    price: float,
    volume: float,
) -> pd.DataFrame:
    stamps, closes = [], []

    cursor = datetime.combine(
        day,
        datetime.min.time(),
        tzinfo=NY_TZ,
    ).replace(hour=start_hour)

    end = (
        datetime.combine(day, datetime.min.time(), tzinfo=NY_TZ)
        + timedelta(minutes=end_minutes)
    )

    while cursor < end:
        stamps.append(cursor)
        closes.append(price)
        cursor += timedelta(minutes=5)

    values = np.array(closes)

    return pd.DataFrame(
        {
            "Open": values,
            "High": values * 1.001,
            "Low": values * 0.999,
            "Close": values,
            "Volume": np.full(len(values), volume),
        },
        index=pd.DatetimeIndex(stamps),
    )


def intraday_history(base: float) -> pd.DataFrame:
    frames = [
        _bars(
            day,
            4,
            9 * 60 + 30,
            base,
            1_000.0,
        )
        for day in _sessions(15)
    ]

    return pd.concat(frames)


def intraday_today(
    base: float,
    gap: float,
    volume: float = 5_000.0,
) -> pd.DataFrame:
    return _bars(
        SESSION,
        4,
        8 * 60 + 45,
        base * (1 + gap),
        volume,
    )


def install_fake_downloads(monkey: dict) -> None:
    """Sustituye las descargas de yfinance por datos sintéticos."""

    def price_of(ticker: str) -> float:
        return BASE_PRICE.get(ticker, 400.0)

    def fake_daily(tickers, market_cfg=None):
        result = market_data.BatchResult()

        for ticker in tickers:
            if ticker == "ROTA":
                result.failed[ticker] = "lote fallido: simulado"
                continue

            result.data[ticker] = daily_frame(price_of(ticker))

        return result

    def fake_hist(tickers, market_cfg=None):
        result = market_data.BatchResult()

        for ticker in tickers:
            if ticker == "ROTA":
                continue

            result.data[ticker] = intraday_history(
                price_of(ticker)
            )

        return result

    def fake_today(tickers, session_date, market_cfg=None):
        result = market_data.BatchResult()

        for ticker in tickers:
            if ticker == "ROTA":
                result.failed[ticker] = "sin barras de hoy"
                continue

            gap = GAP.get(ticker, 0.004)
            volume = 200.0 if ticker == "BARATA" else 5_000.0

            result.data[ticker] = intraday_today(
                price_of(ticker),
                gap,
                volume,
            )

        return result

    monkey["daily"] = market_data.download_daily
    monkey["hist"] = market_data.download_intraday_history
    monkey["today"] = market_data.download_intraday_today

    market_data.download_daily = fake_daily
    market_data.download_intraday_history = fake_hist
    market_data.download_intraday_today = fake_today


def restore_downloads(monkey: dict) -> None:
    market_data.download_daily = monkey["daily"]
    market_data.download_intraday_history = monkey["hist"]
    market_data.download_intraday_today = monkey["today"]


# --------------------------------------------------------------------------- #
# Noticias sintéticas
# --------------------------------------------------------------------------- #

class StaticNews(news_module.NewsProvider):
    name = "test"

    def fetch(self, ticker, since):
        if ticker != "FUERTE":
            return []

        published = (
            utils.now_ny() - timedelta(hours=2)
        ).isoformat()

        return [
            news_module.NewsItem(
                ticker=ticker,
                headline="Company reports record quarterly earnings",
                publisher="Reuters",
                published_at=published,
                age_hours=2.0,
                url="https://example.com/e",
                summary="",
                source=self.name,
                category="earnings",
                category_label="Resultados",
                is_hard_catalyst=True,
            )
        ]


class Env:
    gemini_api_key = ""
    alphavantage_api_key = ""
    gemini_enabled = False
    telegram_enabled = False
    alphavantage_enabled = False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run(use_gemini=False, gemini_stub=None):
    """Ejecuta el pipeline con todo lo externo sustituido."""

    monkey: dict = {}
    install_fake_downloads(monkey)

    original_providers = news_module.build_providers
    original_news_cfg = config.NEWS
    original_analyze = ga.analyze

    news_module.build_providers = (
        lambda env=None, news_cfg=None: [StaticNews()]
    )

    config.NEWS = config.NewsConfig(
        pause_seconds=0.0
    )

    if gemini_stub is not None:
        ga.analyze = gemini_stub
        main.gemini_analyzer.analyze = gemini_stub

    try:
        return main.run_pipeline(
            UNIVERSE,
            SESSION_INFO,
            use_gemini=use_gemini,
            wait_for_snapshot=False,
            env=Env(),
            save_bars=False,
        )
    finally:
        restore_downloads(monkey)
        news_module.build_providers = original_providers
        config.NEWS = original_news_cfg
        ga.analyze = original_analyze
        main.gemini_analyzer.analyze = original_analyze


# --------------------------------------------------------------------------- #
# Helpers de niveles
# --------------------------------------------------------------------------- #

# Niveles que sabemos que existen según _LEVELS_FROM_INDICATORS.
# No se inventan etiquetas como "high_20d" o "prev_high".
REAL_RESISTANCE_LABELS = {
    "max_5d",
    "max_20d",
    "high_52w",
}


def _level_map(analysis: dict) -> dict:
    """Convierte analysis['levels'] en {label: price}."""

    return {
        row["label"]: row["price"]
        for row in analysis.get("levels", [])
        if technical_analysis.is_num(row.get("price"))
    }


def _real_resistance_levels(
    analysis: dict,
    price: float,
) -> dict:
    """Devuelve niveles reales de resistencia por encima del precio."""

    levels = _level_map(analysis)

    return {
        label: value
        for label, value in levels.items()
        if label in REAL_RESISTANCE_LABELS
        and value > price
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_pipeline_completo_sin_gemini():
    result = run(use_gemini=False)

    assert result["session_date"] == "2026-09-21"
    assert result["universe_size"] == 4

    # Un ticker sin datos no detiene el proceso.
    assert "ROTA" in result["failures"]["daily"]
    assert "ROTA" not in result["candidates"]

    # El resto sí se analiza.
    assert set(result["analyses"]) >= {
        "FUERTE",
        "PLANA",
        "BARATA",
    }

    # BARATA queda fuera por precio; PLANA por gap insuficiente.
    log = {
        row["ticker"]: row
        for row in result["filter_log"]
    }

    assert "precio" in log["BARATA"]["exclusion_reason"]
    assert "gap" in log["PLANA"]["exclusion_reason"]

    # FUERTE es la candidata, con catalizador confirmado.
    assert result["candidates"] == ["FUERTE"]
    assert result["news"]["FUERTE"]["catalyst_confirmed"] is True

    # Sin Gemini: informe degradado, sin picks.
    assert result["gemini"]["available"] is False
    assert result["gemini"]["picks"] == []
    assert ga.UNAVAILABLE_NOTE in result["gemini"]["no_trade_reason"]


def test_pipeline_calcula_rvol_y_niveles():
    analysis = run(
        use_gemini=False
    )["analyses"]["FUERTE"]

    assert analysis["premarket"]["premarket_quality"] == "ok"

    assert technical_analysis.is_num(
        analysis["gap_pct"]
    )
    assert analysis["gap_pct"] > 3.0

    assert technical_analysis.is_num(
        analysis["rvol"]["rvol"]
    )
    assert (
        analysis["rvol"]["sessions_used"]
        >= config.RVOL.min_sessions
    )

    labels = {
        row["label"]
        for row in analysis["levels"]
    }

    assert labels >= {
        "prev_close",
        "sma20",
    }


def test_pipeline_con_gemini_valida_los_picks():
    """Gemini propone; Python calcula y valida los campos críticos."""

    captured: dict = {}

    def stub(
        session_date,
        snapshot_ts,
        session_type,
        context,
        candidates,
        api_key="",
        model="",
        direction_mode=None,
        cfg=None,
        client=None,
    ):
        assert candidates, (
            "Gemini no debe recibir cero candidatas"
        )

        captured["payload"] = ga.build_payload(
            session_date,
            snapshot_ts,
            session_type,
            context,
            candidates,
        )

        candidate = candidates[0]
        analysis = candidate["analysis"]

        price = analysis["premarket"]["premarket_price"]
        levels = _level_map(analysis)

        # Debe existir al menos una resistencia real.
        real_resistances = _real_resistance_levels(
            analysis,
            price,
        )

        assert real_resistances, (
            "El fixture debe producir al menos "
            "un nivel real de resistencia."
        )

        # El stop utiliza un nivel real conocido.
        assert "prev_close" in levels
        stop_level = levels["prev_close"]

        # El target utiliza el nivel real disponible más alto.
        target_level = max(
            real_resistances.values()
        )

        raw = {
            "market_view": "Mercado tranquilo",
            "no_trade": False,
            "no_trade_reason": "",
            "picks": [
                {
                    "ticker": candidate["ticker"],
                    "company": "Fuerte SA",
                    "direction": "LONG",
                    "catalyst": "Resultados",
                    "catalyst_source": "Reuters",
                    "trend": "Alcista",
                    "technical_summary": "Sobre medias",
                    "support_level": "prev_close",
                    "resistance_level": "N/A",
                    "fade_risk": "LOW",
                    "fade_risk_explanation": "Volumen alto",

                    # No congelamos un precio absoluto.
                    "entry_zone_low": price - 0.1,
                    "entry_zone_high": price + 0.1,

                    "stop": stop_level,
                    "target_1": target_level,
                    "target_2": None,
                    "exit_time": "15:30 ET",
                    "invalidation": "Pérdida del stop",
                    "decision": "BUY",
                    "confidence": "HIGH",
                    "reason": "Configuración limpia",
                }
            ],
        }

        return ga.validate_response(
            raw,
            {
                c["ticker"]: c["analysis"]
                for c in candidates
            },
            model="fake",
        )

    result = run(
        use_gemini=True,
        gemini_stub=stub,
    )

    assert result["gemini"]["available"] is True
    assert result["gemini"]["picks"]

    pick = result["gemini"]["picks"][0]

    assert pick["ticker"] == "FUERTE"
    assert pick["decision"] == "BUY"

    # Los cálculos críticos deben proceder de Python.
    assert pick["rr_computed_by"] == "python"
    assert technical_analysis.is_num(
        pick["rr"]
    )
    assert technical_analysis.is_num(
        pick["max_valid_entry"]
    )

    # Invariantes de una operación LONG válida.
    assert pick["rr"] > 0
    assert pick["stop"] < pick["max_valid_entry"]
    assert pick["target_1"] > pick["max_valid_entry"]

    # Verificación independiente del RR calculado por Python.
    expected_rr = (
        (pick["target_1"] - pick["max_valid_entry"])
        / (pick["max_valid_entry"] - pick["stop"])
    )

    assert np.isclose(
        pick["rr"],
        expected_rr,
        rtol=1e-9,
        atol=1e-9,
    )

    # Verificación del payload enviado a Gemini.
    payload = captured["payload"]

    assert payload["candidatas"]

    candidate_payload = payload["candidatas"][0]

    assert (
        candidate_payload["news"]["catalyst_confirmed"]
        is True
    )
    assert candidate_payload["levels"]

    assert payload["contexto_mercado"]["regimen"]


def test_niveles_reales_se_conservan_en_el_analisis():
    """El fixture genera niveles históricos reales por encima del precio."""

    analysis = run(
        use_gemini=False
    )["analyses"]["FUERTE"]

    price = analysis["premarket"]["premarket_price"]
    levels = _level_map(analysis)

    assert technical_analysis.is_num(price)

    # Al menos uno de los niveles históricos definidos por
    # _LEVELS_FROM_INDICATORS debe estar por encima del precio.
    real_resistances = _real_resistance_levels(
        analysis,
        price,
    )

    assert real_resistances

    # El fixture fue diseñado específicamente para generar high_52w.
    assert "high_52w" in levels
    assert levels["high_52w"] > price


def test_no_se_fuerza_extension_sintetica_si_existe_nivel_real():
    """La presencia de un nivel real satisface la resistencia."""

    analysis = run(
        use_gemini=False
    )["analyses"]["FUERTE"]

    price = analysis["premarket"]["premarket_price"]

    real_resistances = _real_resistance_levels(
        analysis,
        price,
    )

    # El fixture tiene niveles reales: no necesitamos fabricar
    # una resistencia adicional para el caso normal.
    assert real_resistances

    # El nivel histórico principal está disponible.
    assert "high_52w" in real_resistances


def test_resultado_del_pipeline_es_serializable():
    result = run(use_gemini=False)

    texto = json.dumps(
        utils.sanitize_for_json(result),
        ensure_ascii=False,
    )

    assert "filter_log" in texto
    assert "market_context" in texto


def test_prefiltro_solo_pide_noticias_a_los_tickers_con_gap():
    result = run(use_gemini=False)

    # PLANA (gap 0,1 %) no debe haber pasado
    # el pre-filtro de noticias.
    assert "PLANA" not in result["news"]
    assert "FUERTE" in result["news"]


def test_wait_until_no_espera_si_la_hora_ya_paso():
    llamadas = []

    main.wait_until(
        datetime.min.time(),
        utils.now_ny().date(),
        sleep=llamadas.append,
    )

    assert llamadas == []


def test_wait_until_ignora_fechas_que_no_son_hoy():
    llamadas = []

    main.wait_until(
        config.SCHEDULE.snapshot_time,
        date(2020, 1, 1),
        sleep=llamadas.append,
    )

    assert llamadas == []


def test_no_se_llama_a_gemini_si_no_hay_candidatas():
    def stub(*args, **kwargs):
        raise AssertionError(
            "no debería llamarse a Gemini sin candidatas"
        )

    monkey: dict = {}
    install_fake_downloads(monkey)

    original = ga.analyze
    original_cfg = config.FILTER

    main.gemini_analyzer.analyze = stub

    config.FILTER = config.FilterConfig(
        min_score=99.0
    )

    try:
        result = main.run_pipeline(
            UNIVERSE,
            SESSION_INFO,
            use_gemini=True,
            wait_for_snapshot=False,
            env=Env(),
            save_bars=False,
        )
    finally:
        restore_downloads(monkey)
        main.gemini_analyzer.analyze = original
        config.FILTER = original_cfg

    assert result["candidates"] == []
    assert result["no_trade"] is True
    assert "No se fuerza" in result["no_trade_reason"]
    assert (
        candidate_filter.NO_CANDIDATES_REASON
        == result["no_trade_reason"]
    )