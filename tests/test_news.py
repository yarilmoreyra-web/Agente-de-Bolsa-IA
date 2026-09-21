"""Tests de news.py: parseo de ambos formatos de Yahoo, clasificación y catalizador.

Sin red y sin yfinance: los proveedores se sustituyen por dobles de prueba.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import config
import news
from utils import NY_TZ

REFERENCE = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)


# --------------------------------------------------------------------------- #
# Dobles de prueba
# --------------------------------------------------------------------------- #
class FakeYahoo(news.YahooNewsProvider):
    """Yahoo con una respuesta cruda fija (no toca la red)."""

    def __init__(self, raw, fail: bool = False):
        super().__init__()
        self._raw = raw
        self._fail = fail

    def _raw_news(self, symbol):
        if self._fail:
            raise RuntimeError("Yahoo caído")
        return list(self._raw)


class FakeAlphaVantage(news.AlphaVantageNewsProvider):
    """Alpha Vantage con una respuesta JSON fija."""

    def __init__(self, payload):
        super().__init__(api_key="clave-de-prueba")
        self.payload = payload

    def _request(self, params):
        return self.payload


def _epoch(hours_ago: float) -> int:
    moment = REFERENCE - timedelta(hours=hours_ago)
    return int(moment.astimezone(timezone.utc).timestamp())


def _iso(hours_ago: float) -> str:
    moment = REFERENCE - timedelta(hours=hours_ago)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Clasificación
# --------------------------------------------------------------------------- #
def test_clasificacion_por_palabras_clave():
    assert news.classify_news("NVDA Q3 earnings beat estimates") == "earnings"
    assert news.classify_news("La compañía publica sus resultados") == "earnings"
    assert news.classify_news("FDA approval for phase 3 drug") == "fda"
    assert news.classify_news("Morgan Stanley upgrade to overweight") == "rating"
    assert news.classify_news("Acuerdo de adquisición de su rival") == "m_and_a"
    assert news.classify_news("El sector de semiconductores sube") == "sector"
    assert news.classify_news("La Fed mantiene los tipos de interés") == "macro"
    assert news.classify_news("Un día cualquiera en la bolsa") == "other"


def test_categorias_blandas_no_son_catalizador():
    for soft in config.SOFT_NEWS_CATEGORIES:
        assert not news.is_hard_category(soft)
    assert news.is_hard_category("earnings")
    assert news.is_hard_category("fda")


def test_normalizacion_quita_tildes_y_mayusculas():
    assert news.normalize_text("Adquisición  DE  Rival") == "adquisicion de rival"


# --------------------------------------------------------------------------- #
# Fechas
# --------------------------------------------------------------------------- #
def test_parse_timestamp_admite_varios_formatos():
    assert news.parse_timestamp(_epoch(1)).hour == 7          # epoch en segundos
    assert news.parse_timestamp(_epoch(1) * 1000).hour == 7   # milisegundos
    assert news.parse_timestamp("2026-09-21T11:00:00Z").hour == 7
    assert news.parse_timestamp("20260921T110000").hour == 7
    assert news.parse_timestamp("no es una fecha") is None
    assert news.parse_timestamp(None) is None
    assert news.parse_timestamp(0) is None


# --------------------------------------------------------------------------- #
# Formatos de Yahoo
# --------------------------------------------------------------------------- #
def test_yahoo_formato_plano():
    raw = [{
        "title": "NVDA reports record quarterly results",
        "publisher": "Reuters",
        "link": "https://example.com/nvda-earnings",
        "providerPublishTime": _epoch(3),
    }]
    items = FakeYahoo(raw).fetch("NVDA", REFERENCE - timedelta(hours=96))
    assert len(items) == 1
    item = items[0]
    assert item.publisher == "Reuters"
    assert item.category == "earnings"
    assert item.is_hard_catalyst is True
    assert item.published_at.startswith("2026-09-21")
    assert isinstance(item.age_hours, float)


def test_yahoo_formato_anidado_en_content():
    raw = [{
        "id": "abc",
        "content": {
            "title": "FDA approval granted to the company",
            "summary": "Resumen de la noticia",
            "pubDate": _iso(2),
            "provider": {"displayName": "Bloomberg"},
            "canonicalUrl": {"url": "https://example.com/fda"},
        },
    }]
    items = FakeYahoo(raw).fetch("ABC", REFERENCE - timedelta(hours=96))
    assert len(items) == 1
    assert items[0].publisher == "Bloomberg"
    assert items[0].url == "https://example.com/fda"
    assert items[0].category == "fda"


def test_yahoo_formato_irreconocible_devuelve_lista_vacia():
    raw = [{"algo": "raro"}, "texto suelto", {"title": "Sin fecha"}, None]
    assert FakeYahoo(raw).fetch("XYZ", REFERENCE - timedelta(hours=96)) == []


def test_yahoo_error_no_propaga_excepcion():
    assert FakeYahoo([], fail=True).fetch("XYZ", REFERENCE - timedelta(hours=96)) == []


def test_yahoo_descarta_noticias_anteriores_a_since():
    raw = [{"title": "Earnings de hace una semana", "publisher": "X",
            "link": "https://example.com/vieja", "providerPublishTime": _epoch(24 * 7)}]
    assert FakeYahoo(raw).fetch("XYZ", REFERENCE - timedelta(hours=96)) == []


# --------------------------------------------------------------------------- #
# Deduplicación
# --------------------------------------------------------------------------- #
def test_dedupe_por_url_y_por_titular():
    raw = [
        {"title": "Contract awarded to the company", "publisher": "A",
         "link": "https://www.example.com/deal?utm_source=x",
         "providerPublishTime": _epoch(2)},
        {"title": "Contract awarded to the company", "publisher": "B",
         "link": "https://otro.com/deal", "providerPublishTime": _epoch(3)},
        {"title": "Otro titular distinto", "publisher": "C",
         "link": "https://example.com/deal/", "providerPublishTime": _epoch(4)},
    ]
    items = FakeYahoo(raw).fetch("XYZ", REFERENCE - timedelta(hours=96))
    assert len(items) == 3
    assert len(news.dedupe_items(items)) == 1


# --------------------------------------------------------------------------- #
# Catalizador
# --------------------------------------------------------------------------- #
def _item(headline: str, hours_ago: float, url: str = "") -> news.NewsItem:
    published = REFERENCE - timedelta(hours=hours_ago)
    category = news.classify_news(headline)
    return news.NewsItem(
        ticker="XYZ", headline=headline, publisher="Reuters",
        published_at=published.isoformat(), age_hours=hours_ago,
        url=url or f"https://example.com/{abs(hash(headline))}", summary="",
        source="yahoo", category=category,
        category_label=news.category_label(category),
        is_hard_catalyst=news.is_hard_category(category),
    )


def test_catalizador_confirmado_dentro_de_la_ventana():
    result = news.build_ticker_news(
        "XYZ", [_item("Company beats estimates in Q3 earnings", 5)],
        reference=REFERENCE)
    assert result.catalyst_confirmed is True
    assert result.catalyst_category == "Resultados"
    assert result.catalyst_text != news.NO_CATALYST


def test_catalizador_fuera_de_ventana_no_cuenta():
    result = news.build_ticker_news(
        "XYZ", [_item("Company beats estimates in Q3 earnings", 60)],
        reference=REFERENCE)
    assert result.catalyst_confirmed is False
    assert result.catalyst_text == news.NO_CATALYST


def test_noticia_blanda_no_confirma_catalizador():
    result = news.build_ticker_news(
        "XYZ", [_item("El sector tecnológico rebota", 2)], reference=REFERENCE)
    assert result.catalyst_confirmed is False
    assert result.has_soft_news is True
    assert result.catalyst_text == news.NO_CATALYST


def test_se_elige_la_noticia_dura_mas_reciente():
    items = [
        _item("FDA approval for the new drug", 20),
        _item("Company announces a large contract awarded", 1),
    ]
    result = news.build_ticker_news("XYZ", items, reference=REFERENCE)
    assert result.catalyst_confirmed is True
    assert "contract" in result.catalyst_text.lower()


def test_se_respeta_max_items_por_ticker():
    cfg = config.NewsConfig(max_items_per_ticker=2)
    items = [_item(f"Titular número {i}", i + 1) for i in range(6)]
    result = news.build_ticker_news("XYZ", items, cfg, reference=REFERENCE)
    assert len(result.items) == 2


# --------------------------------------------------------------------------- #
# gather_news
# --------------------------------------------------------------------------- #
class BoomProvider(news.NewsProvider):
    name = "boom"

    def fetch(self, ticker, since):
        raise RuntimeError("proveedor roto")


class StaticProvider(news.NewsProvider):
    name = "static"

    def __init__(self, by_ticker):
        self.by_ticker = by_ticker

    def fetch(self, ticker, since):
        return list(self.by_ticker.get(ticker, []))


def test_gather_news_aisla_fallos_y_respeta_el_limite():
    pauses = []
    provider = StaticProvider({"AAA": [_item("Q3 earnings beat", 2)]})
    cfg = config.NewsConfig(max_tickers=2, pause_seconds=0.1)
    result = news.gather_news(
        ["AAA", "BBB", "CCC"], [provider, BoomProvider()], cfg,
        reference=REFERENCE, sleep=pauses.append,
    )
    assert list(result) == ["AAA", "BBB"]
    assert result["AAA"].catalyst_confirmed is True
    assert result["BBB"].catalyst_text == news.NO_CATALYST
    assert "boom" in result["AAA"].error
    assert pauses == [0.1]          # una pausa entre los dos tickers, no al final


def test_catalyst_map():
    result = news.gather_news(
        ["AAA"], [StaticProvider({"AAA": [_item("FDA approval", 1)]})],
        reference=REFERENCE, sleep=lambda _s: None,
    )
    assert news.catalyst_map(result) == {"AAA": True}


# --------------------------------------------------------------------------- #
# Alpha Vantage (opcional)
# --------------------------------------------------------------------------- #
def test_alphavantage_parsea_el_feed():
    payload = {"feed": [{
        "title": "Company wins a major contract",
        "url": "https://example.com/av",
        "time_published": (REFERENCE - timedelta(hours=1)).astimezone(
            timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "source": "Benzinga",
        "summary": "Resumen",
    }]}
    items = FakeAlphaVantage(payload).fetch("XYZ", REFERENCE - timedelta(hours=96))
    assert len(items) == 1
    assert items[0].source == "alphavantage"
    assert items[0].category == "contract"


def test_alphavantage_se_apaga_al_agotar_la_cuota():
    provider = FakeAlphaVantage({"Note": "límite de peticiones alcanzado"})
    assert provider.fetch("XYZ", REFERENCE - timedelta(hours=96)) == []
    assert provider.exhausted is True
    assert provider.available is False


def test_alphavantage_sin_clave_no_esta_disponible():
    assert news.AlphaVantageNewsProvider("").available is False


def test_build_providers_incluye_alphavantage_solo_con_clave():
    class Env:
        alphavantage_api_key = ""

    assert [p.name for p in news.build_providers(Env())] == ["yahoo"]
    Env.alphavantage_api_key = "clave"
    assert [p.name for p in news.build_providers(Env())] == ["yahoo", "alphavantage"]
