"""news.py — Noticias por ticker, clasificación por categorías y catalizador.

Diseño
------
* ``NewsProvider`` es la interfaz; cada fuente es una clase independiente. Añadir
  una fuente nueva no obliga a tocar el resto del agente.
* ``YahooNewsProvider`` (gratis, sin clave) usa ``yf.Ticker(t).news``. Ese campo
  **no es una API estable**: Yahoo ha servido al menos dos formatos distintos
  (plano y anidado bajo ``content``). El parseo es defensivo: lo que no se
  reconoce se descarta con un aviso en el log y el proceso continúa.
* ⚙️ ``AlphaVantageNewsProvider`` es opcional: solo se instancia si existe
  ``ALPHAVANTAGE_API_KEY`` y solo se usa para las finalistas, porque el plan
  gratuito tiene un límite diario muy bajo (25 peticiones/día en el momento de
  escribir esto; verifica el límite vigente en la documentación de Alpha Vantage).
  Si la cuota se agota, el proveedor se apaga solo y no rompe el informe.

Catalizador
-----------
``catalyst_confirmed`` es ``True`` solo si hay al menos una noticia de una
categoría **dura** publicada dentro de ``NEWS.catalyst_window_hours`` (48 h por
defecto). Las categorías ``sector``, ``macro`` y ``other`` son contexto y nunca
confirman nada. Sin catalizador el texto es «Sin catalizador confirmado».

Regla global 2: ningún dato se inventa. Una noticia sin fecha fiable se descarta.
"""
from __future__ import annotations

import logging
import re
import time as time_module
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import config
from utils import now_ny, to_ny, to_yahoo_symbol

logger = logging.getLogger("trading_agent.news")

NO_CATALYST = "Sin catalizador confirmado"
OTHER = "other"


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@dataclass
class NewsItem:
    """Una noticia normalizada, lista para el informe y para Gemini."""

    ticker: str
    headline: str
    publisher: str
    published_at: str          # ISO 8601 en hora de Nueva York
    age_hours: float
    url: str
    summary: str
    source: str                # proveedor que la trajo
    category: str = OTHER
    category_label: str = ""
    is_hard_catalyst: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TickerNews:
    """Noticias de un ticker y el veredicto sobre su catalizador."""

    ticker: str
    items: list[NewsItem] = field(default_factory=list)
    catalyst_confirmed: bool = False
    catalyst_text: str = NO_CATALYST
    catalyst_category: str = ""
    catalyst_source: str = ""
    catalyst_url: str = ""
    catalyst_time: str = ""
    has_soft_news: bool = False
    providers_used: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["items"] = [item.to_dict() if isinstance(item, NewsItem) else item
                         for item in self.items]
        return data


# --------------------------------------------------------------------------- #
# Clasificación por palabras clave
# --------------------------------------------------------------------------- #
def normalize_text(text: Any) -> str:
    """Pasa a minúsculas y quita tildes, para comparar palabras clave ES/EN."""
    if not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_accents.lower().split())


@lru_cache(maxsize=1024)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Expresión regular de una palabra clave, con límites de palabra.

    Evita falsos positivos por subcadenas: ``"sec"`` no debe casar dentro de
    ``"semiconductores"``.
    """
    return re.compile(rf"\b{re.escape(normalize_text(keyword))}\b")


def _matches(keyword: str, haystack: str) -> bool:
    """True si la palabra clave aparece como palabra completa."""
    normalized = normalize_text(keyword)
    return bool(normalized) and _keyword_pattern(normalized).search(haystack) is not None


def classify_news(headline: str, summary: str = "") -> str:
    """Devuelve la categoría de una noticia según ``config.NEWS_CATEGORY_KEYWORDS``.

    Gana la primera categoría del diccionario cuyas palabras clave aparezcan; el
    titular manda sobre el resumen. Sin coincidencias: ``"other"``.
    """
    title_text = normalize_text(headline)
    body_text = normalize_text(summary)
    for haystack in (title_text, f"{title_text} {body_text}".strip()):
        if not haystack:
            continue
        for category, keywords in config.NEWS_CATEGORY_KEYWORDS.items():
            if any(_matches(word, haystack) for word in keywords):
                return category
    return OTHER


def is_hard_category(category: str) -> bool:
    """True si la categoría puede confirmar un catalizador."""
    return category not in config.SOFT_NEWS_CATEGORIES


def category_label(category: str) -> str:
    """Nombre legible en español de una categoría."""
    return config.NEWS_CATEGORY_LABELS.get(category, category)


# --------------------------------------------------------------------------- #
# Ayudas de parseo
# --------------------------------------------------------------------------- #
def _first_str(mapping: Mapping[str, Any], *keys: str) -> str:
    """Primer valor de texto no vacío entre ``keys``."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _nested_str(mapping: Mapping[str, Any], key: str, *subkeys: str) -> str:
    """Texto dentro de un sub-diccionario (``{"provider": {"displayName": ...}}``)."""
    inner = mapping.get(key)
    if isinstance(inner, Mapping):
        return _first_str(inner, *subkeys)
    if isinstance(inner, str) and inner.strip() and not subkeys:
        return inner.strip()
    return ""


def parse_timestamp(value: Any) -> datetime | None:
    """Convierte epoch, datetime o texto ISO a datetime con zona de Nueva York.

    Acepta ``1712345678`` (segundos), ``1712345678000`` (milisegundos),
    ``"2026-09-21T12:30:00Z"`` y ``"20260921T123000"`` (Alpha Vantage).
    Devuelve ``None`` si no se puede interpretar (la noticia se descartará).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return to_ny(aware)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e11:          # viene en milisegundos
            seconds /= 1000.0
        if seconds <= 0:
            return None
        try:
            return to_ny(datetime.fromtimestamp(seconds, tz=timezone.utc))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return parse_timestamp(int(text))
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:                    # formato compacto de Alpha Vantage
                parsed = datetime.strptime(text, "%Y%m%dT%H%M%S")
            except ValueError:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return to_ny(parsed)
    return None


def _canonical_url(url: str) -> str:
    """URL sin parámetros de seguimiento ni barra final (para deduplicar)."""
    if not url:
        return ""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    host = parts.netloc.lower().removeprefix("www.")
    return f"{host}{path}" if host else path


def dedupe_items(items: Iterable[NewsItem]) -> list[NewsItem]:
    """Elimina repetidas por URL canónica o por titular normalizado."""
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        keys = {k for k in (_canonical_url(item.url), normalize_text(item.headline)) if k}
        if keys & seen:
            continue
        seen |= keys
        unique.append(item)
    return unique


def _build_item(
    ticker: str, headline: str, publisher: str, published: datetime,
    url: str, summary: str, source: str, reference: datetime,
) -> NewsItem:
    """Crea un ``NewsItem`` ya clasificado y con su antigüedad en horas."""
    category = classify_news(headline, summary)
    age = round((reference - published).total_seconds() / 3600.0, 2)
    return NewsItem(
        ticker=ticker,
        headline=headline,
        publisher=publisher or "N/A",
        published_at=published.isoformat(),
        age_hours=age,
        url=url,
        summary=summary,
        source=source,
        category=category,
        category_label=category_label(category),
        is_hard_catalyst=is_hard_category(category),
    )


# --------------------------------------------------------------------------- #
# Interfaz de proveedores
# --------------------------------------------------------------------------- #
class NewsProvider(ABC):
    """Fuente de noticias. Una clase por proveedor."""

    name = "base"

    @abstractmethod
    def fetch(self, ticker: str, since: datetime) -> list[NewsItem]:
        """Noticias de ``ticker`` publicadas a partir de ``since`` (hora NY).

        Nunca debe lanzar excepciones hacia arriba: ante un fallo, registra el
        error y devuelve una lista vacía.
        """

    @property
    def available(self) -> bool:
        """True si el proveedor puede seguir siendo usado (cuota, clave...)."""
        return True


class YahooNewsProvider(NewsProvider):
    """Noticias de Yahoo Finance mediante ``yf.Ticker(t).news``.

    Parseo defensivo de los dos formatos conocidos:

    * plano: ``{"title", "publisher", "link", "providerPublishTime"}``;
    * anidado: ``{"content": {"title", "pubDate", "provider": {"displayName"},
      "canonicalUrl": {"url"}, "summary"}}``.

    Si ningún elemento es interpretable, se registra el aviso y se devuelve [].
    """

    name = "yahoo"

    def __init__(self, news_cfg: config.NewsConfig | None = None) -> None:
        self.cfg = news_cfg or config.NEWS

    @staticmethod
    def _get_ticker(symbol: str) -> Any:
        import yfinance as yf  # noqa: PLC0415 - import diferido (tests sin red)

        return yf.Ticker(symbol)

    def _raw_news(self, symbol: str) -> list[Any]:
        """Devuelve la lista cruda de noticias de yfinance."""
        raw = self._get_ticker(symbol).news
        return list(raw) if isinstance(raw, (list, tuple)) else []

    def parse_item(self, raw: Any, ticker: str, reference: datetime) -> NewsItem | None:
        """Convierte un elemento crudo en ``NewsItem`` o ``None`` si no se entiende."""
        if not isinstance(raw, Mapping):
            return None
        body: Mapping[str, Any] = raw
        content = raw.get("content")
        if isinstance(content, Mapping):
            body = {**raw, **content}

        headline = _first_str(body, "title", "headline")
        if not headline:
            return None

        published = parse_timestamp(
            body.get("providerPublishTime")
            or body.get("pubDate")
            or body.get("displayTime")
            or body.get("published_at")
        )
        if published is None:
            return None

        publisher = (
            _first_str(body, "publisher", "provider_name")
            or _nested_str(body, "provider", "displayName", "name")
        )
        url = (
            _first_str(body, "link", "url")
            or _nested_str(body, "canonicalUrl", "url")
            or _nested_str(body, "clickThroughUrl", "url")
        )
        summary = _first_str(body, "summary", "description")
        return _build_item(ticker, headline, publisher, published, url, summary,
                           self.name, reference)

    def fetch(self, ticker: str, since: datetime) -> list[NewsItem]:
        """Noticias recientes de un ticker; [] si Yahoo falla o cambia de formato."""
        reference = now_ny()
        try:
            raw_items = self._raw_news(to_yahoo_symbol(ticker))
        except Exception as exc:  # noqa: BLE001 - aislamiento por ticker
            logger.error("%s: no se pudieron obtener noticias de Yahoo (%s)", ticker, exc)
            return []

        items: list[NewsItem] = []
        for raw in raw_items:
            try:
                item = self.parse_item(raw, ticker, reference)
            except Exception as exc:  # noqa: BLE001 - un elemento raro no rompe el resto
                logger.warning("%s: noticia de Yahoo ilegible (%s)", ticker, exc)
                continue
            if item is not None and parse_timestamp(item.published_at) >= since:
                items.append(item)

        if raw_items and not items:
            logger.warning(
                "%s: Yahoo devolvió %d noticias pero ninguna en un formato "
                "reconocible o dentro de la ventana.", ticker, len(raw_items),
            )
        return items


class AlphaVantageNewsProvider(NewsProvider):
    """⚙️ Opcional: endpoint ``NEWS_SENTIMENT`` de Alpha Vantage.

    Solo para las finalistas. Lleva su propio contador de llamadas
    (``NEWS.alphavantage_max_calls``) y se desactiva solo si la API responde con
    un aviso de límite alcanzado. Nunca interrumpe el informe.
    """

    name = "alphavantage"

    def __init__(self, api_key: str, news_cfg: config.NewsConfig | None = None) -> None:
        self.api_key = api_key
        self.cfg = news_cfg or config.NEWS
        self.calls_made = 0
        self.exhausted = False

    @property
    def available(self) -> bool:
        return bool(self.api_key) and not self.exhausted \
            and self.calls_made < self.cfg.alphavantage_max_calls

    def _request(self, params: dict[str, str]) -> Mapping[str, Any]:
        import requests  # noqa: PLC0415 - import diferido

        response = requests.get(
            self.cfg.alphavantage_url, params=params, timeout=self.cfg.request_timeout
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, Mapping) else {}

    def fetch(self, ticker: str, since: datetime) -> list[NewsItem]:
        """Noticias del ticker; [] si no hay cuota, hay error o la respuesta es rara."""
        if not self.available:
            return []

        reference = now_ny()
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": since.astimezone(timezone.utc).strftime("%Y%m%dT%H%M"),
            "sort": "LATEST",
            "limit": "50",
            "apikey": self.api_key,
        }
        self.calls_made += 1
        try:
            payload = self._request(params)
        except Exception as exc:  # noqa: BLE001 - degradación sin error
            logger.warning("%s: Alpha Vantage no respondió (%s)", ticker, exc)
            return []

        # Límite alcanzado o petición rechazada: apagamos el proveedor.
        for key in ("Note", "Information", "Error Message"):
            if key in payload:
                self.exhausted = True
                logger.warning(
                    "Alpha Vantage desactivado para el resto de la ejecución: %s",
                    str(payload[key])[:160],
                )
                return []

        feed = payload.get("feed")
        if not isinstance(feed, list):
            logger.warning("%s: respuesta de Alpha Vantage sin campo 'feed'.", ticker)
            return []

        items: list[NewsItem] = []
        for raw in feed:
            if not isinstance(raw, Mapping):
                continue
            headline = _first_str(raw, "title")
            published = parse_timestamp(raw.get("time_published"))
            if not headline or published is None or published < since:
                continue
            items.append(_build_item(
                ticker, headline, _first_str(raw, "source"), published,
                _first_str(raw, "url"), _first_str(raw, "summary"),
                self.name, reference,
            ))
        return items


# --------------------------------------------------------------------------- #
# Construcción del veredicto por ticker
# --------------------------------------------------------------------------- #
def build_ticker_news(
    ticker: str, items: Sequence[NewsItem],
    news_cfg: config.NewsConfig | None = None,
    reference: datetime | None = None,
) -> TickerNews:
    """Ordena, deduplica y decide si hay catalizador confirmado.

    Las noticias se ordenan de más reciente a más antigua (así el pre-market y
    las últimas horas van primero) y se conservan como mucho
    ``max_items_per_ticker``. El catalizador es la noticia dura más reciente
    dentro de ``catalyst_window_hours``.
    """
    news_cfg = news_cfg or config.NEWS
    reference = reference or now_ny()

    unique = dedupe_items(items)
    unique.sort(key=lambda i: i.published_at, reverse=True)
    kept = unique[: news_cfg.max_items_per_ticker]

    window_start = reference - timedelta(hours=news_cfg.catalyst_window_hours)
    result = TickerNews(ticker=ticker, items=kept)
    result.providers_used = sorted({item.source for item in kept})

    for item in kept:
        published = parse_timestamp(item.published_at)
        if published is None or published < window_start:
            continue
        if item.is_hard_catalyst and not result.catalyst_confirmed:
            result.catalyst_confirmed = True
            result.catalyst_text = item.headline
            result.catalyst_category = item.category_label
            result.catalyst_source = item.publisher
            result.catalyst_url = item.url
            result.catalyst_time = item.published_at
        elif not item.is_hard_catalyst:
            result.has_soft_news = True

    if not result.catalyst_confirmed:
        result.catalyst_text = NO_CATALYST
    return result


def build_providers(
    env: Any | None = None, news_cfg: config.NewsConfig | None = None,
) -> list[NewsProvider]:
    """Lista de proveedores disponibles: Yahoo siempre, Alpha Vantage si hay clave."""
    news_cfg = news_cfg or config.NEWS
    env = env if env is not None else config.load_env_settings()
    providers: list[NewsProvider] = [YahooNewsProvider(news_cfg)]
    api_key = getattr(env, "alphavantage_api_key", "") or ""
    if api_key:
        providers.append(AlphaVantageNewsProvider(api_key, news_cfg))
        logger.info("Alpha Vantage activado como segunda fuente de noticias.")
    return providers


def gather_news(
    tickers: Sequence[str],
    providers: Sequence[NewsProvider] | None = None,
    news_cfg: config.NewsConfig | None = None,
    reference: datetime | None = None,
    sleep: Any = time_module.sleep,
) -> dict[str, TickerNews]:
    """Pide noticias de cada ticker a todos los proveedores disponibles.

    Se respeta ``max_tickers`` (los tickers ya llegan ordenados por prioridad
    desde el pre-filtro) y se hace una pausa corta entre tickers. Un fallo de un
    ticker o de un proveedor se registra y no detiene al resto (regla global 5).
    """
    news_cfg = news_cfg or config.NEWS
    reference = reference or now_ny()
    providers = list(providers) if providers is not None else build_providers(
        news_cfg=news_cfg)
    since = reference - timedelta(hours=news_cfg.lookback_hours)

    selected = list(dict.fromkeys(tickers))[: news_cfg.max_tickers]
    if len(tickers) > len(selected):
        logger.info(
            "Noticias: se consultan %d de %d tickers (límite NEWS.max_tickers).",
            len(selected), len(tickers),
        )

    results: dict[str, TickerNews] = {}
    for position, ticker in enumerate(selected):
        collected: list[NewsItem] = []
        errors: list[str] = []
        for provider in providers:
            if not provider.available:
                continue
            try:
                collected.extend(provider.fetch(ticker, since))
            except Exception as exc:  # noqa: BLE001 - proveedor aislado
                errors.append(f"{provider.name}: {exc}")
                logger.error("%s: fallo del proveedor %s (%s)", ticker, provider.name, exc)

        result = build_ticker_news(ticker, collected, news_cfg, reference)
        result.error = "; ".join(errors)
        results[ticker] = result

        if position < len(selected) - 1 and news_cfg.pause_seconds > 0:
            sleep(news_cfg.pause_seconds)

    confirmed = sum(1 for r in results.values() if r.catalyst_confirmed)
    logger.info(
        "Noticias: %d tickers consultados, %d con catalizador confirmado.",
        len(results), confirmed,
    )
    return results


def enrich_with_alphavantage(
    tickers: Sequence[str], current: dict[str, TickerNews], provider: NewsProvider,
    news_cfg: config.NewsConfig | None = None, reference: datetime | None = None,
) -> dict[str, TickerNews]:
    """⚙️ Segunda pasada solo para las finalistas, con Alpha Vantage.

    Mezcla las noticias nuevas con las que ya había y recalcula el catalizador.
    Si el proveedor no está disponible, devuelve ``current`` sin tocar.
    """
    news_cfg = news_cfg or config.NEWS
    reference = reference or now_ny()
    if not provider.available:
        return current

    since = reference - timedelta(hours=news_cfg.lookback_hours)
    for ticker in list(tickers)[: news_cfg.alphavantage_tickers]:
        if not provider.available:
            break
        try:
            extra = provider.fetch(ticker, since)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: fallo de %s (%s)", ticker, provider.name, exc)
            continue
        if not extra:
            continue
        previous = current.get(ticker)
        merged = list(previous.items) if previous else []
        current[ticker] = build_ticker_news(
            ticker, merged + extra, news_cfg, reference)
    return current


def catalyst_map(news: Mapping[str, TickerNews]) -> dict[str, bool]:
    """Diccionario ``ticker -> catalizador confirmado`` (lo usa el filtro)."""
    return {ticker: item.catalyst_confirmed for ticker, item in news.items()}


__all__ = [
    "NO_CATALYST", "NewsItem", "TickerNews", "NewsProvider", "YahooNewsProvider",
    "AlphaVantageNewsProvider", "classify_news", "is_hard_category", "normalize_text",
    "parse_timestamp", "dedupe_items", "build_ticker_news", "build_providers",
    "gather_news", "enrich_with_alphavantage", "catalyst_map",
]
