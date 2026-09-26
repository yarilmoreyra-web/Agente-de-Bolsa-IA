"""Configuración central del agente de análisis pre-market.

Aquí viven TODOS los horarios, umbrales, pesos y rutas. Ninguna otra parte
del código debe tener números "mágicos": si quieres ajustar el comportamiento
del agente, se cambia en este archivo.

Las claves secretas (Gemini, Telegram, Alpha Vantage) NUNCA se escriben aquí:
se leen de variables de entorno (o de un archivo ``.env`` local).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------
# Constantes generales
# --------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
TIMEZONE: str = "America/New_York"

# Modelo de Gemini por defecto. Se puede cambiar SIN tocar el código con la
# variable de entorno GEMINI_MODEL (ejemplo: GEMINI_MODEL=gemini-2.5-flash).
DEFAULT_GEMINI_MODEL: str = "gemini-3.5-flash"

DIRECTION_LONG_ONLY = "long_only"
DIRECTION_BOTH = "both"


# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """Rutas de archivos y carpetas del proyecto."""

    base_dir: Path = BASE_DIR
    universe_file: Path = BASE_DIR / "lista_tickers.xlsx"
    extended_universe_file: Path = BASE_DIR / "lista_extendida.xlsx"
    env_file: Path = BASE_DIR / ".env"
    data_dir: Path = BASE_DIR / "data"
    history_dir: Path = BASE_DIR / "data" / "history"
    history_csv: Path = BASE_DIR / "data" / "history.csv"
    intraday_dir: Path = BASE_DIR / "data" / "intraday"
    reports_dir: Path = BASE_DIR / "data" / "reports"
    backtests_dir: Path = BASE_DIR / "data" / "backtests"
    backtest_summary_csv: Path = BASE_DIR / "data" / "backtest_summary.csv"
    macro_events_file: Path = BASE_DIR / "data" / "macro_events.json"
    logs_dir: Path = BASE_DIR / "logs"
    log_file: Path = BASE_DIR / "logs" / "agent.log"


PATHS = Paths()


# --------------------------------------------------------------------------
# Horarios (todos en hora de Nueva York)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScheduleConfig:
    """Cronograma diario del agente."""

    timezone: str = TIMEZONE
    # Momento en que se toma la "foto" de datos pre-market.
    snapshot_time: time = time(8, 45)
    # Hora a la que se envía el informe (30 min antes de la apertura).
    report_time: time = time(9, 0)
    # Ventana en la que una ejecución programada se considera válida.
    # Sirve para descartar el cron de horario de verano/invierno que no toca.
    run_window_start: time = time(8, 0)
    run_window_end: time = time(9, 25)
    premarket_start: time = time(4, 0)
    regular_open: time = time(9, 30)
    regular_close: time = time(16, 0)


SCHEDULE = ScheduleConfig()


# --------------------------------------------------------------------------
# Universo
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UniverseConfig:
    """Cómo se lee la lista de tickers."""

    ticker_column: str = "Ticker"
    # Filas iniciales del Excel en las que se busca la cabecera "Ticker".
    header_scan_rows: int = 10


UNIVERSE = UniverseConfig()


# --------------------------------------------------------------------------
# Datos de mercado (yfinance)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketDataConfig:
    """Parámetros de descarga de datos."""

    daily_period: str = "1y"
    intraday_interval: str = "5m"
    intraday_baseline_period: str = "1mo"
    batch_size: int = 30
    max_retries: int = 3
    retry_base_delay: float = 2.0
    # Si la última barra pre-market tiene más minutos que esto, el dato se
    # marca como "stale" (desactualizado).
    stale_minutes: int = 20
    # Mínimo de barras pre-market para no marcar el dato como "sparse".
    min_premarket_bars: int = 3
    # Hilos de yf.download (moderado, para no provocar bloqueos).
    download_threads: int = 4
    # Ventana corta para las barras de hoy (luego se filtra el día de hoy).
    intraday_today_period: str = "5d"


MARKET_DATA = MarketDataConfig()


@dataclass(frozen=True)
class RvolConfig:
    """Parámetros del cálculo de RVOL (volumen relativo pre-market)."""

    lookback_sessions: int = 20
    min_sessions: int = 10


RVOL = RvolConfig()


# --------------------------------------------------------------------------
# Filtro de candidatas
# --------------------------------------------------------------------------
# (Ver la definición completa y activa de FilterConfig/FILTER más abajo,
# junto con FilterWeights y FilterPenalties: aquí solo van los pesos
# globales de puntuación, que validate_config() comprueba que sumen 100.)


@dataclass(frozen=True)
class ScoringWeights:
    """Pesos de la puntuación 0-100. Deben sumar 100."""

    rvol: float = 25.0
    gap: float = 20.0
    catalyst: float = 20.0
    liquidity: float = 10.0
    relative_strength: float = 10.0
    technical: float = 10.0
    premarket_behavior: float = 5.0

    def total(self) -> float:
        """Suma de todos los pesos."""
        return (
            self.rvol
            + self.gap
            + self.catalyst
            + self.liquidity
            + self.relative_strength
            + self.technical
            + self.premarket_behavior
        )


WEIGHTS = ScoringWeights()


@dataclass(frozen=True)
class FadeConfig:
    """Umbrales y penalizaciones de riesgo gap-and-fade."""

    # Gap (en %) mayor que N veces el ATR% => extensión excesiva.
    gap_over_atr_multiple: float = 3.0
    # Precio pre-market por debajo del máximo pre-market en más de X %.
    pullback_from_high_pct: float = 2.0
    # Resistencia más cerca que N ATR del precio.
    resistance_min_distance_atr: float = 0.5
    # Penalizaciones (puntos que se restan a la puntuación).
    penalty_extension: float = 15.0
    penalty_pullback: float = 10.0
    penalty_resistance: float = 10.0
    penalty_no_catalyst: float = 10.0
    penalty_low_volume_for_gap: float = 10.0
    penalty_adverse_market: float = 10.0
    # VIX por encima de este valor se considera mercado adverso.
    vix_high: float = 25.0


FADE = FadeConfig()


# --------------------------------------------------------------------------
# Operación (entradas, stops, objetivos)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TradeConfig:
    """Reglas de riesgo/beneficio."""

    direction_mode: str = DIRECTION_LONG_ONLY  # "long_only" o "both"
    min_rr: float = 1.5
    max_picks: int = 3
    atr_stop_multiple: float = 1.0
    atr_target1_multiple: float = 1.5
    atr_target2_multiple: float = 2.5


TRADE = TradeConfig()
MIN_RR = TRADE.min_rr



# --------------------------------------------------------------------------
# Noticias y catalizadores
# --------------------------------------------------------------------------
# (Las palabras clave, categorías y NewsConfig activos están más abajo, junto
# a MarketContextConfig: NEWS_CATEGORY_KEYWORDS, NEWS_CATEGORY_LABELS y NEWS.)
# Gemini y Telegram
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GeminiConfig:
    """Parámetros de la llamada a Gemini y de la validación posterior."""

    # --- Llamada ---
    # Temperatura baja: queremos interpretación estable, no creatividad.
    temperature: float = 0.2
    max_output_tokens: int = 8192
    # Intentos totales ante 429/5xx o JSON inválido (1 llamada + 1 reintento).
    attempts: int = 5
    retry_base_delay: float = 5.0
    timeout_seconds: int = 90
    # Candidatas que se envían en la única llamada.
    max_candidates_sent: int = 10
    # Noticias por candidata que viajan en el prompt.
    max_news_per_candidate: int = 4

    # --- Validación en Python ---
    max_picks: int = 3
    # Un nivel propuesto por Gemini debe estar a menos de esta distancia (en ATR)
    # de algún nivel de la tabla entregada. Si no hay ATR, se usa el porcentaje.
    level_tolerance_atr: float = 0.75
    level_tolerance_pct: float = 1.5
    # Diferencia tolerada entre un número de mercado de Gemini y el de Python
    # antes de registrar un aviso (Python siempre prevalece).
    numeric_mismatch_pct: float = 0.5
    # Un pick con R/R insuficiente pasa a WAIT en lugar de descartarse.
    downgrade_low_rr_to_wait: bool = True


GEMINI = GeminiConfig()


@dataclass(frozen=True)
class TelegramConfig:
    """Parámetros del envío a Telegram."""

    # El límite de Telegram es 4096; dejamos margen por las etiquetas HTML.
    max_message_chars: int = 3800
    max_retries: int = 3
    retry_base_delay: float = 2.0
    timeout_seconds: int = 20


TELEGRAM = TelegramConfig()


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoggingConfig:
    """Parámetros de logging."""

    level: str = "INFO"
    max_bytes: int = 1_000_000
    backup_count: int = 3


LOGGING = LoggingConfig()


# --------------------------------------------------------------------------
# Variables de entorno (secretos)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvSettings:
    """Valores leídos de variables de entorno."""

    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    alphavantage_api_key: str = ""

    @property
    def gemini_enabled(self) -> bool:
        """True si hay clave de Gemini."""
        return bool(self.gemini_api_key)

    @property
    def telegram_enabled(self) -> bool:
        """True si hay token y chat_id de Telegram."""
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def alphavantage_enabled(self) -> bool:
        """True si hay clave de Alpha Vantage (opcional)."""
        return bool(self.alphavantage_api_key)


def load_env_file(path: Path) -> int:
    """Carga un archivo ``.env`` sencillo (CLAVE=valor) en ``os.environ``.

    - Ignora líneas vacías y comentarios (que empiecen por ``#``).
    - NO sobrescribe variables que ya existan en el entorno (así los
      secretos de GitHub Actions tienen prioridad sobre un ``.env``).
    - Devuelve cuántas variables se cargaron.
    """
    if not path.is_file():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def load_env_settings(env_file: Optional[Path] = None) -> EnvSettings:
    """Lee las variables de entorno y devuelve un ``EnvSettings``.

    ``GEMINI_MODEL`` usa ``or``: si la variable no existe o está vacía se usa
    ``DEFAULT_GEMINI_MODEL``. Así un secret vacío no pisa el valor por defecto.
    """
    load_env_file(env_file if env_file is not None else PATHS.env_file)
    return EnvSettings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=(os.getenv("GEMINI_MODEL") or "").strip()
        or DEFAULT_GEMINI_MODEL,
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY", "").strip(),
    )


# --------------------------------------------------------------------------- #
# Noticias
# --------------------------------------------------------------------------- #
# Categorías. Las "blandas" (sector, macro, other) son contexto y NUNCA
# confirman un catalizador; el resto son categorías "duras".
SOFT_NEWS_CATEGORIES: tuple[str, ...] = ("sector", "macro", "other")

# Reglas por palabras clave (minúsculas, sin tildes). Se evalúan en este orden:
# la primera categoría que casa, gana. ES/EN mezclados a propósito.
NEWS_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": (
        "earnings", "quarterly results", "q1 results", "q2 results", "q3 results",
        "q4 results", "eps", "beats estimates", "misses estimates", "revenue",
        "resultados", "beneficio por accion", "ingresos trimestrales", "bpa",
    ),
    "guidance": (
        "guidance", "outlook", "forecast", "raises full-year", "cuts full-year",
        "preannounce", "prevision", "previsiones", "perspectivas", "eleva su guia",
        "recorta su guia",
    ),
    "fda": (
        "fda", "phase 1", "phase 2", "phase 3", "clinical trial", "approval",
        "breakthrough therapy", "ema approval", "ensayo clinico", "aprobacion",
        "fase iii",
    ),
    "m_and_a": (
        "acquisition", "acquires", "merger", "to buy", "takeover", "buyout",
        "stake in", "spin-off", "divestiture", "adquisicion", "fusion", "compra de",
        "opa", "escision",
    ),
    "contract": (
        "contract", "awarded", "order worth", "deal worth", "wins", "selected by",
        "purchase order", "contrato", "adjudicacion", "pedido", "licitacion",
    ),
    "regulatory": (
        "sec ", "doj", "antitrust", "investigation", "probe", "fine", "sanction",
        "regulator", "ftc", "regulacion", "multa", "investigacion", "sancion",
    ),
    "rating": (
        "upgrade", "downgrade", "raised to", "cut to", "initiated at",
        "outperform", "underperform", "overweight", "underweight", "buy rating",
        "sell rating", "mejora la recomendacion", "rebaja la recomendacion",
    ),
    "price_target": (
        "price target", "pt raised", "pt cut", "target price", "precio objetivo",
    ),
    "product": (
        "launch", "unveils", "introduces", "new product", "rollout", "available now",
        "lanza", "presenta", "nuevo producto", "lanzamiento",
    ),
    "partnership": (
        "partnership", "collaboration", "joint venture", "teams up", "agreement with",
        "alianza", "acuerdo con", "colaboracion", "joint venture",
    ),
    "litigation": (
        "lawsuit", "sues", "court", "verdict", "settlement", "patent dispute",
        "demanda", "juicio", "sentencia", "acuerdo extrajudicial", "litigio",
    ),
    "financing": (
        "offering", "private placement", "convertible notes", "buyback",
        "repurchase", "dividend", "capital raise", "dilution", "ampliacion de capital",
        "recompra", "dividendo", "colocacion",
    ),
    "insider": (
        "insider", "form 4", "ceo buys", "ceo sells", "director sells",
        "stake sale by", "compra de directivos", "venta de directivos",
    ),
    "macro": (
        "fed", "fomc", "powell", "cpi", "inflation", "jobs report", "payrolls",
        "rate cut", "rate hike", "treasury yield", "tariff", "inflacion",
        "tipos de interes", "empleo", "arancel",
    ),
    "sector": (
        "sector", "peers", "industry", "rivals", "chip stocks", "bank stocks",
        "competidores", "industria", "el sector",
    ),
}

# Etiquetas legibles para el informe (español).
NEWS_CATEGORY_LABELS: dict[str, str] = {
    "earnings": "Resultados",
    "guidance": "Previsiones",
    "fda": "FDA / ensayos clínicos",
    "m_and_a": "Fusiones y adquisiciones",
    "contract": "Contratos",
    "regulatory": "Regulación",
    "rating": "Recomendaciones (upgrade/downgrade)",
    "price_target": "Precio objetivo",
    "product": "Nuevo producto",
    "partnership": "Acuerdos y alianzas",
    "litigation": "Litigios",
    "financing": "Financiación",
    "insider": "Movimientos de insiders",
    "macro": "Macro / Fed",
    "sector": "Sector",
    "other": "Otros",
}


@dataclass(frozen=True)
class NewsConfig:
    """Parámetros de la búsqueda y clasificación de noticias."""

    # Una noticia de categoría dura dentro de esta ventana confirma catalizador.
    catalyst_window_hours: int = 48
    # Antigüedad máxima de las noticias que se descargan y guardan.
    lookback_hours: int = 96
    # Nunca se piden noticias a más de este número de tickers (limita llamadas).
    max_tickers: int = 25
    # Pausa entre tickers, para no saturar a Yahoo.
    pause_seconds: float = 0.4
    # Noticias que se conservan por ticker (las más recientes).
    max_items_per_ticker: int = 8
    request_timeout: int = 15
    # Alpha Vantage: solo si hay ALPHAVANTAGE_API_KEY y solo para finalistas.
    # ⚙️ El plan gratuito tiene un límite diario reducido (25 peticiones/día en
    # el momento de escribir esto). Verifica el límite vigente en
    # https://www.alphavantage.co/documentation/ antes de subir este número.
    alphavantage_max_calls: int = 10
    alphavantage_tickers: int = 6
    alphavantage_url: str = "https://www.alphavantage.co/query"


NEWS = NewsConfig()


# --------------------------------------------------------------------------- #
# Contexto de mercado
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketContextConfig:
    """Símbolos y umbrales del contexto de mercado."""

    indices: tuple[str, ...] = ("SPY", "QQQ", "DIA")
    futures: tuple[str, ...] = ("ES=F", "NQ=F")
    volatility: tuple[str, ...] = ("^VIX",)
    rates: tuple[str, ...] = ("^TNX",)
    sectors: tuple[str, ...] = (
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    )
    # Régimen de riesgo por VIX.
    vix_elevated: float = 20.0
    vix_high: float = 27.0
    # Futuros: umbral (en %) para considerar el tono positivo o negativo.
    futures_negative_pct: float = -0.40
    futures_positive_pct: float = 0.40
    # Días por delante que se muestran del calendario macro.
    macro_lookahead_days: int = 1

    @property
    def all_symbols(self) -> tuple[str, ...]:
        """Todos los símbolos que hay que descargar, sin duplicados."""
        symbols = (
            self.indices + self.futures + self.volatility + self.rates + self.sectors
        )
        return tuple(dict.fromkeys(symbols))


MARKET_CONTEXT = MarketContextConfig()

# Calendario macro mantenido a mano por el usuario (ver README).
# Si tu dataclass PATHS ya tiene un campo data_dir, usa PATHS.data_dir.
MACRO_EVENTS_FILE: Path = PATHS.history_dir.parent / "macro_events.json"

# --------------------------------------------------------------------------- #
# Filtro de candidatas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FilterWeights:
    """Pesos de la puntuación (suman 100)."""

    rvol: float = 25.0
    gap: float = 20.0
    catalyst: float = 20.0
    liquidity: float = 10.0
    relative_strength: float = 10.0
    technical: float = 10.0
    premarket_behavior: float = 5.0

 
    @property
    def total(self) -> float:
        return (self.rvol + self.gap + self.catalyst + self.liquidity
                + self.relative_strength + self.technical + self.premarket_behavior)


@dataclass(frozen=True)
class FilterPenalties:
    """Puntos que se restan por cada bandera de gap-and-fade."""

    overextended: float = 8.0
    fade_from_high: float = 6.0
    resistance_too_close: float = 6.0
    no_catalyst: float = 6.0
    low_volume_for_gap: float = 6.0
    market_against: float = 4.0


@dataclass(frozen=True)
class FilterConfig:
    """Puertas duras, escalas de puntuación y selección final."""

    # --- Puertas duras (eliminan al ticker) ---
    min_price: float = 5.0
    min_avg_volume_20d: float = 500_000.0
    min_avg_dollar_volume_20d: float = 10_000_000.0
    min_premarket_volume: float = 10_000.0
    min_abs_gap_pct: float = 0.5
    # En "long_only" solo interesan los gaps al alza.
    require_positive_gap_long_only: bool = True

    # --- Pre-filtro barato, previo a pedir noticias ---
    prefilter_abs_gap_pct: float = 0.4
    prefilter_premarket_volume: float = 8_000.0

    # --- Escalas de puntuación (valor mínimo -> 0 puntos; objetivo -> máximo) ---
    rvol_floor: float = 1.0
    rvol_target: float = 3.0
    # Si no hay RVOL se usa pm_pct_of_adv, con la puntuación limitada a la mitad.
    pm_pct_of_adv_target: float = 5.0
    rvol_fallback_cap: float = 0.5
    gap_floor: float = 1.0
    gap_target: float = 5.0
    dollar_volume_target: float = 200_000_000.0
    rs_floor: float = -0.5
    rs_target: float = 2.0
    catalyst_soft_score: float = 0.5

    # --- Estructura técnica (reparto interno del peso "technical") ---
    tech_above_sma20: float = 0.3
    tech_above_sma50: float = 0.2
    tech_ema_stacked: float = 0.3
    tech_room_to_resistance: float = 0.2
    room_atr_min: float = 1.0

    # --- Penalizaciones ---
    overextension_gap_atr: float = 3.0
    fade_from_high_pct: float = 2.0
    resistance_atr_min: float = 0.5
    low_volume_gap_pct: float = 3.0
    low_volume_rvol: float = 1.5

    # --- Selección ---
    min_score: float = 28.0
    min_candidates: int = 3
    max_candidates: int = 10
    target_candidates: int = 6

    weights: FilterWeights = field(default_factory=FilterWeights)
    penalties: FilterPenalties = field(default_factory=FilterPenalties)


FILTER = FilterConfig()

# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def validate_config() -> None:
    """Comprueba que la configuración es coherente.

    Lanza ``ValueError`` con un mensaje claro si algo no cuadra.
    """
    if abs(WEIGHTS.total() - 100.0) > 1e-9:
        raise ValueError(
            f"Los pesos de puntuación deben sumar 100 (suman {WEIGHTS.total()})."
        )
    sched = SCHEDULE
    if not (
        sched.run_window_start
        < sched.snapshot_time
        < sched.report_time
        <= sched.run_window_end
    ):
        raise ValueError(
            "Horarios incoherentes: se exige inicio de ventana < snapshot < "
            "informe <= fin de ventana."
        )
    if sched.report_time >= sched.regular_open:
        raise ValueError("El informe debe enviarse antes de la apertura.")
    if TRADE.direction_mode not in (DIRECTION_LONG_ONLY, DIRECTION_BOTH):
        raise ValueError(
            f"DIRECTION_MODE inválido: {TRADE.direction_mode!r} "
            f"(usa '{DIRECTION_LONG_ONLY}' o '{DIRECTION_BOTH}')."
        )
    if TRADE.min_rr <= 0:
        raise ValueError("min_rr debe ser mayor que 0.")
    if FILTER.min_candidates > FILTER.max_candidates:
        raise ValueError("min_candidates no puede ser mayor que max_candidates.")
    if TELEGRAM.max_message_chars >= 4096:
        raise ValueError("max_message_chars debe ser menor que 4096.")
