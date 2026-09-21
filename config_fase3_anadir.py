# =========================================================================== #
# FASE 3 — Noticias, contexto de mercado y filtro
# Pega este bloque al final de config.py, antes de validate_config().
# No modifica nada de las fases anteriores: solo añade secciones nuevas.
#
# Requiere que config.py tenga ya estos imports (añádelos si faltan):
#     from dataclasses import dataclass, field
#     from pathlib import Path
# =========================================================================== #

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

    overextended: float = 10.0
    fade_from_high: float = 8.0
    resistance_too_close: float = 8.0
    no_catalyst: float = 10.0
    low_volume_for_gap: float = 8.0
    market_against: float = 6.0


@dataclass(frozen=True)
class FilterConfig:
    """Puertas duras, escalas de puntuación y selección final."""

    # --- Puertas duras (eliminan al ticker) ---
    min_price: float = 5.0
    min_avg_volume_20d: float = 1_000_000.0
    min_avg_dollar_volume_20d: float = 20_000_000.0
    min_premarket_volume: float = 25_000.0
    min_abs_gap_pct: float = 1.0
    # En "long_only" solo interesan los gaps al alza.
    require_positive_gap_long_only: bool = True

    # --- Pre-filtro barato, previo a pedir noticias ---
    prefilter_abs_gap_pct: float = 0.8
    prefilter_premarket_volume: float = 15_000.0

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
    catalyst_soft_score: float = 0.35

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
    min_score: float = 40.0
    max_candidates: int = 10
    target_candidates: int = 6

    weights: FilterWeights = field(default_factory=FilterWeights)
    penalties: FilterPenalties = field(default_factory=FilterPenalties)


FILTER = FilterConfig()
