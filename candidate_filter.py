"""candidate_filter.py — Puertas duras, puntuación 0-100 y selección de candidatas.

Cómo funciona
-------------
1. **Pre-filtro barato** (``prefilter_for_news``): antes de pedir noticias se
   descartan los tickers sin gap ni volumen pre-market, para no gastar llamadas.
2. **Puertas duras** (``apply_hard_gates``): precio, liquidez, volumen
   pre-market, gap mínimo y calidad del dato. Quien no pasa, queda fuera; el
   motivo se guarda en el ``filter_log``.
3. **Puntuación 0-100** (``score_candidate``): siete componentes con los pesos
   de ``config.FILTER.weights`` (RVOL 25, gap 20, catalizador 20, liquidez 10,
   fuerza relativa 10, estructura técnica 10, comportamiento pre-market 5).
4. **Penalizaciones de gap-and-fade**: se restan puntos por extensión excesiva,
   alejamiento del máximo pre-market, resistencia demasiado cerca, ausencia de
   catalizador, volumen bajo para el gap y mercado en contra.
5. **Selección**: hasta ``max_candidates`` con puntuación ≥ ``min_score``. Si no
   pasa ninguna, **no se llama a Gemini** y el informe dice NO OPERAR.

Transparencia: cada ticker analizado aparece en el ``filter_log`` con su
puntuación, el desglose por componente, sus penalizaciones y, si procede, el
motivo exacto de exclusión. Ese registro va al histórico.

Dirección: con ``config.DIRECTION_MODE == "long_only"`` solo se consideran gaps
al alza. Con ``"both"``, un gap negativo se evalúa como SHORT y los criterios
técnicos, de fuerza relativa y de fade se invierten (``_sign``).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import config
from technical_analysis import NA, is_num

logger = logging.getLogger("trading_agent.candidate_filter")

LONG, SHORT = "LONG", "SHORT"
NO_CANDIDATES_REASON = (
    "Ninguna acción del universo supera el filtro mínimo "
    "(catalizador, momentum, volumen y estructura). No se fuerza ninguna operación."
)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@dataclass
class Penalty:
    """Una bandera de riesgo con los puntos que resta."""

    flag: str
    points: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilterRow:
    """Resultado del filtro para un ticker (una fila del ``filter_log``)."""

    ticker: str
    direction: str = LONG
    passed_gates: bool = False
    gates_failed: list[str] = field(default_factory=list)
    raw_score: float | str = NA
    penalty_points: float = 0.0
    score: float | str = NA
    components: dict[str, Any] = field(default_factory=dict)
    penalties: list[dict[str, Any]] = field(default_factory=list)
    selected: bool = False
    rank: int | str = NA
    exclusion_reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilterOutcome:
    """Salida completa del filtro."""

    candidates: list[str] = field(default_factory=list)
    rows: list[FilterRow] = field(default_factory=list)
    no_trade: bool = True
    no_trade_reason: str = NO_CANDIDATES_REASON

    @property
    def filter_log(self) -> list[dict[str, Any]]:
        """Registro completo, ordenado, listo para el histórico."""
        return [row.to_dict() for row in self.rows]

    def selected_rows(self) -> list[FilterRow]:
        """Filas seleccionadas, en orden de puntuación."""
        return [row for row in self.rows if row.selected]


# --------------------------------------------------------------------------- #
# Utilidades de puntuación
# --------------------------------------------------------------------------- #
def _scale(value: Any, floor: float, target: float) -> float:
    """Normaliza ``value`` a [0, 1] entre ``floor`` (0) y ``target`` (1)."""
    if not is_num(value) or target == floor:
        return 0.0
    ratio = (float(value) - floor) / (target - floor)
    return max(0.0, min(1.0, ratio))


def _component(weight: float, ratio: float, value: Any, detail: str) -> dict[str, Any]:
    """Estructura homogénea de un componente de la puntuación."""
    return {
        "weight": round(weight, 2),
        "ratio": round(max(0.0, min(1.0, ratio)), 4),
        "points": round(weight * max(0.0, min(1.0, ratio)), 2),
        "value": value,
        "detail": detail,
    }


def _sign(direction: str) -> float:
    """+1 para LONG, -1 para SHORT (invierte los criterios direccionales)."""
    return 1.0 if direction == LONG else -1.0


def _num(value: Any, default: float | None = None) -> float | None:
    """Float si es un número válido; ``default`` en caso contrario."""
    return float(value) if is_num(value) else default


def _premarket_price(analysis: Mapping[str, Any]) -> float | None:
    """Precio pre-market del snapshot, o ``None``."""
    return _num(analysis.get("premarket", {}).get("premarket_price"))


def resolve_direction(analysis: Mapping[str, Any], direction_mode: str) -> str:
    """Dirección propuesta: LONG salvo que el modo sea ``both`` y el gap sea negativo."""
    if direction_mode != "both":
        return LONG
    gap = _num(analysis.get("gap_pct"), 0.0) or 0.0
    return LONG if gap >= 0 else SHORT


# --------------------------------------------------------------------------- #
# 1) Pre-filtro barato (antes de pedir noticias)
# --------------------------------------------------------------------------- #
def passes_prefilter(
    analysis: Mapping[str, Any], cfg: config.FilterConfig | None = None,
    direction_mode: str | None = None, relax_premarket_volume: bool = False,
) -> bool:
    """Criterio mínimo de gap y volumen pre-market para merecer una llamada de noticias."""
    cfg = cfg or config.FILTER
    direction_mode = direction_mode or getattr(config, "DIRECTION_MODE", "long_only")
    premarket = analysis.get("premarket", {})
    if premarket.get("premarket_quality") == "missing":
        return False

    gap = _num(analysis.get("gap_pct"))
    volume = _num(premarket.get("premarket_volume"), 0.0) or 0.0
    if gap is None:
        return False
    if direction_mode != "both" and cfg.require_positive_gap_long_only and gap <= 0:
        return False
    if abs(gap) < cfg.prefilter_abs_gap_pct:
        return False
    return relax_premarket_volume or volume >= cfg.prefilter_premarket_volume


def prefilter_for_news(
    analyses: Mapping[str, Mapping[str, Any]],
    cfg: config.FilterConfig | None = None,
    limit: int | None = None,
    direction_mode: str | None = None,
    relax_premarket_volume: bool = False,
) -> list[str]:
    """Tickers a los que merece la pena pedir noticias, de más a menos prometedor.

    Se ordena por gap absoluto y, a igualdad, por volumen pre-market. El corte lo
    pone ``limit`` (por defecto ``config.NEWS.max_tickers``).
    """
    cfg = cfg or config.FILTER
    limit = limit if limit is not None else config.NEWS.max_tickers

    scored: list[tuple[float, float, str]] = []
    for ticker, analysis in analyses.items():
        if not passes_prefilter(analysis, cfg, direction_mode, relax_premarket_volume):
            continue
        gap = abs(_num(analysis.get("gap_pct"), 0.0) or 0.0)
        volume = _num(analysis.get("premarket", {}).get("premarket_volume"), 0.0) or 0.0
        scored.append((gap, volume, ticker))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected = [ticker for _, _, ticker in scored[:max(0, limit)]]
    logger.info(
        "Pre-filtro de noticias: %d de %d tickers (límite %d)%s.",
        len(selected), len(analyses), limit,
        " [volumen pre-market relajado]" if relax_premarket_volume else "",
    )
    return selected


# --------------------------------------------------------------------------- #
# 2) Puertas duras
# --------------------------------------------------------------------------- #
def apply_hard_gates(
    analysis: Mapping[str, Any], cfg: config.FilterConfig | None = None,
    direction_mode: str | None = None, relax_premarket_volume: bool = False,
) -> list[str]:

    """Motivos por los que el ticker queda descartado. Lista vacía = pasa."""
    cfg = cfg or config.FILTER
    direction_mode = direction_mode or getattr(config, "DIRECTION_MODE", "long_only")
    premarket = analysis.get("premarket", {})
    indicators = analysis.get("indicators", {})
    failures: list[str] = []

    quality = premarket.get("premarket_quality", "missing")
    if quality == "missing":
        failures.append("sin datos pre-market (calidad 'missing')")

    price = _premarket_price(analysis) or _num(indicators.get("last_close"))
    if price is None:
        failures.append("sin precio utilizable")
    elif price < cfg.min_price:
        failures.append(f"precio {price:.2f} < {cfg.min_price:.2f} USD")

    avg_volume = _num(indicators.get("avg_volume_20d"))
    if avg_volume is None:
        failures.append("sin volumen medio de 20 sesiones")
    elif avg_volume < cfg.min_avg_volume_20d:
        failures.append(
            f"volumen medio 20d {avg_volume:,.0f} < {cfg.min_avg_volume_20d:,.0f}")

    dollar_volume = _num(indicators.get("avg_dollar_volume_20d"))
    if dollar_volume is None:
        failures.append("sin volumen medio en dólares")
    elif dollar_volume < cfg.min_avg_dollar_volume_20d:
        failures.append(
            f"volumen en dólares {dollar_volume/1e6:,.1f}M < "
            f"{cfg.min_avg_dollar_volume_20d/1e6:,.0f}M USD")

    if not relax_premarket_volume:
        pm_volume = _num(premarket.get("premarket_volume"))
        if pm_volume is None:
            failures.append("sin volumen pre-market")
        elif pm_volume < cfg.min_premarket_volume:
            failures.append(
                f"volumen pre-market {pm_volume:,.0f} < {cfg.min_premarket_volume:,.0f}")
    

    gap = _num(analysis.get("gap_pct"))
    if gap is None:
        failures.append("gap no calculable")
    else:
        if abs(gap) < cfg.min_abs_gap_pct:
            failures.append(f"gap {gap:+.2f}% < {cfg.min_abs_gap_pct:.1f}% en valor absoluto")
        elif (direction_mode != "both" and cfg.require_positive_gap_long_only
              and gap < 0):
            failures.append(f"gap {gap:+.2f}% a la baja y el modo es solo largos")
    return failures


# --------------------------------------------------------------------------- #
# 3) Componentes de la puntuación
# --------------------------------------------------------------------------- #
def _rvol_component(analysis: Mapping[str, Any], cfg: config.FilterConfig) -> dict[str, Any]:
    """RVOL; si no hay, se usa ``pm_pct_of_adv`` con la puntuación limitada."""
    weight = cfg.weights.rvol
    rvol_data = analysis.get("rvol", {}) or {}
    rvol = rvol_data.get("rvol", NA)
    if is_num(rvol):
        ratio = _scale(rvol, cfg.rvol_floor, cfg.rvol_target)
        return _component(weight, ratio, rvol, f"RVOL {float(rvol):.2f}")

    fallback = analysis.get("pm_pct_of_adv", NA)
    reason = rvol_data.get("reason", "") or "sin RVOL"
    if is_num(fallback):
        ratio = _scale(fallback, 0.0, cfg.pm_pct_of_adv_target) * cfg.rvol_fallback_cap
        return _component(
            weight, ratio, fallback,
            f"RVOL no disponible ({reason}); se usa pm_pct_of_adv "
            f"{float(fallback):.2f}% con puntuación limitada",
        )
    return _component(weight, 0.0, NA, f"RVOL no disponible ({reason}) y sin alternativa")


def _gap_component(analysis: Mapping[str, Any], direction: str,
                   cfg: config.FilterConfig) -> dict[str, Any]:
    """Tamaño del gap en la dirección propuesta."""
    gap = _num(analysis.get("gap_pct"))
    if gap is None:
        return _component(cfg.weights.gap, 0.0, NA, "gap no disponible")
    favorable = gap * _sign(direction)
    ratio = _scale(favorable, cfg.gap_floor, cfg.gap_target)
    return _component(cfg.weights.gap, ratio, round(gap, 2), f"gap {gap:+.2f}%")


def _catalyst_component(ticker_news: Mapping[str, Any] | None,
                        cfg: config.FilterConfig) -> dict[str, Any]:
    """Catalizador confirmado, noticia blanda o nada."""
    weight = cfg.weights.catalyst
    if not ticker_news:
        return _component(weight, 0.0, False, "sin noticias consultadas")
    if ticker_news.get("catalyst_confirmed"):
        category = ticker_news.get("catalyst_category", "") or "categoría dura"
        return _component(weight, 1.0, True, f"catalizador confirmado ({category})")
    if ticker_news.get("has_soft_news"):
        return _component(weight, cfg.catalyst_soft_score, False,
                          "solo noticias de contexto (sector/macro/otros)")
    return _component(weight, 0.0, False, "sin catalizador confirmado")


def _liquidity_component(analysis: Mapping[str, Any],
                         cfg: config.FilterConfig) -> dict[str, Any]:
    """Volumen medio en dólares frente al objetivo de liquidez."""
    dollar_volume = _num(analysis.get("indicators", {}).get("avg_dollar_volume_20d"))
    if dollar_volume is None:
        return _component(cfg.weights.liquidity, 0.0, NA, "sin volumen en dólares")
    ratio = _scale(dollar_volume, cfg.min_avg_dollar_volume_20d, cfg.dollar_volume_target)
    return _component(cfg.weights.liquidity, ratio, round(dollar_volume, 0),
                      f"volumen medio {dollar_volume/1e6:,.0f}M USD")


def _relative_strength_component(analysis: Mapping[str, Any], direction: str,
                                 cfg: config.FilterConfig) -> dict[str, Any]:
    """Fuerza relativa pre-market frente a SPY y QQQ (media de los disponibles)."""
    weight = cfg.weights.relative_strength
    rs = analysis.get("relative_strength", {}) or {}
    values = [float(rs[key]) for key in ("rs_pm_vs_spy", "rs_pm_vs_qqq")
              if is_num(rs.get(key))]
    if not values:
        return _component(weight, 0.0, NA, "fuerza relativa no disponible")
    average = sum(values) / len(values) * _sign(direction)
    ratio = _scale(average, cfg.rs_floor, cfg.rs_target)
    return _component(weight, ratio, round(average, 2),
                      f"{average:+.2f} pp frente a los índices")


def _technical_component(analysis: Mapping[str, Any], direction: str,
                         cfg: config.FilterConfig) -> dict[str, Any]:
    """Estructura técnica: medias, apilamiento de EMAs y espacio hasta el objetivo."""
    weight = cfg.weights.technical
    indicators = analysis.get("indicators", {})
    price = _premarket_price(analysis)
    sign = _sign(direction)
    ratio, notes = 0.0, []

    if price is not None:
        for label, key, share in (
            ("SMA20", "sma20", cfg.tech_above_sma20),
            ("SMA50", "sma50", cfg.tech_above_sma50),
        ):
            level = _num(indicators.get(key))
            if level is None:
                continue
            if (price - level) * sign > 0:
                ratio += share
                notes.append(f"precio {'sobre' if sign > 0 else 'bajo'} {label}")

    ema_fast, ema_slow = _num(indicators.get("ema9")), _num(indicators.get("ema20"))
    if ema_fast is not None and ema_slow is not None and (ema_fast - ema_slow) * sign > 0:
        ratio += cfg.tech_ema_stacked
        notes.append("EMA9/EMA20 alineadas")

    side = "nearest_resistance" if direction == LONG else "nearest_support"
    target_level = analysis.get(side)
    if isinstance(target_level, Mapping):
        distance = _num(target_level.get("distance_atr"))
        if distance is None or abs(distance) >= cfg.room_atr_min:
            ratio += cfg.tech_room_to_resistance
            notes.append("espacio suficiente hasta el siguiente nivel")
    else:
        ratio += cfg.tech_room_to_resistance
        notes.append("sin nivel que estorbe por delante")

    return _component(weight, ratio, round(ratio, 2),
                      ", ".join(notes) or "estructura técnica débil")


def _premarket_behavior_component(analysis: Mapping[str, Any], direction: str,
                                  cfg: config.FilterConfig) -> dict[str, Any]:
    """Posición del precio dentro del rango pre-market (cerca del extremo favorable)."""
    weight = cfg.weights.premarket_behavior
    premarket = analysis.get("premarket", {})
    price = _premarket_price(analysis)
    high, low = _num(premarket.get("premarket_high")), _num(premarket.get("premarket_low"))
    if price is None or high is None or low is None:
        return _component(weight, 0.0, NA, "rango pre-market no disponible")
    span = high - low
    if span <= 0:
        return _component(weight, 0.5, NA, "rango pre-market plano")
    position = (price - low) / span
    if direction == SHORT:
        position = 1.0 - position
    return _component(weight, position, round(position, 3),
                      f"a {position * 100:.0f}% del extremo favorable del rango")


# --------------------------------------------------------------------------- #
# 4) Penalizaciones de gap-and-fade
# --------------------------------------------------------------------------- #
def gap_fade_penalties(
    analysis: Mapping[str, Any], ticker_news: Mapping[str, Any] | None,
    context: Any | None, direction: str, cfg: config.FilterConfig | None = None,
) -> list[Penalty]:
    """Banderas de riesgo de gap-and-fade, con los puntos que restan."""
    cfg = cfg or config.FILTER
    penalties_cfg = cfg.penalties
    premarket = analysis.get("premarket", {})
    sign = _sign(direction)
    found: list[Penalty] = []

    gap_atr = _num(analysis.get("gap_atr"))
    if gap_atr is not None and abs(gap_atr) > cfg.overextension_gap_atr:
        found.append(Penalty(
            "extension_excesiva", penalties_cfg.overextended,
            f"el gap equivale a {abs(gap_atr):.1f} ATR "
            f"(máx. {cfg.overextension_gap_atr:.1f})"))

    price = _premarket_price(analysis)
    extreme = _num(premarket.get("premarket_high" if direction == LONG else "premarket_low"))
    if price is not None and extreme is not None and extreme > 0:
        retreat = (extreme - price) / extreme * 100 * sign
        if retreat >= cfg.fade_from_high_pct:
            label = "máximo" if direction == LONG else "mínimo"
            found.append(Penalty(
                "alejado_del_extremo", penalties_cfg.fade_from_high,
                f"el precio está {retreat:.1f}% por "
                f"{'debajo' if direction == LONG else 'encima'} del {label} pre-market"))

    side = "nearest_resistance" if direction == LONG else "nearest_support"
    level = analysis.get(side)
    if isinstance(level, Mapping):
        distance = _num(level.get("distance_atr"))
        if distance is not None and abs(distance) < cfg.resistance_atr_min:
            found.append(Penalty(
                "nivel_demasiado_cerca", penalties_cfg.resistance_too_close,
                f"{level.get('label', 'nivel')} a solo {abs(distance):.2f} ATR"))

    if not (ticker_news or {}).get("catalyst_confirmed"):
        found.append(Penalty("sin_catalizador", penalties_cfg.no_catalyst,
                             "no hay noticia dura reciente que justifique el movimiento"))

    gap = _num(analysis.get("gap_pct"))
    rvol = (analysis.get("rvol", {}) or {}).get("rvol", NA)
    if gap is not None and abs(gap) >= cfg.low_volume_gap_pct and is_num(rvol) \
            and float(rvol) < cfg.low_volume_rvol:
        found.append(Penalty(
            "volumen_bajo_para_el_gap", penalties_cfg.low_volume_for_gap,
            f"gap de {abs(gap):.1f}% con RVOL {float(rvol):.2f}"))

    if context is not None:
        regime = getattr(context, "regime", None) or {}
        against = (bool(regime.get("against_longs")) if direction == LONG
                   else regime.get("tone") == "positivo")
        if against:
            found.append(Penalty(
                "mercado_en_contra", penalties_cfg.market_against,
                f"régimen {regime.get('label', 'adverso')}, "
                f"tono {regime.get('tone', NA)}"))
    return found


# --------------------------------------------------------------------------- #
# 5) Puntuación completa y selección
# --------------------------------------------------------------------------- #
def score_candidate(
    analysis: Mapping[str, Any], ticker_news: Mapping[str, Any] | None = None,
    context: Any | None = None, cfg: config.FilterConfig | None = None,
    direction: str | None = None,
) -> FilterRow:
    """Calcula la puntuación final de un ticker que ya pasó las puertas duras."""
    cfg = cfg or config.FILTER
    ticker = str(analysis.get("ticker", "?"))
    direction = direction or resolve_direction(
        analysis, getattr(config, "DIRECTION_MODE", "long_only"))

    components = {
        "rvol": _rvol_component(analysis, cfg),
        "gap": _gap_component(analysis, direction, cfg),
        "catalyst": _catalyst_component(ticker_news, cfg),
        "liquidity": _liquidity_component(analysis, cfg),
        "relative_strength": _relative_strength_component(analysis, direction, cfg),
        "technical": _technical_component(analysis, direction, cfg),
        "premarket_behavior": _premarket_behavior_component(analysis, direction, cfg),
    }
    raw_score = sum(component["points"] for component in components.values())

    penalties = gap_fade_penalties(analysis, ticker_news, context, direction, cfg)
    penalty_points = sum(penalty.points for penalty in penalties)
    final = max(0.0, min(100.0, raw_score - penalty_points))

    premarket = analysis.get("premarket", {})
    return FilterRow(
        ticker=ticker,
        direction=direction,
        passed_gates=True,
        raw_score=round(raw_score, 2),
        penalty_points=round(penalty_points, 2),
        score=round(final, 2),
        components=components,
        penalties=[penalty.to_dict() for penalty in penalties],
        metrics={
            "gap_pct": analysis.get("gap_pct", NA),
            "gap_atr": analysis.get("gap_atr", NA),
            "rvol": (analysis.get("rvol", {}) or {}).get("rvol", NA),
            "premarket_volume": premarket.get("premarket_volume", NA),
            "premarket_quality": premarket.get("premarket_quality", "missing"),
            "catalyst_confirmed": bool((ticker_news or {}).get("catalyst_confirmed")),
        },
    )


def filter_candidates(
    analyses: Mapping[str, Mapping[str, Any]],
    news: Mapping[str, Any] | None = None,
    context: Any | None = None,
    cfg: config.FilterConfig | None = None,
    direction_mode: str | None = None,
    relax_premarket_volume: bool = False,
) -> FilterOutcome:   

    """Aplica puertas duras y puntuación a todo el universo y elige las candidatas.

    ``news`` admite ``TickerNews`` o diccionarios equivalentes. Devuelve un
    ``FilterOutcome`` con las candidatas, el ``filter_log`` completo y, si no hay
    ninguna, el motivo del NO OPERAR (en ese caso no debe llamarse a Gemini).
    """
    cfg = cfg or config.FILTER
    direction_mode = direction_mode or getattr(config, "DIRECTION_MODE", "long_only")
    news = news or {}

    rows: list[FilterRow] = []
    for ticker, analysis in analyses.items():
        try:
            failures = apply_hard_gates(analysis, cfg, direction_mode, relax_premarket_volume)
            if failures:
                rows.append(FilterRow(
                    ticker=ticker, passed_gates=False, gates_failed=failures,
                    exclusion_reason="; ".join(failures),
                ))
               continue
            ticker_news = news.get(ticker)
            if ticker_news is not None and hasattr(ticker_news, "to_dict"):
                ticker_news = ticker_news.to_dict()
            rows.append(score_candidate(
                analysis, ticker_news, context, cfg,
                resolve_direction(analysis, direction_mode)))
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: no se pudo aplicar el filtro (%s)", ticker, exc)
            rows.append(FilterRow(
                ticker=ticker, passed_gates=False,
                exclusion_reason=f"error en el filtro: {type(exc).__name__}: {exc}",
            ))

    scored = [row for row in rows if row.passed_gates and is_num(row.score)]
    scored.sort(key=lambda row: float(row.score), reverse=True)

    selected: list[FilterRow] = []
    for row in scored:
        if float(row.score) < cfg.min_score:
            row.exclusion_reason = (
                f"puntuación {float(row.score):.1f} < mínimo {cfg.min_score:.0f}")
            continue
        if len(selected) >= cfg.max_candidates:
            row.exclusion_reason = (
                f"fuera de las {cfg.max_candidates} mejores por puntuación")
            continue
        row.selected = True
        row.rank = len(selected) + 1
        selected.append(row)

    ordered = selected + [row for row in scored if not row.selected] \
        + [row for row in rows if not row.passed_gates]

    outcome = FilterOutcome(
        candidates=[row.ticker for row in selected],
        rows=ordered,
        no_trade=not selected,
        no_trade_reason="" if selected else NO_CANDIDATES_REASON,
    )
    logger.info(
        "Filtro: %d analizados, %d pasan las puertas duras, %d candidatas "
        "(puntuación mínima %.0f)%s.",
        len(rows), len(scored), len(selected), cfg.min_score,
        " [volumen pre-market relajado: fecha no es la de hoy]" if relax_premarket_volume else "",
    )
    if not selected:
        logger.info("Sin candidatas: %s", NO_CANDIDATES_REASON)
    return outcome
                

def candidates_payload(
    outcome: FilterOutcome, analyses: Mapping[str, Mapping[str, Any]],
    news: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Paquete de cada candidata (métricas + noticias) para Gemini y el informe."""
    news = news or {}
    payload: list[dict[str, Any]] = []
    for row in outcome.selected_rows():
        ticker_news = news.get(row.ticker)
        if ticker_news is not None and hasattr(ticker_news, "to_dict"):
            ticker_news = ticker_news.to_dict()
        payload.append({
            "ticker": row.ticker,
            "rank": row.rank,
            "direction": row.direction,
            "score": row.score,
            "score_breakdown": row.components,
            "penalties": row.penalties,
            "analysis": dict(analyses.get(row.ticker, {})),
            "news": ticker_news or {},
        })
    return payload


def passes_prefilter(
    analysis: Mapping[str, Any], cfg: config.FilterConfig | None = None,
    direction_mode: str | None = None, relax_premarket_volume: bool = False,
) -> bool:
    """Criterio mínimo de gap y volumen pre-market para merecer una llamada de noticias."""
    cfg = cfg or config.FILTER
    direction_mode = direction_mode or getattr(config, "DIRECTION_MODE", "long_only")
    premarket = analysis.get("premarket", {})
    if premarket.get("premarket_quality") == "missing":
        return False

    gap = _num(analysis.get("gap_pct"))
    volume = _num(premarket.get("premarket_volume"), 0.0) or 0.0
    if gap is None:
        return False
    if direction_mode != "both" and cfg.require_positive_gap_long_only and gap <= 0:
        return False
    if abs(gap) < cfg.prefilter_abs_gap_pct:
        return False
    return relax_premarket_volume or volume >= cfg.prefilter_premarket_volume


def prefilter_for_news(
    analyses: Mapping[str, Mapping[str, Any]],
    cfg: config.FilterConfig | None = None,
    limit: int | None = None,
    direction_mode: str | None = None,
    relax_premarket_volume: bool = False,
) -> list[str]:
    cfg = cfg or config.FILTER
    limit = limit if limit is not None else config.NEWS.max_tickers

    scored: list[tuple[float, float, str]] = []
    for ticker, analysis in analyses.items():
        if not passes_prefilter(analysis, cfg, direction_mode, relax_premarket_volume):
            continue
        gap = abs(_num(analysis.get("gap_pct"), 0.0) or 0.0)
        volume = _num(analysis.get("premarket", {}).get("premarket_volume"), 0.0) or 0.0
        scored.append((gap, volume, ticker))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected = [ticker for _, _, ticker in scored[:max(0, limit)]]
    logger.info(
        "Pre-filtro de noticias: %d de %d tickers (límite %d)%s.",
        len(selected), len(analyses), limit,
        " [volumen pre-market relajado]" if relax_premarket_volume else "",
    )
    return selected


def apply_hard_gates(
    analysis: Mapping[str, Any], cfg: config.FilterConfig | None = None,
    direction_mode: str | None = None, relax_premarket_volume: bool = False,
) -> list[str]:
    cfg = cfg or config.FILTER
    direction_mode = direction_mode or getattr(config, "DIRECTION_MODE", "long_only")
    premarket = analysis.get("premarket", {})
    indicators = analysis.get("indicators", {})
    failures: list[str] = []

    quality = premarket.get("premarket_quality", "missing")
    if quality == "missing":
        failures.append("sin datos pre-market (calidad 'missing')")

    price = _premarket_price(analysis) or _num(indicators.get("last_close"))
    if price is None:
        failures.append("sin precio utilizable")
    elif price < cfg.min_price:
        failures.append(f"precio {price:.2f} < {cfg.min_price:.2f} USD")

    avg_volume = _num(indicators.get("avg_volume_20d"))
    if avg_volume is None:
        failures.append("sin volumen medio de 20 sesiones")
    elif avg_volume < cfg.min_avg_volume_20d:
        failures.append(
            f"volumen medio 20d {avg_volume:,.0f} < {cfg.min_avg_volume_20d:,.0f}")

    dollar_volume = _num(indicators.get("avg_dollar_volume_20d"))
    if dollar_volume is None:
        failures.append("sin volumen medio en dólares")
    elif dollar_volume < cfg.min_avg_dollar_volume_20d:
        failures.append(
            f"volumen en dólares {dollar_volume/1e6:,.1f}M < "
            f"{cfg.min_avg_dollar_volume_20d/1e6:,.0f}M USD")

    if not relax_premarket_volume:
        pm_volume = _num(premarket.get("premarket_volume"))
        if pm_volume is None:
            failures.append("sin volumen pre-market")
        elif pm_volume < cfg.min_premarket_volume:
            failures.append(
                f"volumen pre-market {pm_volume:,.0f} < {cfg.min_premarket_volume:,.0f}")

    gap = _num(analysis.get("gap_pct"))
    if gap is None:
        failures.append("gap no calculable")
    else:
        if abs(gap) < cfg.min_abs_gap_pct:
            failures.append(f"gap {gap:+.2f}% < {cfg.min_abs_gap_pct:.1f}% en valor absoluto")
        elif (direction_mode != "both" and cfg.require_positive_gap_long_only
              and gap < 0):
            failures.append(f"gap {gap:+.2f}% a la baja y el modo es solo largos")
    return failures


__all__ = [
    "LONG", "SHORT", "NO_CANDIDATES_REASON", "Penalty", "FilterRow", "FilterOutcome",
    "passes_prefilter", "prefilter_for_news", "apply_hard_gates", "score_candidate",
    "gap_fade_penalties", "filter_candidates", "candidates_payload",
    "resolve_direction",
]
