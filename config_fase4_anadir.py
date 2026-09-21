# =========================================================================== #
# FASE 4 — Gemini (salida estructurada y validación)
# Pega este bloque al final de config.py, después del bloque de la Fase 3.
# =========================================================================== #


@dataclass(frozen=True)
class GeminiConfig:
    """Parámetros de la llamada a Gemini y de la validación posterior."""

    # --- Llamada ---
    # Temperatura baja: queremos interpretación estable, no creatividad.
    temperature: float = 0.2
    max_output_tokens: int = 8192
    # Intentos totales ante 429/5xx o JSON inválido (1 llamada + 1 reintento).
    attempts: int = 2
    retry_base_delay: float = 3.0
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
