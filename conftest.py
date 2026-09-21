"""Añade la raíz del proyecto al path para que pytest encuentre los módulos.

Si ya tienes un conftest.py de la Fase 1, basta con conservar este bloque en él.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# stub_run_pipeline: evita que los tests de la CLI (test_main.py) disparen
# red real o se queden esperando en wait_until.
# --------------------------------------------------------------------------- #
def _fake_result(**overrides):
    """Resultado mínimo con la forma que espera print_provisional_summary."""
    base = {
        "session_date": "2026-09-21",
        "session_type": "sesión normal",
        "snapshot_ts": "2026-09-21T08:45:00-04:00",
        "universe_size": 0,
        "market_context": {"summary": "N/A"},
        "analyses": {},
        "snapshots": {},
        "news": {},
        "filter_log": [],
        "candidates": [],
        "candidates_payload": [],
        "gemini": {
            "available": False, "picks": [], "warnings": [], "error": "",
        },
        "no_trade": True,
        "no_trade_reason": "stub de pruebas: run_pipeline no se ejecuta en test_main.py",
        "failures": {},
        "intraday_files": [],
        "disclaimer": "Herramienta informativa, no es asesoramiento financiero.",
    }
    base.update(overrides)
    return base


@pytest.fixture
def stub_run_pipeline(monkeypatch):
    """Sustituye main.run_pipeline por un doble sin red ni espera.

    Úsalo en cualquier test que solo quiera comprobar el comportamiento de la
    CLI (argumentos, calendario, guarda horaria, idempotencia...) sin llegar a
    descargar datos de verdad. Los tests que sí quieran ejercitar el pipeline
    real van en test_pipeline.py, que ya sustituye las descargas de
    market_data por datos sintéticos.

    Devuelve la lista de llamadas (args, kwargs) hechas a run_pipeline, por si
    el test quiere comprobar con qué universo o modo se le llamó.
    """
    calls: list[tuple[tuple, dict]] = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_result()

    monkeypatch.setattr("main.run_pipeline", fake)
    return calls

