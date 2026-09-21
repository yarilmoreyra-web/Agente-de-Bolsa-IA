"""Tests de la guarda horaria (horario de verano e invierno)."""
from datetime import datetime, timezone

import config
import utils

SCHED = config.SCHEDULE


def dentro(moment):
    return utils.is_within_window(moment, SCHED.run_window_start, SCHED.run_window_end)


def test_verano_cron_correcto_esta_dentro():
    # 12:35 UTC en julio = 08:35 en Nueva York (EDT)
    assert dentro(datetime(2026, 7, 15, 12, 35, tzinfo=timezone.utc))


def test_verano_cron_de_invierno_queda_fuera():
    # 13:35 UTC en julio = 09:35 en Nueva York (EDT): fuera de la ventana
    assert not dentro(datetime(2026, 7, 15, 13, 35, tzinfo=timezone.utc))


def test_invierno_cron_correcto_esta_dentro():
    # 13:35 UTC en diciembre = 08:35 en Nueva York (EST)
    assert dentro(datetime(2026, 12, 15, 13, 35, tzinfo=timezone.utc))


def test_invierno_cron_de_verano_queda_fuera():
    # 12:35 UTC en diciembre = 07:35 en Nueva York (EST): fuera de la ventana
    assert not dentro(datetime(2026, 12, 15, 12, 35, tzinfo=timezone.utc))


def test_retraso_del_cron_sigue_dentro():
    # El cron se retrasa 40 min: 09:15 ET, aún dentro de la ventana
    assert dentro(datetime(2026, 7, 15, 13, 15, tzinfo=timezone.utc))


def test_retraso_excesivo_queda_fuera():
    # 09:40 ET: demasiado tarde
    assert not dentro(datetime(2026, 7, 15, 13, 40, tzinfo=timezone.utc))


def test_datetime_sin_zona_horaria_da_error():
    import pytest

    with pytest.raises(ValueError):
        utils.to_ny(datetime(2026, 7, 15, 8, 35))
