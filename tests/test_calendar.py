"""Tests del calendario bursátil (requieren pandas_market_calendars)."""
from datetime import date

import pytest

pytest.importorskip("pandas_market_calendars")

import utils  # noqa: E402


def test_fin_de_semana():
    info = utils.get_session_info(date(2026, 9, 19))  # sábado
    assert not info.is_session
    assert "fin de semana" in info.reason


def test_festivo_labor_day():
    info = utils.get_session_info(date(2026, 9, 7))  # Labor Day 2026
    assert not info.is_session
    assert "festivo" in info.reason


def test_sesion_normal():
    info = utils.get_session_info(date(2026, 9, 21))  # lunes
    assert info.is_session
    assert not info.is_early_close
    assert info.close_time is not None
    assert info.close_time.hour == 16


def test_cierre_anticipado_dia_despues_de_accion_de_gracias():
    info = utils.get_session_info(date(2026, 11, 27))
    assert info.is_session
    assert info.is_early_close
    assert info.close_time is not None
    assert info.close_time.hour == 13
