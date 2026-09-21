"""Tests de utilidades: reintentos, JSON, idempotencia y secretos."""
import json
from datetime import date, datetime, time, timezone

import pytest

import utils


def test_retry_funciona_tras_fallos():
    calls = {"n": 0}

    @utils.retry(attempts=3, base_delay=0, exceptions=(ValueError,))
    def inestable():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("fallo temporal")
        return "ok"

    assert inestable() == "ok"
    assert calls["n"] == 3


def test_retry_relanza_tras_agotar_intentos():
    calls = {"n": 0}

    @utils.retry(attempts=2, base_delay=0, exceptions=(ValueError,))
    def siempre_falla():
        calls["n"] += 1
        raise ValueError("siempre")

    with pytest.raises(ValueError):
        siempre_falla()
    assert calls["n"] == 2


def test_write_json_atomic_limpia_nan_fechas_y_rutas(tmp_path):
    target = tmp_path / "sub" / "datos.json"
    utils.write_json_atomic(target, {
        "nan": float("nan"),
        "inf": float("inf"),
        "fecha": date(2026, 9, 21),
        "hora": datetime(2026, 9, 21, 8, 45, tzinfo=timezone.utc),
        "cierre": time(16, 0),
        "ruta": tmp_path,
        "lista": (1, 2),
    })
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["nan"] is None
    assert data["inf"] is None
    assert data["fecha"] == "2026-09-21"
    assert data["cierre"] == "16:00:00"
    assert data["lista"] == [1, 2]


def test_read_json_devuelve_default_si_falta_o_esta_corrupto(tmp_path):
    assert utils.read_json(tmp_path / "no_existe.json", default={}) == {}
    corrupto = tmp_path / "malo.json"
    corrupto.write_text("{no es json", encoding="utf-8")
    assert utils.read_json(corrupto, default="x") == "x"


def test_already_sent(tmp_path):
    day = date(2026, 9, 21)
    assert not utils.already_sent(tmp_path, day)
    utils.write_json_atomic(tmp_path / "2026-09-21.json", {"sent": False})
    assert not utils.already_sent(tmp_path, day)
    utils.write_json_atomic(tmp_path / "2026-09-21.json", {"sent": True})
    assert utils.already_sent(tmp_path, day)


def test_mask_secret():
    assert utils.mask_secret("") == "(vacío)"
    assert utils.mask_secret("abc") == "***"
    assert utils.mask_secret("ABCDEFGHIJKL") == "AB***KL"
