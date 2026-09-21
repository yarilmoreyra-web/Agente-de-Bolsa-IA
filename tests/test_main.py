"""Tests del flujo de main.py (sin red).

Estos tests comprueban la CLI: argumentos, calendario bursátil, guarda
horaria e idempotencia. Ninguno debe llegar a ejecutar el pipeline de
verdad (eso lo cubre test_pipeline.py, con datos sintéticos), así que
todos usan el fixture ``stub_run_pipeline`` de conftest.py.

Dos cuidados que ya causaron un cuelgue real en esta suite:
1. Sin el fixture, main.main() llama a main.run_pipeline() de verdad, que
   intenta descargar datos de yfinance por la red.
2. Si además la sesión analizada es la de "hoy" (según utils.now_ny, que
   algunos tests congelan con monkeypatch) y la hora está antes del
   snapshot, run_pipeline espera de verdad en wait_until() hasta esa hora.
   Si el reloj está congelado con una lambda constante, esa espera nunca
   termina: el bucle vuelve a preguntar la hora y siempre obtiene la misma.
   Por eso los tests que fijan una hora "dentro de la ventana" pasan
   siempre --no-wait, aparte de usar el stub.
"""
from datetime import datetime

import pytest
from openpyxl import Workbook

import main
import utils


def make_universe(path, tickers):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Ticker"])
    for ticker in tickers:
        sheet.append([ticker])
    workbook.save(path)


@pytest.fixture
def universe_file(tmp_path):
    file = tmp_path / "lista.xlsx"
    make_universe(file, ["aapl", "NVDA", "AAPL", "TSLA"])
    return file


def fijar_hora(monkeypatch, year, month, day, hour, minute):
    fake_now = datetime(year, month, day, hour, minute, tzinfo=utils.NY_TZ)
    monkeypatch.setattr(utils, "now_ny", lambda: fake_now)


def test_dry_run_force_muestra_lista_leida(universe_file, stub_run_pipeline, capsys):
    code = main.main([
        "--dry-run", "--force", "--no-wait", "--date", "2026-09-21",
        "--universe", str(universe_file),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Lista leída del archivo: 3 tickers" in out
    assert len(stub_run_pipeline) == 1  # se llamó a run_pipeline una vez


def test_fecha_invalida_termina_con_codigo_2():
    with pytest.raises(SystemExit) as info:
        main.main(["--date", "21-09-2026"])
    assert info.value.code == 2


def test_universo_inexistente_devuelve_1(tmp_path, capsys):
    code = main.main([
        "--dry-run", "--force", "--date", "2026-09-21",
        "--universe", str(tmp_path / "no_existe.xlsx"),
    ])
    assert code == 1
    assert "ERROR" in capsys.readouterr().out


def test_fin_de_semana_sin_force_no_genera_informe(universe_file, capsys):
    code = main.main([
        "--date", "2026-09-19", "--universe", str(universe_file),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Sin sesión bursátil" in out
    assert "Lista leída" not in out


def test_backtest_aun_no_disponible(capsys):
    code = main.main(["--backtest-date", "2026-09-21"])
    assert code == 2
    assert "Fase 5" in capsys.readouterr().out


def test_guarda_horaria_fuera_de_ventana(universe_file, monkeypatch, capsys):
    fijar_hora(monkeypatch, 2026, 9, 21, 12, 0)  # lunes 12:00 ET
    code = main.main(["--universe", str(universe_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Fuera de la ventana" in out
    assert "Lista leída" not in out


def test_guarda_horaria_dentro_de_ventana(
    universe_file, monkeypatch, stub_run_pipeline, capsys
):
    fijar_hora(monkeypatch, 2026, 9, 21, 8, 35)  # lunes 08:35 ET
    code = main.main(["--no-wait", "--universe", str(universe_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Lista leída del archivo: 3 tickers" in out
    assert len(stub_run_pipeline) == 1


def test_idempotencia_si_ya_se_envio(universe_file, monkeypatch, capsys):
    import config

    fijar_hora(monkeypatch, 2026, 9, 21, 8, 35)
    utils.write_json_atomic(
        config.PATHS.history_dir / "2026-09-21.json", {"sent": True}
    )
    code = main.main(["--universe", str(universe_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert "ya fue enviado" in out
    assert "Lista leída" not in out
