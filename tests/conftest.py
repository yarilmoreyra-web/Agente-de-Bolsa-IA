"""Configuración común de los tests: aísla rutas para no tocar tus datos reales."""
import dataclasses

import pytest

import config
import utils


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Redirige logs, histórico y .env a una carpeta temporal en cada test."""
    paths = dataclasses.replace(
        config.PATHS,
        env_file=tmp_path / ".env",
        data_dir=tmp_path / "data",
        history_dir=tmp_path / "data" / "history",
        logs_dir=tmp_path / "logs",
        log_file=tmp_path / "logs" / "agent.log",
    )
    monkeypatch.setattr(config, "PATHS", paths)
    yield
    utils.shutdown_logging()
