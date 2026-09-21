"""Tests de la configuración y de las variables de entorno."""
import config


def test_configuracion_valida():
    config.validate_config()


def test_pesos_suman_100():
    assert config.WEIGHTS.total() == 100.0


def test_gemini_model_vacio_usa_valor_por_defecto(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_MODEL", "")
    settings = config.load_env_settings(tmp_path / "no_existe.env")
    assert settings.gemini_model == config.DEFAULT_GEMINI_MODEL


def test_gemini_model_personalizado(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-otro-modelo")
    settings = config.load_env_settings(tmp_path / "no_existe.env")
    assert settings.gemini_model == "gemini-otro-modelo"


def test_load_env_file_lee_valores_y_no_pisa_el_entorno(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comentario\n"
        "TELEGRAM_CHAT_ID=12345\n"
        "GEMINI_API_KEY=\"clave-con-comillas\"\n"
        "TELEGRAM_BOT_TOKEN=token-del-archivo\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token-del-entorno")
    settings = config.load_env_settings(env_file)
    assert settings.telegram_chat_id == "12345"
    assert settings.gemini_api_key == "clave-con-comillas"
    assert settings.telegram_bot_token == "token-del-entorno"


def test_flags_enabled(monkeypatch, tmp_path):
    for name in ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                 "ALPHAVANTAGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = config.load_env_settings(tmp_path / "no_existe.env")
    assert not settings.gemini_enabled
    assert not settings.telegram_enabled
    assert not settings.alphavantage_enabled
