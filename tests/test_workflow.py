"""Tests de coherencia: workflow de GitHub Actions, .env.example y main.py."""
import os
import re
from pathlib import Path

import main

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "daily_agent.yml"
ENV_EXAMPLE = ROOT / ".env.example"
GITIGNORE = ROOT / ".gitignore"


def workflow_text():
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_tiene_dos_crons_de_lunes_a_viernes_y_disparo_manual():
    text = workflow_text()
    crons = re.findall(r'cron:\s*"([^"]+)"', text)
    assert len(crons) == 2, "hacen falta dos cron: horario de verano e invierno"
    assert {c.split()[1] for c in crons} == {"12", "13"}   # 12:35 y 13:35 UTC
    assert all(c.endswith("* * 1-5") for c in crons)
    assert "workflow_dispatch:" in text


def test_workflow_permisos_concurrencia_y_timeout():
    text = workflow_text()
    assert "contents: write" in text
    assert "cancel-in-progress: false" in text
    assert "timeout-minutes:" in text


def test_workflow_pasa_secretos_como_secrets_y_el_modelo_como_variable():
    text = workflow_text()
    for name in ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        assert f"{name}: ${{{{ secrets.{name} }}}}" in text
    assert "GEMINI_MODEL: ${{ vars.GEMINI_MODEL }}" in text


def test_todas_las_variables_de_env_example_llegan_al_workflow():
    keys = re.findall(r"^([A-Z][A-Z0-9_]*)=", ENV_EXAMPLE.read_text(encoding="utf-8"), re.M)
    assert {"GEMINI_API_KEY", "GEMINI_MODEL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"} <= set(keys)
    text = workflow_text()
    for key in keys:
        assert f"{key}:" in text, f"{key} está en .env.example pero no en el workflow"


def test_los_flags_del_workflow_existen_en_main():
    flags = set(re.findall(r"args\+=\((--[a-z-]+)", workflow_text()))
    assert {"--date", "--dry-run", "--no-gemini", "--force"} <= flags
    parser = main.build_parser()
    known = {opt for action in parser._actions for opt in action.option_strings}
    for flag in flags:
        assert flag in known, f"main.py no reconoce {flag}"


def test_el_workflow_no_pega_entradas_dentro_del_script():
    # Las entradas manuales deben pasar por variables de entorno (INPUT_*),
    # nunca como ${{ inputs.x }} dentro de un bloque "run:" (riesgo de inyección).
    text = workflow_text()
    assert "INPUT_DATE: ${{ inputs.date }}" in text
    run_blocks = re.findall(r"run: \|\n((?:[ ]{10,}.*\n)+)", text)
    for block in run_blocks:
        assert "${{ inputs." not in block


def test_gitignore_protege_los_secretos():
    ignored = GITIGNORE.read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored
    assert ".venv/" in ignored


def test_no_hay_claves_hardcodeadas():
    patterns = [
        re.compile(r"AIza[0-9A-Za-z_\-]{35}"),                 # claves de Google
        re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b"),        # tokens de Telegram
    ]
    skip_dirs = {".venv", ".git", "__pycache__", ".pytest_cache", "logs", "data"}
    suffixes = {".py", ".yml", ".yaml", ".md", ".txt", ".json", ".ini", ".example"}
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            path = Path(folder) / name
            if path.suffix not in suffixes:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                assert not pattern.search(content), f"posible clave en {path.relative_to(ROOT)}"
