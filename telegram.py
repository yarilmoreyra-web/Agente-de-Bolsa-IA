"""telegram.py — Envío del informe por la Bot API de Telegram (con ``requests``).

* Variables de entorno: ``TELEGRAM_BOT_TOKEN`` y ``TELEGRAM_CHAT_ID``.
* ``parse_mode="HTML"`` (no Markdown) para evitar problemas de escape.
* Límite de 4096 caracteres por mensaje: ``split_message`` divide el HTML por bloques
  (candidata a candidata) en trozos de ``config.TELEGRAM.max_message_chars``
  (3800 por defecto) sin cortar etiquetas: si un bloque tiene que partirse, las
  etiquetas abiertas se cierran al final del trozo y se reabren en el siguiente.
* Reintentos con backoff exponencial ante errores de red y 5xx. Ante 429 se espera lo
  que indique ``retry_after``. Un 400 por HTML no válido se reintenta una vez como texto
  plano. El resto de errores 4xx no se reintentan.
* El token nunca aparece en los logs (se elimina también de los mensajes de error de
  ``requests``, que incluyen la URL).
* Si falla el envío, ``send_report`` devuelve ``ok=False`` y deja un ``ERROR`` en el log;
  el informe ya está guardado en disco y ``main.py`` decide el código de salida.
"""
from __future__ import annotations

import html
import logging
import re
import time as time_module
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

import config
from utils import mask_secret

logger = logging.getLogger("trading_agent.telegram")

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
# Si Telegram pide esperar más que esto, no bloqueamos la ejecución: se da por fallido.
MAX_RETRY_AFTER_SECONDS = 60.0

_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)[^>]*>")
_ATOM_RE = re.compile(r"<[^>]*>|&#?[A-Za-z0-9]+;|\s+|[^\s<&]+|[<&]")


class TelegramError(Exception):
    """Fallo al enviar un mensaje (los textos ya vienen sin el token)."""


class TelegramParseError(TelegramError):
    """Telegram rechazó el HTML del mensaje (HTTP 400 «can't parse entities»)."""


@dataclass
class SendResult:
    """Resultado del envío de un informe."""

    ok: bool
    parts_total: int = 0
    parts_sent: int = 0
    message_ids: list[int] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# División de mensajes
# --------------------------------------------------------------------------- #
def _closing_tags(stack: list[tuple[str, str]]) -> str:
    return "".join(f"</{name}>" for name, _ in reversed(stack))


def _opening_tags(stack: list[tuple[str, str]]) -> str:
    return "".join(raw for _, raw in stack)


def _update_stack(stack: list[tuple[str, str]], atom: str) -> None:
    """Actualiza la pila de etiquetas abiertas con un átomo (si es una etiqueta)."""
    match = _TAG_RE.fullmatch(atom)
    if not match:
        return
    closing, name = match.group(1) == "/", match.group(2).lower()
    if not closing:
        stack.append((name, atom))
    else:
        for index in range(len(stack) - 1, -1, -1):
            if stack[index][0] == name:
                del stack[index:]
                break


def _split_tagged(text: str, limit: int) -> list[str]:
    """Parte un texto HTML largo en trozos <= ``limit`` con las etiquetas equilibradas."""
    pieces: list[str] = []
    stack: list[tuple[str, str]] = []
    current = ""
    has_content = False

    def flush() -> None:
        nonlocal current, has_content
        pieces.append(current.rstrip() + _closing_tags(stack))
        current, has_content = _opening_tags(stack), False

    for atom in _ATOM_RE.findall(text):
        is_tag = _TAG_RE.fullmatch(atom) is not None
        trial = list(stack)
        _update_stack(trial, atom)
        if len(current) + len(atom) + len(_closing_tags(trial)) > limit and has_content:
            flush()
            if atom.isspace():
                continue
            trial = list(stack)
            _update_stack(trial, atom)

        # Átomo que no cabe ni en un trozo vacío (palabra enorme): se corta por longitud.
        while not is_tag and len(current) + len(atom) + len(_closing_tags(stack)) > limit:
            room = max(1, limit - len(current) - len(_closing_tags(stack)))
            current += atom[:room]
            atom = atom[room:]
            has_content = True
            flush()
        current += atom
        has_content = has_content or not (atom.isspace() or is_tag)
        stack[:] = trial if is_tag else stack
    if has_content:
        pieces.append(current + _closing_tags(stack))
    return [p for p in pieces if p.strip()]


def _rebalance(chunks: list[str]) -> list[str]:
    """Garantiza que cada trozo tenga sus etiquetas equilibradas (no-op si ya lo están)."""
    stack: list[tuple[str, str]] = []
    balanced = []
    for chunk in chunks:
        prefix = _opening_tags(stack)
        for atom in _ATOM_RE.findall(chunk):
            _update_stack(stack, atom)
        balanced.append(prefix + chunk + _closing_tags(stack))
    return balanced


def split_message(html_text: str, limit: int | None = None) -> list[str]:
    """Divide un HTML en mensajes de como máximo ``limit`` caracteres.

    Se agrupan bloques enteros (separados por una línea en blanco), de modo que una
    candidata no se parte si cabe en un mensaje. Un bloque demasiado largo se divide por
    líneas y, si una línea aún no cabe, por palabras, siempre con las etiquetas cerradas
    y reabiertas.
    """
    limit = limit or config.TELEGRAM.max_message_chars
    text = html_text.strip("\n")
    if not text.strip():
        return []
    if len(text) <= limit:
        return [text]

    def fit_block(block: str) -> list[str]:
        if len(block) <= limit:
            return [block]
        parts: list[str] = []
        current = ""
        for line in block.split("\n"):
            if len(line) > limit:
                if current:
                    parts.append(current)
                    current = ""
                parts.extend(_split_tagged(line, limit))
                continue
            joined = f"{current}\n{line}" if current else line
            if len(joined) <= limit:
                current = joined
            else:
                parts.append(current)
                current = line
        if current:
            parts.append(current)
        return parts

    chunks: list[str] = []
    current = ""
    for block in (b for b in text.split("\n\n") if b.strip()):
        for piece in fit_block(block):
            joined = f"{current}\n\n{piece}" if current else piece
            if len(joined) <= limit:
                current = joined
            else:
                if current:
                    chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return _rebalance(chunks)


# --------------------------------------------------------------------------- #
# Envío
# --------------------------------------------------------------------------- #
def _scrub(text: str, token: str) -> str:
    """Elimina el token (y su forma ``bot<token>``) de un texto que va al log."""
    return text.replace(token, mask_secret(token)) if token else text


def _plain_text(html_text: str) -> str:
    """Versión sin etiquetas ni entidades de un texto HTML."""
    return html.unescape(re.sub(r"<[^>]+>", "", html_text))


def _json_body(response: Any) -> Mapping[str, Any]:
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - cuerpo no JSON
        return {}
    return body if isinstance(body, Mapping) else {}


def _post_with_retries(
    payload: dict[str, Any], token: str, http_post: Callable[..., Any] | None,
    telegram_cfg: config.TelegramConfig,
) -> Mapping[str, Any]:
    """POST a ``sendMessage`` con reintentos. Devuelve el JSON de éxito o lanza ``TelegramError``."""
    poster = http_post
    if poster is None:
        import requests  # noqa: PLC0415 - import diferido

        poster = requests.post
    url = API_URL.format(token=token)
    attempts = max(1, telegram_cfg.max_retries)
    last_error = "sin detalle"

    for attempt in range(1, attempts + 1):
        backoff = telegram_cfg.retry_base_delay * (2 ** (attempt - 1))
        wait = backoff
        try:
            response = poster(url, json=payload, timeout=telegram_cfg.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - errores de red de requests
            last_error = _scrub(f"{type(exc).__name__}: {exc}", token)
            logger.warning("Telegram: error de red (intento %d/%d): %s",
                           attempt, attempts, last_error)
        else:
            status = response.status_code
            body = _json_body(response)
            if status == 200 and body.get("ok") is not False:
                return body      # 200 = aceptado; no se reintenta para no duplicar el mensaje
            description = _scrub(str(body.get("description", "")), token)
            last_error = f"HTTP {status}: {description}".strip()
            if status == 429:
                params = body.get("parameters") or {}
                retry_after = params.get("retry_after") if isinstance(params, Mapping) else None
                if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
                    if retry_after > MAX_RETRY_AFTER_SECONDS:
                        raise TelegramError(f"{last_error} (retry_after={retry_after}s excede "
                                            f"el máximo de {MAX_RETRY_AFTER_SECONDS:g}s)")
                    wait = float(retry_after)
                logger.warning("Telegram: límite de frecuencia (intento %d/%d); espero %.1fs",
                               attempt, attempts, wait)
            elif status >= 500:
                logger.warning("Telegram: error del servidor (intento %d/%d): %s",
                               attempt, attempts, last_error)
            elif status == 400 and "parse entities" in description.lower():
                raise TelegramParseError(last_error)
            else:
                raise TelegramError(last_error)   # 4xx no reintentable (token, chat, etc.)
        if attempt < attempts:
            time_module.sleep(wait)
    raise TelegramError(f"{last_error} (tras {attempts} intentos)")


def send_message(
    text: str, token: str, chat_id: str, *, parse_mode: str | None = "HTML",
    http_post: Callable[..., Any] | None = None,
    telegram_cfg: config.TelegramConfig | None = None,
) -> int | None:
    """Envía un mensaje y devuelve su ``message_id`` (None si Telegram no lo indica).

    Si Telegram rechaza el HTML, reintenta una vez como texto plano.
    """
    telegram_cfg = telegram_cfg or config.TELEGRAM
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text,
                               "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        body = _post_with_retries(payload, token, http_post, telegram_cfg)
    except TelegramParseError as exc:
        logger.warning("Telegram rechazó el HTML (%s); se reenvía como texto plano", exc)
        payload.pop("parse_mode", None)
        payload["text"] = _plain_text(text)
        body = _post_with_retries(payload, token, http_post, telegram_cfg)
    result = body.get("result")
    message_id = result.get("message_id") if isinstance(result, Mapping) else None
    return int(message_id) if isinstance(message_id, int) else None


def send_report(
    html_text: str, env: config.EnvSettings | None = None,
    http_post: Callable[..., Any] | None = None,
    telegram_cfg: config.TelegramConfig | None = None,
) -> SendResult:
    """Divide el informe en mensajes y los envía en orden.

    Nunca lanza excepciones por fallos de envío: los registra (``ERROR``, sin el token)
    y devuelve ``SendResult(ok=False, ...)``. Si falla una parte, no se envían las siguientes.
    """
    telegram_cfg = telegram_cfg or config.TELEGRAM
    settings = env or config.load_env_settings()
    if not settings.telegram_enabled:
        message = "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID no configurados; informe no enviado"
        logger.error(message)
        return SendResult(ok=False, error=message)

    parts = split_message(html_text, telegram_cfg.max_message_chars)
    result = SendResult(ok=False, parts_total=len(parts))
    if not parts:
        result.error = "informe vacío"
        logger.error("No hay contenido que enviar a Telegram")
        return result

    token, chat_id = settings.telegram_bot_token, settings.telegram_chat_id
    for index, part in enumerate(parts, start=1):
        try:
            message_id = send_message(part, token, chat_id, http_post=http_post,
                                      telegram_cfg=telegram_cfg)
        except TelegramError as exc:
            result.error = _scrub(str(exc), token)
            logger.error("Telegram: fallo al enviar la parte %d/%d: %s", index, len(parts),
                         result.error)
            return result
        result.parts_sent += 1
        if message_id is not None:
            result.message_ids.append(message_id)
    result.ok = True
    logger.info("Informe enviado a Telegram en %d mensaje(s)", len(parts))
    return result
