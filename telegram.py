"""Envío de mensajes a Telegram (Bot API), con reintentos y troceado seguro de HTML.

No se usa el SDK oficial: una llamada HTTP sencilla (inyectable como
``http_post`` para los tests, sin red) es suficiente. El token nunca se
imprime en claro en los logs ni en los mensajes de error (se enmascara con
``utils.mask_secret``).
"""
from __future__ import annotations

import logging
import re
import time as time_module
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import config
import utils

logger = logging.getLogger("trading_agent.telegram")

API_BASE = "https://api.telegram.org"

# Umbral por encima del cual un ``retry_after`` de Telegram se considera
# excesivo: mejor abortar y avisar que bloquear el proceso una hora.
MAX_RETRY_AFTER_SECONDS = 60.0

_TOKEN_RE = re.compile(r"<[^>]+>|&[a-zA-Z]+;|&#\d+;|\s+|[^\s<&]+|<|&")
_TAG_NAME_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9-]*)")
_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")


class TelegramError(Exception):
    """Error al enviar un mensaje o informe a Telegram."""


@dataclass
class SendResult:
    """Resultado de enviar un informe (puede tener varias partes)."""

    ok: bool
    parts_total: int = 0
    parts_sent: int = 0
    message_ids: List[int] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "parts_total": self.parts_total,
            "parts_sent": self.parts_sent,
            "message_ids": self.message_ids,
            "error": self.error,
        }


def _sanitize(text: str, token: str = "") -> str:
    """Quita el token de un mensaje de error/log antes de mostrarlo."""
    if token and token in text:
        return text.replace(token, utils.mask_secret(token))
    return text


def _tag_name(token: str) -> str:
    match = _TAG_NAME_RE.match(token)
    return match.group(1).lower() if match else ""


def _is_atomic(token: str) -> bool:
    """True si el token no debe partirse nunca (una etiqueta o una entidad HTML)."""
    if token.startswith("<") and token.endswith(">"):
        return True
    return bool(_ENTITY_RE.fullmatch(token))


def _stack_after(token: str, stack: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Aplica un token (posible apertura/cierre de etiqueta) a la pila de etiquetas abiertas."""
    if token.startswith("</"):
        name = _tag_name(token)
        for index in range(len(stack) - 1, -1, -1):
            if stack[index][0] == name:
                return stack[:index] + stack[index + 1:]
        return stack
    if token.startswith("<") and token.endswith(">") and not token.endswith("/>"):
        name = _tag_name(token)
        if name:
            return stack + [(name, token)]
    return stack


def _closing_overhead(stack: List[Tuple[str, str]]) -> int:
    return sum(len(name) + 3 for name, _ in stack)  # "</" + name + ">"


def split_message(html_text: str, max_chars: Optional[int] = None) -> List[str]:
    """Divide ``html_text`` en trozos de como mucho ``max_chars`` caracteres.

    Prefiere cortar por bloques (líneas en blanco) enteros; si un bloque no
    cabe, lo parte por líneas; si una línea no cabe, la parte por palabras;
    solo si una palabra sola no cabe se corta a mitad. Nunca corta a mitad de
    una etiqueta HTML ni de una entidad (``&amp;``): si una etiqueta queda
    abierta al final de un trozo, se cierra ahí y se reabre al principio del
    siguiente, para que cada trozo sea HTML válido por sí solo.
    """
    max_chars = max_chars or config.TELEGRAM.max_message_chars
    if not html_text or not html_text.strip():
        return []
    if len(html_text) <= max_chars:
        return [html_text]

    stack: List[Tuple[str, str]] = []
    current: List[str] = []
    current_len = 0
    chunks: List[str] = []

    def overhead() -> int:
        return _closing_overhead(stack)

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        closing = "".join(f"</{name}>" for name, _ in reversed(stack))
        chunks.append("".join(current) + closing)
        prefix = "".join(original for _, original in stack)
        current = [prefix] if prefix else []
        current_len = len(prefix)

    def advance_stack_over(text: str) -> None:
        nonlocal stack
        for tok in _TOKEN_RE.findall(text):
            stack = _stack_after(tok, stack)

    def place_atom(token: str) -> None:
        """Nivel más fino: una etiqueta, una entidad o una palabra suelta.

        Solo aquí se admite el corte a mitad de contenido, y únicamente para
        palabras (nunca para etiquetas ni entidades).
        """
        nonlocal current, current_len, stack
        new_stack = _stack_after(token, stack)
        new_overhead = _closing_overhead(new_stack)
        projected = current_len + len(token) + new_overhead
        if current and projected > max_chars:
            flush()
            new_stack = _stack_after(token, stack)
            new_overhead = _closing_overhead(new_stack)
            projected = current_len + len(token) + new_overhead
        if projected > max_chars and not _is_atomic(token):
            remaining = token
            while remaining:
                budget = max_chars - current_len - overhead()
                if budget <= 0:
                    flush()
                    budget = max_chars - current_len - overhead()
                take = max(budget, 1)
                piece = remaining[:take]
                current.append(piece)
                current_len += len(piece)
                remaining = remaining[len(piece):]
            return
        current.append(token)
        current_len += len(token)
        stack = new_stack

    def place(body: str, sep_prefix: str, seps: Tuple[str, ...]) -> None:
        """Intenta colocar ``body`` entero (pegado a ``sep_prefix`` si el
        trozo actual ya tiene contenido). Si no cabe ni vacío, lo parte por
        el separador más grueso de ``seps`` y coloca cada parte por
        separado; sin separadores ya, cae al nivel de palabras."""
        nonlocal current, current_len, stack
        glued = (sep_prefix + body) if current else body
        projected = current_len + len(glued) + overhead()
        if projected <= max_chars:
            current.append(glued)
            current_len += len(glued)
            advance_stack_over(glued)
            return
        if len(body) + overhead() <= max_chars:
            if current:
                flush()
            current.append(body)
            current_len += len(body)
            advance_stack_over(body)
            return
        if seps:
            sep, rest = seps[0], seps[1:]
            for index, part in enumerate(body.split(sep)):
                place(part, sep if index > 0 else "", rest)
        else:
            for tok in _TOKEN_RE.findall(body):
                place_atom(tok)

    place(html_text, "", ("\n\n", "\n"))
    flush()
    return [chunk for chunk in chunks if chunk.strip()]


# --------------------------------------------------------------------------- #
# Envío HTTP con reintentos
# --------------------------------------------------------------------------- #
def _extract_retry_after(response: Any, default: float) -> Tuple[float, bool]:
    """``(segundos, si_lo_indicó_telegram)`` a partir de un 429."""
    try:
        body = response.json()
        params = body.get("parameters") if isinstance(body, dict) else None
        if isinstance(params, dict) and "retry_after" in params:
            return float(params["retry_after"]), True
    except Exception:  # noqa: BLE001 - cuerpo no es JSON válido
        pass
    return default, False


def _response_description(response: Any) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and body.get("description"):
            return str(body["description"])
    except Exception:  # noqa: BLE001
        pass
    text = getattr(response, "text", "") or ""
    return text or f"HTTP {getattr(response, 'status_code', '?')}"


def _to_plain_text(html_text: str) -> str:
    """Quita etiquetas y desescapa entidades: aproximación de texto plano."""
    import html as _html
    stripped = re.sub(r"<[^>]+>", "", html_text)
    return _html.unescape(stripped)


def send_message(
    html_text: str,
    token: str,
    chat_id: str,
    http_post: Optional[Any] = None,
    max_retries: Optional[int] = None,
) -> Optional[int]:
    """Envía un mensaje HTML. Devuelve el ``message_id`` o ``None`` si no se pudo confirmar.

    Reintenta en errores de red, 429 (respetando ``retry_after`` salvo que
    sea excesivo) y 5xx, con backoff exponencial. Un 4xx no se reintenta,
    salvo que Telegram rechace el HTML: en ese caso se reenvía una vez como
    texto plano. Un 200 siempre cuenta como enviado (nunca se reintenta, para
    no duplicar mensajes).
    """
    if not token or not chat_id:
        logger.error("Telegram: faltan credenciales (token o chat_id).")
        raise TelegramError("faltan credenciales de Telegram")

    cfg = config.TELEGRAM
    max_retries = max_retries or cfg.max_retries
    delay = cfg.retry_base_delay
    url = f"{API_BASE}/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": html_text, "parse_mode": "HTML"}
    poster = http_post

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = poster(url, json=payload, timeout=cfg.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - error de red del cliente HTTP
            last_error = exc
            logger.warning(
                "Telegram: error de red (intento %d/%d): %s",
                attempt, max_retries, _sanitize(str(exc), token),
            )
            if attempt < max_retries:
                time_module.sleep(delay)
                delay *= 2
            continue

        status = getattr(response, "status_code", 200)

        if status == 200:
            try:
                data = response.json()
            except Exception:  # noqa: BLE001 - cuerpo 200 ilegible: se da por enviado
                return None
            if not isinstance(data, dict):
                return None
            return data.get("result", {}).get("message_id")

        if status == 429:
            retry_after, provided = _extract_retry_after(response, default=delay)
            if retry_after > MAX_RETRY_AFTER_SECONDS:
                raise TelegramError(
                    f"Telegram: límite de envíos (429) pide esperar "
                    f"{retry_after:.0f}s, demasiado; se aborta el envío."
                )
            logger.warning("Telegram: límite de envíos (429). Espero %.1fs.", retry_after)
            time_module.sleep(retry_after)
            if not provided:
                delay *= 2
            continue

        if status >= 500:
            last_error = TelegramError(f"HTTP {status}: {_response_description(response)}")
            logger.warning(
                "Telegram: error del servidor (intento %d/%d): %s",
                attempt, max_retries, last_error,
            )
            if attempt < max_retries:
                time_module.sleep(delay)
                delay *= 2
            continue

        # 4xx: sin reintento, salvo el caso especial de HTML rechazado.
        description = _response_description(response)
        if status == 400 and "parse entities" in description.lower():
            logger.warning(
                "Telegram: HTML rechazado (%s); reintento como texto plano.", description
            )
            plain_payload = dict(payload)
            plain_payload.pop("parse_mode", None)
            plain_payload["text"] = _to_plain_text(html_text)
            retry_response = poster(url, json=plain_payload, timeout=cfg.timeout_seconds)
            if getattr(retry_response, "status_code", 400) == 200:
                try:
                    data = retry_response.json()
                except Exception:  # noqa: BLE001
                    return None
                return data.get("result", {}).get("message_id") if isinstance(data, dict) else None
            raise TelegramError(
                f"Telegram: HTML rechazado y el reintento en texto plano también falló "
                f"(HTTP {getattr(retry_response, 'status_code', '?')})."
            )

        raise TelegramError(f"HTTP {status}: {description}")

    detail = _sanitize(str(last_error), token) if last_error else "error desconocido"
    raise TelegramError(f"Telegram: fallo tras {max_retries} intentos: {detail}")


def send_report(
    text: str,
    env: Any,
    http_post: Optional[Any] = None,
    telegram_cfg: Any = None,
) -> SendResult:
    """Trocea ``text`` (ya en HTML) y lo envía por partes con las credenciales de ``env``."""
    cfg = telegram_cfg or config.TELEGRAM

    if not getattr(env, "telegram_enabled", False):
        logger.error("Telegram: credenciales no configurados (falta bot token o chat id).")
        return SendResult(ok=False, error="credenciales de Telegram no configurados")

    if not text or not text.strip():
        logger.error("Telegram: informe vacío, no se envía nada.")
        return SendResult(ok=False, error="informe vacío")

    parts = split_message(text, max_chars=cfg.max_message_chars)
    result = SendResult(ok=True, parts_total=len(parts))
    for index, part in enumerate(parts, start=1):
        try:
            message_id = send_message(
                part, env.telegram_bot_token, env.telegram_chat_id,
                http_post=http_post, max_retries=cfg.max_retries,
            )
        except TelegramError as exc:
            result.ok = False
            result.error = _sanitize(str(exc), getattr(env, "telegram_bot_token", ""))
            logger.error(
                "Telegram: fallo enviando la parte %d/%d: %s",
                index, len(parts), result.error,
            )
            return result
        result.parts_sent += 1
        if message_id is not None:
            result.message_ids.append(message_id)
        logger.info("Telegram: parte %d/%d enviada.", index, len(parts))
    return result
