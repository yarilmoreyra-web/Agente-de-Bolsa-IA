"""Comprobación sencilla de etiquetas HTML equilibradas (para los tests)."""
import re

_TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)[^>]*>")


def is_balanced(text: str) -> bool:
    stack = []
    for match in _TAG.finditer(text):
        closing, name = match.group(1) == "/", match.group(2).lower()
        if not closing:
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def strip_tags(text: str) -> str:
    return _TAG.sub("", text)
