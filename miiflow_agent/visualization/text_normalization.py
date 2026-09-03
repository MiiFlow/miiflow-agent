"""Repair display text that reached us still encoded.

Models routinely hand visualization tools strings that carry another layer of
encoding: ``"Performance Overview \\u2014 Last 90 Days"`` instead of an em
dash, ``"Campaign Structure &amp; Performance"`` instead of an ampersand. The
value is a correct JSON string, so nothing upstream objects — and the browser
renders text nodes verbatim, so the escape sequence is what the reader sees.

Two rules, both chosen because a false positive is essentially impossible in
display copy:

* ``\\uXXXX`` / ``\\UXXXXXXXX`` — no title legitimately contains the six
  characters ``\\u2014``. Decoded by hand rather than via
  ``unicode_escape``, which is latin-1 based and mangles the real non-ASCII
  sitting next to the broken escape.
* HTML character references — ``html.unescape`` leaves a bare ``&`` alone, so
  "Spend & Revenue" is untouched while "&amp;" is repaired.

Payloads whose data is not display text (source code, form values) get their
chrome normalized and their body left exactly as authored.
"""

import html
import re
from typing import Any, Set

#: Visualization types whose `data` is a verbatim payload, not display copy.
#: Source code and form values must survive byte-for-byte.
RAW_DATA_TYPES: Set[str] = {"code_preview", "form"}

#: A backslash-escape that survived into the text. The leading (?<!\\) keeps
#: an already-escaped backslash — "C:\\users" — from being read as an escape.
_UNICODE_ESCAPE_RE = re.compile(r"(?<!\\)\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8})")

#: Guard against pathological nesting; visualization payloads are shallow.
_MAX_DEPTH = 12


def _decode_unicode_escapes(text: str) -> str:
    def replace(match: "re.Match[str]") -> str:
        try:
            return chr(int(match.group(0)[2:], 16))
        except (ValueError, OverflowError):
            return match.group(0)

    return _UNICODE_ESCAPE_RE.sub(replace, text)


def normalize_text(text: str) -> str:
    """Decode stray escape sequences and HTML entities in one display string."""
    if not text:
        return text
    if "\\u" in text or "\\U" in text:
        text = _decode_unicode_escapes(text)
    if "&" in text and ";" in text:
        text = html.unescape(text)
    return text


def normalize_payload(value: Any, _depth: int = 0) -> Any:
    """Apply :func:`normalize_text` to every string reachable in ``value``.

    Containers are rebuilt rather than mutated so a caller's own dict is never
    edited underneath it. Non-string leaves pass through untouched.
    """
    if isinstance(value, str):
        return normalize_text(value)
    if _depth >= _MAX_DEPTH:
        return value
    if isinstance(value, dict):
        return {k: normalize_payload(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_payload(v, _depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_payload(v, _depth + 1) for v in value)
    return value
