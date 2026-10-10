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

The same extra layer reaches ordinary tool arguments: a model named a workflow
"Cross-Platform Paid Media Monitor &amp; Optimizer" and a report "Spend Trend
&amp; CPA", with no escaped text anywhere in its context, and both were stored
and shown with the entity. :func:`normalize_label` repairs one such label;
``core.tools.argument_normalization`` decides which arguments are labels.
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


#: One complete character reference, semicolon included. html.unescape alone
#: also decodes legacy semicolon-less forms, so "&copy2024" would become
#: "©2024"; this shape keeps labels that merely contain "&" untouched.
_CHAR_REF_RE = re.compile(
    r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});"
)

#: A label is escaped once or twice in practice; the bound only stops a
#: pathological value from looping.
_MAX_LABEL_PASSES = 4


def normalize_label(text: str) -> str:
    """Decode a plain-text label until no character reference is left.

    Decoding to a fixed point, not one level, is what makes this idempotent
    for any label escaped fewer than :data:`_MAX_LABEL_PASSES` times: a call
    can pass through more than one entry point (an approval resume re-enters
    the executor), and a second pass must not change the result.
    The cost is that a label whose plain text really contains "&amp;" loses
    it, which no entity name does.
    """
    if not text:
        return text
    if "\\u" in text or "\\U" in text:
        text = _decode_unicode_escapes(text)
    for _ in range(_MAX_LABEL_PASSES):
        if "&" not in text:
            break
        decoded = _CHAR_REF_RE.sub(lambda match: html.unescape(match.group(0)), text)
        if decoded == text:
            break
        text = decoded
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
