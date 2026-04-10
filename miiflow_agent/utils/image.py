"""Image and file conversion utilities for LLM providers."""

import base64
import logging
import re
from io import BytesIO
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import filetype

    FILETYPE_AVAILABLE = True
except ImportError:
    FILETYPE_AVAILABLE = False

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from PIL import Image as _PILImage  # type: ignore

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without Pillow
    PIL_AVAILABLE = False


# ---- Image resize / compression pipeline ------------------------------------
#
# Borrowed conceptually from Claude Code's `src/utils/imageResizer.ts`. Goal:
# bound the size of any image we send to an LLM provider so we don't get
# rejected (Anthropic for example caps base64 image source size). When Pillow
# is not installed, the resizer is a no-op pass-through and just enforces the
# raw cap (raising ImageResizeError if the bytes are still too big).

#: Approximate Anthropic / OpenAI per-image base64 cap. Stay safely under 5MB.
API_IMAGE_MAX_BASE64_SIZE = 5 * 1024 * 1024

#: Hard cap on raw bytes before we even attempt to upload. Pillow will resize
#: down to fit this if needed.
IMAGE_TARGET_RAW_SIZE = 3 * 1024 * 1024

#: Maximum dimensions in pixels. Most providers re-downscale beyond this anyway.
IMAGE_MAX_WIDTH = 1568
IMAGE_MAX_HEIGHT = 1568

#: PDF size at which we should reference rather than inline (caller decision).
PDF_INLINE_MAX_SIZE = 4 * 1024 * 1024

#: MIME types we know how to resize via Pillow.
_RESIZABLE_MIMETYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/tif",
    "image/gif",
}


class ImageResizeError(Exception):
    """Raised when an image cannot be resized to fit provider limits.

    The ``error_type`` attribute classifies the failure for analytics /
    callers, mirroring Claude Code's taxonomy: ``module_load``, ``processing``,
    ``pixel_limit``, ``memory``, ``permission``, ``too_large``, ``unsupported``.
    """

    def __init__(self, message: str, error_type: str = "processing"):
        super().__init__(message)
        self.error_type = error_type


def _encoded_size(raw_len: int) -> int:
    """Approximate base64 encoded length for a given raw byte count."""
    return ((raw_len + 2) // 3) * 4


def resize_image_for_api(
    image_bytes: bytes,
    mimetype: str,
    *,
    max_base64_size: int = API_IMAGE_MAX_BASE64_SIZE,
    target_raw_size: int = IMAGE_TARGET_RAW_SIZE,
    max_width: int = IMAGE_MAX_WIDTH,
    max_height: int = IMAGE_MAX_HEIGHT,
) -> Tuple[bytes, str]:
    """Resize and/or recompress an image so it fits provider limits.

    Returns ``(possibly_new_bytes, possibly_new_mimetype)``. If the input is
    already small enough, the original bytes are returned unchanged. If Pillow
    is not installed, no resizing is attempted; if the raw bytes still exceed
    the cap, an ``ImageResizeError`` is raised.

    Parameters mirror the Claude Code constants but are overridable per-call
    so providers with tighter caps can opt in.
    """
    if not image_bytes:
        raise ImageResizeError("empty image bytes", error_type="processing")

    # Fast path: already small enough.
    if (
        len(image_bytes) <= target_raw_size
        and _encoded_size(len(image_bytes)) <= max_base64_size
    ):
        return image_bytes, mimetype

    if mimetype not in _RESIZABLE_MIMETYPES:
        raise ImageResizeError(
            f"image of type {mimetype!r} exceeds limits and is not resizable",
            error_type="unsupported",
        )

    if not PIL_AVAILABLE:
        raise ImageResizeError(
            "image exceeds size limits and Pillow is not installed; "
            "install with `pip install miiflow-agent[images]` to enable resizing",
            error_type="module_load",
        )

    try:
        img = _PILImage.open(BytesIO(image_bytes))
        img.load()
    except Exception as e:
        raise ImageResizeError(f"failed to decode image: {e}", error_type="processing") from e

    # Convert palette / alpha modes to something JPEG-friendly if we plan to
    # re-encode as JPEG. PNG/WebP can keep alpha.
    out_format = "JPEG" if mimetype in {"image/jpeg", "image/jpg", "image/bmp", "image/tiff", "image/tif"} else None
    if out_format is None:
        # Keep PNG -> PNG, WebP -> WEBP, GIF -> PNG (single frame).
        if mimetype == "image/png":
            out_format = "PNG"
        elif mimetype == "image/webp":
            out_format = "WEBP"
        elif mimetype == "image/gif":
            out_format = "PNG"
        else:
            out_format = "JPEG"

    # Step 1: dimension clamp.
    w, h = img.size
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        try:
            img = img.resize(new_size, _PILImage.LANCZOS)
        except Exception as e:
            raise ImageResizeError(f"resize failed: {e}", error_type="processing") from e

    # Step 2: encode, then iteratively re-encode at lower quality / smaller
    # dimensions until we fit under both caps. JPEG/WebP support quality;
    # PNG falls back to dimension reduction only.
    qualities = [85, 75, 65, 55, 45, 35]
    attempts = 0
    while attempts < 12:
        buf = BytesIO()
        save_kwargs: dict = {}
        if out_format in ("JPEG", "WEBP"):
            quality = qualities[min(attempts, len(qualities) - 1)]
            save_kwargs["quality"] = quality
            if out_format == "JPEG":
                save_kwargs["optimize"] = True
                if img.mode in ("RGBA", "P", "LA"):
                    img_to_save = img.convert("RGB")
                else:
                    img_to_save = img
            else:
                img_to_save = img
        else:
            img_to_save = img
            save_kwargs["optimize"] = True
        try:
            img_to_save.save(buf, format=out_format, **save_kwargs)
        except Exception as e:
            raise ImageResizeError(
                f"encode failed ({out_format}): {e}", error_type="processing"
            ) from e

        out_bytes = buf.getvalue()
        if (
            len(out_bytes) <= target_raw_size
            and _encoded_size(len(out_bytes)) <= max_base64_size
        ):
            new_mime = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }.get(out_format, mimetype)
            return out_bytes, new_mime

        # Still too big — shrink dimensions 20% and try again.
        w, h = img.size
        new_w, new_h = max(64, int(w * 0.8)), max(64, int(h * 0.8))
        if (new_w, new_h) == (w, h):
            break
        try:
            img = img.resize((new_w, new_h), _PILImage.LANCZOS)
        except Exception as e:
            raise ImageResizeError(f"resize failed: {e}", error_type="processing") from e
        attempts += 1

    raise ImageResizeError(
        "image still exceeds API limits after maximum resize attempts",
        error_type="too_large",
    )


# Common image extensions and their MIME types
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".ico": "image/x-icon",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def is_data_uri(uri: str) -> bool:
    """Check if a string is a data URI."""
    return uri.startswith("data:")


def is_http_url(url: str) -> bool:
    """Check if a string is an HTTP(S) URL."""
    return url.startswith(("http://", "https://"))


def extract_mimetype_from_data_uri(data_uri: str) -> str:
    """
    Extract MIME type from a data URI.

    Args:
        data_uri: Data URI string (e.g., 'data:image/png;base64,...')

    Returns:
        MIME type string (e.g., 'image/png')
    """
    if not data_uri.startswith("data:"):
        return "application/octet-stream"

    try:
        if ";" not in data_uri:
            return "application/octet-stream"

        # Extract: data:MIMETYPE;base64,...
        header = data_uri.split(",", 1)[0]
        mimetype = header.split(":")[1].split(";")[0]
        return mimetype or "application/octet-stream"
    except (IndexError, ValueError):
        return "application/octet-stream"


def data_uri_to_bytes(data_uri: str) -> bytes:
    """
    Convert a Base64 data URI to bytes.

    Args:
        data_uri: Data URI string (e.g., 'data:image/png;base64,...')

    Returns:
        Raw bytes

    Raises:
        ValueError: If data URI is invalid
    """
    if not data_uri.startswith("data:"):
        raise ValueError("Not a valid data URI")

    if "," not in data_uri:
        raise ValueError("Invalid data URI format: missing comma separator")

    # Split and get only the base64 part
    base64_data = data_uri.split(",", 1)[1]
    return base64.b64decode(base64_data)


def data_uri_to_bytes_and_mimetype(data_uri: str) -> Tuple[bytes, str]:
    """
    Convert a data URI to bytes and extract MIME type.

    Args:
        data_uri: Data URI string (e.g., 'data:image/png;base64,...')

    Returns:
        Tuple of (bytes, mimetype)
    """
    mimetype = extract_mimetype_from_data_uri(data_uri)
    image_bytes = data_uri_to_bytes(data_uri)
    return image_bytes, mimetype


def data_uri_to_base64_and_mimetype(data_uri: str) -> Tuple[str, str]:
    """
    Extract base64 string and MIME type from a data URI.

    Args:
        data_uri: Data URI string (e.g., 'data:image/png;base64,...')

    Returns:
        Tuple of (base64_string, mimetype)

    Raises:
        ValueError: If data URI is invalid
    """
    if not data_uri.startswith("data:"):
        raise ValueError("Not a valid data URI")

    if "," not in data_uri:
        raise ValueError("Invalid data URI format: missing comma separator")

    # Extract MIME type
    mimetype = extract_mimetype_from_data_uri(data_uri)

    # Extract base64 string (after the comma)
    base64_string = data_uri.split(",", 1)[1]

    return base64_string, mimetype


def bytes_to_data_uri(file_bytes: bytes, mimetype: Optional[str] = None) -> str:
    """
    Convert raw bytes to a base64 data URI.

    Args:
        file_bytes: Raw file bytes
        mimetype: MIME type (will auto-detect if not provided)

    Returns:
        Data URI string
    """
    if mimetype is None:
        mimetype = detect_mimetype_from_bytes(file_bytes)

    base64_str = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mimetype};base64,{base64_str}"


async def url_to_bytes_and_mimetype(
    url: str, timeout: float = 10.0, max_retries: int = 3
) -> Tuple[bytes, str]:
    """
    Download a file from URL and return bytes with MIME type.

    Args:
        url: HTTP(S) URL to download
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for rate limiting

    Returns:
        Tuple of (bytes, mimetype)

    Raises:
        ImportError: If httpx is not available
        Exception: If download fails
    """
    if not HTTPX_AVAILABLE:
        raise ImportError(
            "httpx is required to download images from URLs. "
            "Install with: pip install httpx"
        )

    # Add headers to look like a legitimate browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }

    async with httpx.AsyncClient(
        timeout=timeout, headers=headers, follow_redirects=True
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

        # Get content type from response headers
        content_type = response.headers.get("content-type", "")
        # Remove charset or additional parameters
        if ";" in content_type:
            content_type = content_type.split(";")[0].strip()

        # Fall back to auto-detection if needed
        if not content_type or content_type == "application/octet-stream":
            content_type = detect_mimetype_from_bytes(response.content)

        return response.content, content_type


def url_to_base64_and_mimetype(
    url: str, timeout: float = 30.0, *, resize: bool = False
) -> Tuple[str, str]:
    """
    Download an image from URL and return base64 string with MIME type.

    Synchronous version for use in message formatting (e.g. Anthropic client
    which requires base64 image sources, not URLs).

    Args:
        url: HTTP(S) URL to download
        timeout: Request timeout in seconds

    Returns:
        Tuple of (base64_string, mimetype)

    Raises:
        ImportError: If httpx is not available
        Exception: If download fails
    """
    if not HTTPX_AVAILABLE:
        raise ImportError(
            "httpx is required to download images from URLs. "
            "Install with: pip install httpx"
        )

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if ";" in content_type:
            content_type = content_type.split(";")[0].strip()

        if not content_type or content_type == "application/octet-stream":
            content_type = detect_mimetype_from_bytes(response.content)

        image_bytes = response.content
        if resize:
            try:
                image_bytes, content_type = resize_image_for_api(image_bytes, content_type)
            except ImageResizeError:
                raise
            except Exception as e:
                logger.warning(
                    "image resize unexpectedly failed (%s); using original bytes", e
                )

        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return base64_str, content_type


def detect_mimetype_from_bytes(file_bytes: bytes) -> str:
    """
    Detect MIME type from raw bytes using filetype library.

    Args:
        file_bytes: Raw file bytes

    Returns:
        MIME type string
    """
    if not FILETYPE_AVAILABLE:
        return "application/octet-stream"

    try:
        kind = filetype.guess(file_bytes[:2048])
        if kind is not None:
            return kind.mime
    except Exception:
        pass

    return "application/octet-stream"


async def image_url_to_bytes(
    image_url: str,
    timeout: float = 10.0,
    max_retries: int = 3,
    *,
    resize: bool = False,
) -> Tuple[bytes, str]:
    """
    Convert any image URL format to bytes and MIME type.

    Handles:
    - HTTP(S) URLs: Downloads the image
    - Data URIs: Extracts base64 content

    Args:
        image_url: Image URL (http://, https://, or data:)
        timeout: Request timeout for HTTP URLs
        max_retries: Max retries for HTTP URLs
        resize: If True (default), automatically resize/recompress the image to
            fit provider limits via :func:`resize_image_for_api`. If the image
            cannot be made to fit, raises :class:`ImageResizeError`.

    Returns:
        Tuple of (image_bytes, mimetype)

    Raises:
        ValueError: If URL format is not supported
        ImageResizeError: If resize is requested and the image cannot fit
    """
    if is_data_uri(image_url):
        image_bytes, mimetype = data_uri_to_bytes_and_mimetype(image_url)
    elif is_http_url(image_url):
        image_bytes, mimetype = await url_to_bytes_and_mimetype(image_url, timeout, max_retries)
    else:
        raise ValueError(f"Unsupported image URL format: {image_url}")

    if resize:
        try:
            # Pillow is synchronous and CPU-bound; offload to a worker thread
            # so we don't block the asyncio event loop under ASGI servers.
            import asyncio

            image_bytes, mimetype = await asyncio.to_thread(
                resize_image_for_api, image_bytes, mimetype
            )
        except ImageResizeError:
            raise
        except Exception as e:  # defensive: never crash callers on resize bugs
            logger.warning("image resize unexpectedly failed (%s); using original bytes", e)

    return image_bytes, mimetype
