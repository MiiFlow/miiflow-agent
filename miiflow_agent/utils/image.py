"""Image and file conversion utilities for LLM providers."""

import base64
import re
from io import BytesIO
from typing import Optional, Tuple

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
    image_url: str, timeout: float = 10.0, max_retries: int = 3
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

    Returns:
        Tuple of (image_bytes, mimetype)

    Raises:
        ValueError: If URL format is not supported
    """
    if is_data_uri(image_url):
        return data_uri_to_bytes_and_mimetype(image_url)
    elif is_http_url(image_url):
        return await url_to_bytes_and_mimetype(image_url, timeout, max_retries)
    else:
        raise ValueError(f"Unsupported image URL format: {image_url}")
