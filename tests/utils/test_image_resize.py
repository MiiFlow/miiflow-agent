"""Tests for the image resize / compression pipeline."""

from io import BytesIO

import pytest

from miiflow_agent.utils.image import (
    API_IMAGE_MAX_BASE64_SIZE,
    IMAGE_MAX_HEIGHT,
    IMAGE_MAX_WIDTH,
    IMAGE_TARGET_RAW_SIZE,
    ImageResizeError,
    PIL_AVAILABLE,
    resize_image_for_api,
)


pytestmark = pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")


def _make_jpeg(width: int, height: int, *, color=(200, 100, 50), quality: int = 95) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_png(width: int, height: int) -> bytes:
    from PIL import Image

    img = Image.new("RGBA", (width, height), color=(10, 20, 30, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_small_image_passes_through_unchanged():
    raw = _make_jpeg(64, 64)
    out, mime = resize_image_for_api(raw, "image/jpeg")
    assert out == raw
    assert mime == "image/jpeg"


def test_huge_jpeg_is_recompressed_under_cap():
    # Build a noisy image so it does not compress trivially.
    from PIL import Image
    import os

    pixels = os.urandom(IMAGE_MAX_WIDTH * 3 * IMAGE_MAX_HEIGHT * 3 * 3)
    img = Image.frombytes("RGB", (IMAGE_MAX_WIDTH * 3, IMAGE_MAX_HEIGHT * 3), pixels)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    raw = buf.getvalue()
    assert len(raw) > IMAGE_TARGET_RAW_SIZE  # sanity: actually too big

    out, mime = resize_image_for_api(raw, "image/jpeg")

    assert mime == "image/jpeg"
    assert len(out) <= IMAGE_TARGET_RAW_SIZE
    decoded = Image.open(BytesIO(out))
    assert decoded.width <= IMAGE_MAX_WIDTH
    assert decoded.height <= IMAGE_MAX_HEIGHT


def test_huge_png_recompressed_under_cap():
    # PNG of solid color compresses extremely well, but force a big bitmap.
    raw = _make_png(IMAGE_MAX_WIDTH * 2, IMAGE_MAX_HEIGHT * 2)
    out, mime = resize_image_for_api(raw, "image/png")
    # Either kept as PNG or downgraded; either way must fit.
    assert len(out) <= IMAGE_TARGET_RAW_SIZE
    encoded_b64_len = ((len(out) + 2) // 3) * 4
    assert encoded_b64_len <= API_IMAGE_MAX_BASE64_SIZE
    assert mime in {"image/png", "image/jpeg", "image/webp"}


def test_unsupported_mimetype_raises():
    with pytest.raises(ImageResizeError) as ei:
        resize_image_for_api(b"\x00" * (IMAGE_TARGET_RAW_SIZE + 1), "application/pdf")
    assert ei.value.error_type == "unsupported"


def test_corrupt_bytes_raises_processing_error():
    with pytest.raises(ImageResizeError) as ei:
        resize_image_for_api(b"\x00" * (IMAGE_TARGET_RAW_SIZE + 1), "image/jpeg")
    assert ei.value.error_type == "processing"


def test_empty_bytes_raises():
    with pytest.raises(ImageResizeError):
        resize_image_for_api(b"", "image/jpeg")


def test_small_compressed_image_still_obeys_dimension_limits():
    from PIL import Image

    raw = _make_png(100, 9001)
    assert len(raw) < IMAGE_TARGET_RAW_SIZE
    out, _ = resize_image_for_api(raw, "image/png")
    with Image.open(BytesIO(out)) as decoded:
        assert decoded.width <= IMAGE_MAX_WIDTH
        assert decoded.height <= IMAGE_MAX_HEIGHT


def test_small_corrupt_image_is_rejected():
    with pytest.raises(ImageResizeError, match="decode"):
        resize_image_for_api(b"not actually an image", "image/png")


def test_mime_type_matches_decoded_bytes():
    raw = _make_png(64, 64)
    out, mime = resize_image_for_api(raw, "image/jpeg")
    assert out == raw
    assert mime == "image/png"


def test_small_bmp_is_converted_to_supported_format():
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="BMP")
    out, mime = resize_image_for_api(buf.getvalue(), "image/bmp")
    assert mime == "image/jpeg"
    assert Image.open(BytesIO(out)).format == "JPEG"
