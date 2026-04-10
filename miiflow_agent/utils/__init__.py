"""Utility functions for Miiflow LLM."""

from .env import load_env_file, get_api_key
from .image import (
    is_data_uri,
    is_http_url,
    extract_mimetype_from_data_uri,
    data_uri_to_bytes,
    data_uri_to_bytes_and_mimetype,
    data_uri_to_base64_and_mimetype,
    bytes_to_data_uri,
    url_to_bytes_and_mimetype,
    detect_mimetype_from_bytes,
    image_url_to_bytes,
    resize_image_for_api,
    ImageResizeError,
    API_IMAGE_MAX_BASE64_SIZE,
    IMAGE_TARGET_RAW_SIZE,
    IMAGE_MAX_WIDTH,
    IMAGE_MAX_HEIGHT,
)

__all__ = [
    "load_env_file",
    "get_api_key",
    "is_data_uri",
    "is_http_url",
    "extract_mimetype_from_data_uri",
    "data_uri_to_bytes",
    "data_uri_to_bytes_and_mimetype",
    "data_uri_to_base64_and_mimetype",
    "bytes_to_data_uri",
    "url_to_bytes_and_mimetype",
    "detect_mimetype_from_bytes",
    "image_url_to_bytes",
    "resize_image_for_api",
    "ImageResizeError",
    "API_IMAGE_MAX_BASE64_SIZE",
    "IMAGE_TARGET_RAW_SIZE",
    "IMAGE_MAX_WIDTH",
    "IMAGE_MAX_HEIGHT",
]
