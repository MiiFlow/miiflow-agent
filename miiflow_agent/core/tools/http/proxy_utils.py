"""Proxy detection utilities for HTTP tools.

This module contains the proxy fix implementation that resolves corporate proxy issues.
"""

import os
from urllib.parse import urlparse
from typing import Optional, Dict


def get_proxy_config() -> Optional[Dict[str, str]]:
    """Automatically detect proxy settings from environment variables."""
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
    
    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies["http://"] = http_proxy
        if https_proxy:
            proxies["https://"] = https_proxy
        return proxies
    
    return None


def should_use_proxy(url: str) -> bool:
    """Check if URL should bypass proxy based on NO_PROXY."""
    no_proxy = os.getenv('NO_PROXY') or os.getenv('no_proxy')
    if not no_proxy:
        return True
    
    # Parse hostname from URL
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    
    # Check against NO_PROXY patterns
    for pattern in no_proxy.split(','):
        pattern = pattern.strip()
        if pattern and (hostname == pattern or hostname.endswith(f".{pattern}")):
            return False
    
    return True
