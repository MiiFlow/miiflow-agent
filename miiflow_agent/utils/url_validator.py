"""URL security validation."""
import ipaddress
from urllib.parse import urlparse


class URLSecurityError(Exception):
    pass


def validate_external_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
        
        if parsed.scheme not in ['http', 'https']:
            return False, f"Invalid scheme: {parsed.scheme}"
        
        hostname = parsed.hostname
        if not hostname:
            return False, "No hostname in URL"
        
        if hostname in ['localhost', '127.0.0.1', '0.0.0.0', '::1']:
            return False, "Localhost URLs not allowed"
        
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_link_local or ip.is_loopback:
                return False, f"Private/internal IP not allowed: {hostname}"
        except ValueError:
            blocked_patterns = [
                'internal', 'local', 'corp', 'intranet',
                '192.168.', '10.', '172.16.', '172.17.',
                '172.18.', '172.19.', '172.20.', '172.21.',
                '172.22.', '172.23.', '172.24.', '172.25.',
                '172.26.', '172.27.', '172.28.', '172.29.',
                '172.30.', '172.31.'
            ]
            
            lower_host = hostname.lower()
            for pattern in blocked_patterns:
                if pattern in lower_host:
                    return False, f"Suspicious hostname: {pattern}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"
