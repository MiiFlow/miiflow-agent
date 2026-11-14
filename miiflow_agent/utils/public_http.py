"""Socket-pinned HTTP clients for fetching remote media. TLS keeps the original hostname."""
import ipaddress
import socket
import time

import anyio
import httpcore
import httpx


def public_addresses(host, port):
    addresses = list(dict.fromkeys(row[4][0] for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global or ip.is_multicast or ip.is_reserved:
            raise httpcore.ConnectError("Media host must have only public addresses")
    if not addresses:
        raise httpcore.ConnectError("Media host did not resolve")
    return addresses


class _PublicNetworkBackend(httpcore.SyncBackend):
    def __init__(self, resolve):
        self.resolve = resolve

    def connect_tcp(self, host, port, timeout=None, **kwargs):
        started = time.monotonic()
        addresses = self.resolve(host, port)
        for index, address in enumerate(addresses):
            remaining = None if timeout is None else max(0, timeout - (time.monotonic() - started))
            try:
                return super().connect_tcp(address, port, timeout=None if remaining is None else remaining / (len(addresses) - index), **kwargs)
            except (httpcore.ConnectError, httpcore.ConnectTimeout):
                if index + 1 == len(addresses):
                    raise


class _AsyncPublicNetworkBackend(httpcore.AnyIOBackend):
    def __init__(self, resolve):
        super().__init__()
        self.resolve = resolve

    async def connect_tcp(self, host, port, timeout=None, **kwargs):
        try:
            with anyio.fail_after(timeout) as deadline:
                addresses = await anyio.to_thread.run_sync(self.resolve, host, port, abandon_on_cancel=True)
                for index, address in enumerate(addresses):
                    try:
                        remaining = deadline.deadline - anyio.current_time()
                        per_address = None if timeout is None else max(0, remaining) / (len(addresses) - index)
                        return await super().connect_tcp(address, port, timeout=per_address, **kwargs)
                    except (httpcore.ConnectError, httpcore.ConnectTimeout):
                        if index + 1 == len(addresses):
                            raise
        except TimeoutError as error:
            raise httpcore.ConnectTimeout("Connection timed out") from error


def _transport_options(options):
    # Direct sockets enforce destination policy. Proxy routing and alternate
    # transports would move final destination selection outside the guard.
    if any(key in options for key in ("proxy", "mounts", "transport")):
        raise ValueError("Guarded connections require a direct transport")
    return {key: options[key] for key in ("verify", "cert", "trust_env", "http1", "http2", "limits") if key in options}


def public_http_client(*, resolve=None, **kwargs) -> httpx.Client:
    # HTTPX 0.28 has no public network_backend option. Verify the expected
    # pool interface before replacing it so an incompatible upgrade fails closed.
    transport = httpx.HTTPTransport(**_transport_options(kwargs))
    if not isinstance(transport._pool._network_backend, httpcore.NetworkBackend):
        raise RuntimeError("Unsupported HTTPX socket backend")
    transport._pool._network_backend = _PublicNetworkBackend(resolve or public_addresses)
    return httpx.Client(transport=transport, **kwargs)


def public_async_http_client(*, resolve=None, **kwargs) -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport(**_transport_options(kwargs))
    if not isinstance(transport._pool._network_backend, httpcore.AsyncNetworkBackend):
        raise RuntimeError("Unsupported HTTPX async socket backend")
    transport._pool._network_backend = _AsyncPublicNetworkBackend(resolve or public_addresses)
    return httpx.AsyncClient(transport=transport, **kwargs)


def get_media(url, **kwargs):
    with public_http_client(**kwargs) as client:
        return client.get(url)
