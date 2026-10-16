# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Run the agent through OpenShell's HTTP CONNECT proxy for native TCP.

asyncpg does not honor HTTP proxy environment variables. When PGVector is
configured, proxychains4 intercepts the agent's native TCP sockets and tunnels
them through OpenShell's explicit egress proxy.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
import tempfile
from urllib.parse import urlsplit

DEFAULT_HTTP_PROXY_PORT = 80
REMOTE_DNS_SUBNET = 224
TCP_CONNECT_TIMEOUT_MS = 8000
TCP_READ_TIMEOUT_MS = 15000
EXACT_PROXY_HOST_MASK = "255.255.255.255"
PRIVATE_FILE_MODE = 0o600

PROXY_ENV_VARS = (
    "grpc_proxy",
    "GRPC_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "http_proxy",
    "HTTP_PROXY",
)


def proxy_address(proxy_url: str) -> tuple[str, int]:
    """Return an IPv4 address and port for an HTTP proxy URL."""
    if "://" not in proxy_url:
        proxy_url = f"http://{proxy_url}"

    parsed = urlsplit(proxy_url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise ValueError("OpenShell proxy must use an http:// URL with a host.")
    if parsed.username or parsed.password:
        raise ValueError("Authenticated HTTP proxies are not supported by this wrapper.")

    port = parsed.port or DEFAULT_HTTP_PROXY_PORT
    try:
        address = str(ipaddress.IPv4Address(parsed.hostname))
    except ipaddress.AddressValueError:
        try:
            addresses = socket.getaddrinfo(parsed.hostname, port, socket.AF_INET, socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError(f"Could not resolve HTTP proxy host {parsed.hostname!r}.") from exc
        if not addresses:
            raise ValueError(f"HTTP proxy host {parsed.hostname!r} has no IPv4 address.")
        address = addresses[0][4][0]

    return address, port


def write_proxychains_config(address: str, port: int) -> str:
    """Create a private proxychains config that tunnels through OpenShell."""
    # Bypass proxychains for the proxy connection itself to avoid a proxy loop.
    config = "\n".join(
        (
            "strict_chain",
            "proxy_dns",
            f"remote_dns_subnet {REMOTE_DNS_SUBNET}",
            f"tcp_connect_time_out {TCP_CONNECT_TIMEOUT_MS}",
            f"tcp_read_time_out {TCP_READ_TIMEOUT_MS}",
            f"localnet {address}:{port}/{EXACT_PROXY_HOST_MASK}",
            "[ProxyList]",
            f"http {address} {port}",
            "",
        )
    )
    descriptor, path = tempfile.mkstemp(prefix="openshell-proxychains-", suffix=".conf")
    os.fchmod(descriptor, PRIVATE_FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
        config_file.write(config)
    return path


def main() -> None:
    """Exec the agent, wrapping PGVector's native TCP connections if configured."""
    command = sys.argv[1:]
    if not command:
        raise SystemExit("Usage: proxychains_entrypoint.py COMMAND [ARG ...]")

    if not os.environ.get("PGVECTOR_HOST"):
        os.execv(command[0], command)

    proxy_url = next((os.environ[name] for name in PROXY_ENV_VARS if os.environ.get(name)), None)
    if proxy_url is None:
        raise SystemExit("PGVECTOR_HOST is set, but OpenShell did not provide an HTTP proxy.")

    address, port = proxy_address(proxy_url)
    config_path = write_proxychains_config(address, port)
    os.execv("/usr/bin/proxychains4", ["/usr/bin/proxychains4", "-q", "-f", config_path, *command])


if __name__ == "__main__":
    main()
