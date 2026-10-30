# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Route the agent's TCP traffic through OpenShell egress policies.

gRPC uses a local CONNECT proxy; native database drivers use OpenShell's proxy.
"""

import ipaddress
import os
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

DEFAULT_HTTP_PROXY_PORT = 80
DEFAULT_GRPC_PROXY_PORT = 18080
CONNECT_BUFFER_SIZE = 65536
CONNECT_TIMEOUT_SECONDS = 15
LOCAL_PROXY_HOST = "127.0.0.1"
REMOTE_DNS_SUBNET = 224
TCP_CONNECT_TIMEOUT_MS = 8000
TCP_READ_TIMEOUT_MS = 15000
EXACT_PROXY_HOST_MASK = "255.255.255.255"
PRIVATE_FILE_MODE = 0o600
PROXY_START_TIMEOUT_SECONDS = 10.0
PROXY_POLL_INTERVAL_SECONDS = 0.1

PROXY_ENV_VARS = (
    "grpc_proxy",
    "GRPC_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "http_proxy",
    "HTTP_PROXY",
)
NATIVE_DATABASE_ENV_VARS = ("PGVECTOR_HOST", "NEO4J_URI")
GRPC_PROXY_PORT_ENV = "OPENSHELL_GRPC_PROXY_PORT"
CONNECT_PROXY_MODE = "--openshell-connect-proxy"
CONNECT_HEADER_LIMIT = 8192


def _read_connect_request(connection: socket.socket) -> tuple[bytes, bytes]:
    """Read a CONNECT request line and headers, returning any bytes that follow them."""
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = connection.recv(1024)
        if not chunk:
            raise ValueError("client closed the connection before sending a complete request")
        buffer += chunk
        if len(buffer) > CONNECT_HEADER_LIMIT:
            raise ValueError("request headers too long")
    request, _, leftover = buffer.partition(b"\r\n\r\n")
    return request.split(b"\r\n", 1)[0], leftover


def _pipe(source: socket.socket, destination: socket.socket) -> None:
    """Copy bytes from *source* to *destination* until EOF."""
    try:
        while True:
            chunk = source.recv(CONNECT_BUFFER_SIZE)
            if not chunk:
                break
            destination.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class ConnectHandler(socketserver.BaseRequestHandler):
    """Handle a single HTTP CONNECT tunnel over IPv4."""

    def handle(self) -> None:
        try:
            line, leftover = _read_connect_request(self.request)
            method, target = line.decode("ascii", "replace").split(" ")[:2]
        except (ValueError, IndexError):
            self.request.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return

        if method != "CONNECT" or ":" not in target:
            self.request.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return

        host, _, port = target.rpartition(":")
        try:
            address = socket.gethostbyname(host)
            upstream = socket.create_connection((address, int(port)), timeout=CONNECT_TIMEOUT_SECONDS)
        except (OSError, ValueError):
            self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return

        self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.request.settimeout(None)
        upstream.settimeout(None)
        if leftover:
            upstream.sendall(leftover)
        reverse = threading.Thread(target=_pipe, args=(upstream, self.request), daemon=True)
        reverse.start()
        try:
            _pipe(self.request, upstream)
        finally:
            reverse.join(timeout=5)
            upstream.close()


class ConnectProxy(socketserver.ThreadingTCPServer):
    """Threaded CONNECT proxy that can be quickly restarted."""

    allow_reuse_address = True
    daemon_threads = True


def _run_connect_proxy(host: str, port: int) -> None:
    """Serve local CONNECT requests until the proxy process is stopped."""
    with ConnectProxy((host, port), ConnectHandler) as server:
        server.serve_forever()


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
    except ipaddress.AddressValueError as address_error:
        try:
            addresses = socket.getaddrinfo(parsed.hostname, port, socket.AF_INET, socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError(f"Could not resolve HTTP proxy host {parsed.hostname!r}.") from exc
        if not addresses:
            raise ValueError(f"HTTP proxy host {parsed.hostname!r} has no IPv4 address.") from address_error
        address = addresses[0][4][0]

    return address, port


def _write_config(lines: tuple[str, ...]) -> str:
    """Write a private proxychains config file and return its path."""
    descriptor, path = tempfile.mkstemp(prefix="openshell-proxychains-", suffix=".conf")
    os.fchmod(descriptor, PRIVATE_FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
        config_file.write("\n".join(lines))
    return path


def write_proxychains_config(address: str, port: int) -> str:
    """Create a private proxychains config that tunnels through OpenShell."""
    # Bypass proxychains for the proxy connection itself to avoid a proxy loop.
    return _write_config(
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


def write_loopback_proxychains_config(port: int) -> str:
    """Create a config that tunnels everything through the loopback CONNECT proxy."""
    return _write_config(
        (
            "strict_chain",
            "proxy_dns",
            f"remote_dns_subnet {REMOTE_DNS_SUBNET}",
            f"tcp_connect_time_out {TCP_CONNECT_TIMEOUT_MS}",
            f"tcp_read_time_out {TCP_READ_TIMEOUT_MS}",
            "localnet 127.0.0.0/255.0.0.0",
            "[ProxyList]",
            f"http {LOCAL_PROXY_HOST} {port}",
            "",
        )
    )


def start_loopback_proxy(port: int) -> None:
    """Start this module's CONNECT proxy mode and wait until it listens."""
    # The proxy intentionally outlives this call, so it is not a context manager.
    subprocess.Popen(  # pylint: disable=consider-using-with
        [sys.executable, os.path.abspath(__file__), CONNECT_PROXY_MODE, LOCAL_PROXY_HOST, str(port)],
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        stderr=sys.stderr,
        start_new_session=True,
        close_fds=True,
    )

    deadline = time.monotonic() + PROXY_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((LOCAL_PROXY_HOST, port), timeout=1):
                return
        except OSError:
            time.sleep(PROXY_POLL_INTERVAL_SECONDS)
    raise SystemExit(f"Local gRPC CONNECT proxy did not start on {LOCAL_PROXY_HOST}:{port}.")


def main() -> None:
    """Exec the agent, wrapping TCP connections through proxychains4 if needed."""
    if len(sys.argv) > 1 and sys.argv[1] == CONNECT_PROXY_MODE:
        if len(sys.argv) != 4:
            raise SystemExit(f"Usage: {os.path.basename(__file__)} {CONNECT_PROXY_MODE} HOST PORT")
        _run_connect_proxy(sys.argv[2], int(sys.argv[3]))
        return

    command = sys.argv[1:]
    if not command:
        raise SystemExit("Usage: proxychains_entrypoint.py COMMAND [ARG ...]")

    grpc_proxy_port = os.environ.get(GRPC_PROXY_PORT_ENV, "").strip()
    if grpc_proxy_port:
        port = int(grpc_proxy_port) if grpc_proxy_port.isdigit() else DEFAULT_GRPC_PROXY_PORT
        start_loopback_proxy(port)
        config_path = write_loopback_proxychains_config(port)
    elif any(os.environ.get(name) for name in NATIVE_DATABASE_ENV_VARS):
        proxy_url = next((os.environ[name] for name in PROXY_ENV_VARS if os.environ.get(name)), None)
        if proxy_url is None:
            raise SystemExit("A native database host is configured, but OpenShell did not provide an HTTP proxy.")
        address, port = proxy_address(proxy_url)
        config_path = write_proxychains_config(address, port)
    else:
        os.execv(command[0], command)
        return

    os.execv("/usr/bin/proxychains4", ["/usr/bin/proxychains4", "-q", "-f", config_path, *command])


if __name__ == "__main__":
    main()
