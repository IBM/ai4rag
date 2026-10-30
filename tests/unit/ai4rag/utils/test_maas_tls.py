# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""MaaS trust precedence, public-root preservation, and verified HTTPS coverage."""

import asyncio
import json
import os
import shutil
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import certifi
import httpx
import pytest

from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.template.agentic_rag_template import AgenticRAG
from ai4rag.utils.clients.maas_client import _get_maas_tls_verify, create_maas_client

_FIXTURES = Path(__file__).with_name("fixtures") / "maas_tls"


@pytest.fixture(autouse=True)
def clean_ca_environment(monkeypatch):
    """Keep trust-policy tests independent of the invoking machine."""
    for key in ("MAAS_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


@pytest.fixture
def mounted_bundle(tmp_path, monkeypatch):
    """Emulate DSP's plain PEM bundle in an otherwise unhashed directory."""
    bundle = tmp_path / "ca.crt"
    shutil.copyfile(_FIXTURES / "ca.pem", bundle)
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    return bundle


def test_no_mount_keeps_httpx_defaults(monkeypatch, tmp_path):
    """Neither an absent directory nor a directory without ca.crt overrides trust."""
    assert _get_maas_tls_verify(None) is None
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "absent"))
    assert _get_maas_tls_verify(None) is None
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    assert _get_maas_tls_verify(None) is None


def test_mounted_ca_supplements_public_roots(mounted_bundle):
    """The private root is added without dropping any default public roots."""
    default_roots = set(ssl.create_default_context(cafile=certifi.where()).get_ca_certs(binary_form=True))
    context = _get_maas_tls_verify(None)
    mounted_root = ssl.PEM_cert_to_DER_cert(mounted_bundle.read_text())
    assert set(context.get_ca_certs(binary_form=True)) == default_roots | {mounted_root}
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert "SSL_CERT_FILE" not in os.environ


def test_multiple_directories_find_the_mounted_bundle(mounted_bundle, monkeypatch, tmp_path):
    """Directory lists preserve discovery even when the first entry has no bundle."""
    monkeypatch.setenv("SSL_CERT_DIR", os.pathsep.join((str(tmp_path / "missing"), str(mounted_bundle.parent))))
    assert ssl.PEM_cert_to_DER_cert(mounted_bundle.read_text()) in _get_maas_tls_verify(None).get_ca_certs(
        binary_form=True
    )


def test_explicit_settings_take_precedence(mounted_bundle, monkeypatch):
    """A mounted bundle never replaces explicitly selected trust."""
    mounted_bundle.write_text("invalid mounted certificate")
    monkeypatch.setenv("SSL_CERT_FILE", "/configured/global.pem")
    assert _get_maas_tls_verify(None) is None
    monkeypatch.setenv("MAAS_CA_BUNDLE", "/configured/maas.pem")
    assert _get_maas_tls_verify(None) == "/configured/maas.pem"
    assert _get_maas_tls_verify("/explicit.pem") == "/explicit.pem"


def test_invalid_mounted_bundle_fails_closed(mounted_bundle):
    """A malformed configured bundle is surfaced rather than ignored or retried insecurely."""
    mounted_bundle.write_text("not a PEM certificate")
    with pytest.raises(ssl.SSLError):
        _get_maas_tls_verify(None)


@pytest.fixture
def https_maas():
    """Serve an OpenAI-compatible endpoint signed by a test-only private CA."""

    class Handler(BaseHTTPRequestHandler):
        """Return local model listings and chat completions without external services."""

        def do_GET(self):  # pylint: disable=invalid-name
            """List the endpoint's models."""
            self.reply({"object": "list", "data": []})

        def do_POST(self):  # pylint: disable=invalid-name
            """Return a fixed chat completion."""
            self.rfile.read(int(self.headers["Content-Length"]))
            self.reply(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "trusted"}, "finish_reason": "stop"}
                    ],
                }
            )

        def reply(self, body):
            """Write one JSON response."""
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            """Keep test output free of request logs."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(_FIXTURES / "ca.pem", _FIXTURES / "key.pem")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_private_endpoint_requires_mounted_ca(https_maas, mounted_bundle, monkeypatch):
    """A real TLS handshake fails with default trust and succeeds with the mounted root."""
    monkeypatch.delenv("SSL_CERT_DIR")
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        httpx.get(https_maas, trust_env=False)
    monkeypatch.setenv("SSL_CERT_DIR", str(mounted_bundle.parent))
    with create_maas_client(https_maas, "test-key") as client:
        assert client.chat.completions.create(model="test-model", messages=[]).choices[0].message.content == "trusted"


@pytest.mark.parametrize("configuration", ["mounted", "argument", "maas_env", "ssl_file"])
def test_agent_preserves_sync_and_async_trust(https_maas, mounted_bundle, monkeypatch, configuration):
    """Agent requests retain both inferred and explicitly supplied per-client CA configuration."""
    if configuration != "mounted":
        monkeypatch.delenv("SSL_CERT_DIR")
    if configuration == "maas_env":
        monkeypatch.setenv("MAAS_CA_BUNDLE", str(mounted_bundle))
    elif configuration == "ssl_file":
        monkeypatch.setenv("SSL_CERT_FILE", str(mounted_bundle))
    ca_bundle = str(mounted_bundle) if configuration == "argument" else None
    with create_maas_client(https_maas, "test-key", ca_bundle=ca_bundle) as client:
        foundation = OpenAIFoundationModel(client=client, model_id="test-model")
        agent = AgenticRAG(foundation_model=foundation, retriever=MagicMock())
        chat_model = agent._chat_model()  # pylint: disable=protected-access
        assert chat_model.root_client is client
        assert chat_model.invoke("test").content == "trusted"
        if configuration != "ssl_file":
            assert chat_model.http_async_client.timeout.read == 600.0

        async def invoke():
            try:
                return await chat_model.ainvoke("test")
            finally:
                await chat_model.root_async_client.close()

        assert asyncio.run(invoke()).content == "trusted"
