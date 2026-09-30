# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TEMPLATE_DIR = Path(__file__).resolve().parents[4] / "ai4rag/assets_generator/starter_kit_templates/agentic_rag"


def _import_auth_wrapper(monkeypatch):
    app = SimpleNamespace(add_middleware=lambda middleware: None)
    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(app=app))
    spec = importlib.util.spec_from_file_location("starter_kit_auth_wrapper", _TEMPLATE_DIR / "auth_wrapper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auth_wrapper_rejects_missing_credentials(monkeypatch):
    """An exposed wrapper cannot accidentally run without authentication."""
    monkeypatch.delenv("K8S_API_URL", raising=False)
    monkeypatch.delenv("K8S_REVIEWER_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="required for the auth wrapper"):
        _import_auth_wrapper(monkeypatch)


def test_auth_wrapper_requires_https(monkeypatch, tmp_path):
    """TokenReview credentials cannot be sent over plaintext HTTP."""
    ca_path = tmp_path / "ca.crt"
    ca_path.write_text("test CA", encoding="utf-8")
    monkeypatch.setenv("K8S_API_URL", "http://kubernetes.default.svc")
    monkeypatch.setenv("K8S_REVIEWER_TOKEN", "reviewer-token")
    monkeypatch.setenv("K8S_CA_PATH", str(ca_path))

    with pytest.raises(RuntimeError, match="HTTPS"):
        _import_auth_wrapper(monkeypatch)


def test_auth_wrapper_requires_ca_certificate(monkeypatch, tmp_path):
    """TokenReview starts only with a CA file for TLS verification."""
    monkeypatch.setenv("K8S_API_URL", "https://kubernetes.default.svc")
    monkeypatch.setenv("K8S_REVIEWER_TOKEN", "reviewer-token")
    monkeypatch.setenv("K8S_CA_PATH", str(tmp_path / "missing.crt"))

    with pytest.raises(RuntimeError, match="CA certificate not found"):
        _import_auth_wrapper(monkeypatch)


def test_auth_wrapper_verifies_with_provided_ca(monkeypatch, tmp_path, mocker):
    """The configured CA path is passed to the TokenReview HTTP client."""
    ca_path = tmp_path / "ca.crt"
    ca_path.write_text("test CA", encoding="utf-8")
    monkeypatch.setenv("K8S_API_URL", "https://kubernetes.default.svc")
    monkeypatch.setenv("K8S_REVIEWER_TOKEN", "reviewer-token")
    monkeypatch.setenv("K8S_CA_PATH", str(ca_path))

    module = _import_auth_wrapper(monkeypatch)
    post = mocker.patch.object(module.requests, "post")
    post.return_value.status_code = 201
    post.return_value.json.return_value = {"status": {"authenticated": True, "user": {"username": "allowed"}}}

    assert asyncio.run(module._validate_k8s_token("client-token"))
    assert post.call_args.kwargs["verify"] == str(ca_path)
    makefile = (_TEMPLATE_DIR / "Makefile").read_text(encoding="utf-8")
    assert "K8S_API_INSECURE" not in makefile
    assert "K8S_CA_PATH=/sandbox/k8s-ca.crt" in makefile
