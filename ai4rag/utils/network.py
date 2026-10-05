# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Host and URL trust policy shared by every externally-configurable endpoint.

ai4rag never falls back to unverified TLS. Instead, a plaintext scheme
(``http://``, ``neo4j://``, ``bolt://``, ...) is only tolerated when it targets
a host the deployment itself controls -- loopback, or an in-cluster Kubernetes
service -- and every other endpoint must use an encrypted, verified scheme.
:func:`ensure_safe_url` is the single place that policy is enforced, so every
consumer (MaaS, S3, Milvus, Neo4j, pgvector) stays in agreement about what
counts as trustworthy.
"""

from urllib.parse import urlsplit

__all__ = ["is_local_or_cluster_host", "is_url_scheme_safe", "ensure_safe_url"]

_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_CLUSTER_LOCAL_SUFFIX = "cluster.local"


def _strip_optional_port(host: str) -> str:
    """Strip a trailing ``:port`` from *host*, if present, without mangling an IPv6 literal.

    Handles ``host:port`` (one colon, digits after it) and bracketed IPv6
    (``[::1]:port``). A bare, unbracketed IPv6 literal such as ``::1`` or
    ``fe80::1`` has no unambiguous port separator, so it is returned
    unchanged -- exactly the form :attr:`_LOCAL_HOSTNAMES` already expects.
    """
    if host.startswith("["):
        return host[1 : host.index("]")] if "]" in host else host
    head, sep, tail = host.rpartition(":")
    if sep and tail.isdigit() and ":" not in head:
        return head
    return host


def is_local_or_cluster_host(host: str | None) -> bool:
    """Return whether *host* is a loopback address or an in-cluster DNS name.

    Parameters
    ----------
    host
        A hostname or IP address, with or without a trailing ``:port`` (e.g.
        ``my-svc.svc.cluster.local:2137``) or surrounding IPv6 brackets. This
        accepts both :func:`urllib.parse.urlsplit`'s ``hostname`` attribute
        (already port-free) and a raw ``host:port`` string, including
        ``None`` for a URL with no host.

    Returns
    -------
    bool
        ``True`` for ``localhost``, ``127.0.0.1``, ``::1`` (case- and
        whitespace-insensitive), and any hostname ending in ``cluster.local``
        (Kubernetes' default in-cluster DNS suffix, e.g.
        ``my-svc.my-namespace.svc.cluster.local``), each with or without a
        trailing port; ``False`` otherwise.

    Notes
    -----
    The cluster-local check matches on the ``.``-anchored suffix, not a bare
    substring -- ``cluster.local.evil.com`` does not match, since
    ``cluster.local`` there is a subdomain label, not the actual domain.
    """
    normalized = _strip_optional_port((host or "").strip().lower())
    return normalized in _LOCAL_HOSTNAMES or (
        normalized == _CLUSTER_LOCAL_SUFFIX or normalized.endswith("." + _CLUSTER_LOCAL_SUFFIX)
    )


def is_url_scheme_safe(url: str, *, secure_schemes: frozenset[str], insecure_schemes: frozenset[str]) -> bool:
    """Return whether *url* satisfies ai4rag's scheme/host trust policy.

    A URL using one of *secure_schemes* is always allowed. One using an
    *insecure_schemes* member is allowed only when it targets a local or
    in-cluster host (see :func:`is_local_or_cluster_host`). Any other scheme
    is rejected.

    Parameters
    ----------
    url
        The URL or URI to check.
    secure_schemes
        Scheme names (lowercase, no ``://``) that are always trusted, e.g.
        ``{"https"}``.
    insecure_schemes
        Scheme names that are trusted only against a local/in-cluster host,
        e.g. ``{"http"}``.

    Returns
    -------
    bool
        ``True`` if *url* is allowed under this policy.
    """
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    if scheme in secure_schemes:
        return True
    if scheme in insecure_schemes:
        return is_local_or_cluster_host(parts.hostname or "")
    return False


def ensure_safe_url(
    url: str,
    *,
    context: str,
    secure_schemes: frozenset[str] = frozenset({"https"}),
    insecure_schemes: frozenset[str] = frozenset({"http"}),
) -> None:
    """Raise ``ValueError`` if *url* fails ai4rag's scheme/host trust policy.

    Parameters
    ----------
    url
        The URL or URI to validate.
    context
        Short description of the caller/field, used in the error message
        (e.g. ``"MaaS base_url"``, ``"MilvusConfig.uri"``).
    secure_schemes, insecure_schemes
        See :func:`is_url_scheme_safe`. Default to the ``http``/``https``
        pair.

    Raises
    ------
    ValueError
        If *url* is rejected by the policy.
    """
    if not is_url_scheme_safe(url, secure_schemes=secure_schemes, insecure_schemes=insecure_schemes):
        raise ValueError(
            f"{context} ({url!r}) uses an insecure scheme against a non-local, non-cluster host. "
            f"Use one of {sorted(secure_schemes)} for any endpoint outside localhost/127.0.0.1/::1 "
            "or a '*.cluster.local' in-cluster service -- ai4rag never falls back to unverified TLS."
        )
