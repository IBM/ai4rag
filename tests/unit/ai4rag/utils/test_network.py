# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.utils.network import ensure_safe_url, is_local_or_cluster_host, is_url_scheme_safe


class TestIsLocalOrClusterHost:
    """Test suite for :func:`is_local_or_cluster_host`."""

    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "LOCALHOST",
            "  localhost  ",
            "127.0.0.1",
            "::1",
            "foo.bar.svc.cluster.local",
            "my-svc.my-namespace.svc.cluster.local",
            "CLUSTER.LOCAL",
        ],
    )
    def test_true_for_local_or_cluster_hosts(self, host):
        """Loopback hosts and any '*cluster.local*' hostname are trusted."""
        assert is_local_or_cluster_host(host) is True

    @pytest.mark.parametrize("host", ["example.com", "8.8.8.8", "", None])
    def test_false_for_everything_else(self, host):
        """An external host (or an empty/missing one) is not trusted."""
        assert is_local_or_cluster_host(host) is False

    @pytest.mark.parametrize(
        "host",
        [
            "cluster.local.evil.com",
            "evil.com.cluster.local.evil.com",
            "notcluster.local",
            "evilcluster.local",
        ],
    )
    def test_false_for_cluster_local_as_subdomain_not_suffix(self, host):
        """'cluster.local' must anchor the actual domain suffix, not appear anywhere in the hostname.

        A substring check would let an attacker register 'cluster.local.evil.com' or embed
        'cluster.local' as a misleading subdomain label and have it wrongly trusted.
        """
        assert is_local_or_cluster_host(host) is False

    @pytest.mark.parametrize(
        "host",
        [
            "localhost:8080",
            "127.0.0.1:5432",
            "foo.bar.svc.cluster.local:2137",
            "my-svc.my-namespace.svc.cluster.local:2137",
            "CLUSTER.LOCAL:2137",
            "[::1]:8080",
            "[::1]",
        ],
    )
    def test_true_for_local_or_cluster_hosts_with_port(self, host):
        """A trailing ':port' (or bracketed IPv6) must not defeat the local/cluster check.

        A bare host field (e.g. PGVectorConfig.host) isn't guaranteed to be port-free before
        reaching this function, so the check must tolerate a port suffix.
        """
        assert is_local_or_cluster_host(host) is True

    def test_true_for_bare_ipv6_without_brackets(self):
        """A bare, unbracketed IPv6 literal with no port must still match '::1' unchanged."""
        assert is_local_or_cluster_host("::1") is True

    @pytest.mark.parametrize("host", ["example.com:8080", "cluster.local.evil.com:2137", "8.8.8.8:53"])
    def test_false_for_external_host_with_port(self, host):
        """A port suffix must not make an external host look local/cluster-local."""
        assert is_local_or_cluster_host(host) is False


class TestIsUrlSchemeSafe:
    """Test suite for :func:`is_url_scheme_safe`."""

    def _check(self, url: str) -> bool:
        return is_url_scheme_safe(url, secure_schemes=frozenset({"https"}), insecure_schemes=frozenset({"http"}))

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com",
            "https://localhost:8080",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://foo.svc.cluster.local",
        ],
    )
    def test_allowed_urls(self, url):
        """https anywhere, and http only against a local/cluster host, are allowed."""
        assert self._check(url) is True

    @pytest.mark.parametrize("url", ["http://example.com", "http://8.8.8.8", "ftp://localhost"])
    def test_rejected_urls(self, url):
        """http against an external host, and unrecognized schemes, are rejected."""
        assert self._check(url) is False


class TestEnsureSafeUrl:
    """Test suite for :func:`ensure_safe_url`."""

    def test_allows_https_to_external_host(self):
        """https to any host raises nothing."""
        ensure_safe_url("https://example.com", context="test field")

    def test_allows_http_to_local_host(self):
        """http to localhost raises nothing."""
        ensure_safe_url("http://localhost:8080", context="test field")

    def test_rejects_http_to_external_host(self):
        """http to a non-local host raises ValueError naming the context and URL."""
        with pytest.raises(ValueError, match="test field"):
            ensure_safe_url("http://example.com", context="test field")

    def test_error_message_includes_url(self):
        """The raised error includes the offending URL for debuggability."""
        with pytest.raises(ValueError, match=r"http://example\.com"):
            ensure_safe_url("http://example.com", context="test field")

    def test_custom_scheme_pairs(self):
        """A caller-supplied secure/insecure scheme pair (e.g. Neo4j's) is honored."""
        ensure_safe_url(
            "neo4j+s://example.com",
            context="Neo4jConfig.uri",
            secure_schemes=frozenset({"neo4j+s"}),
            insecure_schemes=frozenset({"neo4j"}),
        )
        with pytest.raises(ValueError, match="Neo4jConfig.uri"):
            ensure_safe_url(
                "neo4j://example.com",
                context="Neo4jConfig.uri",
                secure_schemes=frozenset({"neo4j+s"}),
                insecure_schemes=frozenset({"neo4j"}),
            )
