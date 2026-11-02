# Starter Kit

Packages a published RAG pattern into a deployable agentic-RAG microservice project.

::: ai4rag.assets_generator.starter_kit
    options:
      members:
        - generate_starter_kit


The generated `agent_config.json` includes the pattern's `template_id`. The runtime
resolves it through the template registry, selecting `AgenticRAG` or `SequentialRAG`.
Neo4j configurations retain effective graph retrieval settings and pass them to the
retriever. Graph retrieval also works when these options are omitted: the store
uses its built-in defaults. Persisting the options reproduces any custom graph
settings used during optimization. The application/package name remains
`agentic_rag` for all templates.

`template_id` is required by the generated runtime configuration. Regenerate older
starter kits when upgrading, and migrate explicit `simple_rag` pattern IDs to
`sequential_rag` before generation.

The generator reads the boolean `settings.agentic.enable_context_compression` from
the published pattern, defaulting to `True`, and writes it as
`agentic.enable_context_compression` in `agent_config.json`. Set it to `false` in
that JSON file to disable the agentic
[context-overflow fallback](../../architecture/rag-components.md#context-overflow-fallback).
This option does not affect sequential execution.

## OpenShell Deployment

Generated starter kits can deploy an agent backed by Milvus, PostgreSQL/pgvector,
or Neo4j. The generated `agent_config.json` selects the provider and existing
collection from the published pattern. Neo4j patterns use graph retrieval and
require a collection that has already been indexed; document indexing is a separate
pipeline step.

Deployment requires an OpenShift cluster with the Agent Sandbox operator, a logged-in
`oc` client, `podman`, `uv`, `python3`, and `curl`. The generated Makefile runs the OpenShell CLI from the
RHOAI image in a container and keeps the gateway registration and the mTLS client
certificates in `~/.config/openshell`, so no local OpenShell CLI installation is needed.
It also requires an OpenShell gateway that is already installed on the cluster.
Setting up the gateway is a one-time cluster prerequisite and is not part of the
generated starter kit. MaaS must provide reachable chat and embedding models, and
agentic templates require a chat model that supports tool calling.

The generated Makefile registers the `openshell-gateway` route as gateway `openshift`
and stores the mTLS client certificates from the `openshell-client-tls` secret when
you run `make refresh-certs`. Run it when a command reports `Unknown gateway
'openshift'` or that the gateway is not registered or reachable, and after the cluster
gateway is recreated or its certificates are rotated.

Cluster administrators can use the repository's
[`scripts/setup-openshell-gateway.sh`](https://github.com/IBM/ai4rag/blob/main/scripts/setup-openshell-gateway.sh)
for one-time gateway provisioning. Run it from an ai4rag checkout with `oc`,
Helm, `curl`, `jq`, and `podman` installed and the required cluster permissions:

```bash
bash scripts/setup-openshell-gateway.sh
```

The script installs the downstream Red Hat OpenShell 3.6 gateway with mTLS
authentication. It reads gateway, sandbox runtime, supervisor, and CLI image
references from the installed RHOAI ClusterServiceVersion (CSV), preserving image
digests. The Helm chart is pinned by `OPENSHELL_CHART_REPO` and
`OPENSHELL_CHART_REF`; these replace the former `OPENSHELL_VERSION` setting.
Run `bash scripts/setup-openshell-gateway.sh --help` for all setup overrides.

!!! warning "Migrating an upstream 0.0.99 gateway"
    The downstream 3.6 gateway cannot read the upstream 0.0.99 gateway database.
    A cluster administrator must reset that database during migration with
    `bash scripts/setup-openshell-gateway.sh --reset` (or `OPENSHELL_RESET=true`).
    This deletes gateway records for sandboxes, policies, and stored credentials;
    workspace PVCs are preserved. Recreate the required sandbox deployments afterward.

Gateway setup automatically mirrors sandbox runtime and supervisor images into the
cluster registry when the `add-imagepullsecrets` Kyverno policy is installed.
`OPENSHELL_MIRROR_RUNTIME_IMAGES=true` or `false` overrides this detection. Sandbox
creation also retries the specific Pod-update failure caused by duplicated pull
secrets, with `SANDBOX_CREATE_ATTEMPTS=3` by default.

The generated Makefile has no `setup-gateway` or `teardown-gateway` targets.
Manage the shared cluster gateway separately from starter-kit deployment.

Set database credentials in `.env` or the Kubernetes secret referenced by
`db_secret_name` in `values.yaml`. Non-empty environment settings take precedence
over database secret values. The secret referenced by `maas_secret_name` supplies
`MAAS_BASE_URL` and `MAAS_API_KEY`.

| Store | Connection settings | Optional private CA |
|---|---|---|
| Milvus | `MILVUS_URI`, optional `MILVUS_TOKEN` | `MILVUS_CA_CERT` |
| PostgreSQL/pgvector | `PGVECTOR_HOST`, optional `PGVECTOR_PORT`, `PGVECTOR_DB`, `PGVECTOR_USER`, `PGVECTOR_PASSWORD` | `PGVECTOR_CA_CERT` |
| Neo4j | Required `NEO4J_URI` and `NEO4J_PASSWORD`; `NEO4J_USERNAME` and `NEO4J_DATABASE` default to `neo4j` | `NEO4J_CA_CERT` |

CA variables accept inline PEM certificate text; pgvector deployment also accepts
base64-encoded PEM. The Makefile loads these values from the database secret when
needed, transports CA values in base64 across shell boundaries, and restores PEM
text for the agent. Use `MILVUS_CA_CERT` in place of the obsolete `MILVUS_SERVER_CERT` key.
For Neo4j, use a verified TLS URI such as `neo4j+s://host:7687`; egress setup reads
the host and port from that URI, defaulting to port `7687` when it is omitted.

The generated OpenShell image includes Python 3.12 and `proxychains4`. The Makefile
sets `OPENSHELL_GRPC_PROXY_PORT=18080` when starting the agent, enabling a local
IPv4 HTTP CONNECT proxy for Milvus gRPC and native pgvector/Neo4j TCP traffic
under OpenShell's egress policies. For Kubernetes service-account authentication,
deployment uses the API endpoint reported by `oc whoami --show-server` and selects
CA material from the cluster trust bundles.

If the entrypoint is used without `OPENSHELL_GRPC_PROXY_PORT`, setting
`PGVECTOR_HOST` or `NEO4J_URI` selects the direct OpenShell proxy mode. That mode
reads `grpc_proxy`, `GRPC_PROXY`, `https_proxy`, `HTTPS_PROXY`, `http_proxy`, or
`HTTP_PROXY` (first non-empty value wins); the proxy must use HTTP, have no
authentication, and resolve to IPv4. With neither a local-proxy setting nor a
native database host, the entrypoint launches the agent directly.

`make build-openshell` selects the Red Hat sandbox base image from the latest
succeeded RHOAI CSV and assembles its Python runtime using UBI Python 3.12.
Set `OPENSHELL_BASE_IMAGE` to override the build's base image; editing the
Containerfile's default `BASE_IMAGE` alone does not override the Makefile.
The CLI can fall back to a matching image under `quay.io/rhoai`, and failed
`registry.redhat.io` sandbox-image pulls are retried through `quay.io`.

| Setting | Default | Purpose |
|---|---|---|
| `RHOAI_CSV_NAMESPACE` | `redhat-ods-operator` | Namespace queried for the installed RHOAI CSV |
| `OPENSHELL_CLI_IMAGE` | RHOAI CSV CLI image | Override the containerized CLI image |
| `OPENSHELL_BASE_IMAGE` | RHOAI CSV sandbox image | Override the sandbox build's base image |
| `OPENSHELL_FALLBACK_REGISTRY` | `quay.io/rhoai` | Registry prefix used for CLI image fallback |
| `OPENSHELL_NAMESPACE` | `openshell` | Namespace of the gateway, certificates, and agent route |
| `OPENSHELL_GATEWAY_NAME` | `openshift` | Gateway registration selected by the CLI |
| `OPENSHELL_CLI_STATE` | `~/.config/openshell` | Host directory holding gateway registration and mTLS credentials |
| `OPENSHELL_PODMAN_USERNS` | `--userns=keep-id` | User-namespace flags for the CLI container |

Export environment settings before running `make`. Make variables such as
`OPENSHELL_GATEWAY_NAME` and `OPENSHELL_CLI_STATE` can also be passed on its command
line. To use a compatible locally installed CLI, pass `OPENSHELL_CLI=openshell`.

Run these commands from the extracted starter-kit directory:

```bash
uv lock --upgrade-package ai4rag
make build-openshell
make refresh-certs
make deploy-openshell
```

Run `make refresh-certs` if the gateway registration or the mTLS client certificates
are missing or stale.

Sandbox creation checks gateway connectivity before replacing an existing sandbox.
Run `make undeploy-openshell` to remove the sandbox and agent route; shared gateway
resources and built images remain available.

Regenerate older starter kits to obtain the updated Makefile and deployment guidance.
