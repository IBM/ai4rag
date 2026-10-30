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
`oc` client, `podman`, and `uv`. The generated Makefile runs the OpenShell CLI from the
RHOAI image in a container, so no local OpenShell CLI installation is needed. It also
requires an OpenShell gateway that is already installed on the cluster and registered
in the CLI as gateway `openshift`.
Setting up the gateway is a one-time cluster prerequisite and is not part of the
generated starter kit. MaaS must provide reachable chat and embedding models, and
agentic templates require a chat model that supports tool calling.

Cluster administrators can use the repository's
[`scripts/setup-openshell-gateway.sh`](https://github.com/IBM/ai4rag/blob/main/scripts/setup-openshell-gateway.sh)
for one-time gateway provisioning. Run it from an ai4rag checkout with `oc`,
Helm, `curl`, `jq`, and `podman` installed and the required cluster permissions:

```bash
bash scripts/setup-openshell-gateway.sh
```

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

For pgvector and Neo4j, the generated OpenShell image includes `proxychains4` and an
entrypoint that tunnels native TCP connections through OpenShell's HTTP CONNECT proxy.
When either `PGVECTOR_HOST` or `NEO4J_URI` is set, startup requires a proxy supplied through `grpc_proxy`,
`GRPC_PROXY`, `https_proxy`, `HTTPS_PROXY`, `http_proxy`, or `HTTP_PROXY` (first
non-empty value wins). The proxy must use HTTP, have no authentication, and resolve
to an IPv4 address. If both database variables are unset, the entrypoint launches
the agent directly.

For production deployments, the generated README recommends
`registry.redhat.io/rhoai/odh-openshell-sandbox-rhel9` as the `BASE_IMAGE` in
`Containerfile.openshell`, with a tag or digest appropriate for your environment.

Run these commands from the extracted starter-kit directory:

```bash
uv lock --upgrade-package ai4rag
make build-openshell
make deploy-openshell
```

Regenerate older starter kits to obtain the updated Makefile and deployment guidance.
