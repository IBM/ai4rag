# Starter Kit

Packages a published RAG pattern into a deployable agentic-RAG microservice project.

::: ai4rag.assets_generator.starter_kit
    options:
      members:
        - generate_starter_kit


The generated `agent_config.json` includes the pattern's `template_id`. The runtime
resolves it through the template registry, selecting `ReActRAG` or `SequentialRAG`.
Neo4j configurations retain effective graph retrieval settings and pass them to the
retriever. Graph retrieval also works when these options are omitted: the store
uses its built-in defaults. Persisting the options reproduces any custom graph
settings used during optimization. The application/package name remains
`agentic_rag` for all templates.

## OpenShell Deployment

Generated starter kits can deploy an agent backed by Milvus, PostgreSQL/pgvector,
or Neo4j. The generated `agent_config.json` selects the provider and existing
collection from the published pattern. Neo4j patterns use graph retrieval and
require a collection that has already been indexed; document indexing is a separate
pipeline step.

Deployment requires an OpenShift cluster with the Agent Sandbox operator, a logged-in
`oc` client, the `openshell` CLI, Helm 3, and `uv`. MaaS must provide reachable chat
and embedding models, and ReAct templates require a chat model that supports tool calling.

Set database credentials in `.env` or the Kubernetes secret referenced by
`db_secret_name` in `values.yaml`. Non-empty environment settings take precedence
over database secret values. The secret referenced by `maas_secret_name` supplies
`MAAS_BASE_URL` and `MAAS_API_KEY`.

| Store | Connection settings | Optional private CA |
|---|---|---|
| Milvus | `MILVUS_URI`, optional `MILVUS_TOKEN` | `MILVUS_CA_CERT` |
| PostgreSQL/pgvector | `PGVECTOR_HOST`, optional `PGVECTOR_PORT`, `PGVECTOR_DB`, `PGVECTOR_USER`, `PGVECTOR_PASSWORD` | `PGVECTOR_CA_CERT` |
| Neo4j | Required `NEO4J_URI` and `NEO4J_PASSWORD`; `NEO4J_USERNAME` and `NEO4J_DATABASE` default to `neo4j` | `NEO4J_CA_CERT` |

CA variables contain inline PEM certificate text. The Makefile loads these values
from the database secret when needed and forwards them to both the sandbox and
agent. Use `MILVUS_CA_CERT` in place of the obsolete `MILVUS_SERVER_CERT` key.
For Neo4j, use a verified TLS URI such as `neo4j+s://host:7687`; egress setup reads
the host and port from that URI, defaulting to port `7687` when it is omitted.

Run these commands from the extracted starter-kit directory:

```bash
uv lock --upgrade-package ai4rag
make setup-gateway  # Once per cluster
make build-openshell
make deploy-openshell
```

Regenerate older starter kits to obtain the updated Makefile and deployment guidance.
