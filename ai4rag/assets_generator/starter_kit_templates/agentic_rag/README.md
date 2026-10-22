# Agentic RAG Starter Kit: `__PATTERN_NAME__`

This project was generated from an optimized RAG pattern. Before deploying the
application, make sure MaaS and the selected vector database are reachable.

The indexed production collection is an input to this starter kit. Do not run
`make load-docs` against it. Re-indexing belongs to the separate document
indexing pipeline and could overwrite or corrupt the shared collection.

The application exposes `POST /v1/responses` and `GET /health`. The agent can
call `rewrite_query` to refine a search, call `retriever` to fetch documents,
and answer using the evidence found.

## OpenShell deployment

### Prerequisites

- An OpenShift cluster with `oc` logged in and permission to create the required
  resources.
- The `openshell` CLI and Helm 3 installed locally.
- `uv` installed locally to update the dependency lockfile.
- An OpenShift AI MaaS deployment exposing chat and embedding models, together
  with credentials. The chat model must support tool calling so the agent can
  invoke the retriever.
- A reachable Milvus, PostgreSQL/pgvector, or Neo4j instance and credentials,
  provided through the Kubernetes secrets referenced by `values.yaml`
  (`maas_secret_name` and `db_secret_name`).
  - For Milvus, provide `MILVUS_URI` and optionally `MILVUS_TOKEN` and
    `MILVUS_CA_CERT` for TLS.
  - For pgvector, provide `PGVECTOR_HOST` and optionally `PGVECTOR_PORT`,
    `PGVECTOR_DB`, `PGVECTOR_USER`, `PGVECTOR_PASSWORD`, and `PGVECTOR_CA_CERT`
    (PEM text for a private or self-signed CA).
  - For Neo4j, provide `NEO4J_URI` (for example `neo4j+s://host:7687`) and
    `NEO4J_PASSWORD`. `NEO4J_USERNAME` and `NEO4J_DATABASE` default to `neo4j`.
    For a private or self-signed CA, provide `NEO4J_CA_CERT` as PEM text.
- Neo4j patterns use graph search and require a collection already indexed in
  Neo4j. The starter kit serves retrieval from that collection; it does not
  index documents.
- The Red Hat build of the Agent Sandbox operator installed in the cluster.

> **Hint:** For production deployments, we recommend using the Red Hat OpenShell
> sandbox image `registry.redhat.io/rhoai/odh-openshell-sandbox-rhel9` as the
> `BASE_IMAGE` in `Containerfile.openshell`, with a tag or digest appropriate for
> your environment.

### Commands

Run the following commands from the starter-kit root directory:

```bash
# Generate the lockfile to the latest ai4rag release before building.
uv lock --upgrade-package ai4rag

# Run once per cluster.
make setup-gateway

make build-openshell
make deploy-openshell
```

`make deploy-openshell` creates and configures the sandbox, starts the agent,
and exposes it through an OpenShift route.

The generated `agent_config.json` stays in the starter kit and is injected into
the sandbox during deployment. You can edit values such as `temperature` or
`system_message` and rerun `make deploy-openshell` without rebuilding the image.
Rebuild the image only after changing application code or dependencies.

## API and Swagger

After deployment, the Swagger/OpenAPI documentation is available at
`https://<agent-route>/docs`.

The Responses API can be called at `https://<agent-route>/v1/responses`:

```bash
curl -sk https://<agent-route>/v1/responses \
  -H "X-Api-Key: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"input":"What is vLLM?"}'
```

Additional optimized patterns are available in the RAG patterns artifact:

```text
s3://<bucket>/<run-id>/rag-templates-optimization/<artifact-id>/rag_patterns/
```
