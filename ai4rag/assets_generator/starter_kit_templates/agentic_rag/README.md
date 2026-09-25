# Agentic RAG starter kit

This project is generated from an optimized RAG pattern. Make sure you have
access to MaaS and the vector database before deploying the application.

The indexed production collection is an input to this starter kit. Do not run
`make load-docs` against the production collection. Re-indexing belongs to the
separate documents-indexing pipeline and can overwrite or corrupt the shared
collection.

The application exposes `POST /v1/responses` and `GET /health`.

## OpenShell deployment

### Prerequisites

- An OpenShift cluster with `oc` logged in and permission to create the required resources.
- The `openshell` CLI and Helm 3 installed locally.
- An OpenShift AI MaaS deployment exposing chat and embedding models, together with credentials for accessing it.
- A reachable Milvus instance and credentials for accessing it, provided through the Kubernetes secrets referenced by `values.yaml` (`maas_secret_name` and `vector_db_secret_name`). The vector database secret must contain the Milvus CA certificate as `MILVUS_SERVER_CERT` when TLS is enabled.
- The Red Hat build of the Agent Sandbox operator installed in the cluster.

### Commands

Run the following commands from the starter-kit root directory:

```bash
# Run once per cluster.
make setup-gateway

make build-openshell
make deploy-openshell
```

`make deploy-openshell` creates and configures the sandbox, starts the agent, and exposes it through an OpenShift route.

The generated `agent_config.json` stays in the starter-kit and is injected into the sandbox automatically during deployment. You can edit values such as `temperature` or `system_message` and rerun `make deploy-openshell` without rebuilding the image. Rebuild the image only after changing application code or dependencies.

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
