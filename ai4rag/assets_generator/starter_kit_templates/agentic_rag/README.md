# Agentic RAG starter kit

This project is generated from an optimized RAG pattern. Configure the empty
credentials in `.env` before running the application.

The indexed production collection is an input to this starter kit. Do not run
`make load-docs` against the production collection. Re-indexing belongs to the
separate documents-indexing pipeline and can overwrite or corrupt the shared
collection.

The application exposes `POST /chat/completions`, `GET /health`, and a local
playground at `GET /`.

## OpenShell deployment

### Prerequisites

- An OpenShift cluster with `oc` logged in and permission to create the required resources.
- The `openshell` CLI and Helm 3 installed locally.
- An OpenShift AI MaaS deployment exposing chat and embedding models.
- A reachable Milvus instance and Kubernetes secrets referenced by `values.yaml` (`maas_secret_name` and `vector_db_secret_name`). The vector database secret must contain the Milvus CA certificate as `MILVUS_SERVER_CERT` when TLS is enabled.
- The Red Hat build of the Agent Sandbox operator installed in the cluster.

### Commands

Run the following commands from the starter-kit root directory:

```bash
make init

# Run once per cluster.
make setup-gateway

make build-openshell
make deploy-openshell
```

`make deploy-openshell` creates and configures the sandbox, starts the agent, and exposes it through an OpenShift route.
