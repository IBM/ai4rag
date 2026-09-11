SHELL          := bash
OPENSHELL_SANDBOX ?= $(shell . ./.env 2>/dev/null && echo $${OPENSHELL_SANDBOX_NAME:-rag-sandbox})
OPENSHELL_IMAGE   := openshell-agentic-rag

.PHONY: init env run-app run-cli refresh-certs setup-gateway teardown-gateway build-openshell create-sandbox wait-sandbox setup-egress start-agent expose-agent deploy-openshell undeploy-openshell playground-sandbox test help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

init: ## Copy .env.example to .env for configuration
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example — edit it with your configuration"; else echo ".env already exists — skipping"; fi

env: ## Create venv and install dependencies
	@echo "==> Creating virtual environment and installing dependencies..." && \
	  rm -rf .venv && \
	  uv sync --python 3.12 && \
	  echo "" && \
	  echo "Done. Activate with: source .venv/bin/activate" && \
	  echo "Then edit .env with your MaaS credentials and run: make run-app"

run-app: ## Run agent locally with hot-reload
	@source .venv/bin/activate && set -a && source .env && set +a && \
	  if lsof -ti:$${PORT:-8000} >/dev/null 2>&1; then \
	    echo "ERROR: Port $${PORT:-8000} is already in use." && \
	    exit 1; \
	  fi && \
	  uv run $${MLFLOW_TRACKING_URI:+--extra tracing} uvicorn main:app --host 127.0.0.1 --port $${PORT:-8000} --reload --reload-exclude .venv

run-cli: ## Run interactive CLI chat (no web server)
	@source .venv/bin/activate && set -a && source .env && set +a && \
	  echo "CLI mode not yet configured. Use: make run-app"

test: ## Run unit tests
	uv run --extra dev python -m pytest tests/ $(PYTEST_ARGS)

## ── openShell sandbox targets ─────────────────────────────────────────────────

OPENSHELL_VERSION ?= 0.0.99

refresh-certs: ## Refresh openShell client certificates from cluster
	@echo "==> Refreshing openShell client certificates..." && \
	  MTLS_DIR=~/.config/openshell/gateways/openshift/mtls && mkdir -p $$MTLS_DIR && \
	  oc -n openshell get secret openshell-client-tls -o jsonpath='{.data.ca\.crt}'  | base64 -d > $$MTLS_DIR/ca.crt && \
	  oc -n openshell get secret openshell-client-tls -o jsonpath='{.data.tls\.crt}' | base64 -d > $$MTLS_DIR/tls.crt && \
	  oc -n openshell get secret openshell-client-tls -o jsonpath='{.data.tls\.key}' | base64 -d > $$MTLS_DIR/tls.key && \
	  echo "    Certificates refreshed."

setup-gateway: ## Install openShell gateway, create Route, register CLI (one-time)
	@set -e && \
	  APPS_DOMAIN=$$(oc get ingresses.config cluster -o jsonpath='{.spec.domain}') && \
	  echo "==> Detected apps domain: $$APPS_DOMAIN" && \
	  echo "==> Ensuring openshell namespace and SCCs..." && \
	  oc create ns openshell 2>/dev/null || true && \
	  oc adm policy add-scc-to-user privileged -z openshell-sandbox -n openshell 2>/dev/null || true && \
	  oc adm policy add-scc-to-user privileged -z openshell -n openshell 2>/dev/null || true && \
	  echo "==> Granting TokenReview RBAC to sandbox SA..." && \
	  oc create clusterrolebinding openshell-tokenreview \
	    --clusterrole=system:auth-delegator \
	    --serviceaccount=openshell:openshell-sandbox 2>/dev/null || true && \
	  echo "==> Installing openShell gateway $(OPENSHELL_VERSION)..." && \
	  helm upgrade --install openshell oci://ghcr.io/nvidia/openshell/helm-chart \
	    --version $(OPENSHELL_VERSION) \
	    --namespace openshell \
	    --set securityContext.runAsUser=null \
	    --set securityContext.runAsNonRoot=false \
	    --set securityContext.allowPrivilegeEscalation=true \
	    --set securityContext.capabilities=null \
	    --set server.auth.allowUnauthenticatedUsers=true \
	    --set "pkiInitJob.serverDnsNames={*.openshell.$$APPS_DOMAIN,openshell.$$APPS_DOMAIN,openshell-gateway-openshell.$$APPS_DOMAIN}" && \
	  echo "==> Waiting for gateway to be ready..." && \
	  oc -n openshell rollout status statefulset/openshell --timeout=120s && \
	  echo "==> Disabling mTLS client cert requirement..." && \
	  TOML=$$(oc -n openshell get configmap openshell-config -o jsonpath='{.data.gateway\.toml}' | sed '/^client_ca_path/d') && \
	  oc -n openshell patch configmap openshell-config --type merge \
	    -p "$$(printf '{"data":{"gateway.toml":%s}}' "$$(printf '%s' "$$TOML" | jq -Rs .)")" && \
	  oc -n openshell delete pod openshell-0 && \
	  oc -n openshell rollout status statefulset/openshell --timeout=120s && \
	  echo "==> Creating gateway Route..." && \
	  oc -n openshell create route passthrough openshell-gateway \
	    --service=openshell --port=8080 2>/dev/null || true && \
	  GATEWAY_URL="https://openshell-gateway-openshell.$$APPS_DOMAIN" && \
	  echo "==> Registering gateway in CLI..." && \
	  openshell gateway remove openshift 2>/dev/null || true && \
	  MTLS_DIR=~/.config/openshell/gateways/openshift/mtls && mkdir -p $$MTLS_DIR && \
	  oc -n openshell get secret openshell-client-tls -o jsonpath='{.data.ca\.crt}'  | base64 -d > $$MTLS_DIR/ca.crt && \
	  oc -n openshell get secret openshell-client-tls -o jsonpath='{.data.tls\.crt}' | base64 -d > $$MTLS_DIR/tls.crt && \
	  oc -n openshell get secret openshell-client-tls -o jsonpath='{.data.tls\.key}' | base64 -d > $$MTLS_DIR/tls.key && \
	  openshell gateway add "$$GATEWAY_URL" --local --name openshift && \
	  echo "==> Verifying connection..." && \
	  export OPENSHELL_GATEWAY_INSECURE=true && \
	  openshell status && \
	  echo "" && \
	  echo "Gateway ready. Next: make build-openshell"

teardown-gateway: ## Uninstall openShell gateway and clean up
	@openshell gateway remove openshift 2>/dev/null || true && \
	  helm uninstall openshell -n openshell 2>/dev/null || true && \
	  oc -n openshell delete pvc --all 2>/dev/null || true && \
	  oc -n openshell delete secret openshell-server-tls openshell-client-tls openshell-jwt-keys 2>/dev/null || true && \
	  oc -n openshell delete route --all 2>/dev/null || true && \
	  oc delete clusterrolebinding openshell-tokenreview 2>/dev/null || true && \
	  oc adm policy remove-scc-from-user privileged -z openshell-sandbox -n openshell 2>/dev/null || true && \
	  oc adm policy remove-scc-from-user privileged -z openshell -n openshell 2>/dev/null || true && \
	  echo "Gateway removed."

_openshell-env = set -a && . ./.env && set +a && export OPENSHELL_GATEWAY_INSECURE=true && \
	SANDBOX=$${OPENSHELL_SANDBOX_NAME:-rag-sandbox} && \
	NS=$$(oc project -q) && \
	IMAGE="image-registry.openshift-image-registry.svc:5000/$$NS/$(OPENSHELL_IMAGE):latest" && \
	APPS_DOMAIN=$$(oc get ingresses.config cluster -o jsonpath='{.spec.domain}') && \
	MAAS_SECRET_NAME=$$(sed -n 's/^[[:space:]]*maas_secret_name:[[:space:]]*"\([^"\]*\)".*/\1/p' values.yaml) && \
	VECTOR_DB_SECRET_NAME=$${VECTOR_DB_SECRET_NAME:-$$(sed -n 's/^[[:space:]]*vector_db_secret_name:[[:space:]]*"\([^"\]*\)".*/\1/p' values.yaml)} && \
	secret_value() { oc get secret "$$1" -n "$$NS" -o jsonpath="{.data.$$2}" 2>/dev/null | base64 -d; } && \
	if [ -n "$$MAAS_SECRET_NAME" ] && [ "$$MAAS_SECRET_NAME" != "__MAAS_SECRET_NAME__" ]; then \
	  value=$$(secret_value "$$MAAS_SECRET_NAME" MAAS_BASE_URL); [ -z "$$value" ] || MAAS_BASE_URL="$$value"; \
	  value=$$(secret_value "$$MAAS_SECRET_NAME" MAAS_API_KEY); [ -z "$$value" ] || MAAS_API_KEY="$$value"; \
	fi && \
	if [ -n "$$VECTOR_DB_SECRET_NAME" ] && [ "$$VECTOR_DB_SECRET_NAME" != "__VECTOR_DB_SECRET_NAME__" ]; then \
	  for key in MILVUS_URI MILVUS_TOKEN MILVUS_SERVER_CERT MILVUS_SERVER_NAME PGVECTOR_HOST PGVECTOR_PORT PGVECTOR_DB PGVECTOR_DATABASE PGVECTOR_USER PGVECTOR_PASSWORD; do \
	    value=$$(secret_value "$$VECTOR_DB_SECRET_NAME" "$$key"); current=$$(printenv "$$key"); [ -n "$$current" ] || [ -z "$$value" ] || export "$$key=$$value"; \
	  done; \
	fi && export MAAS_BASE_URL MAAS_API_KEY MILVUS_URI MILVUS_TOKEN MILVUS_SERVER_CERT MILVUS_SERVER_NAME PGVECTOR_HOST PGVECTOR_PORT PGVECTOR_DB PGVECTOR_DATABASE PGVECTOR_USER PGVECTOR_PASSWORD

build-openshell: ## Build openShell sandbox image via OpenShift BuildConfig
	@oc new-build --strategy=docker --binary \
	    --name=$(OPENSHELL_IMAGE) --to=$(OPENSHELL_IMAGE):latest 2>/dev/null || true && \
	  oc patch bc $(OPENSHELL_IMAGE) \
	    -p '{"spec":{"strategy":{"dockerStrategy":{"dockerfilePath":"Containerfile.openshell"}}}}' && \
	  oc start-build $(OPENSHELL_IMAGE) --from-dir=. --follow && \
	  NS=$$(oc project -q) && \
	  echo "" && \
	  echo "Image ready: image-registry.openshift-image-registry.svc:5000/$$NS/$(OPENSHELL_IMAGE):latest" && \
	  echo "Next: make deploy-openshell"

create-sandbox: ## Delete existing sandbox and create a new one (with 120s timeout)
	@$(_openshell-env) && \
	  echo "==> Granting image-pull to openshell namespace..." && \
	  oc policy add-role-to-user system:image-puller "system:serviceaccount:openshell:openshell-sandbox" -n "$$NS" 2>/dev/null || true
	@$(_openshell-env) && \
	  oc policy add-role-to-user system:image-puller "system:serviceaccount:openshell:default" -n "$$NS" 2>/dev/null || true
	@$(_openshell-env) && \
	  echo "==> Deleting old sandbox $$SANDBOX..." && \
	  openshell sandbox delete "$$SANDBOX" 2>/dev/null || true
	@$(_openshell-env) && \
	  echo "==> Creating sandbox $$SANDBOX from $$IMAGE..." && \
	  echo "    (timeout 120s — this may take a minute while the image is pulled)..." && \
	  openshell sandbox create --name "$$SANDBOX" --from "$$IMAGE" \
	    --env "MAAS_BASE_URL=$$MAAS_BASE_URL" \
	    --env "CHAT_BASE_URL=$${CHAT_BASE_URL:-$$BASE_URL}" \
	    --env "MAAS_API_KEY=$$MAAS_API_KEY" \
	    --env "MODEL_ID=$$MODEL_ID" \
	    --env "EMBEDDING_MODEL_ID=$$EMBEDDING_MODEL_ID" \
	    --env "EMBEDDING_DIMENSION=$${EMBEDDING_DIMENSION:-768}" \
	    --env "PROVIDER_TYPE=$${PROVIDER_TYPE:-milvus}" \
	    --env "RETRIEVAL_METHOD=$${RETRIEVAL_METHOD:-simple}" \
	    --env "NUMBER_OF_CHUNKS=$${NUMBER_OF_CHUNKS:-5}" \
	    --env "SEARCH_MODE=$${SEARCH_MODE:-vector}" \
	    --env "RANKER_STRATEGY=$$RANKER_STRATEGY" \
	    --env "RANKER_ALPHA=$$RANKER_ALPHA" \
	    --env "MILVUS_URI=$$MILVUS_URI" \
	    --env "MILVUS_TOKEN=$$MILVUS_TOKEN" \
	    --env "PORT=8080" </dev/null & \
	  CPID=$$!; ( sleep 80 && kill $$CPID 2>/dev/null ) & TPID=$$!; \
	  wait $$CPID 2>/dev/null; RC=$$?; kill $$TPID 2>/dev/null; wait $$TPID 2>/dev/null; \
	  if [ $$RC -ne 0 ]; then echo "    sandbox create exited with $$RC (may still be starting — wait-sandbox will poll)"; fi; true

wait-sandbox: ## Poll until sandbox phase is Ready (max 150s)
	@$(_openshell-env) && \
	  echo "==> Waiting for sandbox $$SANDBOX to be ready..." && \
	  i=0; while [ $$i -lt 30 ]; do \
	    i=$$((i+1)); \
	    phase=$$(openshell sandbox list 2>/dev/null | grep "$$SANDBOX" | awk '{print $$NF}' | sed 's/\x1b\[[0-9;]*m//g'); \
	    if [ "$$phase" = "Ready" ]; then echo "    Sandbox ready."; exit 0; fi; \
	    echo "    status: $$phase (attempt $$i/30, waiting 5s...)"; \
	    sleep 5; \
	  done; \
	  echo "ERROR: sandbox not ready after 150s"; exit 1

setup-egress: ## Add MaaS + Milvus egress policies for agent Python binary
	@$(_openshell-env) && \
	  echo "==> Using agent Python binary: /usr/bin/python3.12" && \
	  PYTHON_BIN="/usr/bin/python3.12" && \
	  echo "==> Adding egress policy for MaaS..." && \
	  MAAS_HOST=$$(echo "$${CHAT_BASE_URL:-$${BASE_URL:-$$MAAS_BASE_URL}}" | sed -E 's|^https?://||;s|/.*||') && \
	  openshell policy update "$$SANDBOX" --add-endpoint "$$MAAS_HOST:443" --binary "$$PYTHON_BIN" --wait && \
	  echo "==> Adding egress policies for K8s API + Milvus..." && \
	  MILVUS_HOSTPORT=$$(echo "$$MILVUS_URI" | sed -E 's|^https?://||;s|/.*||') && \
	  MILVUS_HOST_ONLY=$$(echo "$$MILVUS_HOSTPORT" | cut -d: -f1) && \
	  MILVUS_PORT_ONLY=$$(echo "$$MILVUS_HOSTPORT" | cut -d: -f2) && \
	  POLICY_FILE=$$(mktemp /tmp/openshell-policy.XXXXXX) && \
	  export MILVUS_H="$$MILVUS_HOST_ONLY" MILVUS_P="$$MILVUS_PORT_ONLY" PY_BIN="$$PYTHON_BIN" && \
	  openshell policy get "$$SANDBOX" --full -o json 2>/dev/null | sed -n '/^{/,$$p' | \
	    python3 -c 'import json,sys,os;d=json.load(sys.stdin);p=d["policy"];p["network_policies"]["k8s_api"]={"name":"k8s-api","endpoints":[{"host":"kubernetes.default.svc","port":443,"tls":"skip"}],"binaries":[{"path":os.environ["PY_BIN"]}]};p["network_policies"]["milvus_grpc"]={"name":"milvus-grpc","endpoints":[{"host":os.environ["MILVUS_H"],"port":int(os.environ["MILVUS_P"]),"tls":"skip"}],"binaries":[{"path":os.environ["PY_BIN"]}]};json.dump(p,sys.stdout,indent=2)' > "$$POLICY_FILE" && \
	  openshell policy set "$$SANDBOX" --policy "$$POLICY_FILE" --wait && \
	  rm -f "$$POLICY_FILE" && \
	  echo "Done!"

start-agent: refresh-certs ## Start agent with K8s ServiceAccount auth inside sandbox
	@$(_openshell-env) && \
	  echo "==> Creating agent-client ServiceAccount and token..." && \
	  oc create sa agent-client 2>/dev/null || true && \
	  CLIENT_TOKEN=$$(oc create token agent-client --duration=168h) && \
	  oc delete secret agent-client-token 2>/dev/null || true && \
	  oc create secret generic agent-client-token --from-literal=token="$$CLIENT_TOKEN" && \
	  echo "==> Generating sandbox reviewer token..." && \
	  K8S_API_URL="https://kubernetes.default.svc" && \
	  K8S_REVIEWER_TOKEN=$$(oc create token openshell-sandbox -n openshell --duration=168h) && \
	  ALLOWED_SA_USERNAME="system:serviceaccount:$${NS}:agent-client" && \
	  echo "==> Stopping any running agent..." && \
	  openshell sandbox exec -n "$$SANDBOX" -- bash -c "pkill -f '[/]usr/bin/python3.12 -m uvicorn auth_wrapper:app'" 2>/dev/null || true && \
	  sleep 5 && \
	  echo "==> Starting agent (auth: enabled)..." && \
	  openshell sandbox exec -n "$$SANDBOX" -- bash -c '\
	    export MAAS_API_KEY="'"$$MAAS_API_KEY"'" && \
	    export MAAS_BASE_URL="'"$$MAAS_BASE_URL"'" && \
	    export CHAT_BASE_URL="'"$${CHAT_BASE_URL:-$$BASE_URL}"'" && \
	    export MODEL_ID="'"$$MODEL_ID"'" && \
	    export EMBEDDING_MODEL_ID="'"$$EMBEDDING_MODEL_ID"'" && \
	    export EMBEDDING_DIMENSION="'"$$EMBEDDING_DIMENSION"'" && \
	    export PROVIDER_TYPE="'"$${PROVIDER_TYPE:-milvus}"'" && \
	    export RETRIEVAL_METHOD="'"$${RETRIEVAL_METHOD:-simple}"'" && \
	    export NUMBER_OF_CHUNKS="'"$${NUMBER_OF_CHUNKS:-5}"'" && \
	    export SEARCH_MODE="'"$${SEARCH_MODE:-vector}"'" && \
	    export RANKER_STRATEGY="'"$$RANKER_STRATEGY"'" && \
	    export RANKER_ALPHA="'"$$RANKER_ALPHA"'" && \
	    export MILVUS_URI="'"$$MILVUS_URI"'" && \
	    export MILVUS_TOKEN="'"$$MILVUS_TOKEN"'" && \
	    export MILVUS_COLLECTION_NAME="'"$$MILVUS_COLLECTION_NAME"'" && \
	    export MILVUS_SERVER_CERT="'"$$MILVUS_SERVER_CERT"'" && \
	    export MILVUS_SERVER_NAME="'"$$MILVUS_SERVER_NAME"'" && \
	    export K8S_API_URL="'"$$K8S_API_URL"'" && \
	    export K8S_REVIEWER_TOKEN="'"$$K8S_REVIEWER_TOKEN"'" && \
	    export K8S_API_INSECURE="true" && \
	    export PLAYGROUND_TOKEN="'"$$CLIENT_TOKEN"'" && \
	    export ALLOWED_SA_USERNAME="'"$$ALLOWED_SA_USERNAME"'" && \
	    export PYTHONPATH=/sandbox/.local/lib/python3.12/site-packages:/sandbox/.local/lib64/python3.12/site-packages:/sandbox && \
	    cd /sandbox && \
	    nohup /usr/bin/python3.12 -m uvicorn auth_wrapper:app --host 0.0.0.0 --port 8080 > /sandbox/agent.log 2>&1 &\
	    echo $$! > /sandbox/agent.pid && \
	    sleep 3 && \
	    if kill -0 $$! 2>/dev/null; then echo "Agent process alive"; else echo "ERROR: Agent died on startup"; cat /sandbox/agent.log; exit 1; fi' && \
	  echo "    Agent started in background."

expose-agent: ## Expose agent service and create Route
	@$(_openshell-env) && \
	  echo "==> Exposing service URL..." && \
	  openshell service expose "$$SANDBOX" 8080 agent 2>/dev/null || true && \
	  sleep 2 && \
	  SVC_HOST="default--$$SANDBOX--agent.openshell.$$APPS_DOMAIN" && \
	  printf 'apiVersion: route.openshift.io/v1\nkind: Route\nmetadata:\n  name: openshell-rag-agent\n  namespace: openshell\nspec:\n  host: "%s"\n  to:\n    kind: Service\n    name: openshell\n  port:\n    targetPort: grpc\n  tls:\n    termination: passthrough\n' "$$SVC_HOST" | oc apply -f - && \
	  echo "==> Waiting for agent to initialize (10s)..." && \
	  sleep 10 && \
	  echo "==> Checking agent health..." && \
	  curl -sk -m 3 "https://$$SVC_HOST/health" 2>/dev/null || echo "Agent not ready yet (check logs)" && \
	  echo "" && \
	  echo "Agent URL: https://$$SVC_HOST" && \
	  echo "" && \
	  echo "Get token:" && \
	  echo "  TOKEN=\$$(oc get secret agent-client-token -o jsonpath='{.data.token}' | base64 -d)" && \
	  echo "" && \
	  echo "Query:" && \
	  echo "  curl -sk https://$$SVC_HOST/chat/completions \\" && \
	  echo "    -H \"X-Api-Key: \$$TOKEN\" \\" && \
	  echo "    -H 'Content-Type: application/json' \\" && \
	  echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"

deploy-openshell: create-sandbox wait-sandbox setup-egress start-agent expose-agent ## Full sandbox deploy: create, start, expose

undeploy-openshell: ## Remove openShell sandbox and route
	@export OPENSHELL_GATEWAY_INSECURE=true && \
	  (openshell sandbox delete $(OPENSHELL_SANDBOX) 2>/dev/null || true) && \
	  oc -n openshell delete route openshell-rag-agent 2>/dev/null || true && \
	  echo "Sandbox $(OPENSHELL_SANDBOX) and route removed." && \
	  sleep 10 2>/dev/null || true

playground-sandbox: ## Start sandbox playground web UI (auto-fetches agent URL and SA token)
	@. .venv/bin/activate && \
	  echo "Starting sandbox playground on http://localhost:5002" && \
	  echo "Auto-fetching agent URL from OpenShift Route..." && \
	  echo "Auto-fetching SA token from Secret agent-client-token..." && \
	  echo "" && \
	  python playground-sandbox/app.py
