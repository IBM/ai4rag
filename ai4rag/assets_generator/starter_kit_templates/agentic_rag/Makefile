
SHELL          := bash
OPENSHELL_SANDBOX   ?= $(shell if [ -f .env ]; then . ./.env; fi; echo $${OPENSHELL_SANDBOX_NAME:-rag-sandbox})
OPENSHELL_NAMESPACE ?= $(shell if [ -f .env ]; then . ./.env; fi; echo $${OPENSHELL_NAMESPACE:-openshell})
OPENSHELL_IMAGE     := openshell-agentic-rag

.PHONY: env run-app run-cli refresh-certs build-openshell create-sandbox wait-sandbox setup-egress start-agent expose-agent deploy-openshell undeploy-openshell test help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

env: ## Create venv and install dependencies
	@echo "==> Creating virtual environment and installing dependencies..." && \
	  rm -rf .venv && \
	  uv sync --python 3.12 && \
	  echo "" && \
	  echo "Done. Activate with: source .venv/bin/activate" && \
	  echo "Export MAAS_BASE_URL and MAAS_API_KEY, then run: make run-app"

run-app: ## Run agent locally with hot-reload
	@source .venv/bin/activate && set -a && if [ -f .env ]; then . ./.env; fi && set +a && \
	  if lsof -ti:$${PORT:-8000} >/dev/null 2>&1; then \
	    echo "ERROR: Port $${PORT:-8000} is already in use." && \
	    exit 1; \
	  fi && \
	  uv run uvicorn main:app --host 127.0.0.1 --port $${PORT:-8000} --reload --reload-exclude .venv

run-cli: ## Run interactive CLI chat (no web server)
	@source .venv/bin/activate && set -a && source .env && set +a && \
	  echo "CLI mode not yet configured. Use: make run-app"

test: ## Run unit tests
	uv run --extra dev python -m pytest tests/ $(PYTEST_ARGS)

## ── openShell sandbox targets ─────────────────────────────────────────────────

# The default CLI comes from the RHOAI CSV, with a same-digest Quay fallback.
# Set OPENSHELL_CLI=openshell to use a compatible local CLI.
# Container runs mount the kit at /work for policy files and the CLI state
# directory at /cli, which holds the gateway registration and the mTLS client
# certificates written by `make refresh-certs`.
OPENSHELL_CLI          ?= openshell_cli
OPENSHELL_CLI_STATE    ?= $(HOME)/.config/openshell
OPENSHELL_GATEWAY_NAME ?= openshift
OPENSHELL_PODMAN_USERNS ?= --userns=keep-id
OPENSHELL ?= $(OPENSHELL_CLI) -g $(OPENSHELL_GATEWAY_NAME)

# Retries create-sandbox when a Kyverno policy duplicates pull secrets on the
# driver's post-create Pod update ("pod updates may not change fields").
SANDBOX_CREATE_ATTEMPTS ?= 3

_openshell-cli := openshell_cli() { \
	local csv_json image fallback_registry fallback_path fallback_namespace fallback_image; \
	image=$${OPENSHELL_CLI_IMAGE:-}; \
	if [ -z "$$image" ]; then \
	  csv_json=$$(oc -n "$${RHOAI_CSV_NAMESPACE:-redhat-ods-operator}" get csv -o json 2>/dev/null || printf '{"items":[]}'); \
	  image=$$(python3 -c "import json,sys; d=json.load(sys.stdin); c=[x for x in d.get(\"items\",[]) if x.get(\"status\",{}).get(\"phase\")==\"Succeeded\" and x.get(\"metadata\",{}).get(\"name\",\"\").startswith(\"rhods-operator.\")]; c.sort(key=lambda x:x.get(\"metadata\",{}).get(\"creationTimestamp\",\"\")); r=c[-1].get(\"spec\",{}).get(\"relatedImages\",[]) if c else []; print(next((x.get(\"image\",\"\") for x in r if x.get(\"name\")==\"odh_openshell_cli_image\"),\"\"))" <<< "$$csv_json"); \
	fi; \
	[ -n "$$image" ] || { echo "ERROR: OpenShell CLI image is missing from the RHOAI CSV; set OPENSHELL_CLI_IMAGE." >&2; return 1; }; \
	fallback_registry=$${OPENSHELL_FALLBACK_REGISTRY:-quay.io/rhoai}; \
	fallback_namespace=$$(basename "$$fallback_registry"); \
	fallback_path=$$(printf '%s\n' "$$image" | cut -d/ -f2-); \
	case "$$fallback_path" in "$$fallback_namespace"/*) fallback_path=$$(printf '%s\n' "$$fallback_path" | cut -d/ -f2-);; esac; \
	fallback_registry=$$(printf '%s\n' "$$fallback_registry" | sed 's|/*$$||'); \
	fallback_image="$$fallback_registry/$$fallback_path"; \
	if podman image exists "$$image"; then :; \
	elif [ "$$image" != "$$fallback_image" ] && podman image exists "$$fallback_image"; then \
	  image="$$fallback_image"; \
	elif ! podman pull "$$image" >/dev/null 2>&1; then \
	  if [ "$$image" = "$$fallback_image" ]; then echo "ERROR: unable to pull OpenShell CLI image $$image." >&2; return 1; fi; \
	  echo "OpenShell CLI: $$image is not pullable here; using $$fallback_image..." >&2; \
	  image="$$fallback_image" && podman pull "$$image" >/dev/null 2>&1 || return 1; \
	fi; \
	podman run --rm -i $(OPENSHELL_PODMAN_USERNS) -e XDG_CONFIG_HOME=/cli -e HOME=/cli \
	  -v "$(OPENSHELL_CLI_STATE):/cli/openshell" -v "$$(pwd):/work:ro" "$$image" "$$@"; \
}

refresh-certs: ## Refresh openShell mTLS client certificates from the cluster
	@$(_openshell-cli) && \
	  echo "==> Refreshing openShell client certificates..." && \
	  MTLS_DIR="$(OPENSHELL_CLI_STATE)/gateways/$(OPENSHELL_GATEWAY_NAME)/mtls" && mkdir -p "$$MTLS_DIR" && \
	  oc -n $(OPENSHELL_NAMESPACE) get secret openshell-client-tls -o jsonpath='{.data.ca\.crt}'  | base64 -d > "$$MTLS_DIR/ca.crt" && \
	  oc -n $(OPENSHELL_NAMESPACE) get secret openshell-client-tls -o jsonpath='{.data.tls\.crt}' | base64 -d > "$$MTLS_DIR/tls.crt" && \
	  oc -n $(OPENSHELL_NAMESPACE) get secret openshell-client-tls -o jsonpath='{.data.tls\.key}' | base64 -d > "$$MTLS_DIR/tls.key" && \
	  chmod 600 "$$MTLS_DIR/tls.key" && \
	  echo "    Certificates refreshed."
	@$(_openshell-cli) && \
	  if ! $(OPENSHELL) status 2>/dev/null | grep -q "Connected"; then \
	    echo "ERROR: gateway '$(OPENSHELL_GATEWAY_NAME)' is not registered or not reachable over mTLS." >&2; \
	    echo "       The gateway is a cluster prerequisite: run scripts/setup-openshell-gateway.sh." >&2; \
	    exit 1; \
	  fi

_openshell-env = $(_openshell-cli) && set -a && if [ -f .env ]; then . ./.env; fi && set +a && \
	SANDBOX=$${OPENSHELL_SANDBOX_NAME:-rag-sandbox} && \
	NS=$$(oc project -q) && \
	IMAGE="image-registry.openshift-image-registry.svc:5000/$$NS/$(OPENSHELL_IMAGE):latest" && \
	APPS_DOMAIN=$$(oc get ingresses.config cluster -o jsonpath='{.spec.domain}') && \
	[ -f agent_config.json ] || { echo "ERROR: agent_config.json not found."; exit 1; } && \
	AGENT_CONFIG_B64=$$(base64 < agent_config.json | tr -d '\n') && \
	VECTOR_STORE_PROVIDER=$$(python3 -c 'import json; print(json.load(open("agent_config.json"))["vector_store"]["provider_type"])') && \
	MAAS_SECRET_NAME=$$(sed -n 's/^[[:space:]]*maas_secret_name:[[:space:]]*"\([^"\]*\)".*/\1/p' values.yaml) && \
	DB_SECRET_NAME=$${DB_SECRET_NAME:-$$(sed -n 's/^[[:space:]]*db_secret_name:[[:space:]]*"\([^"\]*\)".*/\1/p' values.yaml)} && \
	secret_value() { oc get secret "$$1" -n "$$NS" -o jsonpath="{.data.$$2}" 2>/dev/null | base64 -d; } && \
	if [ -n "$$MAAS_SECRET_NAME" ] && [ "$$MAAS_SECRET_NAME" != "__MAAS_SECRET_NAME__" ]; then \
	  value=$$(secret_value "$$MAAS_SECRET_NAME" MAAS_BASE_URL); [ -z "$$value" ] || MAAS_BASE_URL="$$value"; \
	  value=$$(secret_value "$$MAAS_SECRET_NAME" MAAS_API_KEY); [ -z "$$value" ] || MAAS_API_KEY="$$value"; \
	fi && \
	if [ -n "$$DB_SECRET_NAME" ] && [ "$$DB_SECRET_NAME" != "__DB_SECRET_NAME__" ]; then \
	  for key in MILVUS_URI MILVUS_TOKEN MILVUS_CA_CERT PGVECTOR_HOST PGVECTOR_PORT PGVECTOR_DB PGVECTOR_USER PGVECTOR_PASSWORD PGVECTOR_CA_CERT NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD NEO4J_DATABASE NEO4J_CA_CERT; do \
	    value=$$(secret_value "$$DB_SECRET_NAME" "$$key"); current=$$(printenv "$$key"); [ -n "$$current" ] || [ -z "$$value" ] || export "$$key=$$value"; \
	  done; \
	fi && \
	case "$$MILVUS_CA_CERT" in *"-----BEGIN CERTIFICATE-----"*) MILVUS_CA_CERT=$$(printf "%s" "$$MILVUS_CA_CERT" | base64 | tr -d "\n");; esac && \
	case "$$PGVECTOR_CA_CERT" in *"-----BEGIN CERTIFICATE-----"*) PGVECTOR_CA_CERT=$$(printf "%s" "$$PGVECTOR_CA_CERT" | base64 | tr -d "\\n");; esac && \
	case "$$NEO4J_CA_CERT" in *"-----BEGIN CERTIFICATE-----"*) NEO4J_CA_CERT=$$(printf "%s" "$$NEO4J_CA_CERT" | base64 | tr -d "\\n");; esac && \
	if [ "$$VECTOR_STORE_PROVIDER" = "pgvector" ]; then \
	  [ -n "$$PGVECTOR_HOST" ] || { echo "ERROR: PGVECTOR_HOST is required for pgvector deployment."; exit 1; }; \
	  PGVECTOR_PORT=$${PGVECTOR_PORT:-5432}; \
	  PGVECTOR_DB=$${PGVECTOR_DB:-postgres}; \
	  PGVECTOR_USER=$${PGVECTOR_USER:-postgres}; \
	elif [ "$$VECTOR_STORE_PROVIDER" = "neo4j" ]; then \
	  [ -n "$$NEO4J_URI" ] || { echo "ERROR: NEO4J_URI is required for Neo4j deployment."; exit 1; }; \
	  [ -n "$$NEO4J_PASSWORD" ] || { echo "ERROR: NEO4J_PASSWORD is required for Neo4j deployment."; exit 1; }; \
	  NEO4J_USERNAME=$${NEO4J_USERNAME:-neo4j}; \
	  NEO4J_DATABASE=$${NEO4J_DATABASE:-neo4j}; \
	fi && \
	export MAAS_BASE_URL MAAS_API_KEY MILVUS_URI MILVUS_TOKEN MILVUS_CA_CERT PGVECTOR_HOST PGVECTOR_PORT PGVECTOR_DB PGVECTOR_USER PGVECTOR_PASSWORD PGVECTOR_CA_CERT NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD NEO4J_DATABASE NEO4J_CA_CERT

build-openshell: ## Build openShell sandbox image via OpenShift BuildConfig
	@set -a && if [ -f .env ]; then . ./.env; fi && set +a && \
	  oc new-build --strategy=docker --binary \
	    --name=$(OPENSHELL_IMAGE) --to=$(OPENSHELL_IMAGE):latest 2>/dev/null || true && \
	  oc patch bc $(OPENSHELL_IMAGE) \
	    -p '{"spec":{"strategy":{"dockerStrategy":{"dockerfilePath":"Containerfile.openshell"}}}}' && \
	  CSV_JSON=$$(oc -n "$${RHOAI_CSV_NAMESPACE:-redhat-ods-operator}" get csv -o json 2>/dev/null || printf '{"items":[]}') && \
	  BASE_IMAGE=$${OPENSHELL_BASE_IMAGE:-$$(python3 -c 'import json,sys; d=json.load(sys.stdin); c=[x for x in d.get("items",[]) if x.get("status",{}).get("phase")=="Succeeded" and x.get("metadata",{}).get("name","").startswith("rhods-operator.")]; c.sort(key=lambda x:x.get("metadata",{}).get("creationTimestamp","")); r=c[-1].get("spec",{}).get("relatedImages",[]) if c else []; print(next((x.get("image","") for x in r if x.get("name")=="odh_openshell_sandbox_image"),""))' <<< "$$CSV_JSON")} && \
	  if [ -z "$$BASE_IMAGE" ]; then \
	    echo "ERROR: sandbox image is absent from the RHOAI CSV. Set RHOAI_CSV_NAMESPACE or OPENSHELL_BASE_IMAGE."; \
	    exit 1; \
	  else \
	    echo "    Using sandbox base image from the RHOAI CSV: $$BASE_IMAGE"; \
	  fi && \
	  set_base_image() { \
	    BUILD_ARGS_JSON=$$(python3 -c 'import json,sys; print(json.dumps({"spec":{"strategy":{"dockerStrategy":{"buildArgs":[{"name":"BASE_IMAGE","value":sys.argv[1]}]}}}}))' "$$1") && \
	    oc patch bc $(OPENSHELL_IMAGE) --type=merge -p "$$BUILD_ARGS_JSON"; \
	  } && \
	  run_build() { \
	    BUILD_BEFORE=$$(oc get builds -l build=$(OPENSHELL_IMAGE) --sort-by=.metadata.creationTimestamp -o name | tail -n1) || return 1; \
	    oc start-build $(OPENSHELL_IMAGE) --from-dir=. --follow || true; \
	    BUILD_NAME=$$(oc get builds -l build=$(OPENSHELL_IMAGE) --sort-by=.metadata.creationTimestamp -o name | tail -n1) || return 1; \
	    [ -n "$$BUILD_NAME" ] || return 1; \
	    [ "$$BUILD_NAME" != "$$BUILD_BEFORE" ] || { echo "ERROR: OpenShift did not create a new build."; return 1; }; \
	    BUILD_PHASE=$$(oc get "$$BUILD_NAME" -o jsonpath='{.status.phase}') || return 1; \
	    echo "    Build $$BUILD_NAME finished with phase $$BUILD_PHASE."; \
	  } && \
	  set_base_image "$$BASE_IMAGE" && run_build && \
	  if [ "$$BUILD_PHASE" != "Complete" ]; then \
	    FALLBACK_BASE_IMAGE="$${BASE_IMAGE/registry.redhat.io/quay.io}"; \
	    if [ "$$FALLBACK_BASE_IMAGE" = "$$BASE_IMAGE" ]; then \
	      echo "ERROR: build failed and no registry.redhat.io to quay.io fallback applies."; exit 1; \
	    fi; \
	    echo "    Build failed; retrying with $$FALLBACK_BASE_IMAGE"; \
	    BASE_IMAGE="$$FALLBACK_BASE_IMAGE" && set_base_image "$$BASE_IMAGE" && run_build; \
	  fi && \
	  if [ "$$BUILD_PHASE" != "Complete" ]; then echo "ERROR: OpenShift image build did not complete."; exit 1; fi && \
	  NS=$$(oc project -q) && \
	  echo "" && \
	  echo "Image ready: image-registry.openshift-image-registry.svc:5000/$$NS/$(OPENSHELL_IMAGE):latest" && \
	  echo "Next: make deploy-openshell"

create-sandbox: ## Delete existing sandbox and create a new one (with 120s timeout)
	@$(_openshell-env) && \
	  echo "==> Granting image-pull to $(OPENSHELL_NAMESPACE) namespace..." && \
	  oc policy add-role-to-user system:image-puller "system:serviceaccount:$(OPENSHELL_NAMESPACE):openshell-sandbox" -n "$$NS" 2>/dev/null || true
	@$(_openshell-env) && \
	  oc policy add-role-to-user system:image-puller "system:serviceaccount:$(OPENSHELL_NAMESPACE):default" -n "$$NS" 2>/dev/null || true
	@$(_openshell-env) && \
	  echo "==> Deleting old sandbox $$SANDBOX..." && \
	  $(OPENSHELL) sandbox delete "$$SANDBOX" 2>/dev/null || true
	@$(_openshell-env) && \
	  wait_gone() { \
	    i=0; while [ $$i -lt 30 ]; do \
	      sandbox_present=0; pod_present=0; \
	      oc -n $(OPENSHELL_NAMESPACE) get sandbox "default--$$SANDBOX" >/dev/null 2>&1 && sandbox_present=1; \
	      oc -n "$$NS" get pod "$$SANDBOX" >/dev/null 2>&1 && pod_present=1; \
	      [ $$sandbox_present -eq 0 ] && [ $$pod_present -eq 0 ] && return 0; \
	      i=$$((i+1)); sleep 2; \
	    done; \
	    return 0; \
	  } && \
	  echo "==> Waiting for the previous sandbox to be removed..." && \
	  wait_gone && \
	  echo "==> Creating sandbox $$SANDBOX from $$IMAGE..." && \
	  echo "    (timeout 120s — this may take a minute while the image is pulled)..." && \
	  CREATE_LOG=$$(mktemp /tmp/openshell-create.XXXXXX) && \
	  run_create() { \
	    $(OPENSHELL) sandbox create --name "$$SANDBOX" --from "$$IMAGE" \
	      --env "MAAS_BASE_URL=$$MAAS_BASE_URL" \
	      --env "MAAS_API_KEY=$$MAAS_API_KEY" \
	      --env "AGENT_CONFIG_B64=$$AGENT_CONFIG_B64" \
	      --env "MILVUS_URI=$$MILVUS_URI" \
	      --env "MILVUS_TOKEN=$$MILVUS_TOKEN" \
	      --env "MILVUS_CA_CERT=$$MILVUS_CA_CERT" \
	      --env "PGVECTOR_HOST=$$PGVECTOR_HOST" \
	      --env "PGVECTOR_PORT=$$PGVECTOR_PORT" \
	      --env "PGVECTOR_DB=$$PGVECTOR_DB" \
	      --env "PGVECTOR_USER=$$PGVECTOR_USER" \
	      --env "PGVECTOR_PASSWORD=$$PGVECTOR_PASSWORD" \
	      --env "PGVECTOR_CA_CERT=$$PGVECTOR_CA_CERT" \
	      --env "NEO4J_URI=$$NEO4J_URI" \
	      --env "NEO4J_USERNAME=$$NEO4J_USERNAME" \
	      --env "NEO4J_PASSWORD=$$NEO4J_PASSWORD" \
	      --env "NEO4J_DATABASE=$$NEO4J_DATABASE" \
	      --env "NEO4J_CA_CERT=$$NEO4J_CA_CERT" \
	      --env "PORT=8080" </dev/null >"$$CREATE_LOG" 2>&1 & \
	    CPID=$$!; ( sleep 80 && kill $$CPID 2>/dev/null ) >/dev/null 2>&1 & TPID=$$!; \
	    wait $$CPID 2>/dev/null; RC=$$?; kill $$TPID 2>/dev/null; wait $$TPID 2>/dev/null; \
	    return $$RC; \
	  } && \
	  attempt=1; RC=0; \
	  while :; do \
	    echo "    Attempt $$attempt/$(SANDBOX_CREATE_ATTEMPTS)..." && \
	    run_create; RC=$$?; \
	    [ $$RC -eq 0 ] && break; \
	    if ! grep -q "pod updates may not change fields" "$$CREATE_LOG"; then break; fi; \
	    if [ $$attempt -ge $(SANDBOX_CREATE_ATTEMPTS) ]; then break; fi; \
	    echo "    Sandbox Pod update was rejected by the cluster Kyverno imagePullSecrets policy" && \
	    echo "    (duplicate pull secret). Recreating the sandbox and retrying..." && \
	    $(OPENSHELL) sandbox delete "$$SANDBOX" >/dev/null 2>&1 || true; \
	    wait_gone; sleep 5; \
	    attempt=$$((attempt+1)); \
	  done; \
	  if [ $$RC -ne 0 ]; then cat "$$CREATE_LOG"; fi; \
	  rm -f "$$CREATE_LOG"; \
	  if [ $$RC -ne 0 ]; then \
	    echo "    sandbox create exited with $$RC (may still be starting — wait-sandbox will poll)"; \
	  else \
	    echo "    Sandbox created."; \
	  fi; true

wait-sandbox: ## Poll until sandbox phase is Ready (max 300s)
	@$(_openshell-env) && \
	  echo "==> Waiting for sandbox $$SANDBOX to be ready..." && \
	  i=0; while [ $$i -lt 60 ]; do \
	    i=$$((i+1)); \
	    phase=$$($(OPENSHELL) sandbox list 2>/dev/null | grep "$$SANDBOX" | awk '{print $$NF}' | sed 's/\x1b\[[0-9;]*m//g'); \
	    if [ "$$phase" = "Ready" ]; then echo "    Sandbox ready."; exit 0; fi; \
	    echo "    status: $$phase (attempt $$i/60, waiting 5s...)"; \
	    sleep 5; \
	  done; \
	  echo "ERROR: sandbox not ready after 300s"; exit 1

setup-egress: ## Add MaaS + vector database egress policies for agent Python binary
	@$(_openshell-env) && \
	  echo "==> Using agent Python binary: /usr/bin/python3.12" && \
	  PYTHON_BIN="/usr/bin/python3.12" && \
	  echo "==> Adding egress policy for MaaS..." && \
	  MAAS_HOST=$$(echo "$$MAAS_BASE_URL" | sed -E 's|^https?://||;s|/.*||') && \
	  $(OPENSHELL) policy update "$$SANDBOX" --add-endpoint "$$MAAS_HOST:443" --binary "$$PYTHON_BIN" --wait && \
	  echo "==> Adding egress policies for K8s API + $$VECTOR_STORE_PROVIDER..." && \
	  if [ "$$VECTOR_STORE_PROVIDER" = "pgvector" ]; then \
	    DB_HOST=$${PGVECTOR_HOST:-localhost}; DB_PORT=$${PGVECTOR_PORT:-5432}; \
	  elif [ "$$VECTOR_STORE_PROVIDER" = "milvus" ]; then \
	    MILVUS_HOSTPORT=$$(echo "$$MILVUS_URI" | sed -E 's|^https?://||;s|/.*||'); \
	    DB_HOST=$$(echo "$$MILVUS_HOSTPORT" | cut -d: -f1); \
	    DB_PORT=$$(echo "$$MILVUS_HOSTPORT" | cut -d: -f2); \
	  elif [ "$$VECTOR_STORE_PROVIDER" = "neo4j" ]; then \
	    NEO4J_ENDPOINT=$$(python3 -c 'import sys; from urllib.parse import urlsplit; endpoint=urlsplit(sys.argv[1]); print(endpoint.hostname or ""); print(endpoint.port or 7687)' "$$NEO4J_URI") && \
	    DB_HOST=$$(printf '%s\n' "$$NEO4J_ENDPOINT" | sed -n '1p') && \
	    DB_PORT=$$(printf '%s\n' "$$NEO4J_ENDPOINT" | sed -n '2p'); \
	  else \
	    echo "ERROR: Unsupported deployment provider $$VECTOR_STORE_PROVIDER"; exit 1; \
	  fi && \
	  [ -n "$$DB_HOST" ] && [ -n "$$DB_PORT" ] || { echo "ERROR: Vector database host and port are required."; exit 1; } && \
	  POLICY_FILE="$$(pwd)/.openshell-policy.yaml" && \
	  trap 'rm -f "$$POLICY_FILE"' EXIT && \
	  case "$(OPENSHELL_CLI)" in \
	    openshell_cli|podman*|docker*) POLICY_ARG="/work/.openshell-policy.yaml";; \
	    *) POLICY_ARG="$$POLICY_FILE";; \
	  esac && \
	  K8S_API_URL=$$(oc whoami --show-server) && \
	  export DB_HOST DB_PORT K8S_API_URL PY_BIN="$$PYTHON_BIN" && \
	  $(OPENSHELL) policy get "$$SANDBOX" --full -o json 2>/dev/null | sed -n '/^{/,$$p' | \
	    python3 -c 'import json,sys,os,urllib.parse;d=json.load(sys.stdin);p=d["policy"];api=urllib.parse.urlsplit(os.environ["K8S_API_URL"]);p["network_policies"]["k8s_api"]={"name":"k8s-api","endpoints":[{"host":api.hostname,"port":api.port or 443,"tls":"skip"}],"binaries":[{"path":os.environ["PY_BIN"]}]};p["network_policies"]["vector_db"]={"name":"vector-db","endpoints":[{"host":os.environ["DB_HOST"],"port":int(os.environ["DB_PORT"]),"tls":"skip"}],"binaries":[{"path":os.environ["PY_BIN"]}]};json.dump(p,sys.stdout,indent=2)' > "$$POLICY_FILE" && \
	  $(OPENSHELL) policy set "$$SANDBOX" --policy "$$POLICY_ARG" --wait && \
	  echo "Done!"

start-agent: refresh-certs ## Start agent with K8s ServiceAccount auth inside sandbox
	@$(_openshell-env) && \
	  echo "==> Creating agent-client ServiceAccount and token..." && \
	  oc create sa agent-client 2>/dev/null || true && \
	  CLIENT_TOKEN=$$(oc create token agent-client --duration=168h) && \
	  oc delete secret agent-client-token 2>/dev/null || true && \
	  oc create secret generic agent-client-token --from-literal=token="$$CLIENT_TOKEN" && \
	  echo "==> Generating sandbox reviewer token..." && \
	  K8S_API_URL=$$(oc whoami --show-server) && \
	  K8S_REVIEWER_TOKEN=$$(oc create token openshell-sandbox -n $(OPENSHELL_NAMESPACE) --duration=168h) && \
	  if oc -n openshift-config-managed get configmap trusted-ca-bundle >/dev/null 2>&1; then K8S_CA_SOURCE=trusted-ca-bundle; \
	  elif oc get configmap kube-root-ca.crt -n "$$NS" >/dev/null 2>&1; then K8S_CA_SOURCE=kube-root-ca.crt; \
	  else echo "ERROR: Kubernetes API CA certificate not found." >&2; exit 1; fi && \
	  K8S_CA_PATH=/sandbox/k8s-ca.crt && \
	  ALLOWED_SA_USERNAME="system:serviceaccount:$${NS}:agent-client" && \
	  echo "==> Stopping any running agent..." && \
	  $(OPENSHELL) sandbox exec -n "$$SANDBOX" -- /usr/bin/python3.12 -c "import os,signal; from pathlib import Path; pid=Path('/sandbox/agent.pid'); os.kill(int(pid.read_text().strip()),signal.SIGTERM) if pid.is_file() else None" 2>/dev/null || true && \
	  sleep 5 && \
	  echo "==> Uploading the cluster API CA ($$K8S_CA_SOURCE)..." && \
	  if [ "$$K8S_CA_SOURCE" = "trusted-ca-bundle" ]; then \
	    oc -n openshift-config-managed get configmap trusted-ca-bundle -o jsonpath='{.data.ca-bundle\.crt}'; \
	  else \
	    oc get configmap kube-root-ca.crt -n "$$NS" -o jsonpath='{.data.ca\.crt}'; \
	  fi | $(OPENSHELL) sandbox exec -n "$$SANDBOX" -- bash -c 'cat > /sandbox/k8s-ca.crt' && \
	  PGVECTOR_PASSWORD_B64=$$(printf '%s' "$$PGVECTOR_PASSWORD" | base64 | tr -d '\n') && \
	  NEO4J_URI_QUOTED=$$(python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$$NEO4J_URI") && \
	  NEO4J_USERNAME_QUOTED=$$(python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$$NEO4J_USERNAME") && \
	  NEO4J_PASSWORD_QUOTED=$$(python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$$NEO4J_PASSWORD") && \
	  NEO4J_DATABASE_QUOTED=$$(python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$$NEO4J_DATABASE") && \
	  echo "==> Starting agent (auth: enabled)..." && \
	  { $(OPENSHELL) sandbox exec -n "$$SANDBOX" -- bash -c '\
	    export MAAS_API_KEY="'"$$MAAS_API_KEY"'" && \
	    export MAAS_BASE_URL="'"$$MAAS_BASE_URL"'" && \
	    export AGENT_CONFIG_B64="'"$$AGENT_CONFIG_B64"'" && \
	    export MILVUS_URI="'"$$MILVUS_URI"'" && \
	    export MILVUS_TOKEN="'"$$MILVUS_TOKEN"'" && \
	    export MILVUS_CA_CERT="$$(printf %s '"'"$$MILVUS_CA_CERT"'"' | base64 -d)" && \
	    export PGVECTOR_HOST="'"$$PGVECTOR_HOST"'" && \
	    export PGVECTOR_PORT="'"$$PGVECTOR_PORT"'" && \
	    export PGVECTOR_DB="'"$$PGVECTOR_DB"'" && \
	    export PGVECTOR_USER="'"$$PGVECTOR_USER"'" && \
	    export PGVECTOR_PASSWORD=$$(printf %s "'"$$PGVECTOR_PASSWORD_B64"'" | base64 -d) && \
	    case "$$PGVECTOR_CA_CERT" in *"-----BEGIN CERTIFICATE-----"*) ;; *) PGVECTOR_CA_CERT=$$(printf %s "$$PGVECTOR_CA_CERT" | base64 -d);; esac && \
	    export PGVECTOR_CA_CERT && \
	    export NEO4J_URI='"$$NEO4J_URI_QUOTED"' && \
	    export NEO4J_USERNAME='"$$NEO4J_USERNAME_QUOTED"' && \
	    export NEO4J_PASSWORD='"$$NEO4J_PASSWORD_QUOTED"' && \
	    export NEO4J_DATABASE='"$$NEO4J_DATABASE_QUOTED"' && \
	    export NEO4J_CA_CERT="$$(printf %s '"'"$$NEO4J_CA_CERT"'"' | base64 -d)" && \
	    export K8S_API_URL="'"$$K8S_API_URL"'" && \
	    export K8S_REVIEWER_TOKEN="'"$$K8S_REVIEWER_TOKEN"'" && \
	    export K8S_CA_PATH="'"$$K8S_CA_PATH"'" && \
	    export ALLOWED_SA_USERNAME="'"$$ALLOWED_SA_USERNAME"'" && \
	    export OPENSHELL_GRPC_PROXY_PORT=18080 && \
	    export PYTHONPATH=/sandbox/.local/lib/python3.12/site-packages:/sandbox/.local/lib64/python3.12/site-packages:/sandbox && \
	    cd /sandbox && \
	    : > /sandbox/agent.log && rm -f /sandbox/agent.pid && \
	    /usr/bin/python3.12 -c "import subprocess,sys,time; log=open(\"/sandbox/agent.log\",\"ab\",buffering=0); cmd=[\"/usr/bin/python3.12\",\"/sandbox/proxychains_entrypoint.py\",\"/usr/bin/python3.12\",\"-m\",\"uvicorn\",\"auth_wrapper:app\",\"--host\",\"0.0.0.0\",\"--port\",\"8080\"]; process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True); open(\"/sandbox/agent.pid\",\"w\").write(str(process.pid)); time.sleep(3); alive=process.poll() is None; print(\"Agent process alive\" if alive else \"ERROR: Agent died on startup\"); sys.exit(0 if alive else 1)"' || \
	    { echo "    Checking agent health after OpenShell exec transport error..." && \
	      $(OPENSHELL) sandbox exec -n "$$SANDBOX" -- /usr/bin/python3.12 -c 'import sys,urllib.request; response=urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3); sys.exit(0 if response.status == 200 else 1)'; }; } && \
	  echo "    Agent started in background."

expose-agent: ## Expose agent service and create Route
	@$(_openshell-env) && \
	  echo "==> Exposing service URL..." && \
	  $(OPENSHELL) service expose "$$SANDBOX" 8080 agent 2>/dev/null || true && \
	  sleep 2 && \
	  SVC_HOST="default--$$SANDBOX--agent.$(OPENSHELL_NAMESPACE).$$APPS_DOMAIN" && \
	  printf 'apiVersion: route.openshift.io/v1\nkind: Route\nmetadata:\n  name: openshell-rag-agent\n  namespace: $(OPENSHELL_NAMESPACE)\nspec:\n  host: "%s"\n  to:\n    kind: Service\n    name: openshell\n  port:\n    targetPort: grpc\n  tls:\n    termination: passthrough\n' "$$SVC_HOST" | oc apply -f - && \
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
	  echo "  curl -sk https://$$SVC_HOST/v1/responses \\" && \
	  echo "    -H \"X-Api-Key: \$$TOKEN\" \\" && \
	  echo "    -H 'Content-Type: application/json' \\" && \
	  echo "    -d '{\"input\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"

deploy-openshell: create-sandbox wait-sandbox setup-egress start-agent expose-agent ## Full sandbox deploy: create, start, expose

undeploy-openshell: ## Remove openShell sandbox and route
	@$(_openshell-cli) && \
	  ($(OPENSHELL) sandbox delete $(OPENSHELL_SANDBOX) 2>/dev/null || true) && \
	  oc -n $(OPENSHELL_NAMESPACE) delete route openshell-rag-agent 2>/dev/null || true && \
	  echo "Sandbox $(OPENSHELL_SANDBOX) and route removed." && \
	  sleep 10 2>/dev/null || true
