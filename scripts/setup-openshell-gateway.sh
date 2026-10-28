#!/bin/bash

# Installs the OpenShell gateway on an OpenShift cluster, exposes it through a
# passthrough Route, and registers it in the local OpenShell CLI.
#
# This is a cluster-level, one-time prerequisite for deploying ai4rag starter
# kits. It is intentionally kept in the ai4rag repository and is not included
# in generated starter_kit.zip archives.

set -euo pipefail

OPENSHELL_VERSION="${OPENSHELL_VERSION:-0.0.99}"
OPENSHELL_GATEWAY_NAME="${OPENSHELL_GATEWAY_NAME:-openshift}"
OPENSHELL_NAMESPACE="${OPENSHELL_NAMESPACE:-openshell}"
MTLS_DIR="${HOME}/.config/openshell/gateways/${OPENSHELL_GATEWAY_NAME}/mtls"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--help]

Installs the OpenShell gateway and registers it with the OpenShell CLI.

Environment variables:
  OPENSHELL_VERSION       Helm chart version to install (default: ${OPENSHELL_VERSION})
  OPENSHELL_GATEWAY_NAME  Name used when registering the gateway (default: ${OPENSHELL_GATEWAY_NAME})
  OPENSHELL_NAMESPACE     Namespace for the gateway (default: ${OPENSHELL_NAMESPACE})

Requirements: oc, helm, jq, and the openshell CLI, all logged in/installed locally.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

for tool in oc helm jq openshell; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "ERROR: '$tool' is required but was not found in PATH." >&2
    exit 1
  }
done

APPS_DOMAIN=$(oc get ingresses.config cluster -o jsonpath='{.spec.domain}')
echo "==> Detected apps domain: ${APPS_DOMAIN}"

echo "==> Ensuring ${OPENSHELL_NAMESPACE} namespace and SCCs..."
oc create namespace "${OPENSHELL_NAMESPACE}" 2>/dev/null || true
oc adm policy add-scc-to-user privileged -z openshell-sandbox -n "${OPENSHELL_NAMESPACE}" 2>/dev/null || true
oc adm policy add-scc-to-user privileged -z openshell -n "${OPENSHELL_NAMESPACE}" 2>/dev/null || true

echo "==> Granting TokenReview RBAC to sandbox SA..."
oc create clusterrolebinding openshell-tokenreview \
  --clusterrole=system:auth-delegator \
  --serviceaccount="${OPENSHELL_NAMESPACE}:openshell-sandbox" 2>/dev/null || true

echo "==> Installing OpenShell gateway ${OPENSHELL_VERSION}..."
helm upgrade --install openshell oci://ghcr.io/nvidia/openshell/helm-chart \
  --version "${OPENSHELL_VERSION}" \
  --namespace "${OPENSHELL_NAMESPACE}" \
  --set securityContext.runAsUser=null \
  --set securityContext.runAsNonRoot=false \
  --set securityContext.allowPrivilegeEscalation=true \
  --set securityContext.capabilities=null \
  --set server.auth.allowUnauthenticatedUsers=true \
  --set "pkiInitJob.serverDnsNames={*.openshell.${APPS_DOMAIN},openshell.${APPS_DOMAIN},openshell-gateway-openshell.${APPS_DOMAIN}}"

echo "==> Waiting for gateway to be ready..."
oc -n "${OPENSHELL_NAMESPACE}" rollout status statefulset/openshell --timeout=120s

echo "==> Disabling mTLS client cert requirement..."
TOML=$(oc -n "${OPENSHELL_NAMESPACE}" get configmap openshell-config -o jsonpath='{.data.gateway\.toml}' | sed '/^client_ca_path/d')
oc -n "${OPENSHELL_NAMESPACE}" patch configmap openshell-config --type merge \
  -p "$(printf '{"data":{"gateway.toml":%s}}' "$(printf '%s' "${TOML}" | jq -Rs .)")"
oc -n "${OPENSHELL_NAMESPACE}" delete pod openshell-0 --ignore-not-found
oc -n "${OPENSHELL_NAMESPACE}" rollout status statefulset/openshell --timeout=120s

echo "==> Creating gateway Route..."
oc -n "${OPENSHELL_NAMESPACE}" create route passthrough openshell-gateway \
  --service=openshell --port=8080 2>/dev/null || true

GATEWAY_URL="https://openshell-gateway-${OPENSHELL_NAMESPACE}.${APPS_DOMAIN}"

echo "==> Registering gateway in CLI as '${OPENSHELL_GATEWAY_NAME}'..."
openshell gateway remove "${OPENSHELL_GATEWAY_NAME}" 2>/dev/null || true
mkdir -p "${MTLS_DIR}"
oc -n "${OPENSHELL_NAMESPACE}" get secret openshell-client-tls -o jsonpath='{.data.ca\.crt}' | base64 -d >"${MTLS_DIR}/ca.crt"
oc -n "${OPENSHELL_NAMESPACE}" get secret openshell-client-tls -o jsonpath='{.data.tls\.crt}' | base64 -d >"${MTLS_DIR}/tls.crt"
oc -n "${OPENSHELL_NAMESPACE}" get secret openshell-client-tls -o jsonpath='{.data.tls\.key}' | base64 -d >"${MTLS_DIR}/tls.key"
openshell gateway add "${GATEWAY_URL}" --local --name "${OPENSHELL_GATEWAY_NAME}"

echo "==> Verifying connection..."
export OPENSHELL_GATEWAY_INSECURE=true
openshell status
echo ""
echo "Gateway ready. Deploy a starter kit with 'make build-openshell' and 'make deploy-openshell'."
