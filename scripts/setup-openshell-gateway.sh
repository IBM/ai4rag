#!/bin/bash

# Installs the downstream Red Hat OpenShell 3.6 gateway on an OpenShift cluster
# and exposes it through a passthrough Route.
#
# This is a cluster-level, one-time prerequisite for deploying ai4rag starter
# kits. It is intentionally kept in the ai4rag repository and is not included
# in generated starter_kit.zip archives.

set -euo pipefail

OPENSHELL_NAMESPACE="${OPENSHELL_NAMESPACE:-openshell}"
OPENSHELL_GATEWAY_NAME="${OPENSHELL_GATEWAY_NAME:-openshift}"
RHOAI_CSV_NAMESPACE="${RHOAI_CSV_NAMESPACE:-redhat-ods-operator}"
OPENSHELL_IMAGE_REGISTRY="${OPENSHELL_IMAGE_REGISTRY:-}"
OPENSHELL_FALLBACK_REGISTRY="${OPENSHELL_FALLBACK_REGISTRY:-quay.io/rhoai}"
OPENSHELL_CHART_REPO="${OPENSHELL_CHART_REPO:-https://github.com/red-hat-data-services/openshell}"
OPENSHELL_CHART_REF="${OPENSHELL_CHART_REF:-2aea2c5375cf9a4a284f79c4b97f9ab6aba439bc}"
OPENSHELL_CLI_IMAGE="${OPENSHELL_CLI_IMAGE:-}"
OPENSHELL_CLI_STATE="${OPENSHELL_CLI_STATE:-${HOME}/.config/openshell}"
if [[ -z "${OPENSHELL_PODMAN_USERNS:-}" ]]; then
  OPENSHELL_PODMAN_USERNS="--userns=keep-id"
fi
OPENSHELL_RESET="${OPENSHELL_RESET:-false}"
OPENSHELL_MIRROR_RUNTIME_IMAGES="${OPENSHELL_MIRROR_RUNTIME_IMAGES:-auto}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--reset] [--help]

Installs the downstream Red Hat OpenShell 3.6 gateway and exposes it through a
Route named openshell-gateway.

Options:
  --reset   Drop the gateway database before installing. Required when migrating
            from the upstream 0.0.99 chart, whose database the 3.6 gateway
            cannot read. Deletes sandboxes, policies and stored credentials;
            workspace PVCs are kept.
  --help    Show this help.

Environment variables:
  OPENSHELL_NAMESPACE               Gateway namespace (default: ${OPENSHELL_NAMESPACE})
  OPENSHELL_GATEWAY_NAME            Name used to register the gateway in the CLI (default: ${OPENSHELL_GATEWAY_NAME})
  RHOAI_CSV_NAMESPACE               Namespace containing the installed RHOAI CSV (default: ${RHOAI_CSV_NAMESPACE})
  OPENSHELL_IMAGE_REGISTRY          Optional registry override (default: full CSV references)
  OPENSHELL_FALLBACK_REGISTRY       Registry prefix for CLI fallback (default: ${OPENSHELL_FALLBACK_REGISTRY})
  OPENSHELL_CHART_REPO              Chart repository (default: ${OPENSHELL_CHART_REPO})
  OPENSHELL_CHART_REF               Chart commit (default: ${OPENSHELL_CHART_REF})
  OPENSHELL_RESET                   Same as --reset (default: ${OPENSHELL_RESET})
  OPENSHELL_CLI_IMAGE               Optional CLI image override (default: from the RHOAI CSV)
  OPENSHELL_CLI_STATE               CLI state directory holding the gateway registration (default: ${OPENSHELL_CLI_STATE})
  OPENSHELL_PODMAN_USERNS           Extra `podman run` flags for the CLI container (default: ${OPENSHELL_PODMAN_USERNS})
  OPENSHELL_MIRROR_RUNTIME_IMAGES   Copy the sandbox runtime and supervisor images into the
                                    cluster registry: auto|true|false (default: auto, which
                                    mirrors only when the add-imagepullsecrets Kyverno policy
                                    is installed)

Image references, including registry and digest, are read from the installed
RHOAI CSV. Override OPENSHELL_IMAGE_REGISTRY to select another registry while
preserving each CSV image name and digest. The CLI falls back to the matching
OPENSHELL_FALLBACK_REGISTRY reference when the CSV image cannot be pulled.

Requirements: oc, helm, curl, jq, and podman, all logged in/installed locally.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset) OPENSHELL_RESET=true ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

for tool in oc helm curl jq podman; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "ERROR: '$tool' is required but was not found in PATH." >&2
    exit 1
  }
done

APPS_DOMAIN=$(oc get ingresses.config cluster -o jsonpath='{.spec.domain}')
echo "==> Detected apps domain: ${APPS_DOMAIN}"

echo "==> Reading OpenShell image references from the RHOAI CSV..."
RHOAI_CSV_JSON=$(oc -n "${RHOAI_CSV_NAMESPACE}" get csv -o json 2>/dev/null || printf '{"items":[]}')
RHOAI_CSV_NAME=$(jq -r '
  [.items[] | select(.status.phase == "Succeeded" and (.metadata.name | startswith("rhods-operator.")))]
  | sort_by(.metadata.creationTimestamp) | last | .metadata.name // empty
' <<<"${RHOAI_CSV_JSON}")
csv_image() {
  [[ -n "${RHOAI_CSV_NAME}" ]] || return 1
  jq -er --arg csv "${RHOAI_CSV_NAME}" --arg name "$1" '
    .items[] | select(.metadata.name == $csv) | .spec.relatedImages[]
    | select(.name == $name) | .image
  ' <<<"${RHOAI_CSV_JSON}"
}
image_registry() { local ref="${1%@*}"; printf '%s\n' "${ref%/*}"; }
image_repository() { local ref="${1%@*}"; printf '%s\n' "${ref##*/}"; }
image_digest() { printf '%s\n' "${1##*@}"; }
fallback_image_ref() {
  local ref="$1" path="${1#*/}" namespace="${OPENSHELL_FALLBACK_REGISTRY##*/}"
  if [[ "${path}" == "${namespace}/"* ]]; then
    path="${path#"${namespace}/"}"
  fi
  printf '%s/%s\n' "${OPENSHELL_FALLBACK_REGISTRY%/}" "${path}"
}

CSV_GATEWAY_REF=$(csv_image odh_openshell_gateway_image || true)
CSV_SANDBOX_RUNTIME_REF=$(csv_image odh_openshell_sandbox_image || true)
CSV_SUPERVISOR_REF=$(csv_image odh_openshell_supervisor_image || true)
CSV_CLI_REF=$(csv_image odh_openshell_cli_image || true)
if [[ -z "${CSV_GATEWAY_REF}" || -z "${CSV_SANDBOX_RUNTIME_REF}" || -z "${CSV_SUPERVISOR_REF}" ]]; then
  echo "ERROR: gateway, sandbox runtime or supervisor image is missing from the RHOAI CSV in ${RHOAI_CSV_NAMESPACE}." >&2
  exit 1
fi
if [[ -z "${CSV_CLI_REF}" && -z "${OPENSHELL_CLI_IMAGE}" ]]; then
  echo "ERROR: OpenShell CLI image is missing from the RHOAI CSV in ${RHOAI_CSV_NAMESPACE}." >&2
  exit 1
fi

QUAY_CLI_REF=$(fallback_image_ref "${CSV_CLI_REF:-${OPENSHELL_CLI_IMAGE}}")

# The add-imagepullsecrets Kyverno policy duplicates pull secrets on Pod updates,
# which breaks the OpenShell driver's post-create update. Serve the runtime images
# from the cluster registry so the policy does not match them.
openshell_needs_internal_images() {
  case "${OPENSHELL_MIRROR_RUNTIME_IMAGES:-auto}" in
    true) return 0 ;;
    false) return 1 ;;
    *) oc get clusterpolicy add-imagepullsecrets >/dev/null 2>&1 ;;
  esac
}

mirror_runtime_image() {
  local imagestream="$1" source_ref="$2" tag internal_ref candidate
  local candidates=("${source_ref}")
  tag="$(image_digest "${source_ref}")"
  tag="${tag#sha256:}"
  tag="${tag:0:12}"
  if [[ "${source_ref}" == registry.redhat.io/* ]]; then
    candidates+=("${source_ref/registry.redhat.io/quay.io}")
  fi
  oc -n "${OPENSHELL_NAMESPACE}" secrets link default pull-secret-brew --for=pull >/dev/null 2>&1 || true
  for candidate in "${candidates[@]}"; do
    oc -n "${OPENSHELL_NAMESPACE}" tag --reference-policy=local "${candidate}" "${imagestream}:${tag}" >/dev/null 2>&1 || continue
    internal_ref=""
    for _ in $(seq 1 25); do
      internal_ref=$(oc -n "${OPENSHELL_NAMESPACE}" get istag "${imagestream}:${tag}" -o jsonpath='{.image.dockerImageReference}' 2>/dev/null)
      [[ -n "${internal_ref}" ]] && break
      sleep 4
    done
    if [[ -n "${internal_ref}" ]]; then
      printf '%s\n' "${internal_ref}"
      return 0
    fi
  done
  return 1
}

OPENSHELL_IMAGE_REGISTRY="${OPENSHELL_IMAGE_REGISTRY:-$(image_registry "${CSV_GATEWAY_REF}")}"
OPENSHELL_CLI_IMAGE="${OPENSHELL_CLI_IMAGE:-${CSV_CLI_REF}}"

echo "==> Ensuring ${OPENSHELL_NAMESPACE} namespace and SCCs..."
oc create namespace "${OPENSHELL_NAMESPACE}" 2>/dev/null || true
oc adm policy add-scc-to-user privileged -z openshell-sandbox -n "${OPENSHELL_NAMESPACE}" 2>/dev/null || true
oc adm policy add-scc-to-user privileged -z openshell -n "${OPENSHELL_NAMESPACE}" 2>/dev/null || true

echo "==> Granting TokenReview RBAC to sandbox SA..."
# The binding name includes the namespace so several gateways can coexist and a
# second install does not silently keep a binding that points at another SA.
oc create clusterrolebinding "openshell-tokenreview-${OPENSHELL_NAMESPACE}" \
  --clusterrole=system:auth-delegator \
  --serviceaccount="${OPENSHELL_NAMESPACE}:openshell-sandbox" 2>/dev/null || true

if [[ "${OPENSHELL_RESET}" == "true" ]]; then
  echo "==> Resetting gateway state..."
  oc -n "${OPENSHELL_NAMESPACE}" scale statefulset/openshell --replicas=0 2>/dev/null || true
  oc -n "${OPENSHELL_NAMESPACE}" delete pvc openshell-data-openshell-0 --ignore-not-found
  oc -n "${OPENSHELL_NAMESPACE}" scale statefulset/openshell --replicas=1 2>/dev/null || true
  oc -n "${OPENSHELL_NAMESPACE}" rollout status statefulset/openshell --timeout=180s 2>/dev/null || true
fi

echo "==> Downloading downstream OpenShell chart..."
CHART_DIR=$(mktemp -d)
trap 'rm -rf "${CHART_DIR}"' EXIT
curl -fsSL "${OPENSHELL_CHART_REPO}/archive/${OPENSHELL_CHART_REF}.tar.gz" |
  tar -xz -C "${CHART_DIR}" --strip-components=1

HELM_SANDBOX_RUNTIME_IMAGE=(
  --set sandboxRuntime.image.repository="$(image_repository "${CSV_SANDBOX_RUNTIME_REF}")"
  --set sandboxRuntime.image.digest="$(image_digest "${CSV_SANDBOX_RUNTIME_REF}")"
)
HELM_SUPERVISOR_IMAGE=(
  --set supervisor.image.repository="$(image_repository "${CSV_SUPERVISOR_REF}")"
  --set supervisor.image.digest="$(image_digest "${CSV_SUPERVISOR_REF}")"
)
if openshell_needs_internal_images; then
  echo "==> Mirroring sandbox runtime and supervisor images into the cluster registry..."
  SANDBOX_RUNTIME_LOCAL_REF=$(mirror_runtime_image sandbox-runtime "${CSV_SANDBOX_RUNTIME_REF}" || true)
  SUPERVISOR_LOCAL_REF=$(mirror_runtime_image supervisor "${CSV_SUPERVISOR_REF}" || true)
  if [[ -n "${SANDBOX_RUNTIME_LOCAL_REF}" && -n "${SUPERVISOR_LOCAL_REF}" ]]; then
    echo "    Using ${SANDBOX_RUNTIME_LOCAL_REF}"
    echo "    Using ${SUPERVISOR_LOCAL_REF}"
    HELM_SANDBOX_RUNTIME_IMAGE=(
      --set sandboxRuntime.image.registry="$(image_registry "${SANDBOX_RUNTIME_LOCAL_REF}")"
      --set sandboxRuntime.image.repository="$(image_repository "${SANDBOX_RUNTIME_LOCAL_REF}")"
      --set sandboxRuntime.image.digest="$(image_digest "${SANDBOX_RUNTIME_LOCAL_REF}")"
    )
    HELM_SUPERVISOR_IMAGE=(
      --set supervisor.image.registry="$(image_registry "${SUPERVISOR_LOCAL_REF}")"
      --set supervisor.image.repository="$(image_repository "${SUPERVISOR_LOCAL_REF}")"
      --set supervisor.image.digest="$(image_digest "${SUPERVISOR_LOCAL_REF}")"
    )
  else
    echo "    WARNING: could not mirror the runtime images; falling back to the upstream references." >&2
    echo "             Sandbox creation may fail on clusters with the add-imagepullsecrets policy." >&2
  fi
fi

echo "==> Installing downstream OpenShell gateway..."
oc -n "${OPENSHELL_NAMESPACE}" delete configmap openshell-config --ignore-not-found 2>/dev/null || true
oc -n "${OPENSHELL_NAMESPACE}" delete statefulset openshell --ignore-not-found 2>/dev/null || true
helm upgrade --install openshell "${CHART_DIR}/deploy/helm/openshell" \
  --namespace "${OPENSHELL_NAMESPACE}" \
  --set global.image.registry="${OPENSHELL_IMAGE_REGISTRY}" \
  --set gateway.image.repository="$(image_repository "${CSV_GATEWAY_REF}")" \
  --set gateway.image.digest="$(image_digest "${CSV_GATEWAY_REF}")" \
  ${HELM_SANDBOX_RUNTIME_IMAGE[@]} \
  ${HELM_SUPERVISOR_IMAGE[@]} \
  --set securityContext.runAsUser=null \
  --set securityContext.runAsNonRoot=false \
  --set securityContext.allowPrivilegeEscalation=true \
  --set securityContext.capabilities=null \
  --set server.auth.allowUnauthenticatedUsers=false \
  --set "pkiInitJob.serverDnsNames={*.${OPENSHELL_NAMESPACE}.${APPS_DOMAIN},${OPENSHELL_NAMESPACE}.${APPS_DOMAIN},openshell-gateway-${OPENSHELL_NAMESPACE}.${APPS_DOMAIN}}"

echo "==> Waiting for gateway to be ready..."
if ! oc -n "${OPENSHELL_NAMESPACE}" rollout status statefulset/openshell --timeout=180s; then
  echo "ERROR: gateway did not become ready." >&2
  echo "       When upgrading an existing gateway from 0.0.99, its database is incompatible." >&2
  echo "       Re-run this script with --reset and retry." >&2
  exit 1
fi

echo "==> Disabling inherited sandbox image pull secrets..."
TOML=$(oc -n "${OPENSHELL_NAMESPACE}" get configmap openshell-config -o jsonpath='{.data.gateway\.toml}' |
  sed '/^[[:space:]]*image_pull_secrets[[:space:]]*=/d' |
  awk '{print} /^\[openshell\.drivers\.kubernetes\]$/ && !d {print "image_pull_secrets = []"; d=1}')
for key in image_pull_secrets supervisor_image sandbox_runtime_image; do
  if ! grep -Fq "${key}" <<<"${TOML}"; then
    echo "ERROR: ${key} is missing from the gateway configuration." >&2
    exit 1
  fi
done
oc -n "${OPENSHELL_NAMESPACE}" patch configmap openshell-config --type merge \
  -p "$(printf '{"data":{"gateway.toml":%s}}' "$(printf '%s' "${TOML}" | jq -Rs .)")"
oc -n "${OPENSHELL_NAMESPACE}" delete pod openshell-0 --ignore-not-found
oc -n "${OPENSHELL_NAMESPACE}" rollout status statefulset/openshell --timeout=120s

echo "==> Creating gateway Route..."
oc -n "${OPENSHELL_NAMESPACE}" create route passthrough openshell-gateway \
  --service=openshell --port=8080 2>/dev/null || true

ROUTE_HOST=$(oc -n "${OPENSHELL_NAMESPACE}" get route openshell-gateway -o jsonpath='{.spec.host}' 2>/dev/null)
GATEWAY_URL="https://${ROUTE_HOST:-openshell-gateway-${OPENSHELL_NAMESPACE}.${APPS_DOMAIN}}"

echo "==> Verifying gateway connection with mTLS..."
if podman image exists "${OPENSHELL_CLI_IMAGE}"; then :
elif [ "${OPENSHELL_CLI_IMAGE}" != "${QUAY_CLI_REF}" ] && podman image exists "${QUAY_CLI_REF}"; then
  OPENSHELL_CLI_IMAGE="${QUAY_CLI_REF}"
elif ! podman pull "${OPENSHELL_CLI_IMAGE}" >/dev/null 2>&1; then
  echo "    Using ${QUAY_CLI_REF} because ${OPENSHELL_CLI_IMAGE} is not pullable here."
  podman pull "${QUAY_CLI_REF}" >/dev/null 2>&1
  OPENSHELL_CLI_IMAGE="${QUAY_CLI_REF}"
fi
# The gateway requires a client certificate. Store it in the CLI state directory
# and register the gateway there, so starter kits only consume the registration.
CLI_MTLS_DIR="${OPENSHELL_CLI_STATE}/gateways/${OPENSHELL_GATEWAY_NAME}/mtls"
mkdir -p "${CLI_MTLS_DIR}"
for key in ca.crt tls.crt tls.key; do
  oc -n "${OPENSHELL_NAMESPACE}" get secret openshell-client-tls \
    -o jsonpath="{.data.${key//./\\.}}" | base64 -d >"${CLI_MTLS_DIR}/${key}"
done
chmod 600 "${CLI_MTLS_DIR}/tls.key"

openshell_cli() {
  podman run --rm ${OPENSHELL_PODMAN_USERNS} -e XDG_CONFIG_HOME=/cli -e HOME=/cli \
    -v "${OPENSHELL_CLI_STATE}:/cli/openshell" "${OPENSHELL_CLI_IMAGE}" "$@"
}

echo "==> Registering gateway '${OPENSHELL_GATEWAY_NAME}' in ${OPENSHELL_CLI_STATE}..."
openshell_cli gateway remove "${OPENSHELL_GATEWAY_NAME}" >/dev/null 2>&1 || true
openshell_cli gateway add "${GATEWAY_URL}" --local --name "${OPENSHELL_GATEWAY_NAME}"
CLI_STATUS=$(openshell_cli -g "${OPENSHELL_GATEWAY_NAME}" status 2>&1) || true
printf '%s\n' "${CLI_STATUS}"
if ! grep -q "Connected" <<<"${CLI_STATUS}"; then
  echo "ERROR: the OpenShell CLI could not authenticate to the gateway over mTLS." >&2
  exit 1
fi

echo ""
echo "Gateway ready at ${GATEWAY_URL}."
echo "Deploy a starter kit with 'make build-openshell' and 'make deploy-openshell'."
echo "The starter-kit Makefile reads the same client certificate secret, so no local"
echo "OpenShell CLI or certificate files are required."
