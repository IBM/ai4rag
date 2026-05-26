#!/bin/bash

set -o pipefail

TARGETS=("ai4rag")

if [[ "$1" == "--all" ]]; then
    TARGETS=("ai4rag" "dev_utils" "tests")
fi

echo "Running formatters on: ${TARGETS[*]}"

echo ""
echo "==> isort"
uv run --extra code_check isort "${TARGETS[@]}"

echo ""
echo "==> black"
uv run --extra code_check black "${TARGETS[@]}"

echo ""
echo "==> copyright_check"
bash scripts/copyright_check.sh --fix
