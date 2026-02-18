#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/ops/ops.env"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-forecast/bin/python}"
ASOF="${1:-${NGF_ASOF:-}}"

CMD=("${PYTHON_BIN}" "${ROOT_DIR}/main.py")
if [[ -n "${ASOF}" ]]; then
  CMD+=(--asof "${ASOF}")
fi
if [[ "${NGF_SKIP_DATA_REFRESH:-0}" == "1" ]]; then
  CMD+=(--skip-data-refresh)
fi
if [[ "${NGF_REFRESH_RETRIEVE_STEO:-0}" == "1" ]]; then
  CMD+=(--refresh-retrieve-steo)
fi
if [[ "${NGF_REFRESH_RETRIEVE_CONTEXT:-0}" == "1" ]]; then
  CMD+=(--refresh-retrieve-context)
fi
if [[ "${NGF_RETIRE_DATA_NEW:-0}" == "1" ]]; then
  CMD+=(--retire-data-new)
fi

"${CMD[@]}"
