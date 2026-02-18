#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/ops/ops.env"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-forecast/bin/python}"
OUTPUT_DIR="${1:-${ROOT_DIR}/ops/viz/output}"
RUNTIME_ROOT="${NGF_RUNTIME_REPORT_ROOT:-${ROOT_DIR}/data/reports/sprint9}"
RUNTIME_PNG_DIR="${NGF_RUNTIME_PNG_DIR:-${ROOT_DIR}/ops/viz/runtime_png}"
STRICT_MISSING="${NGF_VIZ_STRICT_MISSING:-0}"

EXTRA_ARGS=()
if [[ "${STRICT_MISSING}" == "1" ]]; then
  EXTRA_ARGS+=(--strict-missing)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/ops/viz/generate_pipeline_visuals.py" \
  --output-dir "${OUTPUT_DIR}" \
  --clean 1 \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" "${ROOT_DIR}/ops/viz/generate_runtime_pngs.py" \
  --runtime-root "${RUNTIME_ROOT}" \
  --output-dir "${RUNTIME_PNG_DIR}" \
  --clean 1
