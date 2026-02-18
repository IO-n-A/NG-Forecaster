#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/ops/ops.env"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-forecast/bin/python}"
PYTEST_BIN="${PYTEST_BIN:-${ROOT_DIR}/.venv-forecast/bin/pytest}"
RUFF_BIN="${RUFF_BIN:-${ROOT_DIR}/.venv-forecast/bin/ruff}"
MYPY_BIN="${MYPY_BIN:-${ROOT_DIR}/.venv-forecast/bin/mypy}"

cd "${ROOT_DIR}"

"${PYTEST_BIN}" tests/leakage tests/data_contracts tests/features tests/models tests/evaluation tests/orchestration -q
"${PYTEST_BIN}" tests/integration -q

"${PYTHON_BIN}" scripts/qa/check_preprocess_gate.py
"${PYTHON_BIN}" scripts/qa/check_target_month_gate.py
"${PYTHON_BIN}" scripts/qa/check_n4_acceptance.py
"${PYTHON_BIN}" scripts/qa/check_n5_policy_audit.py
"${PYTHON_BIN}" scripts/qa/check_n6_adoption_ready.py
"${PYTHON_BIN}" scripts/qa/check_ops_readiness.py

"${RUFF_BIN}" check .
"${MYPY_BIN}"

printf 'PASS: quality suite complete\n'
