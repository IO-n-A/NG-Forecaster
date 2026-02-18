"""Weekly orchestration runtime helpers for Airflow-compatible DAG execution."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ng_forecaster.errors import ContractViolation


@dataclass(frozen=True)
class RetryPolicy:
    """Retry/backoff policy for orchestration tasks."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.2
    backoff_multiplier: float = 2.0
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class TaskSpec:
    """Static DAG task contract used by tests and optional Airflow binding."""

    task_id: str
    upstream_task_ids: tuple[str, ...] = field(default_factory=tuple)
    retries: int = 3
    retry_delay_seconds: float = 0.2


@dataclass(frozen=True)
class DAGSpec:
    """Static DAG contract with weekly schedule policy."""

    dag_id: str
    schedule: str
    task_specs: tuple[TaskSpec, ...]


@dataclass(frozen=True)
class SecretValue:
    """Secret payload with redacted representation for logging."""

    name: str
    value: str
    redacted: str


@dataclass(frozen=True)
class TransientHTTPError(RuntimeError):
    """Exception raised for retryable HTTP status conditions."""

    status_code: int
    detail: str

    def __str__(self) -> str:
        return f"status_code={self.status_code}; detail={self.detail}"


def _redact_secret(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def resolve_secret(
    *,
    name: str,
    env_var: str,
    required: bool,
) -> SecretValue:
    """Resolve secret from environment with redacted logging output."""

    value = os.getenv(env_var, "")
    if required and not value:
        raise ContractViolation(
            "missing_secret",
            key=name,
            detail=f"required secret env var is not set: {env_var}",
        )
    return SecretValue(name=name, value=value, redacted=_redact_secret(value))


def build_idempotency_key(
    *,
    dag_id: str,
    task_id: str,
    context: Mapping[str, Any],
) -> str:
    """Build deterministic idempotency key from DAG/task context."""

    canonical = json.dumps(
        dict(context), sort_keys=True, default=str, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{dag_id}|{task_id}|{canonical}".encode("utf-8")
    ).hexdigest()
    return digest


def idempotency_marker_path(
    *,
    dag_id: str,
    task_id: str,
    key: str,
    root: str | Path = "data/artifacts/orchestration/idempotency",
) -> Path:
    """Resolve marker path for a task idempotency key."""

    return Path(root) / dag_id / task_id / f"{key}.json"


def write_idempotency_marker(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist task completion marker for idempotent reruns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )


def run_with_retries(
    fn: Callable[[], Any],
    *,
    retry_policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Execute callable with exponential backoff on transient HTTP errors."""

    attempt = 1
    delay = max(0.0, float(retry_policy.initial_delay_seconds))

    while True:
        try:
            return fn()
        except TransientHTTPError as exc:
            if (
                exc.status_code not in retry_policy.retry_status_codes
                or attempt >= retry_policy.max_attempts
            ):
                raise
            sleep(delay)
            delay = delay * retry_policy.backoff_multiplier
            attempt += 1


def run_task_with_reliability(
    *,
    dag_id: str,
    task_id: str,
    context: Mapping[str, Any],
    task_fn: Callable[[], Mapping[str, Any]],
    retry_policy: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Run task with idempotency and retry/backoff reliability wiring."""

    policy = retry_policy or RetryPolicy()
    key = build_idempotency_key(dag_id=dag_id, task_id=task_id, context=context)
    marker = idempotency_marker_path(dag_id=dag_id, task_id=task_id, key=key)

    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return {
            "task_id": task_id,
            "idempotency_key": key,
            "status": "skipped",
            "result": payload.get("result", {}),
        }

    def _runner() -> dict[str, Any]:
        result = dict(task_fn())
        write_idempotency_marker(
            marker,
            {
                "dag_id": dag_id,
                "task_id": task_id,
                "idempotency_key": key,
                "status": "done",
                "result": result,
            },
        )
        return {
            "task_id": task_id,
            "idempotency_key": key,
            "status": "done",
            "result": result,
        }

    return dict(run_with_retries(_runner, retry_policy=policy))


def execute_task_graph(
    *,
    dag_spec: DAGSpec,
    context: Mapping[str, Any],
    task_functions: Mapping[
        str, Callable[[dict[str, dict[str, Any]]], Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    """Execute task graph in declared order with dependency, retry, and idempotency guards."""

    task_results: dict[str, dict[str, Any]] = {}
    for task_spec in dag_spec.task_specs:
        if task_spec.task_id not in task_functions:
            raise ContractViolation(
                "missing_task_callable",
                key=task_spec.task_id,
                detail=f"task callable is not registered for DAG {dag_spec.dag_id}",
            )

        for upstream_task_id in task_spec.upstream_task_ids:
            if upstream_task_id not in task_results:
                raise ContractViolation(
                    "invalid_task_dependency",
                    key=task_spec.task_id,
                    detail=(
                        "upstream task has not completed before dependent task run: "
                        f"{upstream_task_id}"
                    ),
                )

        task_context = {
            **dict(context),
            "dag_id": dag_spec.dag_id,
            "task_id": task_spec.task_id,
        }

        def _run() -> Mapping[str, Any]:
            return task_functions[task_spec.task_id](task_results)

        result = run_task_with_reliability(
            dag_id=dag_spec.dag_id,
            task_id=task_spec.task_id,
            context=task_context,
            task_fn=lambda: dict(_run()),
            retry_policy=RetryPolicy(
                max_attempts=int(task_spec.retries),
                initial_delay_seconds=float(task_spec.retry_delay_seconds),
                backoff_multiplier=2.0,
            ),
        )
        task_results[task_spec.task_id] = dict(result["result"])

    return {
        "dag_id": dag_spec.dag_id,
        "schedule": dag_spec.schedule,
        "tasks": task_results,
    }
