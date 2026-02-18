"""Deterministic rerun comparator for artifact snapshot directories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

_METADATA_JSON_KEYS = {
    "generated_at",
    "created_at",
    "updated_at",
    "run_timestamp",
    "metadata_timestamp",
}
_METADATA_CSV_COLUMNS = {
    "generated_at",
    "created_at",
    "updated_at",
    "run_timestamp",
    "metadata_timestamp",
}


@dataclass(frozen=True)
class DeterminismResult:
    """Result object for snapshot comparison."""

    matches: bool
    left_snapshot: Path
    right_snapshot: Path
    left_hash: str
    right_hash: str
    first_diff: str | None
    reason_code: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "left_snapshot": str(self.left_snapshot),
            "right_snapshot": str(self.right_snapshot),
            "left_hash": self.left_hash,
            "right_hash": self.right_hash,
            "first_diff": self.first_diff,
            "reason_code": self.reason_code,
        }

    def to_text(self) -> str:
        if self.matches:
            return (
                "PASS: rerun snapshots are deterministic "
                f"(hash={self.left_hash[:16]})"
            )
        return (
            "FAIL: rerun snapshots differ "
            f"(reason_code={self.reason_code}; first_diff={self.first_diff})"
        )


def _iter_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


def _normalize_json(value: Any, ignored_keys: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _normalize_json(item, ignored_keys)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
            if key not in ignored_keys
        }
    if isinstance(value, list):
        return [_normalize_json(item, ignored_keys) for item in value]
    return value


def _canonical_json_bytes(path: Path, ignored_keys: set[str]) -> bytes:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    normalized = _normalize_json(payload, ignored_keys)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return encoded


def _canonical_csv_frame(path: Path, ignored_columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    dropped = [name for name in frame.columns if name in ignored_columns]
    if dropped:
        frame = frame.drop(columns=dropped)

    ordered_columns = sorted(frame.columns.tolist())
    if ordered_columns:
        frame = (
            frame[ordered_columns].sort_values(ordered_columns).reset_index(drop=True)
        )
    return frame.fillna("")


def _canonical_csv_bytes(path: Path, ignored_columns: set[str]) -> bytes:
    frame = _canonical_csv_frame(path, ignored_columns)
    csv_text = str(frame.to_csv(index=False))
    return csv_text.encode("utf-8")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _first_json_diff(left: Any, right: Any, prefix: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{prefix} type mismatch {type(left).__name__}!={type(right).__name__}"

    if isinstance(left, Mapping):
        left_keys = sorted(left.keys())
        right_keys = sorted(right.keys())
        if left_keys != right_keys:
            return f"{prefix} key mismatch left={left_keys} right={right_keys}"
        for key in left_keys:
            diff = _first_json_diff(left[key], right[key], f"{prefix}.{key}")
            if diff:
                return diff
        return None

    if isinstance(left, list):
        if len(left) != len(right):
            return f"{prefix} length mismatch {len(left)}!={len(right)}"
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            diff = _first_json_diff(left_item, right_item, f"{prefix}[{idx}]")
            if diff:
                return diff
        return None

    if left != right:
        return f"{prefix} value mismatch left={left} right={right}"
    return None


def _first_csv_diff(left: pd.DataFrame, right: pd.DataFrame, path: str) -> str | None:
    left_cols = left.columns.tolist()
    right_cols = right.columns.tolist()
    if left_cols != right_cols:
        return f"{path} column mismatch left={left_cols} right={right_cols}"

    if len(left) != len(right):
        return f"{path} row_count mismatch left={len(left)} right={len(right)}"

    for row_idx in range(len(left)):
        for col in left_cols:
            left_val = str(left.iloc[row_idx][col])
            right_val = str(right.iloc[row_idx][col])
            if left_val != right_val:
                return (
                    f"{path} row={row_idx} column={col} "
                    f"left={left_val} right={right_val}"
                )
    return None


def _aggregate_hash(parts: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for rel_path, file_hash in sorted(parts):
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"|")
        digest.update(file_hash.encode("utf-8"))
    return digest.hexdigest()


def compare_rerun_snapshots(
    left_snapshot: str | Path,
    right_snapshot: str | Path,
    *,
    ignore_json_keys: Iterable[str] | None = None,
    ignore_csv_columns: Iterable[str] | None = None,
) -> DeterminismResult:
    """Compare two artifact snapshots with deterministic hashes and typed diff output."""

    left_root = Path(left_snapshot)
    right_root = Path(right_snapshot)

    left_files = _iter_files(left_root)
    right_files = _iter_files(right_root)

    left_set = {str(path) for path in left_files}
    right_set = {str(path) for path in right_files}
    if left_set != right_set:
        missing_left = sorted(right_set - left_set)
        missing_right = sorted(left_set - right_set)
        first_diff = ""
        if missing_left:
            first_diff = f"missing_in_left={missing_left[0]}"
        elif missing_right:
            first_diff = f"missing_in_right={missing_right[0]}"
        return DeterminismResult(
            matches=False,
            left_snapshot=left_root,
            right_snapshot=right_root,
            left_hash="",
            right_hash="",
            first_diff=first_diff,
            reason_code="file_set_mismatch",
        )

    json_ignored = set(ignore_json_keys or _METADATA_JSON_KEYS)
    csv_ignored = set(ignore_csv_columns or _METADATA_CSV_COLUMNS)

    left_parts: list[tuple[str, str]] = []
    right_parts: list[tuple[str, str]] = []

    for rel_path in left_files:
        left_path = left_root / rel_path
        right_path = right_root / rel_path
        rel_str = str(rel_path)

        if rel_path.suffix.lower() == ".json":
            left_bytes = _canonical_json_bytes(left_path, json_ignored)
            right_bytes = _canonical_json_bytes(right_path, json_ignored)

            if left_bytes != right_bytes:
                left_payload = json.loads(left_bytes.decode("utf-8"))
                right_payload = json.loads(right_bytes.decode("utf-8"))
                diff = _first_json_diff(left_payload, right_payload)
                return DeterminismResult(
                    matches=False,
                    left_snapshot=left_root,
                    right_snapshot=right_root,
                    left_hash=_aggregate_hash(left_parts),
                    right_hash=_aggregate_hash(right_parts),
                    first_diff=f"{rel_str}: {diff}",
                    reason_code="json_value_mismatch",
                )

        elif rel_path.suffix.lower() == ".csv":
            left_frame = _canonical_csv_frame(left_path, csv_ignored)
            right_frame = _canonical_csv_frame(right_path, csv_ignored)
            diff = _first_csv_diff(left_frame, right_frame, rel_str)
            if diff:
                return DeterminismResult(
                    matches=False,
                    left_snapshot=left_root,
                    right_snapshot=right_root,
                    left_hash=_aggregate_hash(left_parts),
                    right_hash=_aggregate_hash(right_parts),
                    first_diff=diff,
                    reason_code="csv_value_mismatch",
                )

            left_bytes = left_frame.to_csv(index=False).encode("utf-8")
            right_bytes = right_frame.to_csv(index=False).encode("utf-8")

        else:
            left_bytes = left_path.read_bytes()
            right_bytes = right_path.read_bytes()
            if left_bytes != right_bytes:
                return DeterminismResult(
                    matches=False,
                    left_snapshot=left_root,
                    right_snapshot=right_root,
                    left_hash=_aggregate_hash(left_parts),
                    right_hash=_aggregate_hash(right_parts),
                    first_diff=f"{rel_str}: binary content mismatch",
                    reason_code="binary_mismatch",
                )

        left_parts.append((rel_str, _hash_bytes(left_bytes)))
        right_parts.append((rel_str, _hash_bytes(right_bytes)))

    left_hash = _aggregate_hash(left_parts)
    right_hash = _aggregate_hash(right_parts)
    return DeterminismResult(
        matches=(left_hash == right_hash),
        left_snapshot=left_root,
        right_snapshot=right_root,
        left_hash=left_hash,
        right_hash=right_hash,
        first_diff=None if left_hash == right_hash else "aggregate hash mismatch",
        reason_code=None if left_hash == right_hash else "hash_mismatch",
    )
