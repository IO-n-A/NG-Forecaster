#!/usr/bin/env python3
"""Generate PNG charts for Sprint runtime outputs using Plotly Coconut palette."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import warnings

import pandas as pd
import pypalettes  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]


def _load_coconut_palette() -> list[str]:
    raw = pypalettes.load_palette("Coconut", keep_first_n=6)
    base = [str(color)[:7] for color in raw]
    return (base * 3)[:18]


_COCONUT = _load_coconut_palette()


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _slugify(path: Path) -> str:
    raw = path.as_posix().replace("/", "__")
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    raw = raw.replace(".csv", "").replace(".json", "")
    return raw.strip("._") or "runtime_output"


def _is_datetime_series(series: pd.Series) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    valid_ratio = float(parsed.notna().mean()) if len(parsed) > 0 else 0.0
    return valid_ratio >= 0.7


def _pick_datetime_col(frame: pd.DataFrame) -> str | None:
    priority = [
        "target_month",
        "timestamp",
        "asof",
        "run_end",
        "vintage_month",
        "month",
        "date",
    ]
    for col in priority:
        if col in frame.columns and _is_datetime_series(frame[col]):
            return str(col)
    for col in frame.columns:
        if _is_datetime_series(frame[col]):
            return str(col)
    return None


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in frame.columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().sum() > 0:
            columns.append(str(col))
    return columns


def _pick_categorical(frame: pd.DataFrame, exclude: set[str]) -> str | None:
    for col in frame.columns:
        if col in exclude:
            continue
        if str(frame[col].dtype) in {"object", "string", "category"}:
            return str(col)
    return None


def _apply_layout(
    fig: go.Figure,
    *,
    title: str,
    x_title: str,
    y_title: str,
) -> None:
    fig.update_layout(
        template="plotly_white",
        colorway=_COCONUT,
        title={"text": title, "x": 0.02},
        hovermode="x unified",
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "traceorder": "normal",
        },
        margin={"l": 70, "r": 220, "t": 70, "b": 60},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")


def _error_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 15},
    )
    return fig


def _table_figure(frame: pd.DataFrame, *, title: str) -> go.Figure:
    if frame.empty:
        return _error_figure("No rows available")
    preview = frame.head(20).copy()
    preview = preview.fillna("")
    for col in preview.columns:
        preview[col] = preview[col].astype(str)
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": list(preview.columns),
                    "fill_color": _COCONUT[0],
                    "font": {"color": "white", "size": 12},
                    "align": "left",
                },
                cells={
                    "values": [preview[col].tolist() for col in preview.columns],
                    "fill_color": "#f8fafc",
                    "align": "left",
                },
            )
        ]
    )
    _apply_layout(fig, title=title, x_title="", y_title="")
    return fig


def _build_csv_figure(source: Path) -> tuple[go.Figure, str]:
    frame = pd.read_csv(source)
    if frame.empty:
        return _error_figure("CSV is empty"), "empty"

    dt_col = _pick_datetime_col(frame)
    num_cols = _numeric_columns(frame)
    kind = "table"

    if dt_col and num_cols:
        plot = frame.copy()
        plot[dt_col] = pd.to_datetime(plot[dt_col], errors="coerce")
        plot = plot.dropna(subset=[dt_col]).sort_values(dt_col)
        fig = go.Figure()
        for idx, col in enumerate(num_cols[:6]):
            values = pd.to_numeric(plot[col], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=plot[dt_col],
                    y=values,
                    mode="lines+markers",
                    name=col,
                    line={"width": 2.2, "color": _COCONUT[idx % len(_COCONUT)]},
                    marker={"size": 6},
                )
            )
        _apply_layout(
            fig,
            title=f"{source.name} (time series)",
            x_title=dt_col,
            y_title="value",
        )
        return fig, "line"

    if num_cols:
        cat_col = _pick_categorical(frame, set(num_cols))
        if cat_col:
            plot = frame[[cat_col, num_cols[0]]].copy()
            plot[num_cols[0]] = pd.to_numeric(plot[num_cols[0]], errors="coerce")
            plot = (
                plot.dropna(subset=[cat_col, num_cols[0]])
                .groupby(cat_col, as_index=False)[num_cols[0]]
                .mean()
                .sort_values(num_cols[0], ascending=False)
                .head(30)
            )
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=plot[cat_col],
                        y=plot[num_cols[0]],
                        marker={"color": _COCONUT[1]},
                        name=num_cols[0],
                    )
                ]
            )
            _apply_layout(
                fig,
                title=f"{source.name} (category summary)",
                x_title=cat_col,
                y_title=num_cols[0],
            )
            return fig, "bar"

    table = _table_figure(frame, title=f"{source.name} (table preview)")
    return table, kind


def _json_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        rows = []
        for key, value in payload.items():
            rows.append(
                {
                    "key": str(key),
                    "value": (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else str(value)
                    ),
                }
            )
        return pd.DataFrame(rows)
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return pd.DataFrame(payload)
        return pd.DataFrame({"value": [str(item) for item in payload]})
    return pd.DataFrame({"value": [str(payload)]})


def _build_json_figure(source: Path) -> tuple[go.Figure, str]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    frame = _json_to_frame(payload)
    if frame.empty:
        return _error_figure("JSON has no rows"), "empty"
    if {"key", "value"}.issubset(frame.columns):
        table = _table_figure(frame, title=f"{source.name} (json key/value)")
        return table, "table"

    dt_col = _pick_datetime_col(frame)
    num_cols = _numeric_columns(frame)
    if dt_col and num_cols:
        plot = frame.copy()
        plot[dt_col] = pd.to_datetime(plot[dt_col], errors="coerce")
        plot = plot.dropna(subset=[dt_col]).sort_values(dt_col)
        fig = go.Figure()
        for idx, col in enumerate(num_cols[:6]):
            fig.add_trace(
                go.Scatter(
                    x=plot[dt_col],
                    y=pd.to_numeric(plot[col], errors="coerce"),
                    mode="lines+markers",
                    name=col,
                    line={"width": 2.2, "color": _COCONUT[idx % len(_COCONUT)]},
                    marker={"size": 6},
                )
            )
        _apply_layout(
            fig,
            title=f"{source.name} (json time series)",
            x_title=dt_col,
            y_title="value",
        )
        return fig, "line"

    return _table_figure(frame, title=f"{source.name} (json table preview)"), "table"


def _write_png(fig: go.Figure, out_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fig.write_image(out_path, format="png", scale=2, width=1600, height=900)


def _cleanup_stale(output_dir: Path, keep_files: set[str]) -> None:
    for candidate in output_dir.iterdir():
        if not candidate.is_file():
            continue
        if candidate.name in keep_files:
            continue
        if candidate.suffix not in {".png", ".json", ".md"}:
            continue
        candidate.unlink()


def generate_runtime_pngs(
    *,
    runtime_root: str | Path = "data/reports/sprint9",
    output_dir: str | Path = "ops/viz/runtime_png",
    clean: bool = True,
) -> dict[str, Any]:
    source_root = Path(runtime_root)
    out = _ensure_dir(output_dir)

    sources = sorted(
        [
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}
        ]
    )
    generated: list[dict[str, Any]] = []

    for source in sources:
        relative = source.relative_to(source_root)
        png_name = f"{_slugify(relative)}.png"
        png_path = out / png_name

        status = "ok"
        kind = "unknown"
        try:
            if source.suffix.lower() == ".csv":
                fig, kind = _build_csv_figure(source)
            else:
                fig, kind = _build_json_figure(source)
            _write_png(fig, png_path)
        except Exception as exc:
            status = "error"
            kind = "error"
            error_fig = _error_figure(f"Failed to render {source.name}: {exc}")
            _apply_layout(
                error_fig,
                title=f"{source.name} (render error)",
                x_title="",
                y_title="",
            )
            _write_png(error_fig, png_path)

        generated.append(
            {
                "source": str(source),
                "relative_source": relative.as_posix(),
                "png": str(png_path),
                "kind": kind,
                "status": status,
            }
        )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "runtime_root": str(source_root),
        "output_dir": str(out),
        "source_count": int(len(sources)),
        "png_count": int(len(generated)),
        "error_count": int(sum(1 for row in generated if row["status"] != "ok")),
        "generated": generated,
    }

    manifest_path = out / "runtime_png_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8"
    )

    dashboard_lines = [
        "# Runtime Output PNG Dashboard",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Runtime root: `{source_root}`",
        f"- Source files processed: `{summary['source_count']}`",
        f"- PNG files generated: `{summary['png_count']}`",
        f"- Render errors: `{summary['error_count']}`",
        "",
        "## Files",
    ]
    for row in generated:
        dashboard_lines.append(
            f"- `{row['relative_source']}` -> `{Path(str(row['png'])).as_posix()}` (`{row['kind']}`, `{row['status']}`)"
        )
    dashboard_path = out / "runtime_png_dashboard.md"
    dashboard_path.write_text("\n".join(dashboard_lines) + "\n", encoding="utf-8")

    if clean:
        keep = {Path(str(row["png"])).name for row in generated}
        keep.add(manifest_path.name)
        keep.add(dashboard_path.name)
        _cleanup_stale(out, keep)

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", default="data/reports/sprint9")
    parser.add_argument("--output-dir", default="ops/viz/runtime_png")
    parser.add_argument(
        "--clean",
        type=int,
        default=1,
        choices=(0, 1),
        help="Set to 1 to remove stale generated files in output-dir.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = generate_runtime_pngs(
        runtime_root=args.runtime_root,
        output_dir=args.output_dir,
        clean=bool(int(args.clean)),
    )
    payload = {
        "runtime_root": summary["runtime_root"],
        "output_dir": summary["output_dir"],
        "source_count": summary["source_count"],
        "png_count": summary["png_count"],
        "error_count": summary["error_count"],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
