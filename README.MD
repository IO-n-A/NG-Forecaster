# NG-Forecaster

A production-focused forecasting system for monthly U.S. dry natural gas production, built around policy-safe release timing, champion/challenger model evaluation, regime-aware fusion, and reproducible operations.

The project provides:

- **Monthly nowcasting with release-policy control** (target month is derived from as-of date and publication lag rules)
- **Champion/challenger model stack** (`wpd_lstm_one_layer`, `wpd_vmd_lstm1`, `wpd_vmd_lstm2`, `challenger_bsts`)
- **Fusion policy search and ablation support** (including BSTS on/off and regime-aware calibration policy)
- **First-class interval evaluation** (coverage, WIS, pinball, calibration tables)
- **Operational QA and leakage auditing** (target-month gate, preprocess gate, policy checks, no-future-data checks)
- **Visualization pipeline** with Plotly HTML + PNG exports and Coconut palette standardization

# Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [File Structure](#file-structure)
- [Setup and Configuration](#setup-and-configuration)
- [Usage Workflows](#usage-workflows)
- [Evaluation Methodology](#evaluation-methodology)
- [Latest Operational Results](#latest-operational-results)
- [Visualization Outputs](#visualization-outputs)
- [Troubleshooting](#troubleshooting)

# Key Features

## Policy-safe monthly target resolution

- As-of dates are converted into valid target months using release-calendar constraints.
- Replay metadata is explicit (`target_month_start/end`, `asof_start/end`, `asof_schedule`) for auditability.

## Champion/challenger integration

- Standard validation bundles include challenger BSTS.
- Direct challenger token usage is supported in 24-run replay commands.

## Regime-aware fusion and calibration

- Fusion supports explicit constraint paths such as BSTS-off and multi-shock gating controls.
- Calibration policy is searchable (enable/weight/cap) and can be selected by regime.

## Interval quality as a primary signal

- Interval exports include:
  - `interval_scorecard.csv`
  - `interval_scorecard_by_regime.csv`
  - `interval_calibration_table.csv`
  - `interval_calibration_by_regime.csv`

## Operations and quality gates

- Built-in checks include policy readiness, preprocess integrity, and adoption-readiness checks.
- Dedicated leakage audit script validates that features and targets do not use future information.

## Visualization and reporting

- Standard chart pack in `ops/viz/output` is generated as both `.html` and `.png`.
- Runtime artifact plot pack in `ops/viz/runtime_png` renders one PNG per CSV/JSON report artifact.

# Architecture Overview

The system follows a layered research-to-operations architecture:

- **Data Layer**: retrieval, migration, and monthly feature panel refresh
- **Model Layer**: champion/challenger forecasting and fusion runtime
- **Evaluation Layer**: rolling 24-run validation, DM statistics, interval metrics, ablations
- **QA Layer**: gate checks, policy audits, leakage detection
- **Ops Layer**: orchestrated pipeline execution, artifact packaging, visual reporting

Core implementation files:

- `src/ng_forecaster/evaluation/validation_24m.py`
- `src/ng_forecaster/models/fusion.py`
- `src/ng_forecaster/evaluation/interval_metrics.py`
- `src/ng_forecaster/evaluation/bsts_marginal_benefit.py`
- `scripts/ops/wpdlstm2_sanity_report.py`
- `scripts/qa/leakage_audit.py`
- `ops/viz/generate_pipeline_visuals.py`
- `ops/viz/generate_runtime_pngs.py`

# File Structure

```text
NG-Forecaster/
├── README.MD
├── main.py
├── pyproject.toml
├── requirements-dev.txt
├── mypy.ini
├── configs/
│   └── experiments/
├── src/
│   └── ng_forecaster/
├── scripts/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── ops/
│   ├── orchestration/
│   └── qa/
├── ops/
│   ├── run_workflow.sh
│   ├── run_quality_suite.sh
│   ├── run_visuals.sh
│   └── viz/
│       ├── generate_pipeline_visuals.py
│       ├── generate_runtime_pngs.py
│       ├── output/
│       └── runtime_png/
└── tests/
```

# Setup and Configuration

## Requirements

- Python 3.10+
- Local virtual environment (recommended: `.venv-forecast`)
- API keys for full online retrieval:
  - `EIA_API_KEY` (recommended for production/API mode)
  - `NASA_EARTHDATA_TOKEN` (only for NASA weather retrieval scripts)

## Installation

```bash
python3 -m venv .venv-forecast
source .venv-forecast/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
pip install pandas numpy scipy pyyaml requests python-dateutil pytest pytest-cov scikit-learn statsmodels pymc arviz
```

## Environment

Optional but recommended:

```bash
export PYTHONPATH=src
export TZ=UTC
export EIA_API_KEY="<your_key>"
export NASA_EARTHDATA_TOKEN="<your_token_if_used>"
```

For repo-local configuration, you can also place these keys in `/ops/ops.env`
(copy from `/ops/ops.env.example`):

```bash
EIA_API_KEY="<your_key>"
NASA_EARTHDATA_TOKEN="<your_token_if_used>"
```

# Usage Workflows

## 1) Full pipeline run (without Airflow scheduler)

```bash
python3 main.py --asof 2026-02-14
```

This runs ingestion, nowcast, backtest, ablation, QA gates, and visualization.

## 2) 24-run model replay and scorecards

```bash
PYTHONPATH=src .venv-forecast/bin/python scripts/evaluation/run_24_month_validation.py \
  --variants full_plus_regime \
  --runs 24 \
  --regime_split 1 \
  --report-root data/reports/validation_latest
```

## 3) BSTS enabled vs forced-off marginal benefit

```bash
PYTHONPATH=src .venv-forecast/bin/python -m ng_forecaster.evaluation.bsts_marginal_benefit \
  --enabled_root data/reports/fusion_ablation/enabled \
  --forced_off_root data/reports/fusion_ablation/forced_off \
  --out_root data/reports/fusion_ablation
```

## 4) Sanity and leakage checks

```bash
PYTHONPATH=src .venv-forecast/bin/python scripts/ops/wpdlstm2_sanity_report.py --out_root data/reports/wpdlstm2_sanity
PYTHONPATH=src .venv-forecast/bin/python scripts/qa/leakage_audit.py --runs 24 --report-root data/reports/wpdlstm2_sanity
```

## 5) Visualization regeneration

```bash
./ops/run_visuals.sh
```

# Evaluation Methodology

The evaluation workflow is evidence-first and fixed-window reproducible:

- **Window**: 24 monthly runs with explicit as-of schedule and target-month bounds
- **Model metrics**: MAPE, MAE, RMSE (overall and by regime)
- **Statistical tests**: DM one-sided improve p-values and two-sided p-values
- **Interval metrics**: empirical coverage, WIS-like score, pinball losses
- **Regime diagnostics**: freeze_off, multi_shock, transfer_dispersion slices
- **Ablation and attribution**: policy and block-level impact tables by regime
- **Leakage control**: timestamp checks across features/availability/targets

# Latest Operational Results

Latest validated 24-run evidence shows:

- BSTS enabled-vs-off deltas (`enabled - forced_off`, MAPE points):
  - `wpd_lstm_one_layer`: `-0.0502`
  - `wpd_vmd_lstm1`: `-0.0484`
  - `wpd_vmd_lstm2`: `-1.9392`
- Promoted replay MAPE snapshot:
  - `challenger_bsts`: `1.8806`
  - `wpd_lstm_one_layer`: `0.9684`
  - `wpd_vmd_lstm1`: `1.0465`
  - `wpd_vmd_lstm2`: `3.0584`
- WPD-VMD-LSTM2 sanity check improved from `4.6115` to `2.8292` MAPE on identical replay window.
- Leakage audit reported `0` violations across `24` runs.

# Visualization Outputs

## Standard charts

Will be generated in `ops/viz/output` as both HTML and PNG, including:

- `validation_24m_point_vs_official_releases.*`
- `validation_24m_point_vs_release_36m.*`
- `nowcast_latest_vs_previous_delta.*`
- `ablation_mae.*`
- `ablation_dm_pvalue.*`
- `dm_policy_adjusted_pvalue.*`
- `backtest_rmse.*`

## Runtime artifact plot library

- Output path: `ops/viz/runtime_png`
- Source: report CSV/JSON artifacts
- Index files:
  - `ops/viz/runtime_png/runtime_png_manifest.json`

# Troubleshooting

- If validation or fusion runs fail, confirm `PYTHONPATH=src` is set.
- If visual outputs are incomplete, inspect `ops/viz/output/visualization_summary.json` for missing-source diagnostics.
- If retrieval scripts fail, verify API tokens in environment (`EIA_API_KEY`, `NASA_EARTHDATA_TOKEN` where needed).
- If QA gates fail after `main.py`, inspect `data/reports/pipeline_run_summary.json` gate payloads and rerun with focused script-level checks from `scripts/qa/`.
