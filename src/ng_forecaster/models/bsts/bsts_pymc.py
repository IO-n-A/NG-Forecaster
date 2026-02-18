"""PyMC-backed BSTS challenger backend with posterior export support."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation

# ArviZ defaults to user-home cache. In restricted runtimes that path may be
# read-only, so route cache writes to a repo-local fallback unless already set.
repo_root = Path(__file__).resolve().parents[4]
if "XDG_CACHE_HOME" not in os.environ:
    cache_root = repo_root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)

# PyTensor (PyMC backend) writes compiled artifacts under user home by default.
# In restricted environments, pin a local compiledir if caller did not provide one.
pytensor_root = repo_root / ".pytensor"
pytensor_root.mkdir(parents=True, exist_ok=True)
_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "").strip()
if "base_compiledir=" not in _pytensor_flags:
    if _pytensor_flags:
        os.environ["PYTENSOR_FLAGS"] = (
            f"{_pytensor_flags},base_compiledir={pytensor_root}"
        )
    else:
        os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={pytensor_root}"

az: Any
pm: Any
try:
    az = importlib.import_module("arviz")
    pm = importlib.import_module("pymc")
except Exception:  # pragma: no cover - optional dependency
    az = None
    pm = None


@dataclass(frozen=True)
class PyMCBSTSResult:
    """Output payload for PyMC BSTS backend."""

    forecast: pd.DataFrame
    state_summary: dict[str, float]
    diagnostics: dict[str, Any]


def _ensure_pymc() -> None:
    if pm is None or az is None:
        raise ContractViolation(
            "missing_dependency",
            key="pymc",
            detail=(
                "PyMC backend requested but pymc/arviz are not installed; "
                "install optional dependencies or switch challenger engine"
            ),
        )


def _forecast_quantiles(alpha: float) -> tuple[float, float]:
    lower = (1.0 - float(alpha)) / 2.0
    upper = 1.0 - lower
    return float(lower), float(upper)


def _coerce_regressors(
    regressors: pd.DataFrame | None,
    *,
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if regressors is None or regressors.empty:
        return np.zeros((n_obs, 0), dtype=float), np.zeros((0,), dtype=float), []

    reg = regressors.copy()
    for column in list(reg.columns):
        reg[column] = pd.to_numeric(reg[column], errors="coerce")
    reg = reg.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(reg) < n_obs:
        raise ContractViolation(
            "insufficient_training_data",
            key="challenger_pymc_regressors",
            detail=(
                "regressor design matrix must have at least as many rows as "
                f"observations; rows={len(reg)} observations={n_obs}"
            ),
        )
    reg = reg.tail(n_obs).reset_index(drop=True)
    matrix = reg.to_numpy(dtype=float)
    future = reg.iloc[-1].to_numpy(dtype=float) if matrix.shape[1] > 0 else np.zeros(0)
    return matrix, future, [str(value) for value in reg.columns.tolist()]


def fit_bsts_with_pymc(
    observed: pd.Series,
    *,
    timestamps: pd.Series,
    horizons: Sequence[int],
    alpha: float,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    regressors: pd.DataFrame | None = None,
    steo_observation: pd.Series | None = None,
    random_seed: int = 42,
    artifact_dir: str | Path | None = None,
) -> PyMCBSTSResult:
    """Fit a PyMC structural model and emit horizon forecasts."""

    _ensure_pymc()
    y = pd.to_numeric(observed, errors="coerce").dropna().astype(float).to_numpy()
    if len(y) < 24:
        raise ContractViolation(
            "insufficient_training_data",
            key="challenger_pymc",
            detail="PyMC BSTS backend requires at least 24 observations",
        )
    ts = pd.to_datetime(timestamps, errors="coerce")
    if ts.isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key="timestamps",
            detail="challenger PyMC backend received invalid timestamps",
        )
    n_obs = int(len(y))
    ts = ts.iloc[-n_obs:].reset_index(drop=True)

    mean = float(np.mean(y))
    std = float(np.std(y))
    if std <= 1e-12:
        std = 1.0
    y_scaled = (y - mean) / std
    x = np.arange(n_obs, dtype=float)
    month_idx = ts.dt.month.to_numpy(dtype=int) - 1
    x_reg, future_reg, reg_columns = _coerce_regressors(regressors, n_obs=n_obs)
    n_regressors = int(x_reg.shape[1])
    steo_scaled: np.ndarray | None = None
    if steo_observation is not None:
        steo = pd.to_numeric(steo_observation, errors="coerce")
        if len(steo) >= n_obs:
            steo = steo.tail(n_obs).reset_index(drop=True)
            if steo.notna().all():
                steo_scaled = ((steo.to_numpy(dtype=float) - mean) / std).astype(float)

    with pm.Model():
        sigma_level = pm.HalfNormal("sigma_level", sigma=0.25)
        sigma_slope = pm.HalfNormal("sigma_slope", sigma=0.1)
        level = pm.GaussianRandomWalk("level", sigma=sigma_level, shape=n_obs)
        slope = pm.GaussianRandomWalk("slope", sigma=sigma_slope, shape=n_obs)
        seasonal_sigma = pm.HalfNormal("seasonal_sigma", sigma=0.2)
        seasonal = pm.GaussianRandomWalk(
            "seasonal_month",
            sigma=seasonal_sigma,
            shape=12,
        )
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Exponential("nu_minus_two", lam=1.0) + 2.0
        mu = level + slope * x + seasonal[month_idx]
        beta = None
        if n_regressors > 0:
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
            beta = pm.Normal("beta", mu=0.0, sigma=sigma_beta, shape=n_regressors)
            mu = mu + pm.math.dot(x_reg, beta)
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y_scaled)
        if steo_scaled is not None:
            sigma_steo = pm.HalfNormal("sigma_steo", sigma=1.0)
            pm.Normal("y_steo_obs", mu=mu, sigma=sigma_steo, observed=steo_scaled)
        idata = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=1,
            target_accept=0.9,
            progressbar=False,
            return_inferencedata=True,
            random_seed=int(random_seed),
        )

    posterior = idata.posterior
    level_draws = posterior["level"].values.reshape(-1, n_obs)
    slope_draws = posterior["slope"].values.reshape(-1, n_obs)
    seasonal_draws = posterior["seasonal_month"].values.reshape(-1, 12)
    sigma_draws = posterior["sigma"].values.reshape(-1)
    nu_draws = posterior["nu_minus_two"].values.reshape(-1) + 2.0
    beta_draws = (
        posterior["beta"].values.reshape(-1, n_regressors)
        if n_regressors > 0 and "beta" in posterior.data_vars
        else None
    )

    lower_q, upper_q = _forecast_quantiles(alpha)
    parsed_horizons = sorted({int(h) for h in horizons})
    rows: list[dict[str, Any]] = []
    for horizon in parsed_horizons:
        future_index = float(horizon)
        future_month = (pd.Timestamp(ts.iloc[-1]).to_period("M") + horizon).month - 1
        mu_draws = (
            level_draws[:, -1]
            + slope_draws[:, -1] * future_index
            + seasonal_draws[:, int(future_month)]
        )
        if beta_draws is not None:
            mu_draws = mu_draws + (beta_draws * future_reg).sum(axis=1)
        scale_draws = np.maximum(sigma_draws, 1e-6)
        df_draws = np.maximum(nu_draws, 2.01)
        rng = np.random.default_rng(int(random_seed) + int(horizon) * 97)
        predictive = mu_draws + rng.standard_t(df=df_draws) * scale_draws
        mean_forecast = float(np.mean(predictive) * std + mean)
        lower = float(np.quantile(predictive, lower_q) * std + mean)
        upper = float(np.quantile(predictive, upper_q) * std + mean)
        rows.append(
            {
                "horizon": int(horizon),
                "mean_forecast": mean_forecast,
                "lower_95": lower,
                "upper_95": upper,
                "residual_scale": float(np.mean(scale_draws) * std),
            }
        )

    posterior_path = ""
    if artifact_dir is not None:
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        posterior_file = root / "posterior.nc"
        idata.to_netcdf(posterior_file)
        posterior_path = str(posterior_file)

    state_summary = {
        "center": mean,
        "scale": std,
        "level_last_mean": float(np.mean(level_draws[:, -1])),
        "slope_last_mean": float(np.mean(slope_draws[:, -1])),
        "sigma_mean": float(np.mean(sigma_draws)),
        "nu_mean": float(np.mean(nu_draws)),
        "n_regressors": float(n_regressors),
    }
    diagnostics = {
        "engine": "pymc",
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "random_seed": int(random_seed),
        "posterior_path": posterior_path,
        "observations": int(len(y)),
        "regressor_columns": reg_columns,
        "measurement_equation_steo": bool(steo_scaled is not None),
    }
    return PyMCBSTSResult(
        forecast=pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True),
        state_summary=state_summary,
        diagnostics=diagnostics,
    )
