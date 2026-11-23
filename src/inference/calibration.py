"""BLIS-Trace-inspired step-level NNLS latency coefficient calibrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import nnls

from inference.schema import LatencyCoeffs

ATTN_QUAD_SCALE = 1e6


class StepObservation(BaseModel):
    """Single observed forward iteration from live vLLM / SGLang step telemetry."""

    prefill_tokens: float = Field(ge=0.0)
    attn_quad_work: float = Field(ge=0.0, description="sum(p_i * c_i) / 1e6")
    decode_tokens: float = Field(ge=0.0)
    kv_read_gb: float = Field(ge=0.0)
    observed_step_ms: float = Field(gt=0.0)


def read_observations_jsonl(path: str | Path) -> list[StepObservation]:
    """Load step observations from JSONL."""
    obs: list[StepObservation] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            obs.append(StepObservation.model_validate(json.loads(stripped)))
    return obs


def fit_latency_coeffs(
    observations: Iterable[StepObservation],
    source_label: str = "nnls_trace_calibrated",
) -> tuple[LatencyCoeffs, dict[str, float]]:
    """Fit non-negative step latency coefficients via NNLS and return validation error metrics."""
    obs_list = list(observations)
    if len(obs_list) < 5:
        raise ValueError("Need at least 5 step observations to fit 5 latency coefficients.")

    design = np.array(
        [
            [
                1.0,
                o.prefill_tokens,
                o.attn_quad_work,
                o.decode_tokens,
                o.kv_read_gb,
            ]
            for o in obs_list
        ],
        dtype=np.float64,
    )
    target = np.array([o.observed_step_ms for o in obs_list], dtype=np.float64)

    weights, _ = nnls(design, target)
    preds = design @ weights
    abs_err = np.abs(preds - target)
    rel_err = abs_err / np.maximum(target, 1e-6)

    ss_res = float(np.sum((target - preds) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    coeffs = LatencyCoeffs(
        alpha_0_ms=round(float(weights[0]), 6),
        alpha_prefill_ms_per_tok=round(float(weights[1]), 6),
        alpha_attn_quad_ms=round(float(weights[2]), 6),
        alpha_decode_ms_per_tok=round(float(weights[3]), 6),
        alpha_kv_ms_per_gb=round(float(weights[4]), 6),
        calibration_source=source_label,
    )
    stats = {
        "num_observations": float(len(obs_list)),
        "mae_ms": round(float(np.mean(abs_err)), 4),
        "mape_pct": round(float(np.mean(rel_err) * 100.0), 4),
        "p95_abs_err_ms": round(float(np.percentile(abs_err, 95)), 4),
        "r2": round(float(r2), 6),
    }
    return coeffs, stats
