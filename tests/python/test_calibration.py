"""Tests for NNLS step-latency calibration."""

from __future__ import annotations

import random

from inference.calibration import StepObservation, fit_latency_coeffs


def test_nnls_recovers_ground_truth_step_coefficients() -> None:
    rng = random.Random(2026)
    true_a0 = 2.0
    true_ap = 0.068
    true_aq = 1.9
    true_ad = 0.36
    true_akv = 2.5

    obs: list[StepObservation] = []
    for _ in range(160):
        prefill = float(rng.choice([0, 0, 256, 512, 1024, 2048, 4096]))
        ctx = prefill * rng.uniform(1.0, 3.0) if prefill > 0 else 0.0
        attn_quad = (prefill * ctx) / 1e6
        decode = float(rng.randint(0, 48))
        kv_gb = decode * rng.uniform(0.2, 1.8)
        noise = rng.gauss(0.0, 0.25)
        y = max(
            0.5,
            true_a0
            + true_ap * prefill
            + true_aq * attn_quad
            + true_ad * decode
            + true_akv * kv_gb
            + noise,
        )
        obs.append(
            StepObservation(
                prefill_tokens=prefill,
                attn_quad_work=attn_quad,
                decode_tokens=decode,
                kv_read_gb=kv_gb,
                observed_step_ms=y,
            )
        )

    coeffs, stats = fit_latency_coeffs(obs, source_label="test_h100_trace")
    assert coeffs.calibration_source == "test_h100_trace"
    assert stats["mape_pct"] < 5.0
    assert stats["r2"] > 0.98
    assert abs(coeffs.alpha_prefill_ms_per_tok - true_ap) < 0.005
    assert abs(coeffs.alpha_attn_quad_ms - true_aq) < 0.25
