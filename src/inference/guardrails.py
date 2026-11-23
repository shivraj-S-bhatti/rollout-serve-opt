"""Zero-Score completeness, logprob-KL drift, and batch-invariance determinism guardrails."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from inference.schema import GuardrailReport


class EvalSample(BaseModel):
    """Paired evaluation outcome with completeness, logprob vector/scalar, and greedy outputs."""

    sample_id: str
    passed: bool
    completed_cleanly: bool = True
    schema_valid: bool = True
    logprobs: list[float] = Field(default_factory=list)
    greedy_trial_outputs: list[str] = Field(default_factory=list)


def read_eval_samples_jsonl(path: str | Path) -> list[EvalSample]:
    """Load evaluation samples from JSONL."""
    samples: list[EvalSample] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            samples.append(EvalSample.model_validate(json.loads(stripped)))
    return samples


def _effective_zero_score_pass(sample: EvalSample) -> bool:
    """Zero-Score guardrail: incomplete, truncated, or schema-invalid outputs score 0."""
    return bool(sample.passed and sample.completed_cleanly and sample.schema_valid)


def _mean_kl_approx(p_logprobs: list[float], q_logprobs: list[float]) -> tuple[float, float]:
    """Compute token-level symmetric KL proxy and max absolute logprob difference."""
    if not p_logprobs or not q_logprobs:
        return 0.0, 0.0
    n = min(len(p_logprobs), len(q_logprobs))
    kls: list[float] = []
    max_diff = 0.0
    for i in range(n):
        lp = p_logprobs[i]
        lq = q_logprobs[i]
        diff = abs(lp - lq)
        if diff > max_diff:
            max_diff = diff
        p = min(1.0, max(1e-12, math.exp(lp)))
        q = min(1.0, max(1e-12, math.exp(lq)))
        sym_kl = 0.5 * (p * (lp - lq) + q * (lq - lp))
        kls.append(max(0.0, sym_kl))
    return sum(kls) / max(1, len(kls)), max_diff


def evaluate_guardrails(
    baseline_samples: Iterable[EvalSample],
    candidate_samples: Iterable[EvalSample],
    max_pass_rate_drop: float = 0.01,
    max_mean_kl: float = 0.02,
    min_determinism_rate: float = 0.99,
) -> GuardrailReport:
    """Validate candidate config against Zero-Score completeness, logprob drift, and batch invariance."""
    base_list = list(baseline_samples)
    cand_list = list(candidate_samples)
    if not base_list or not cand_list:
        raise ValueError("Both baseline_samples and candidate_samples must be non-empty.")

    base_pass_rate = sum(1 for s in base_list if _effective_zero_score_pass(s)) / len(base_list)
    cand_pass_rate = sum(1 for s in cand_list if _effective_zero_score_pass(s)) / len(cand_list)
    pass_rate_delta = cand_pass_rate - base_pass_rate

    zero_score_violations = sum(
        1 for s in cand_list if not (s.completed_cleanly and s.schema_valid)
    )

    base_by_id = {s.sample_id: s for s in base_list}
    kl_values: list[float] = []
    max_abs_diff = 0.0
    for cand in cand_list:
        base = base_by_id.get(cand.sample_id)
        if base is not None and base.logprobs and cand.logprobs:
            kl, abs_d = _mean_kl_approx(base.logprobs, cand.logprobs)
            kl_values.append(kl)
            if abs_d > max_abs_diff:
                max_abs_diff = abs_d

    mean_logprob_kl = sum(kl_values) / max(1, len(kl_values))

    det_checked = 0
    det_consistent = 0
    for cand in cand_list:
        if len(cand.greedy_trial_outputs) >= 2:
            det_checked += 1
            if len(set(cand.greedy_trial_outputs)) == 1:
                det_consistent += 1
    greedy_determinism_rate = (
        det_consistent / det_checked if det_checked > 0 else 1.0
    )

    failures: list[str] = []
    if zero_score_violations > 0:
        failures.append(
            f"zero_score_violations={zero_score_violations} (incomplete or schema-invalid outputs)"
        )
    if pass_rate_delta < -max_pass_rate_drop:
        failures.append(
            f"pass_rate_delta={pass_rate_delta:.4f} below allowed -{max_pass_rate_drop:.4f}"
        )
    if mean_logprob_kl > max_mean_kl:
        failures.append(
            f"mean_logprob_kl={mean_logprob_kl:.5f} exceeds max {max_mean_kl:.5f}"
        )
    if greedy_determinism_rate < min_determinism_rate:
        failures.append(
            f"greedy_determinism_rate={greedy_determinism_rate:.4f} below min {min_determinism_rate:.4f}"
        )

    return GuardrailReport(
        passed=len(failures) == 0,
        baseline_pass_rate=round(base_pass_rate, 4),
        candidate_pass_rate=round(cand_pass_rate, 4),
        pass_rate_delta=round(pass_rate_delta, 4),
        zero_score_violations=zero_score_violations,
        mean_logprob_kl=round(mean_logprob_kl, 6),
        max_logprob_abs_diff=round(max_abs_diff, 6),
        greedy_determinism_rate=round(greedy_determinism_rate, 4),
        failure_reasons=failures,
    )
