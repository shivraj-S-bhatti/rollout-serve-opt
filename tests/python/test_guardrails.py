"""Tests for Zero-Score completeness, logprob KL drift, and determinism guardrails."""

from __future__ import annotations

from inference.guardrails import EvalSample, evaluate_guardrails


def _baseline() -> list[EvalSample]:
    return [
        EvalSample(
            sample_id=f"s{i}",
            passed=True,
            completed_cleanly=True,
            schema_valid=True,
            logprobs=[-0.2, -0.35, -0.15, -0.5],
            greedy_trial_outputs=["ok", "ok", "ok"],
        )
        for i in range(10)
    ]


def test_clean_candidate_passes_guardrails() -> None:
    base = _baseline()
    cand = [
        EvalSample(
            sample_id=f"s{i}",
            passed=True,
            completed_cleanly=True,
            schema_valid=True,
            logprobs=[-0.205, -0.348, -0.152, -0.503],
            greedy_trial_outputs=["ok", "ok", "ok"],
        )
        for i in range(10)
    ]
    report = evaluate_guardrails(base, cand)
    assert report.passed is True
    assert report.zero_score_violations == 0
    assert report.greedy_determinism_rate == 1.0


def test_incomplete_or_schema_invalid_triggers_zero_score_failure() -> None:
    base = _baseline()
    cand = _baseline()
    cand[0] = cand[0].model_copy(update={"completed_cleanly": False, "passed": True})
    report = evaluate_guardrails(base, cand)
    assert report.passed is False
    assert report.zero_score_violations == 1
    assert report.candidate_pass_rate == 0.9


def test_logprob_kl_drift_failure() -> None:
    base = _baseline()
    cand = _baseline()
    for i in range(len(cand)):
        cand[i] = cand[i].model_copy(update={"logprobs": [-1.8, -2.1, -1.5, -2.4]})
    report = evaluate_guardrails(base, cand, max_mean_kl=0.02)
    assert report.passed is False
    assert any("mean_logprob_kl" in msg for msg in report.failure_reasons)


def test_nondeterministic_greedy_trials_fail_guardrail() -> None:
    base = _baseline()
    cand = _baseline()
    cand[0] = cand[0].model_copy(update={"greedy_trial_outputs": ["ans_a", "ans_b", "ans_a"]})
    report = evaluate_guardrails(base, cand, min_determinism_rate=0.99)
    assert report.passed is False
    assert report.greedy_determinism_rate == 0.9
    assert any("greedy_determinism_rate" in msg for msg in report.failure_reasons)
