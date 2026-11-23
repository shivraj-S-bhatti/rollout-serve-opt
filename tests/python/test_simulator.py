"""Physics and security invariant tests for InferenceSimulator."""

from __future__ import annotations

from pathlib import Path
import yaml

from inference.schema import HardwareProfile, TopologyConfig, TraceRequest
from inference.simulator import InferenceSimulator
from inference.workloads import generate_rl_group_rollouts, generate_tool_agent, generate_workload

ROOT = Path(__file__).resolve().parents[2]


def _load_hw(name: str) -> HardwareProfile:
    raw = yaml.safe_load((ROOT / "configs" / "hardware" / f"{name}.yaml").read_text(encoding="utf-8"))
    return HardwareProfile.model_validate(raw)


def _load_topo(name: str) -> TopologyConfig:
    raw = yaml.safe_load((ROOT / "configs" / "topologies" / f"{name}.yaml").read_text(encoding="utf-8"))
    return TopologyConfig.model_validate(raw)


def test_tenant_isolation_and_zero_unique_cache_hits() -> None:
    hw = _load_hw("h100_8x_nvlink")
    topo = _load_topo("monolithic_apc_prefix").model_copy(update={"num_gpus": 1})
    shared_hashes = [f"shared_blk_{i}" for i in range(16)]

    reqs = [
        TraceRequest(
            request_id="r0",
            session_id="s0",
            tenant_id="tenant_alpha",
            workload_class="short_chat",
            arrival_ms=0.0,
            input_tokens=1024,
            output_tokens=64,
            prefix_block_hashes=shared_hashes,
            ttft_sla_ms=1000.0,
            itl_sla_ms=50.0,
        ),
        TraceRequest(
            request_id="r1",
            session_id="s1",
            tenant_id="tenant_beta",
            workload_class="short_chat",
            arrival_ms=5000.0,
            input_tokens=1024,
            output_tokens=64,
            prefix_block_hashes=shared_hashes,
            ttft_sla_ms=1000.0,
            itl_sla_ms=50.0,
        ),
    ]
    m = InferenceSimulator(hw, topo).run(reqs)
    assert m.l1_prefix_hit_rate == 0.0
    assert m.l2_prefix_hit_rate == 0.0
    assert m.calibration_source == "analytical_prior_uncalibrated"


def test_cold_simultaneous_arrivals_cannot_see_future_cache_fills() -> None:
    hw = _load_hw("h100_8x_nvlink")
    topo = _load_topo("monolithic_apc_prefix").model_copy(update={"num_gpus": 1})
    shared_hashes = [f"cold_blk_{i}" for i in range(64)]
    reqs = [
        TraceRequest(
            request_id=f"cold_{i}",
            session_id=f"cold_s_{i}",
            tenant_id="tenant_same",
            workload_class="short_chat",
            arrival_ms=0.0,
            input_tokens=4096,
            output_tokens=64,
            prefix_block_hashes=shared_hashes,
            ttft_sla_ms=2000.0,
            itl_sla_ms=80.0,
        )
        for i in range(2)
    ]
    m = InferenceSimulator(hw, topo).run(reqs)
    assert m.l1_prefix_hit_rate == 0.0
    assert m.l2_prefix_hit_rate == 0.0


def test_prefix_aware_beats_round_robin_on_tool_and_rl_workloads() -> None:
    hw = _load_hw("h100_8x_nvlink")
    rr = _load_topo("monolithic_rr")
    apc = _load_topo("monolithic_apc_prefix")

    for trace in (
        generate_tool_agent(count=90, seed=42),
        generate_rl_group_rollouts(count=80, seed=42, group_size=8),
    ):
        m_rr = InferenceSimulator(hw, rr).run(trace)
        m_apc = InferenceSimulator(hw, apc).run(trace)
        assert (m_apc.l1_prefix_hit_rate + m_apc.l2_prefix_hit_rate) > 0.35
        assert m_rr.l1_prefix_hit_rate == 0.0
        assert m_apc.p99_ttft_ms < m_rr.p99_ttft_ms


def test_disaggregated_pd_reduces_decode_itl_interference_on_nvlink() -> None:
    hw = _load_hw("h100_8x_nvlink")
    rr_unchunked = _load_topo("monolithic_rr")
    pd = _load_topo("disaggregated_pd")
    trace = generate_workload("mixed_agentic", count=100, seed=42)

    m_rr = InferenceSimulator(hw, rr_unchunked).run(trace)
    m_pd = InferenceSimulator(hw, pd).run(trace)
    assert m_pd.p999_itl_ms < m_rr.p999_itl_ms
    assert m_pd.max_itl_ms < m_rr.max_itl_ms
    assert m_pd.avg_kv_transfer_ms < 15.0

    # Under higher sustained arrival concurrency (~9 RPS), unchunked prefills drive p99.9 / max ITL spikes
    burst_trace = [r.model_copy(update={"arrival_ms": round(r.arrival_ms * 0.2, 3)}) for r in trace]
    m_rr_burst = InferenceSimulator(hw, rr_unchunked).run(burst_trace)
    m_pd_burst = InferenceSimulator(hw, pd).run(burst_trace)
    assert m_pd_burst.p999_itl_ms < m_rr_burst.p999_itl_ms
    assert m_pd_burst.max_itl_ms < m_rr_burst.max_itl_ms


def test_honest_negative_result_pd_loses_ttft_on_slow_ethernet() -> None:
    hw_eth = _load_hw("l40s_multi_eth")
    apc = _load_topo("monolithic_apc_prefix")
    pd = _load_topo("disaggregated_pd")
    trace = generate_workload("mixed_agentic", count=100, seed=42)

    m_apc = InferenceSimulator(hw_eth, apc).run(trace)
    m_pd = InferenceSimulator(hw_eth, pd).run(trace)
    assert m_pd.avg_kv_transfer_ms > 200.0
    assert m_pd.p99_ttft_ms > m_apc.p99_ttft_ms
