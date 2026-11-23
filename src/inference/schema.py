"""Typed Pydantic schemas for traces, hardware calibration, topologies, metrics, and guardrails."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator

WorkloadClass = Literal[
    "short_chat",
    "tool_agent",
    "batch_docs",
    "reasoning_heavy",
    "rl_group_rollouts",
]


class TraceRequest(BaseModel):
    """Privacy-safe inference request trace record (no raw prompt text)."""

    request_id: str
    session_id: str
    tenant_id: str = "tenant_default"
    model_revision: str = "model_r1"
    group_id: str | None = None
    turn_idx: int = Field(default=0, ge=0)
    workload_class: WorkloadClass
    arrival_ms: float = Field(ge=0.0)
    t_pause_ms: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    prefix_block_hashes: list[str] = Field(default_factory=list)
    ttft_sla_ms: float = Field(gt=0.0)
    itl_sla_ms: float = Field(gt=0.0)
    task_ok: bool = True


class LatencyCoeffs(BaseModel):
    """Non-negative step-latency regression coefficients (NNLS calibrated or analytical prior)."""

    alpha_0_ms: float = Field(ge=0.0)
    alpha_prefill_ms_per_tok: float = Field(ge=0.0)
    alpha_attn_quad_ms: float = Field(ge=0.0)
    alpha_decode_ms_per_tok: float = Field(ge=0.0)
    alpha_kv_ms_per_gb: float = Field(ge=0.0)
    calibration_source: str = "analytical_prior_uncalibrated"


class HardwareProfile(BaseModel):
    """Cluster GPU memory, PCIe, interconnect, and latency-coefficient profile."""

    name: str
    description: str = ""
    gpu_model: str = "H100-SXM-80GB"
    num_gpus: int = Field(gt=0)
    kv_hbm_gb_per_gpu: float = Field(gt=0.0)
    host_ram_kv_gb_per_gpu: float = Field(ge=0.0)
    pcie_bw_gbps: float = Field(gt=0.0)
    interconnect_name: str
    interconnect_bw_gbps: float = Field(gt=0.0)
    transfer_handshake_ms: float = Field(ge=0.0)
    kv_gb_per_token: float = Field(gt=0.0)
    block_size_tokens: int = Field(default=64, gt=0)
    coeffs: LatencyCoeffs

    @property
    def l1_capacity_blocks(self) -> int:
        """Number of KV cache blocks that fit in per-GPU L1 HBM."""
        gb_per_block = self.block_size_tokens * self.kv_gb_per_token
        return max(1, int(self.kv_hbm_gb_per_gpu / gb_per_block))

    @property
    def l2_capacity_blocks(self) -> int:
        """Number of KV cache blocks that fit in per-GPU L2 Host RAM."""
        if self.host_ram_kv_gb_per_gpu <= 0.0:
            return 0
        gb_per_block = self.block_size_tokens * self.kv_gb_per_token
        return max(0, int(self.host_ram_kv_gb_per_gpu / gb_per_block))


class TopologyConfig(BaseModel):
    """Candidate serving cluster topology, batching, routing, and tiering configuration."""

    name: str
    description: str = ""
    mode: Literal["monolithic", "disaggregated", "workload_shaped"]
    routing_policy: Literal["round_robin", "least_loaded", "prefix_aware"]
    num_gpus: int = Field(gt=0)
    prefill_gpus: int = Field(default=0, ge=0)
    decode_gpus: int = Field(default=0, ge=0)
    pool_allocation: dict[str, int] = Field(default_factory=dict)
    max_num_seqs: int = Field(default=64, gt=0)
    max_num_batched_tokens: int = Field(default=4096, gt=0)
    chunked_prefill_tokens: int = Field(default=2048, ge=0)
    prefix_caching: bool = True
    l2_ram_offload: bool = False
    prefix_weight: float = Field(default=0.65, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_mode_allocations(self) -> "TopologyConfig":
        if self.mode == "disaggregated":
            if self.prefill_gpus <= 0 or self.decode_gpus <= 0:
                raise ValueError("Disaggregated topology requires prefill_gpus > 0 and decode_gpus > 0.")
            if self.prefill_gpus + self.decode_gpus != self.num_gpus:
                raise ValueError("prefill_gpus + decode_gpus must equal num_gpus.")
        elif self.mode == "workload_shaped":
            if not self.pool_allocation or sum(self.pool_allocation.values()) != self.num_gpus:
                raise ValueError("workload_shaped pool_allocation must sum to num_gpus.")
            if any(v <= 0 for v in self.pool_allocation.values()):
                raise ValueError("All pool allocations must be positive integers.")
        return self


class SimMetrics(BaseModel):
    """Summary serving, SLA, goodput, and cache-efficiency metrics from a simulation run."""

    topology_name: str
    hardware_name: str
    calibration_source: str
    total_requests: int
    completed_requests: int
    oom_rejected_requests: int = 0
    sla_compliant_requests: int
    zero_score_failed_requests: int
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    p50_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    p999_itl_ms: float = 0.0
    max_itl_ms: float = 0.0
    total_output_tokens: int
    sim_duration_s: float
    gpu_hours: float
    aggregate_tok_per_s: float
    goodput_tasks_per_gpu_hr: float
    goodput_tokens_per_gpu_hr: float
    l1_prefix_hit_rate: float
    l2_prefix_hit_rate: float
    avg_kv_transfer_ms: float
    peak_hbm_utilization: float
    cost_per_completed_task_gpu_s: float


class GuardrailReport(BaseModel):
    """Quality, completeness, logprob-drift, and determinism guardrail verdict."""

    passed: bool
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_delta: float
    zero_score_violations: int
    mean_logprob_kl: float
    max_logprob_abs_diff: float
    greedy_determinism_rate: float
    failure_reasons: list[str] = Field(default_factory=list)
