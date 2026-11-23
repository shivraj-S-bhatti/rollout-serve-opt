# `rollout-serve-opt` — Architecture & Modeling Spec

> **Core Thesis:** Measure workload shape first (prefix trees, ISL/OSL skew, inter-turn tool/verifier dwell time `t_pause_ms`), calibrate an offline digital twin via Non-Negative Least Squares (NNLS) step regression against live engine traces, sweep the serving Pareto frontier under strict `p99 TTFT / ITL` SLAs and **Zero-Score completeness & logprob-determinism guardrails**, and validate top candidates on live `SGLang` / `vLLM v1` endpoints.

---

## 1. Stack Alignment (2026)

| Layer | Target Ecosystem | Implementation in `rollout-serve-opt` |
| :--- | :--- | :--- |
| **Trace Format** | Mooncake traces + OpenTelemetry spans | Privacy-safe JSONL (`session_id`, `tenant_id`, `model_revision`, `group_id`, `turn_idx`, `prefix_block_hashes[]`, `t_pause_ms`, SLAs, `task_ok`). Zero raw prompt text stored. |
| **Offline Calibrated Twin** | `llm-d` BLIS (`inference-sim` + NNLS trace calibration) & NVIDIA `AIConfigurator` | `src/inference/` implements NNLS step-latency calibration, continuous batching, chunked prefill, L1/L2 KV tiering, P/D transfer physics, and Pareto search. |
| **Data Plane** | **SGLang** (RadixAttention, HiCache, deterministic mode) & **vLLM v1** (`NixlConnector`, APC) | Topologies directly model chunked prefill, APC, L2 Host-RAM offload, latency-aware prefix routing, and P/D disaggregation. |
| **Load Generation** | NVIDIA **`AIPerf`** | External replay driver for Stage 2 live step-observation collection and A/B trace replay. |
| **Reproducibility & RL Guardrails** | `batch_invariant_ops`-style determinism + logprob KL + Zero-Score completeness | `src/inference/guardrails.py` enforces pass-rate bounds, `mean_logprob_kl`, `max_logprob_abs_diff`, and greedy batch-invariance rate. |

---

## 2. Five Canonical Workload Archetypes

1. `short_chat` — Low ISL (128–768), bursty arrivals, short OSL (64–256), minimal prefix reuse, tight TTFT (500 ms) and ITL (40 ms).
2. `tool_agent` — Multi-turn agent trajectories (5 turns/session) appending tool outputs onto monotonically growing `prefix_block_hashes`, separated by `t_pause_ms` (1.5–12 s) testing HBM residency vs L2 offload.
3. `batch_docs` — High ISL (4k–12k), shared rubric prefix, short OSL (64–256), relaxed TTFT (4000 ms), throughput-bound.
4. `reasoning_heavy` — Long chain-of-thought generations (1024–2400 OSL), decode memory-bandwidth bound, tight ITL (50 ms).
5. `rl_group_rollouts` — GRPO-style post-training batches (`group_id`) where 1 shared prompt fans out to $G=8$ rollouts sharing 100% of prompt blocks, with verifier dwell pauses and strict logprob consistency.

---

## 3. Mathematical Step-Latency, KV & Interconnect Model

For an active iteration batch $B_t$ on a worker:

$$T_{\text{step}}(B_t) = \alpha_0 + \alpha_p \sum_{i \in B_t} p_i + \alpha_{\text{attn}} \frac{\sum_{i \in B_t} p_i \cdot c_i}{10^6} + \alpha_d \sum_{i \in B_t} d_i + \alpha_{\text{kv}} \cdot \text{KV\_read\_GB}(B_t)$$

where $p_i$ is uncached prefill tokens in step $t$ (capped by `chunked_prefill_tokens`), $c_i$ is context length attended over, $d_i \in \{0,1\}$ is decode tokens in step $t$, and $\text{KV\_read\_GB}(B_t)$ is total active KV bytes streamed from HBM. Coefficients $(\alpha_0, \alpha_p, \alpha_{\text{attn}}, \alpha_d, \alpha_{\text{kv}}) \ge 0$ are recovered via `scipy.optimize.nnls`.

For **L2 Host-RAM restore** (`l2_ram_offload=True`), L2 prefix hits bypass prefill compute at the cost of PCIe restore time:
$$T_{\text{pcie}} = \frac{\text{L2\_hit\_blocks} \cdot \text{block\_tokens} \cdot \text{kv\_gb\_per\_token}}{\text{pcie\_bw\_gbps}} \times 1000\text{ ms}$$

For **Disaggregated P/D**, cross-pool KV handoff adds explicit interconnect latency:
$$T_{\text{transfer}} = T_{\text{handshake}} + \frac{\text{transfer\_tokens} \cdot \text{kv\_gb\_per\_token}}{\text{interconnect\_bw\_gbps}} \times 1000\text{ ms}$$

---

## 4. Risk Register & Anti-Signals

- **`[CONFIDENCE: High]`** NNLS step calibration, prefix-aware latency-cost routing + chunked prefill on `tool_agent` / `rl_group_rollouts`, L2 RAM tiering under multi-second `t_pause_ms`, and Zero-Score + logprob-KL guardrails.
- **`[CONFIDENCE: Medium]`** Net P/D disaggregation gains—requires fast RDMA/NVLink interconnects AND sufficient prefill-vs-decode interference that splitting pools does not starve decode HBM capacity.
- **`[CONFIDENCE: Low]`** Static `workload_shaped_pools` at $\le 8\text{–}16$ GPUs—bin-packing fragmentation often drops `goodput_tokens_per_gpu_hr` below a shared continuous-batching pool.
- **`[ANTI-SIGNALS]`** Never quote synthetic priors as measured GPU speedups; never benchmark against an APC-disabled strawman without also comparing against `monolithic_apc_prefix`; never trade task completeness (`Zero-Score`) for raw tokens/s.
