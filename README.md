# `rollout-serve-opt`

**Workload-calibrated LLM inference & RL rollout topology optimizer** — measure trace shapes first (prefix trees, ISL/OSL skew, inter-turn verifier/tool dwell time `t_pause_ms`), fit hardware step-latency coefficients via Non-Negative Least Squares (`scipy.optimize.nnls`), sweep serving topologies across the SLA-constrained `goodput_tokens_per_gpu_hr` Pareto frontier, and gate candidate optimizations behind zero-score completeness, logprob-KL drift, and greedy batch-invariance guardrails (`vLLM`, `SGLang`, `llm-d`, `NVIDIA Dynamo`, `batch_invariant_ops`-aligned).

> **Provenance Notice:** The built-in hardware profiles under `configs/hardware/` are tagged `analytical_prior_uncalibrated`. Use `rollout-serve-opt calibrate` on live `SGLang` / `vLLM` step telemetry (`AIPerf` replay) before citing absolute production numbers.

---

## Architecture & Core Capabilities

1. **Five Canonical Workload Trace Archetypes (`src/inference/workloads.py`):**
   - `short_chat` — bursty arrivals, low ISL/OSL, tight TTFT (500 ms) and ITL (40 ms).
   - `tool_agent` — multi-turn trajectories with monotonically growing prefix trees (`prefix_block_hashes`) and realistic tool dwell pauses (`t_pause_ms` 1.5–12 s).
   - `batch_docs` — high-ISL (4k–12k) throughput sweeps with shared system/rubric prefixes.
   - `reasoning_heavy` — decode memory-bandwidth-bound chain-of-thought generations (1k–2.4k OSL).
   - `rl_group_rollouts` — GRPO-style `1 -> G` parallel rollout groups sharing 100% prompt-prefix hashes with verifier pauses and strict logprob-drift constraints.
   - Also includes a `load_mooncake_trace` converter for standard `(timestamp, input_length, output_length, hash_ids)` traces.

2. **Continuous-Batching & Interconnect-Aware Simulator (`src/inference/simulator.py`, `src/inference/kv_cache.py`):**
   - Iteration-level continuous batching with configurable `chunked_prefill_tokens` (`0` reproduces real unchunked prefill-vs-decode head-of-line stalls).
   - Tenant- and model-revision-scoped block prefix KV cache across **L1 HBM** and **L2 Host RAM** (`pcie_bw_gbps` restore penalty).
   - Explicit **Prefill/Decode (P/D) disaggregation** modeling link handshake + bandwidth transfer delay across **NVLink4** (450 GB/s), **RoCEv2** (40 GB/s), and **TCP Ethernet** (3 GB/s).
   - Multi-pool `workload_shaped_pools` partitioning to evaluate specialization gains vs. bin-packing fragmentation.

3. **NNLS Step Calibrator (`src/inference/calibration.py`):**
   - Recovers non-negative coefficients $(\alpha_0, \alpha_{\text{prefill}}, \alpha_{\text{attn\_quad}}, \alpha_{\text{decode}}, \alpha_{\text{kv}})$ from live GPU step observations (`scipy.optimize.nnls`).

4. **Reproducibility & RL Stability Guardrails (`src/inference/guardrails.py`):**
   - **Zero-Score Completeness Gate:** Incomplete, truncated, or schema-invalid outputs score `0` (prevents trading task validity for raw tokens/s).
   - **Logprob KL & Max-Diff Bounds:** Tracks token-level logprob divergence to prevent silent on-policy RL drift.
   - **Greedy Batch-Invariance Determinism Rate:** Measures output consistency across repeated trials under dynamic batch-size shifts.

---

## Key Physics Bugs Caught & Resolved

| Failure Mode Caught | Root Cause | Fix & Verified Invariant |
| :--- | :--- | :--- |
| **Cold-arrival cache leakage** | Inserting prompt blocks into L1 at admission let simultaneous cold arrivals illegally hit uncomputed KV blocks. | Added `active_uncomputed_blocks` HBM reservation; blocks only enter lookup-visible L1 after prefill completes (`0.0%` hit on simultaneous cold bursts). |
| **Hot-replica pile-up on `rl_group_rollouts`** | Bounded `[0, 1]` queue penalty let `prefix_weight = 0.65` route all parallel group rollouts behind one overloaded GPU. | Replaced bounded score with estimated TTFT latency cost (`est_queue_ms + decode_wait + uncached_prefill + pcie_restore`), balancing locality against backlog. |
| **Interconnect cliff & pool HBM starvation** | Static P/D assumes free KV movement and ignores decode-pool halving on decode-heavy traffic. | Modeled explicit `T_transfer` and per-worker HBM bounds: NVLink4 P/D cuts max ITL from `998.6 -> 27.7 ms`, whereas TCP Ethernet collapses P/D `p99 TTFT` (`43.3 s` vs. `1.83 s` monolithic+APC). |

---

## Synthetic Sweep Results (`mixed_agentic`, `seed=42`, `count=120`)

> All numbers below are generated deterministically by `src/inference/` under `analytical_prior_uncalibrated` hardware priors (`reports/sweep_nvlink.json`, `reports/sweep_eth.json`).

### 8× H100 SXM (`NVLink4`, 450 GB/s)

| Topology | Pareto | SLA Pass | p99 TTFT (ms) | p99 / max ITL (ms) | Goodput (tok/GPU-hr) | L1 / L2 Hit | Avg KV Xfer (ms) | Peak HBM |
| :--- | :---: | :---: | ---: | :---: | ---: | :---: | ---: | ---: |
| `disaggregated_pd` | no | 120/120 | 899.1 | 26.2 / 27.7 | 298,771 | 32.4% / 1.9% | 1.78 | 100.0% |
| `monolithic_apc_prefix` | YES | 119/120 | 868.2 | 17.0 / 169.1 | 364,617 | 34.3% / 0.0% | 0.00 | 100.0% |
| `monolithic_rr` | YES | 120/120 | 1004.8 | 19.1 / 998.6 | 365,919 | 0.0% / 0.0% | 0.00 | 100.0% |
| `workload_shaped_pools` | no | 120/120 | 868.2 | 19.2 / 169.1 | 326,805 | 34.5% / 0.0% | 0.00 | 100.0% |

### 8× L40S (`TCP_Ethernet`, 3 GB/s — Honest Negative Result for P/D)

| Topology | Pareto | SLA Pass | p99 TTFT (ms) | p99 / max ITL (ms) | Goodput (tok/GPU-hr) | L1 / L2 Hit | Avg KV Xfer (ms) | Peak HBM |
| :--- | :---: | :---: | ---: | :---: | ---: | :---: | ---: | ---: |
| `disaggregated_pd` | no | 69/120 | 43302.7 | 54.9 / 58.9 | 50,162 | 35.1% / 0.3% | 242.99 | 100.0% |
| `monolithic_apc_prefix` | YES | 119/120 | 1825.6 | 47.6 / 377.9 | 252,472 | 33.6% / 0.0% | 0.00 | 100.0% |
| `monolithic_rr` | no | 115/120 | 5555.0 | 48.4 / 2103.6 | 240,149 | 0.0% / 0.0% | 0.00 | 100.0% |
| `workload_shaped_pools` | no | 106/120 | 15306.8 | 53.1 / 371.1 | 153,248 | 34.1% / 0.0% | 0.00 | 100.0% |

---

## Quickstart & Live GPU Calibration Track

```bash
# Run verification suite
PYTHONPATH=src python3 -m pytest tests/python -v

# Generate synthetic workload traces
PYTHONPATH=src python3 -m inference.cli generate-traces --workload all --count 120 --seed 42 --out .artifacts/traces

# Sweep candidate serving topologies
PYTHONPATH=src python3 -m inference.cli sweep \
  --trace .artifacts/traces/mixed_agentic.jsonl \
  --hardware configs/hardware/h100_8x_nvlink.yaml \
  --topologies configs/topologies \
  --out reports/sweep_nvlink.json

# Fit real hardware coefficients from live SGLang / vLLM step observations (AIPerf replay)
PYTHONPATH=src python3 -m inference.cli calibrate \
  --observations path/to/step_spans.jsonl \
  --hardware-template configs/hardware/h100_8x_nvlink.yaml \
  --hardware-out configs/hardware/h100_calibrated.yaml \
  --source-label sglang_h100_live_nnls

# Gate candidate configuration with Zero-Score + logprob-KL + batch-invariance checks
PYTHONPATH=src python3 -m inference.cli guardrail \
  --baseline path/to/baseline_eval.jsonl \
  --candidate path/to/candidate_eval.jsonl
```

### Live Validation Roadmap
1. Replay generated `tool_agent` and `rl_group_rollouts` traces against live **SGLang** (RadixAttention + deterministic batch-invariant ops) and **vLLM v1** (`NixlConnector` + APC) via **AIPerf**.
2. Calibrate step coefficients with `calibrate`, verify `<10%` p95 sim-vs-real error, and log batch-invariance determinism & logprob-KL drift via `guardrail`.
3. See [`docs/SPEC.md`](docs/SPEC.md) for the mathematical model, stack alignment, and risk register.
