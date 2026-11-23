"""Discrete-event continuous-batching, KV-tiering, and P/D disaggregated serving simulator."""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from inference.calibration import ATTN_QUAD_SCALE
from inference.kv_cache import BlockKVCache
from inference.schema import HardwareProfile, SimMetrics, TopologyConfig, TraceRequest

WorkerRole = Literal["mono", "prefill", "decode"]

WORKLOAD_TO_POOL: dict[str, str] = {
    "tool_agent": "context_reuse",
    "rl_group_rollouts": "context_reuse",
    "short_chat": "low_latency",
    "reasoning_heavy": "low_latency",
    "batch_docs": "batch_throughput",
}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return float(ordered[idx])


@dataclass
class _RequestState:
    req: TraceRequest
    prompt_keys: list[str]
    all_keys: list[str]
    remaining_prefill: int
    cached_context_tokens: int = 0
    decoded_so_far: int = 0
    reserved_blocks: int = 0
    prefill_admitted: bool = False
    decode_admitted: bool = False
    pending_pcie_ms: float = 0.0
    ttft_ms: float | None = None
    completed_ms: float | None = None
    itl_samples: list[float] = field(default_factory=list)


class _Worker:
    def __init__(
        self,
        worker_id: str,
        role: WorkerRole,
        pool_name: str,
        hw: HardwareProfile,
        topo: TopologyConfig,
    ) -> None:
        self.worker_id = worker_id
        self.role = role
        self.pool_name = pool_name
        self.hw = hw
        self.topo = topo
        self.cache = BlockKVCache(
            l1_capacity_blocks=hw.l1_capacity_blocks,
            l2_capacity_blocks=hw.l2_capacity_blocks,
            l2_ram_offload=topo.l2_ram_offload if topo.prefix_caching else False,
        )
        self.waiting_prefill: deque[_RequestState] = deque()
        self.active_prefill: list[_RequestState] = []
        self.waiting_decode: deque[_RequestState] = deque()
        self.active_decode: list[_RequestState] = []
        self.stepping: bool = False
        self.busy_until_ms: float = 0.0

    def queued_and_active_tokens(self) -> int:
        waiting = sum(r.remaining_prefill for r in self.waiting_prefill) + sum(
            r.req.output_tokens for r in self.waiting_decode
        )
        active = sum(r.remaining_prefill for r in self.active_prefill) + sum(
            max(0, r.req.output_tokens - r.decoded_so_far) for r in self.active_decode
        )
        return waiting + active

    def active_count(self) -> int:
        return len(self.active_prefill) + len(self.active_decode)


class InferenceSimulator:
    """Calibrated discrete-event inference simulator with continuous batching and KV tiering."""

    def __init__(self, hardware: HardwareProfile, topology: TopologyConfig) -> None:
        self.hw = hardware
        self.topo = topology
        self.workers: list[_Worker] = []
        self.prefill_workers: list[_Worker] = []
        self.decode_workers: list[_Worker] = []
        self.pools: dict[str, list[_Worker]] = {}
        self._rr_cursors: dict[str, int] = {}
        self._init_workers()

    def _init_workers(self) -> None:
        if self.topo.mode == "monolithic":
            for i in range(self.topo.num_gpus):
                w = _Worker(f"gpu-{i}", "mono", "mono", self.hw, self.topo)
                self.workers.append(w)
            self.pools["mono"] = self.workers
        elif self.topo.mode == "disaggregated":
            for i in range(self.topo.prefill_gpus):
                w = _Worker(f"prefill-{i}", "prefill", "prefill", self.hw, self.topo)
                self.workers.append(w)
                self.prefill_workers.append(w)
            for j in range(self.topo.decode_gpus):
                w = _Worker(f"decode-{j}", "decode", "decode", self.hw, self.topo)
                self.workers.append(w)
                self.decode_workers.append(w)
            self.pools["prefill"] = self.prefill_workers
            self.pools["decode"] = self.decode_workers
        elif self.topo.mode == "workload_shaped":
            idx = 0
            for pool_name, count in self.topo.pool_allocation.items():
                pool_list: list[_Worker] = []
                for _ in range(count):
                    w = _Worker(f"{pool_name}-{idx}", "mono", pool_name, self.hw, self.topo)
                    self.workers.append(w)
                    pool_list.append(w)
                    idx += 1
                self.pools[pool_name] = pool_list

    def _build_keys(self, req: TraceRequest) -> tuple[list[str], list[str]]:
        if self.topo.prefix_caching and req.prefix_block_hashes:
            prompt_keys = list(req.prefix_block_hashes)
        else:
            n_prompt_blocks = max(1, math.ceil(req.input_tokens / self.hw.block_size_tokens))
            prompt_keys = [f"nocache:{req.request_id}:p:{i}" for i in range(n_prompt_blocks)]

        n_out_blocks = max(1, math.ceil(req.output_tokens / self.hw.block_size_tokens))
        out_keys = [f"out:{req.request_id}:d:{i}" for i in range(n_out_blocks)]
        return prompt_keys, prompt_keys + out_keys

    def _eligible_pool(self, req: TraceRequest) -> tuple[str, list[_Worker]]:
        if self.topo.mode == "monolithic":
            return "mono", self.pools["mono"]
        if self.topo.mode == "disaggregated":
            return "prefill", self.pools["prefill"]
        target_pool = WORKLOAD_TO_POOL.get(req.workload_class, "low_latency")
        if target_pool in self.pools:
            return target_pool, self.pools[target_pool]
        fallback_name = max(self.pools, key=lambda k: len(self.pools[k]))
        return fallback_name, self.pools[fallback_name]

    def _select_worker(
        self,
        pool_key: str,
        candidates: list[_Worker],
        req: TraceRequest,
        prompt_keys: list[str],
        now_ms: float,
    ) -> _Worker:
        if self.topo.routing_policy == "round_robin":
            cursor = self._rr_cursors.get(pool_key, 0)
            chosen = candidates[cursor % len(candidates)]
            self._rr_cursors[pool_key] = cursor + 1
            return chosen

        if self.topo.routing_policy == "least_loaded" or not self.topo.prefix_caching:
            return min(candidates, key=lambda w: (w.queued_and_active_tokens(), w.worker_id))

        coeffs = self.hw.coeffs
        pw = self.topo.prefix_weight

        def _estimated_cost_ms(w: _Worker) -> tuple[float, str]:
            rem_step_ms = max(0.0, w.busy_until_ms - now_ms)
            queued_prefill_toks = sum(r.remaining_prefill for r in w.waiting_prefill) + sum(
                r.remaining_prefill for r in w.active_prefill
            )
            est_queue_ms = rem_step_ms + queued_prefill_toks * coeffs.alpha_prefill_ms_per_tok
            decode_step_cost = sum(
                coeffs.alpha_decode_ms_per_tok
                + (r.req.input_tokens + r.decoded_so_far)
                * self.hw.kv_gb_per_token
                * coeffs.alpha_kv_ms_per_gb
                for r in w.active_decode
            )
            decode_wait_penalty = (1.0 - pw) * decode_step_cost * 8.0 + len(w.waiting_decode) * 25.0

            l1_hits, l2_hits = w.cache.lookup_prefix(req.tenant_id, req.model_revision, prompt_keys)
            cached_toks = min(req.input_tokens - 1, (l1_hits + l2_hits) * self.hw.block_size_tokens)
            uncached_toks = max(1, req.input_tokens - cached_toks)
            est_own_prefill_ms = (
                uncached_toks * coeffs.alpha_prefill_ms_per_tok
                + (uncached_toks * req.input_tokens / ATTN_QUAD_SCALE) * coeffs.alpha_attn_quad_ms
            )
            est_pcie_ms = (
                (l2_hits * self.hw.block_size_tokens * self.hw.kv_gb_per_token)
                / self.hw.pcie_bw_gbps
            ) * 1000.0
            total_cost = est_queue_ms + decode_wait_penalty + est_own_prefill_ms + est_pcie_ms
            return (total_cost, w.worker_id)

        return min(candidates, key=_estimated_cost_ms)

    def run(self, trace: list[TraceRequest]) -> SimMetrics:
        """Replay a trace deterministically across the configured topology and hardware."""
        requests = sorted(trace, key=lambda r: (r.arrival_ms, r.request_id))
        if not requests:
            raise ValueError("Cannot simulate an empty trace.")

        event_heap: list[tuple[float, int, str, object]] = []
        seq_counter = 0

        def push_event(at_ms: float, kind: str, payload: object) -> None:
            nonlocal seq_counter
            heapq.heappush(event_heap, (at_ms, seq_counter, kind, payload))
            seq_counter += 1

        states_by_id: dict[str, _RequestState] = {}
        oom_rejected = 0
        total_prompt_blocks_checked = 0
        total_l1_hit_blocks = 0
        total_l2_hit_blocks = 0
        kv_transfers_ms: list[float] = []

        for req in requests:
            prompt_keys, all_keys = self._build_keys(req)
            st = _RequestState(
                req=req,
                prompt_keys=prompt_keys,
                all_keys=all_keys,
                remaining_prefill=req.input_tokens,
            )
            states_by_id[req.request_id] = st
            push_event(req.arrival_ms, "ARRIVE", st)

        def start_worker_step(w: _Worker, now_ms: float) -> None:
            nonlocal oom_rejected, total_prompt_blocks_checked, total_l1_hit_blocks, total_l2_hit_blocks

            # Admit waiting prefill requests by reserving uncomputed HBM capacity without exposing blocks early
            while w.waiting_prefill and w.active_count() < self.topo.max_num_seqs:
                cand = w.waiting_prefill[0]
                reserve_count = len(set(cand.prompt_keys if w.role == "prefill" else cand.all_keys))
                if not w.cache.can_reserve_uncomputed(reserve_count):
                    if w.active_count() == 0:
                        w.waiting_prefill.popleft()
                        oom_rejected += 1
                        continue
                    break

                w.waiting_prefill.popleft()
                if self.topo.prefix_caching:
                    l1_hits, l2_hits = w.cache.lookup_prefix(
                        cand.req.tenant_id, cand.req.model_revision, cand.prompt_keys
                    )
                else:
                    l1_hits, l2_hits = 0, 0

                total_prompt_blocks_checked += len(cand.prompt_keys)
                total_l1_hit_blocks += l1_hits
                total_l2_hit_blocks += l2_hits

                cached_tokens = min(
                    cand.req.input_tokens - 1,
                    (l1_hits + l2_hits) * self.hw.block_size_tokens,
                )
                cand.cached_context_tokens = cached_tokens
                cand.remaining_prefill = max(1, cand.req.input_tokens - cached_tokens)
                cand.pending_pcie_ms = (
                    (l2_hits * self.hw.block_size_tokens * self.hw.kv_gb_per_token)
                    / self.hw.pcie_bw_gbps
                ) * 1000.0
                cand.reserved_blocks = reserve_count
                cand.prefill_admitted = True
                w.cache.reserve_uncomputed(reserve_count)
                w.active_prefill.append(cand)

            # Admit waiting decode requests (only in disaggregated decode role)
            while w.waiting_decode and w.active_count() < self.topo.max_num_seqs:
                cand = w.waiting_decode[0]
                if not w.cache.available_l1_capacity_for_pinning(
                    cand.req.tenant_id, cand.req.model_revision, cand.all_keys
                ):
                    if w.active_count() == 0:
                        w.waiting_decode.popleft()
                        oom_rejected += 1
                        continue
                    break
                w.waiting_decode.popleft()
                cand.decode_admitted = True
                w.cache.pin_and_insert(cand.req.tenant_id, cand.req.model_revision, cand.all_keys)
                w.active_decode.append(cand)

            if not w.active_prefill and not w.active_decode:
                w.stepping = False
                return

            decode_tokens_step = len(w.active_decode)
            kv_read_gb = 0.0
            for d_st in w.active_decode:
                seq_len = d_st.req.input_tokens + d_st.decoded_so_far
                kv_read_gb += seq_len * self.hw.kv_gb_per_token

            prefill_tokens_step = 0
            attn_quad_work = 0.0
            pcie_ms_step = 0.0
            prefill_allocs: list[tuple[_RequestState, int]] = []

            if self.topo.chunked_prefill_tokens == 0:
                for p_st in w.active_prefill:
                    chunk = p_st.remaining_prefill
                    ctx = p_st.cached_context_tokens + chunk
                    prefill_tokens_step += chunk
                    attn_quad_work += (chunk * ctx) / ATTN_QUAD_SCALE
                    pcie_ms_step += p_st.pending_pcie_ms
                    p_st.pending_pcie_ms = 0.0
                    prefill_allocs.append((p_st, chunk))
            else:
                budget = max(
                    1,
                    min(
                        self.topo.chunked_prefill_tokens,
                        self.topo.max_num_batched_tokens - decode_tokens_step,
                    ),
                )
                for p_st in w.active_prefill:
                    if budget <= 0:
                        break
                    chunk = min(p_st.remaining_prefill, budget)
                    ctx = p_st.cached_context_tokens + chunk
                    prefill_tokens_step += chunk
                    attn_quad_work += (chunk * ctx) / ATTN_QUAD_SCALE
                    pcie_ms_step += p_st.pending_pcie_ms
                    p_st.pending_pcie_ms = 0.0
                    prefill_allocs.append((p_st, chunk))
                    budget -= chunk

            coeffs = self.hw.coeffs
            step_ms = (
                coeffs.alpha_0_ms
                + coeffs.alpha_prefill_ms_per_tok * prefill_tokens_step
                + coeffs.alpha_attn_quad_ms * attn_quad_work
                + coeffs.alpha_decode_ms_per_tok * decode_tokens_step
                + coeffs.alpha_kv_ms_per_gb * kv_read_gb
                + pcie_ms_step
            )
            step_end_ms = now_ms + step_ms
            w.stepping = True
            w.busy_until_ms = step_end_ms
            push_event(step_end_ms, "STEP_DONE", (w, prefill_allocs, list(w.active_decode), step_ms))

        while event_heap:
            now_ms, _, kind, payload = heapq.heappop(event_heap)

            if kind == "ARRIVE":
                st: _RequestState = payload  # type: ignore[assignment]
                if len(set(st.all_keys)) > self.hw.l1_capacity_blocks:
                    oom_rejected += 1
                    continue
                pool_key, candidates = self._eligible_pool(st.req)
                worker = self._select_worker(pool_key, candidates, st.req, st.prompt_keys, now_ms)
                worker.waiting_prefill.append(st)
                if not worker.stepping:
                    start_worker_step(worker, now_ms)

            elif kind == "KV_TRANSFER_DONE":
                d_worker, st = payload  # type: ignore[misc]
                d_worker.waiting_decode.append(st)
                if not d_worker.stepping:
                    start_worker_step(d_worker, now_ms)

            elif kind == "STEP_DONE":
                w, prefill_allocs, decoded_list, step_ms = payload  # type: ignore[misc]

                next_active_decode: list[_RequestState] = []
                decoded_set = set(id(s) for s in decoded_list)
                for d_st in w.active_decode:
                    if id(d_st) not in decoded_set:
                        next_active_decode.append(d_st)
                        continue
                    if d_st.ttft_ms is None:
                        d_st.ttft_ms = now_ms - d_st.req.arrival_ms
                    else:
                        d_st.itl_samples.append(step_ms)
                    d_st.decoded_so_far += 1
                    if d_st.decoded_so_far >= d_st.req.output_tokens:
                        d_st.completed_ms = now_ms
                        w.cache.unpin(d_st.req.tenant_id, d_st.req.model_revision, d_st.all_keys)
                    else:
                        next_active_decode.append(d_st)
                w.active_decode = next_active_decode

                still_prefilling: list[_RequestState] = []
                alloc_map = {id(s): chunk for s, chunk in prefill_allocs}
                for p_st in w.active_prefill:
                    chunk = alloc_map.get(id(p_st), 0)
                    p_st.remaining_prefill -= chunk
                    p_st.cached_context_tokens += chunk
                    if p_st.remaining_prefill > 0:
                        still_prefilling.append(p_st)
                        continue

                    w.cache.release_uncomputed(p_st.reserved_blocks)
                    p_st.reserved_blocks = 0

                    if w.role == "prefill":
                        # Prefill complete: insert computed prompt blocks into prefill cache and unpin
                        w.cache.pin_and_insert(
                            p_st.req.tenant_id, p_st.req.model_revision, p_st.prompt_keys
                        )
                        w.cache.unpin(
                            p_st.req.tenant_id, p_st.req.model_revision, p_st.prompt_keys
                        )
                        d_worker = self._select_worker(
                            "decode", self.decode_workers, p_st.req, p_st.prompt_keys, now_ms
                        )
                        if self.topo.prefix_caching:
                            d_l1, _ = d_worker.cache.lookup_prefix(
                                p_st.req.tenant_id, p_st.req.model_revision, p_st.prompt_keys
                            )
                        else:
                            d_l1 = 0
                        transfer_tokens = max(
                            self.hw.block_size_tokens,
                            p_st.req.input_tokens - d_l1 * self.hw.block_size_tokens,
                        )
                        transfer_ms = (
                            self.hw.transfer_handshake_ms
                            + ((transfer_tokens * self.hw.kv_gb_per_token) / self.hw.interconnect_bw_gbps)
                            * 1000.0
                        )
                        kv_transfers_ms.append(transfer_ms)
                        push_event(now_ms + transfer_ms, "KV_TRANSFER_DONE", (d_worker, p_st))
                    else:
                        # Monolithic / workload_shaped: pin computed blocks on entry to decode
                        w.cache.pin_and_insert(
                            p_st.req.tenant_id, p_st.req.model_revision, p_st.all_keys
                        )
                        p_st.ttft_ms = now_ms - p_st.req.arrival_ms
                        p_st.decoded_so_far = 1
                        if p_st.decoded_so_far >= p_st.req.output_tokens:
                            p_st.completed_ms = now_ms
                            w.cache.unpin(
                                p_st.req.tenant_id, p_st.req.model_revision, p_st.all_keys
                            )
                        else:
                            w.active_decode.append(p_st)
                w.active_prefill = still_prefilling

                start_worker_step(w, now_ms)

        completed_states = [s for s in states_by_id.values() if s.completed_ms is not None]
        ttfts = [s.ttft_ms for s in completed_states if s.ttft_ms is not None]
        all_itls = [itl for s in completed_states for itl in s.itl_samples]

        sla_compliant = 0
        sla_output_tokens = 0
        total_output_tokens = 0
        for s in completed_states:
            total_output_tokens += s.req.output_tokens
            req_p95_itl = _percentile(s.itl_samples, 0.95) if s.itl_samples else 0.0
            if (
                s.req.task_ok
                and s.ttft_ms is not None
                and s.ttft_ms <= s.req.ttft_sla_ms
                and req_p95_itl <= s.req.itl_sla_ms
            ):
                sla_compliant += 1
                sla_output_tokens += s.req.output_tokens

        min_arrival = min(r.arrival_ms for r in requests)
        max_done = max(
            (s.completed_ms for s in completed_states if s.completed_ms is not None),
            default=min_arrival + 1.0,
        )
        sim_duration_s = max(0.001, (max_done - min_arrival) / 1000.0)
        gpu_hours = (self.topo.num_gpus * sim_duration_s) / 3600.0

        peak_hbm = max(w.cache.peak_l1_blocks / w.cache.l1_capacity_blocks for w in self.workers)
        zero_score_failed = len(requests) - sla_compliant

        return SimMetrics(
            topology_name=self.topo.name,
            hardware_name=self.hw.name,
            calibration_source=self.hw.coeffs.calibration_source,
            total_requests=len(requests),
            completed_requests=len(completed_states),
            oom_rejected_requests=oom_rejected,
            sla_compliant_requests=sla_compliant,
            zero_score_failed_requests=zero_score_failed,
            p50_ttft_ms=round(_percentile(ttfts, 0.50), 2),
            p95_ttft_ms=round(_percentile(ttfts, 0.95), 2),
            p99_ttft_ms=round(_percentile(ttfts, 0.99), 2),
            p50_itl_ms=round(_percentile(all_itls, 0.50), 2),
            p95_itl_ms=round(_percentile(all_itls, 0.95), 2),
            p99_itl_ms=round(_percentile(all_itls, 0.99), 2),
            p999_itl_ms=round(_percentile(all_itls, 0.999), 2),
            max_itl_ms=round(max(all_itls), 2) if all_itls else 0.0,
            total_output_tokens=total_output_tokens,
            sim_duration_s=round(sim_duration_s, 3),
            gpu_hours=round(gpu_hours, 6),
            aggregate_tok_per_s=round(total_output_tokens / sim_duration_s, 2),
            goodput_tasks_per_gpu_hr=round(sla_compliant / max(gpu_hours, 1e-9), 2),
            goodput_tokens_per_gpu_hr=round(sla_output_tokens / max(gpu_hours, 1e-9), 2),
            l1_prefix_hit_rate=round(
                total_l1_hit_blocks / max(1, total_prompt_blocks_checked), 4
            ),
            l2_prefix_hit_rate=round(
                total_l2_hit_blocks / max(1, total_prompt_blocks_checked), 4
            ),
            avg_kv_transfer_ms=round(
                sum(kv_transfers_ms) / max(1, len(kv_transfers_ms)), 3
            ),
            peak_hbm_utilization=round(min(1.0, peak_hbm), 4),
            cost_per_completed_task_gpu_s=round(
                (self.topo.num_gpus * sim_duration_s) / max(1, sla_compliant), 4
            ),
        )
