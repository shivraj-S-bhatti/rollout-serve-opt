"""Seeded trace generators for 5 canonical serving archetypes plus Mooncake trace loading."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

from inference.schema import TraceRequest, WorkloadClass

ALL_WORKLOAD_TYPES = (
    "short_chat",
    "tool_agent",
    "batch_docs",
    "reasoning_heavy",
    "rl_group_rollouts",
    "mixed_agentic",
)


def _block_hash(namespace: str, idx: int) -> str:
    raw = f"{namespace}:blk:{idx}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _make_hashes(namespace: str, num_tokens: int, block_size: int = 64) -> list[str]:
    n_blocks = max(1, math.ceil(num_tokens / block_size))
    return [_block_hash(namespace, i) for i in range(n_blocks)]


def generate_short_chat(
    count: int, seed: int = 42, block_size: int = 64, start_ms: float = 0.0
) -> list[TraceRequest]:
    """Bursty short chat requests with minimal shared prefixes and strict latency SLAs."""
    rng = random.Random(seed)
    reqs: list[TraceRequest] = []
    t = start_ms
    for i in range(count):
        gap = rng.expovariate(1.0 / 35.0)
        if i % 15 == 0 and i > 0:
            gap += 180.0
        t += gap
        isl = rng.randint(128, 768)
        osl = rng.randint(64, 256)
        reqs.append(
            TraceRequest(
                request_id=f"chat_{i:05d}",
                session_id=f"chat_sess_{i:05d}",
                tenant_id=f"tenant_{i % 2}",
                turn_idx=0,
                workload_class="short_chat",
                arrival_ms=round(t, 3),
                t_pause_ms=0.0,
                input_tokens=isl,
                output_tokens=osl,
                prefix_block_hashes=_make_hashes(f"chat_uniq_{i}", isl, block_size),
                ttft_sla_ms=500.0,
                itl_sla_ms=40.0,
                task_ok=True,
            )
        )
    return reqs


def generate_tool_agent(
    count: int, seed: int = 42, block_size: int = 64, start_ms: float = 0.0
) -> list[TraceRequest]:
    """Multi-turn tool-using agent trajectories with growing prefix trees and tool dwell pauses."""
    rng = random.Random(seed)
    reqs: list[TraceRequest] = []
    turns_per_session = 5
    num_sessions = max(1, math.ceil(count / turns_per_session))

    shared_sys_tokens = 1536
    sys_hashes_by_family = [
        _make_hashes(f"agent_sys_fam_{fam}", shared_sys_tokens, block_size) for fam in range(4)
    ]

    for s in range(num_sessions):
        tenant = f"tenant_{s % 2}"
        fam = s % len(sys_hashes_by_family)
        sess_blocks = list(sys_hashes_by_family[fam])
        t = start_ms + s * 220.0 + rng.uniform(0.0, 90.0)
        total_input = shared_sys_tokens
        for turn in range(turns_per_session):
            if len(reqs) >= count:
                break
            delta_tokens = rng.randint(512, 1536) if turn > 0 else rng.randint(384, 768)
            total_input += delta_tokens
            new_blocks = max(1, math.ceil(delta_tokens / block_size))
            base_idx = len(sess_blocks)
            for k in range(new_blocks):
                sess_blocks.append(_block_hash(f"agent_s{s}_t{turn}", base_idx + k))

            osl = rng.randint(128, 320)
            pause_ms = float(rng.randint(1500, 12000)) if turn < turns_per_session - 1 else 0.0
            reqs.append(
                TraceRequest(
                    request_id=f"agent_s{s:04d}_t{turn}",
                    session_id=f"agent_s{s:04d}",
                    tenant_id=tenant,
                    turn_idx=turn,
                    workload_class="tool_agent",
                    arrival_ms=round(t, 3),
                    t_pause_ms=round(pause_ms, 2),
                    input_tokens=total_input,
                    output_tokens=osl,
                    prefix_block_hashes=list(sess_blocks),
                    ttft_sla_ms=1500.0,
                    itl_sla_ms=65.0,
                    task_ok=True,
                )
            )
            est_service_ms = 250.0 + osl * 16.0
            t += est_service_ms + pause_ms

    reqs.sort(key=lambda r: (r.arrival_ms, r.request_id))
    return reqs[:count]


def generate_batch_docs(
    count: int, seed: int = 42, block_size: int = 64, start_ms: float = 0.0
) -> list[TraceRequest]:
    """High-ISL document extraction/eval batches with shared system rubric and relaxed TTFT."""
    rng = random.Random(seed)
    reqs: list[TraceRequest] = []
    rubric_tokens = 512
    rubric_hashes = _make_hashes("batch_doc_rubric", rubric_tokens, block_size)
    t = start_ms
    for i in range(count):
        t += rng.uniform(45.0, 110.0)
        doc_tokens = rng.randint(3584, 11776)
        isl = rubric_tokens + doc_tokens
        osl = rng.randint(64, 256)
        doc_hashes = _make_hashes(f"doc_body_{i}", doc_tokens, block_size)
        reqs.append(
            TraceRequest(
                request_id=f"batch_{i:05d}",
                session_id=f"batch_job_{i // 10:04d}",
                tenant_id="tenant_batch",
                turn_idx=0,
                workload_class="batch_docs",
                arrival_ms=round(t, 3),
                t_pause_ms=0.0,
                input_tokens=isl,
                output_tokens=osl,
                prefix_block_hashes=rubric_hashes + doc_hashes,
                ttft_sla_ms=4000.0,
                itl_sla_ms=120.0,
                task_ok=True,
            )
        )
    return reqs


def generate_reasoning_heavy(
    count: int, seed: int = 42, block_size: int = 64, start_ms: float = 0.0
) -> list[TraceRequest]:
    """Decode-heavy chain-of-thought requests with long output sequences."""
    rng = random.Random(seed)
    reqs: list[TraceRequest] = []
    sys_tokens = 512
    sys_hashes = _make_hashes("reasoning_sys", sys_tokens, block_size)
    t = start_ms
    for i in range(count):
        t += rng.expovariate(1.0 / 90.0)
        prompt_body = rng.randint(1536, 5120)
        isl = sys_tokens + prompt_body
        osl = rng.randint(1024, 2400)
        reqs.append(
            TraceRequest(
                request_id=f"reason_{i:05d}",
                session_id=f"reason_sess_{i:05d}",
                tenant_id=f"tenant_{i % 2}",
                turn_idx=0,
                workload_class="reasoning_heavy",
                arrival_ms=round(t, 3),
                t_pause_ms=0.0,
                input_tokens=isl,
                output_tokens=osl,
                prefix_block_hashes=sys_hashes + _make_hashes(f"reason_q_{i}", prompt_body, block_size),
                ttft_sla_ms=1800.0,
                itl_sla_ms=50.0,
                task_ok=True,
            )
        )
    return reqs


def generate_rl_group_rollouts(
    count: int,
    seed: int = 42,
    group_size: int = 8,
    block_size: int = 64,
    start_ms: float = 0.0,
) -> list[TraceRequest]:
    """GRPO-style async RL rollouts: 1 shared prompt fans out to G rollouts after warmer prefill."""
    rng = random.Random(seed)
    reqs: list[TraceRequest] = []
    num_groups = max(1, math.ceil(count / group_size))
    env_tokens = 1536
    env_hashes = _make_hashes("rl_shared_env", env_tokens, block_size)
    t = start_ms
    for g in range(num_groups):
        prompt_tail = rng.randint(1024, 3072)
        isl = env_tokens + prompt_tail
        prompt_hashes = env_hashes + _make_hashes(f"rl_prompt_g{g}", prompt_tail, block_size)
        for member in range(group_size):
            if len(reqs) >= count:
                break
            osl = rng.randint(256, 640)
            pause_ms = float(rng.randint(800, 4500))
            # Member 0 warms the prompt prefix; subsequent group rollouts arrive staggered behind it.
            offset_ms = 0.0 if member == 0 else 450.0 + (member - 1) * 20.0
            reqs.append(
                TraceRequest(
                    request_id=f"rl_g{g:04d}_m{member}",
                    session_id=f"rl_g{g:04d}_m{member}",
                    tenant_id="tenant_rl",
                    group_id=f"grpo_grp_{g:04d}",
                    turn_idx=0,
                    workload_class="rl_group_rollouts",
                    arrival_ms=round(t + offset_ms, 3),
                    t_pause_ms=round(pause_ms, 2),
                    input_tokens=isl,
                    output_tokens=osl,
                    prefix_block_hashes=list(prompt_hashes),
                    ttft_sla_ms=1500.0,
                    itl_sla_ms=55.0,
                    task_ok=True,
                )
            )
        t += rng.uniform(550.0, 900.0)
    reqs.sort(key=lambda r: (r.arrival_ms, r.request_id))
    return reqs


def _stretch_window(reqs: list[TraceRequest], target_span_ms: float) -> list[TraceRequest]:
    if len(reqs) <= 1:
        return reqs
    t0 = min(r.arrival_ms for r in reqs)
    t1 = max(r.arrival_ms for r in reqs)
    span = max(1.0, t1 - t0)
    scale = target_span_ms / span
    return [
        r.model_copy(update={"arrival_ms": round((r.arrival_ms - t0) * scale, 3)})
        for r in reqs
    ]


def generate_mixed_agentic(
    count: int, seed: int = 42, block_size: int = 64
) -> list[TraceRequest]:
    """Combined realistic workload interleaving chat, tool agents, batch docs, and reasoning."""
    n_chat = max(1, int(round(count * 0.35)))
    n_agent = max(1, int(round(count * 0.35)))
    n_batch = max(1, int(round(count * 0.15)))
    n_reason = max(1, count - n_chat - n_agent - n_batch)

    agent_reqs = generate_tool_agent(n_agent, seed=seed + 2, block_size=block_size)
    active_span_ms = max(12_000.0, max((r.arrival_ms for r in agent_reqs), default=12_000.0) * 0.85)

    chat_reqs = _stretch_window(
        generate_short_chat(n_chat, seed=seed + 1, block_size=block_size), active_span_ms
    )
    batch_reqs = _stretch_window(
        generate_batch_docs(n_batch, seed=seed + 3, block_size=block_size), active_span_ms
    )
    reason_reqs = _stretch_window(
        generate_reasoning_heavy(n_reason, seed=seed + 4, block_size=block_size), active_span_ms
    )

    combined = chat_reqs + agent_reqs + batch_reqs + reason_reqs
    combined.sort(key=lambda r: (r.arrival_ms, r.request_id))
    return combined[:count]


def generate_workload(workload: str, count: int, seed: int = 42) -> list[TraceRequest]:
    """Dispatch to a named canonical workload generator."""
    generators = {
        "short_chat": generate_short_chat,
        "tool_agent": generate_tool_agent,
        "batch_docs": generate_batch_docs,
        "reasoning_heavy": generate_reasoning_heavy,
        "rl_group_rollouts": generate_rl_group_rollouts,
        "mixed_agentic": generate_mixed_agentic,
    }
    if workload not in generators:
        raise ValueError(f"Unknown workload '{workload}'. Expected one of {sorted(generators)}.")
    return generators[workload](count=count, seed=seed)


def write_trace_jsonl(path: str | Path, requests: Iterable[TraceRequest]) -> None:
    """Write trace requests to JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for req in requests:
            fh.write(req.model_dump_json() + "\n")


def read_trace_jsonl(path: str | Path) -> list[TraceRequest]:
    """Load trace requests from JSONL."""
    reqs: list[TraceRequest] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            reqs.append(TraceRequest.model_validate_json(stripped))
    return reqs


def load_mooncake_trace(
    path: str | Path,
    default_workload: WorkloadClass = "tool_agent",
    ttft_sla_ms: float = 1500.0,
    itl_sla_ms: float = 60.0,
) -> list[TraceRequest]:
    """Convert a Mooncake-style JSONL trace (`timestamp`, `input_length`, `output_length`, `hash_ids`) into TraceRequest."""
    reqs: list[TraceRequest] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            hashes = [str(h) for h in row.get("hash_ids", [])]
            isl = int(row["input_length"])
            osl = int(row["output_length"])
            if not hashes:
                hashes = _make_hashes(f"mooncake_row_{idx}", isl)
            reqs.append(
                TraceRequest(
                    request_id=str(row.get("request_id", f"mc_{idx:06d}")),
                    session_id=str(row.get("session_id", f"mc_sess_{idx:06d}")),
                    tenant_id=str(row.get("tenant_id", "tenant_mooncake")),
                    model_revision=str(row.get("model_revision", "model_r1")),
                    workload_class=row.get("workload_class", default_workload),
                    arrival_ms=float(row["timestamp"]),
                    t_pause_ms=float(row.get("t_pause_ms", 0.0)),
                    input_tokens=isl,
                    output_tokens=osl,
                    prefix_block_hashes=hashes,
                    ttft_sla_ms=float(row.get("ttft_sla_ms", ttft_sla_ms)),
                    itl_sla_ms=float(row.get("itl_sla_ms", itl_sla_ms)),
                    task_ok=bool(row.get("task_ok", True)),
                )
            )
    reqs.sort(key=lambda r: (r.arrival_ms, r.request_id))
    return reqs
