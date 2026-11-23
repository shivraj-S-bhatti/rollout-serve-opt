"""Tests for canonical workload generators, JSONL round-trip, and Mooncake trace loading."""

from __future__ import annotations

import json
from pathlib import Path

from inference.workloads import (
    generate_rl_group_rollouts,
    generate_tool_agent,
    generate_workload,
    load_mooncake_trace,
    read_trace_jsonl,
    write_trace_jsonl,
)


def test_generators_are_deterministic() -> None:
    for wname in ("short_chat", "tool_agent", "batch_docs", "reasoning_heavy", "rl_group_rollouts", "mixed_agentic"):
        run_a = generate_workload(wname, count=40, seed=123)
        run_b = generate_workload(wname, count=40, seed=123)
        assert [r.model_dump() for r in run_a] == [r.model_dump() for r in run_b]
        assert len(run_a) == 40


def test_tool_agent_grows_prefix_tree_across_turns() -> None:
    reqs = generate_tool_agent(count=25, seed=7)
    by_session: dict[str, list] = {}
    for r in reqs:
        by_session.setdefault(r.session_id, []).append(r)

    multi_turn = [turns for turns in by_session.values() if len(turns) >= 3]
    assert multi_turn, "Expected multi-turn agent trajectories"
    turns = sorted(multi_turn[0], key=lambda x: x.turn_idx)
    for idx in range(1, len(turns)):
        prev_h = turns[idx - 1].prefix_block_hashes
        curr_h = turns[idx].prefix_block_hashes
        assert len(curr_h) > len(prev_h)
        assert curr_h[: len(prev_h)] == prev_h
        assert turns[idx].arrival_ms > turns[idx - 1].arrival_ms


def test_rl_group_rollouts_share_full_prompt_prefix() -> None:
    reqs = generate_rl_group_rollouts(count=24, seed=19, group_size=8)
    groups: dict[str, list] = {}
    for r in reqs:
        assert r.group_id is not None
        groups.setdefault(r.group_id, []).append(r)

    assert len(groups) == 3
    for members in groups.values():
        assert len(members) == 8
        first_hashes = members[0].prefix_block_hashes
        assert all(m.prefix_block_hashes == first_hashes for m in members)


def test_jsonl_roundtrip_and_mooncake_loader(tmp_path: Path) -> None:
    reqs = generate_workload("mixed_agentic", count=30, seed=99)
    jsonl_path = tmp_path / "mixed.jsonl"
    write_trace_jsonl(jsonl_path, reqs)
    loaded = read_trace_jsonl(jsonl_path)
    assert loaded == reqs

    mc_path = tmp_path / "mooncake.jsonl"
    rows = [
        {"timestamp": 100.0, "input_length": 512, "output_length": 64, "hash_ids": ["a", "b", "c"]},
        {"timestamp": 250.0, "input_length": 1024, "output_length": 128, "hash_ids": ["a", "b", "d", "e"]},
    ]
    mc_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    mc_reqs = load_mooncake_trace(mc_path)
    assert len(mc_reqs) == 2
    assert mc_reqs[0].prefix_block_hashes == ["a", "b", "c"]
    assert mc_reqs[1].input_tokens == 1024
