"""Tests for two-tier tenant-isolated BlockKVCache."""

from __future__ import annotations

from inference.kv_cache import BlockKVCache


def test_contiguous_prefix_stops_at_first_mismatch() -> None:
    cache = BlockKVCache(l1_capacity_blocks=16, l2_capacity_blocks=32, l2_ram_offload=True)
    assert cache.pin_and_insert("t1", "r1", ["b0", "b1", "b2", "b3"])
    cache.unpin("t1", "r1", ["b0", "b1", "b2", "b3"])

    l1_hits, l2_hits = cache.lookup_prefix("t1", "r1", ["b0", "b1", "diff", "b3"])
    assert (l1_hits, l2_hits) == (2, 0)


def test_tenant_and_model_revision_isolation() -> None:
    cache = BlockKVCache(l1_capacity_blocks=16, l2_capacity_blocks=32, l2_ram_offload=True)
    assert cache.pin_and_insert("tenant_a", "model_r1", ["b0", "b1", "b2"])
    cache.unpin("tenant_a", "model_r1", ["b0", "b1", "b2"])

    assert cache.lookup_prefix("tenant_b", "model_r1", ["b0", "b1", "b2"]) == (0, 0)
    assert cache.lookup_prefix("tenant_a", "model_r2", ["b0", "b1", "b2"]) == (0, 0)
    assert cache.lookup_prefix("tenant_a", "model_r1", ["b0", "b1", "b2"]) == (3, 0)


def test_l1_eviction_to_l2_and_promotion_on_hit() -> None:
    cache = BlockKVCache(l1_capacity_blocks=4, l2_capacity_blocks=8, l2_ram_offload=True)
    assert cache.pin_and_insert("t1", "r1", ["a0", "a1", "a2"])
    cache.unpin("t1", "r1", ["a0", "a1", "a2"])

    assert cache.pin_and_insert("t1", "r1", ["x0", "x1", "x2", "x3"])
    cache.unpin("t1", "r1", ["x0", "x1", "x2", "x3"])

    l1_hits, l2_hits = cache.lookup_prefix("t1", "r1", ["a0", "a1", "a2"])
    assert l1_hits == 0
    assert l2_hits == 3

    # Re-pinning promotes back into L1
    assert cache.pin_and_insert("t1", "r1", ["a0", "a1", "a2"])
    cache.unpin("t1", "r1", ["a0", "a1", "a2"])
    l1_after, l2_after = cache.lookup_prefix("t1", "r1", ["a0", "a1", "a2"])
    assert (l1_after, l2_after) == (3, 0)


def test_l2_disabled_drops_evicted_blocks() -> None:
    cache = BlockKVCache(l1_capacity_blocks=3, l2_capacity_blocks=8, l2_ram_offload=False)
    assert cache.pin_and_insert("t1", "r1", ["a0", "a1"])
    cache.unpin("t1", "r1", ["a0", "a1"])
    assert cache.pin_and_insert("t1", "r1", ["b0", "b1", "b2"])
    cache.unpin("t1", "r1", ["b0", "b1", "b2"])
    assert cache.lookup_prefix("t1", "r1", ["a0", "a1"]) == (0, 0)
