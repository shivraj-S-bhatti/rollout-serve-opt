"""Tenant- and model-revision-scoped block prefix KV cache across L1 HBM and L2 Host RAM."""

from __future__ import annotations

from collections import OrderedDict

CacheKey = tuple[str, str, str]


class BlockKVCache:
    """Two-tier LRU block KV cache with uncomputed prefill reservation and tenant isolation."""

    def __init__(
        self,
        l1_capacity_blocks: int,
        l2_capacity_blocks: int = 0,
        l2_ram_offload: bool = False,
    ) -> None:
        self.l1_capacity_blocks = max(1, l1_capacity_blocks)
        self.l2_capacity_blocks = max(0, l2_capacity_blocks)
        self.l2_ram_offload = l2_ram_offload and self.l2_capacity_blocks > 0

        self._l1: OrderedDict[CacheKey, None] = OrderedDict()
        self._l2: OrderedDict[CacheKey, None] = OrderedDict()
        self._pins: dict[CacheKey, int] = {}
        self.active_uncomputed_blocks: int = 0
        self.peak_l1_blocks: int = 0

    @property
    def pinned_blocks_count(self) -> int:
        return sum(1 for count in self._pins.values() if count > 0)

    def _scope_keys(
        self, tenant_id: str, model_revision: str, block_hashes: list[str]
    ) -> list[CacheKey]:
        return [(tenant_id, model_revision, bh) for bh in block_hashes]

    def _update_peak(self) -> None:
        used = min(self.l1_capacity_blocks, len(self._l1) + self.active_uncomputed_blocks)
        if used > self.peak_l1_blocks:
            self.peak_l1_blocks = used

    def lookup_prefix(
        self, tenant_id: str, model_revision: str, block_hashes: list[str]
    ) -> tuple[int, int]:
        """Return contiguous (l1_hit_blocks, l2_hit_blocks) starting at block 0."""
        l1_hits = 0
        l2_hits = 0
        for key in self._scope_keys(tenant_id, model_revision, block_hashes):
            if key in self._l1:
                l1_hits += 1
            elif self.l2_ram_offload and key in self._l2:
                l2_hits += 1
            else:
                break
        return l1_hits, l2_hits

    def available_l1_capacity_for_pinning(
        self, tenant_id: str, model_revision: str, block_hashes: list[str]
    ) -> bool:
        """Check whether pinning these blocks fits inside L1 HBM alongside active reservations."""
        unique_keys = set(self._scope_keys(tenant_id, model_revision, block_hashes))
        needed_new = sum(1 for key in unique_keys if self._pins.get(key, 0) == 0)
        return (
            self.pinned_blocks_count + self.active_uncomputed_blocks + needed_new
            <= self.l1_capacity_blocks
        )

    def can_reserve_uncomputed(self, num_blocks: int) -> bool:
        """Check if HBM has headroom to reserve blocks for an in-flight prefill."""
        return (
            self.pinned_blocks_count + self.active_uncomputed_blocks + num_blocks
            <= self.l1_capacity_blocks
        )

    def reserve_uncomputed(self, num_blocks: int) -> bool:
        """Reserve HBM blocks for an active prefill without exposing them to prefix lookup."""
        if not self.can_reserve_uncomputed(num_blocks):
            return False
        while len(self._l1) + self.active_uncomputed_blocks + num_blocks > self.l1_capacity_blocks:
            if not self._evict_one_unpinned_l1():
                return False
        self.active_uncomputed_blocks += num_blocks
        self._update_peak()
        return True

    def release_uncomputed(self, num_blocks: int) -> None:
        """Release HBM blocks reserved for an in-flight prefill before promoting to computed L1."""
        self.active_uncomputed_blocks = max(0, self.active_uncomputed_blocks - num_blocks)

    def _evict_one_unpinned_l1(self) -> bool:
        for candidate in self._l1:
            if self._pins.get(candidate, 0) == 0:
                self._l1.pop(candidate)
                if self.l2_ram_offload:
                    self._insert_l2(candidate)
                return True
        return False

    def _insert_l2(self, key: CacheKey) -> None:
        if not self.l2_ram_offload or self.l2_capacity_blocks <= 0:
            return
        if key in self._l2:
            self._l2.move_to_end(key)
            return
        while len(self._l2) >= self.l2_capacity_blocks:
            self._l2.popitem(last=False)
        self._l2[key] = None

    def pin_and_insert(
        self, tenant_id: str, model_revision: str, block_hashes: list[str]
    ) -> bool:
        """Pin prefix blocks in L1 (promoting L2 hits if needed); return False if pinned HBM is full."""
        if not self.available_l1_capacity_for_pinning(tenant_id, model_revision, block_hashes):
            return False

        for key in self._scope_keys(tenant_id, model_revision, block_hashes):
            if key in self._l1:
                self._l1.move_to_end(key)
            else:
                if key in self._l2:
                    self._l2.pop(key)
                while len(self._l1) + self.active_uncomputed_blocks >= self.l1_capacity_blocks:
                    if not self._evict_one_unpinned_l1():
                        return False
                self._l1[key] = None
            self._pins[key] = self._pins.get(key, 0) + 1

        self._update_peak()
        return True

    def unpin(self, tenant_id: str, model_revision: str, block_hashes: list[str]) -> None:
        """Release active sequence pins while leaving blocks reusable in L1 LRU."""
        for key in self._scope_keys(tenant_id, model_revision, block_hashes):
            cnt = self._pins.get(key, 0)
            if cnt <= 1:
                self._pins.pop(key, None)
            else:
                self._pins[key] = cnt - 1
            if key in self._l1:
                self._l1.move_to_end(key)
