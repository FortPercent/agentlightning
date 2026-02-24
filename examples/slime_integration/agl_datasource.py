"""SLIME DataSource backed by Agent Lightning's LightningStore.

`AGLDataSource` is the core bridge between the two frameworks:

- **Producer side**: Agent Lightning runners execute `LitAgent.rollout_async()`,
  record spans, and mark rollouts as `succeeded` in `LightningStore`.
- **Consumer side**: SLIME's training loop calls `get_samples(n)` to fetch a
  batch of `Sample` groups for the next PPO/GRPO update step.

`AGLDataSource` runs a background daemon thread with its own `asyncio` event
loop so that the async `LightningStore` API can be polled without blocking
SLIME's synchronous training code.

CLI usage: instantiate and pass to `OnlineRLLoop`; see `train.py`.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from argparse import Namespace
from typing import Optional

from transformers import PreTrainedTokenizer

from agentlightning.store.base import LightningStore

from .converter import convert

try:
    from slime.rollout.data_source import DataSource
    from slime.utils.types import Sample
except ImportError as exc:
    raise ImportError(
        "slime package not found. Install it from https://github.com/zai-org/slime"
    ) from exc

logger = logging.getLogger(__name__)

__all__ = ["AGLDataSource"]


class AGLDataSource(DataSource):
    """SLIME `DataSource` that streams completed rollouts from `LightningStore`.

    Each Agent Lightning rollout is converted to a SLIME `Sample` via
    `converter.convert(args, rollout, spans, tokenizer)`. The converter type
    is controlled by ``args.converter_type`` (``"single_turn"`` or
    ``"multistep"``) and ``args.custom_converter_path`` — the same dispatch
    pattern as SLIME's ``rm_hub.async_rm``.

    Rollouts sharing the same ``group_id`` metadata key are collected into a
    single group before being handed to the training loop — this supports the
    ``n_samples_per_prompt`` grouping that GRPO and GSPO require.

    Args:
        store: Agent Lightning `LightningStore` instance (in-memory or mongo).
        tokenizer: HuggingFace tokenizer used for token-ID encoding.
        args: Argument `Namespace`. Relevant converter fields:
            ``converter_type``, ``custom_converter_path``,
            ``converter_llm_match``.
        n_samples_per_prompt: Expected group size. Defaults to `1`.
        poll_interval: Seconds between `LightningStore` polls.
        max_wait_seconds: How long `get_samples` will wait before returning a
            partial batch (with a warning).
    """

    def __init__(
        self,
        store: LightningStore,
        tokenizer: PreTrainedTokenizer,
        args: Namespace,
        *,
        n_samples_per_prompt: int = 1,
        poll_interval: float = 0.5,
        max_wait_seconds: float = 300.0,
    ) -> None:
        self._store = store
        self._tokenizer = tokenizer
        self._args = args
        self._n_samples = n_samples_per_prompt
        self._poll_interval = poll_interval
        self._max_wait = max_wait_seconds

        # Thread-safe queue of ready Sample groups sent to SLIME.
        self._ready: queue.Queue[list[Sample]] = queue.Queue()

        # Rollout IDs already handed to SLIME — never converted twice.
        self._consumed_ids: set[str] = set()
        self._consumed_lock = threading.Lock()

        # Partial groups: group_id → list of Samples waiting to be completed.
        self._pending_groups: dict[str, list[Sample]] = {}

        # Background polling thread lifecycle control.
        self._stop_event = threading.Event()
        self._poll_thread = threading.Thread(
            target=self._polling_thread, name="AGLDataSource-poll", daemon=True
        )
        self._poll_thread.start()
        logger.info("AGLDataSource started (poll_interval=%.1fs)", poll_interval)

    # ------------------------------------------------------------------
    # DataSource ABC implementation
    # ------------------------------------------------------------------

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Block until `num_samples` Sample groups are ready, then return them.

        If `max_wait_seconds` elapses before enough groups arrive, a partial
        list is returned with a warning.

        Args:
            num_samples: Number of prompt groups requested by SLIME.

        Returns:
            List of `num_samples` Sample groups (each inner list has
            `n_samples_per_prompt` elements).
        """
        result: list[list[Sample]] = []
        deadline = time.monotonic() + self._max_wait

        while len(result) < num_samples:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "AGLDataSource: timeout waiting for samples — got %d/%d",
                    len(result),
                    num_samples,
                )
                break
            try:
                group = self._ready.get(timeout=min(1.0, remaining))
                result.append(group)
            except queue.Empty:
                continue

        return result

    def add_samples(self, samples: list[list[Sample]]) -> None:
        """Inject pre-built Sample groups directly (e.g. for replay buffers)."""
        for group in samples:
            self._ready.put(group)

    def save(self, rollout_id: int) -> None:  # noqa: D102
        # State is held in LightningStore; nothing to persist here.
        pass

    def load(self, rollout_id: Optional[int] = None) -> None:  # noqa: D102
        pass

    def __len__(self) -> int:
        """Return the number of ready groups currently in the queue."""
        return self._ready.qsize()

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._poll_thread.join(timeout=10.0)
        logger.info("AGLDataSource stopped")

    def _polling_thread(self) -> None:
        """Entry point for the background daemon thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._poll_loop())
        finally:
            loop.close()

    async def _poll_loop(self) -> None:
        """Continuously poll LightningStore until `stop()` is called."""
        while not self._stop_event.is_set():
            try:
                await self._fetch_and_enqueue()
            except Exception:
                logger.exception("AGLDataSource: error during polling — will retry")
            await asyncio.sleep(self._poll_interval)

    async def _fetch_and_enqueue(self) -> None:
        """Fetch newly completed rollouts and push ready groups to the queue."""
        rollouts = await self._store.query_rollouts(status_in=["succeeded"])

        for rollout in rollouts:
            # Skip already-consumed rollouts.
            with self._consumed_lock:
                if rollout.rollout_id in self._consumed_ids:
                    continue
                self._consumed_ids.add(rollout.rollout_id)

            # Fetch spans for the latest attempt.
            spans = await self._store.query_spans(
                rollout.rollout_id, attempt_id="latest"
            )

            sample = convert(self._args, rollout, list(spans), self._tokenizer)
            if sample is None:
                logger.warning(
                    "AGLDataSource: could not convert rollout %s — skipping",
                    rollout.rollout_id,
                )
                continue

            # Group by group_id from rollout metadata (set by OnlineRLLoop).
            meta = rollout.metadata or {}
            group_id: str = meta.get("group_id", rollout.rollout_id)
            group_size: int = int(meta.get("group_size", self._n_samples))

            self._pending_groups.setdefault(group_id, [])
            self._pending_groups[group_id].append(sample)

            if len(self._pending_groups[group_id]) >= group_size:
                group = self._pending_groups.pop(group_id)
                # Assign stable indices for SLIME's reward normalisation.
                for i, s in enumerate(group):
                    s.group_index = abs(hash(group_id)) % (2**31)
                    s.index = i
                self._ready.put(group)
                logger.debug(
                    "AGLDataSource: enqueued group %s (%d samples)", group_id, len(group)
                )
