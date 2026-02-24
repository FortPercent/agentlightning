"""Online RL training loop coordinating Agent Lightning and SLIME.

`OnlineRLLoop` drives the three-phase cycle that closes the RL loop:

1. **Collect** — enqueue `rollout_batch_size` tasks into `LightningStore`.
   Agent Lightning runners execute `LitAgent.rollout_async()` in the background,
   producing completed rollouts that `AGLDataSource` converts to SLIME Samples.
2. **Train** — call the user-supplied `slime_train_fn(iteration, data_source)`.
   SLIME pulls samples from `AGLDataSource.get_samples()` internally and runs
   a PPO/GRPO gradient update.
3. **Sync** — call `weight_sync_fn()` (e.g. `actor_model.update_weights()`) to
   push the new policy weights to SGLang so the next rollout uses the updated model.

CLI usage: see `train.py` for a full wiring example.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import Any, Callable, Iterable, Optional

from agentlightning import LitAgent, Trainer
from agentlightning.store.base import LightningStore
from agentlightning.types.core import EnqueueRolloutRequest
from agentlightning.types.resources import NamedResources

from .agl_datasource import AGLDataSource

logger = logging.getLogger(__name__)

__all__ = ["OnlineRLLoop"]

# Type aliases for the two user-supplied callables.
SlimeTrainFn = Callable[[int, AGLDataSource], None]
WeightSyncFn = Callable[[], None]


class OnlineRLLoop:
    """Coordinates Agent Lightning data collection with SLIME policy training.

    The loop is intentionally simple: collect → train → sync, repeated for
    `num_iterations`.  Both Agent Lightning runners and SLIME training run in
    the same process on separate threads so they can share state through
    `LightningStore` without inter-process serialisation.

    Args:
        agent: `LitAgent` subclass implementing task-specific rollout logic.
        store: `LightningStore` instance shared between runners and the loop.
        data_source: `AGLDataSource` that bridges completed rollouts to SLIME.
        slime_train_fn: Called as ``slime_train_fn(iteration, data_source)``
            once per collect phase.  Inside, call SLIME's `RolloutManager` and
            `actor_model` Ray actors; the function must be synchronous (Ray
            calls block internally).
        weight_sync_fn: Called after each training step, e.g.
            ``lambda: actor_model.update_weights()``.
        n_runners: Number of parallel Agent Lightning runner workers.
        initial_resources: Optional initial `NamedResources` (e.g. a system
            prompt) registered in `LightningStore` before the first iteration.
    """

    def __init__(
        self,
        *,
        agent: LitAgent,
        store: LightningStore,
        data_source: AGLDataSource,
        slime_train_fn: SlimeTrainFn,
        weight_sync_fn: WeightSyncFn,
        n_runners: int = 4,
        initial_resources: Optional[NamedResources] = None,
    ) -> None:
        self._agent = agent
        self._store = store
        self._data_source = data_source
        self._slime_train_fn = slime_train_fn
        self._weight_sync_fn = weight_sync_fn
        self._n_runners = n_runners
        self._initial_resources = initial_resources

        # Trainer with no algorithm — we drive the loop ourselves.
        self._trainer = Trainer(
            store=store,
            n_runners=n_runners,
            algorithm=None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        train_dataset: Iterable[Any],
        *,
        num_iterations: int = 100,
        rollout_batch_size: int = 32,
        n_samples_per_prompt: int = 1,
    ) -> None:
        """Run the online RL loop.

        Args:
            train_dataset: Iterable of task dicts (``{"prompt": ..., "label": ...}``).
                The dataset is repeated cyclically to fill each batch.
            num_iterations: Total number of collect → train → sync cycles.
            rollout_batch_size: Number of *prompt groups* per iteration (each
                group spawns `n_samples_per_prompt` rollouts).
            n_samples_per_prompt: How many independent rollouts to run for each
                prompt (used for GRPO group advantage estimation).
        """
        logger.info(
            "OnlineRLLoop: starting — %d iterations × %d groups × %d samples",
            num_iterations,
            rollout_batch_size,
            n_samples_per_prompt,
        )

        resources_id: Optional[str] = None
        if self._initial_resources:
            resources_id = asyncio.run(
                self._register_resources(self._initial_resources)
            )
            logger.info("OnlineRLLoop: registered initial resources (id=%s)", resources_id)

        tasks = list(train_dataset)
        runner_thread = self._start_runners()

        try:
            for iteration in range(num_iterations):
                logger.info(
                    "=== Iteration %d / %d ===", iteration + 1, num_iterations
                )

                # Phase 1: enqueue rollouts into LightningStore.
                self._enqueue_rollouts(
                    tasks,
                    rollout_batch_size,
                    n_samples_per_prompt,
                    resources_id,
                )

                # Phase 2: SLIME training step.
                # `data_source.get_samples()` blocks here until rollouts complete.
                logger.info("OnlineRLLoop: running SLIME training step...")
                self._slime_train_fn(iteration, self._data_source)

                # Phase 3: push updated weights to SGLang.
                logger.info("OnlineRLLoop: syncing weights to SGLang...")
                self._weight_sync_fn()

                logger.info("OnlineRLLoop: iteration %d complete", iteration + 1)

        finally:
            self._data_source.stop()
            # The runner Trainer has no built-in stop signal in algorithm=None mode;
            # the daemon thread exits when the process does.
            runner_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _register_resources(self, resources: NamedResources) -> str:
        """Register initial resources in the store and return their ID."""
        from agentlightning.types.resources import ResourcesUpdate

        update = ResourcesUpdate(resources=resources)
        stored = await self._store.put_resources(update)
        return stored.resources_id

    def _enqueue_rollouts(
        self,
        tasks: list[Any],
        batch_size: int,
        n_samples_per_prompt: int,
        resources_id: Optional[str],
    ) -> None:
        """Build `batch_size × n_samples_per_prompt` rollout requests and enqueue them.

        Each prompt group shares a unique `group_id` in the rollout metadata so
        that `AGLDataSource` can re-assemble groups before sending to SLIME.
        """
        requests: list[EnqueueRolloutRequest] = []

        # Cycle through tasks to fill the batch.
        task_cycle = (tasks * ((batch_size // max(len(tasks), 1)) + 1))[:batch_size]

        for task in task_cycle:
            group_id = str(uuid.uuid4())
            task_input = task if isinstance(task, dict) else {"prompt": str(task)}

            for _ in range(n_samples_per_prompt):
                requests.append(
                    EnqueueRolloutRequest(
                        input=task_input,
                        mode="train",
                        resources_id=resources_id,
                        metadata={
                            "group_id": group_id,
                            "group_size": n_samples_per_prompt,
                        },
                    )
                )

        asyncio.run(self._store.enqueue_many_rollouts(requests))
        logger.info(
            "OnlineRLLoop: enqueued %d rollouts (%d groups × %d samples/group)",
            len(requests),
            batch_size,
            n_samples_per_prompt,
        )

    def _start_runners(self) -> threading.Thread:
        """Launch Agent Lightning runners in a background daemon thread.

        The `Trainer` with `algorithm=None` simply keeps runners alive,
        claiming rollouts from the store and executing the agent until the
        process exits.
        """

        def _run() -> None:
            try:
                self._trainer.fit(self._agent)
            except Exception:
                logger.exception("OnlineRLLoop: runner thread raised an exception")

        thread = threading.Thread(target=_run, name="AGL-runners", daemon=True)
        thread.start()
        logger.info("OnlineRLLoop: AGL runners started (%d workers)", self._n_runners)
        return thread
