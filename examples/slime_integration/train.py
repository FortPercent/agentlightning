"""Online RL training: Agent Lightning data collection + SLIME policy training.

This script wires Agent Lightning's rollout infrastructure to SLIME's
distributed PPO/GRPO training backend, forming a closed online RL loop:

    Collect (AGL runners + SGLang) → Train (SLIME Megatron/FSDP) → Sync weights

Usage::

    # Single-node example (4 training GPUs, 2 rollout GPUs):
    python train.py \\
        --model-path /path/to/hf-model \\
        --sglang-url http://localhost:30000 \\
        --actor-num-nodes 1 --actor-num-gpus-per-node 4 \\
        --rollout-num-gpus 2 \\
        --num-iterations 50 \\
        --rollout-batch-size 32 \\
        --n-samples-per-prompt 4

Customise the agent and reward function by editing `MyAgent` below.
"""

from __future__ import annotations

import argparse
import logging
import sys
from argparse import Namespace
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports — defer Ray and SLIME so the script can be imported for unit tests
# without a full SLIME install.
# ---------------------------------------------------------------------------


def _import_heavy() -> tuple[Any, Any, Any]:
    """Import Ray, SLIME, and transformers lazily."""
    import ray
    from transformers import AutoTokenizer

    return ray, AutoTokenizer, None


# ---------------------------------------------------------------------------
# User-customisable agent
# ---------------------------------------------------------------------------

from agentlightning.store.memory import InMemoryLightningStore  # noqa: E402

from agl_datasource import AGLDataSource  # noqa: E402
from online_rl_loop import OnlineRLLoop  # noqa: E402
from sglang_lit_agent import SGLangLitAgent  # noqa: E402


class MyAgent(SGLangLitAgent):
    """Replace this with your task-specific agent.

    Override `compute_reward` to implement your reward signal and optionally
    override `build_prompt` to customise the prompt format.

    Example for math reasoning::

        async def compute_reward(self, task: dict, response: str) -> float:
            from math_utils import grade_math_answer
            return 1.0 if grade_math_answer(response, task["label"]) else 0.0
    """

    async def compute_reward(self, task: dict, response: str) -> float:
        """TODO: implement your reward function here."""
        raise NotImplementedError(
            "Implement compute_reward() in MyAgent before running."
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Agent Lightning × SLIME online RL training"
    )

    # Agent Lightning settings
    agl = parser.add_argument_group("Agent Lightning")
    agl.add_argument(
        "--sglang-url",
        default="http://localhost:30000",
        help="SGLang router base URL (default: http://localhost:30000)",
    )
    agl.add_argument(
        "--n-runners",
        type=int,
        default=4,
        help="Number of parallel AGL runner workers (default: 4)",
    )
    agl.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="SGLang max_new_tokens per generation (default: 512)",
    )
    agl.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    # SLIME / model settings
    slime = parser.add_argument_group("SLIME / Model")
    slime.add_argument(
        "--model-path",
        required=True,
        help="Path to the HuggingFace model checkpoint (for tokenizer + training)",
    )
    slime.add_argument(
        "--actor-num-nodes",
        type=int,
        default=1,
        help="Number of training nodes (default: 1)",
    )
    slime.add_argument(
        "--actor-num-gpus-per-node",
        type=int,
        default=4,
        help="Training GPUs per node (default: 4)",
    )
    slime.add_argument(
        "--rollout-num-gpus",
        type=int,
        default=2,
        help="Total GPUs allocated to SGLang rollout (default: 2)",
    )
    slime.add_argument(
        "--train-backend",
        choices=["megatron", "fsdp"],
        default="megatron",
        help="SLIME training backend (default: megatron)",
    )

    # Training loop settings
    loop = parser.add_argument_group("Training loop")
    loop.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="Total collect → train → sync iterations (default: 50)",
    )
    loop.add_argument(
        "--rollout-batch-size",
        type=int,
        default=32,
        help="Number of prompt groups per iteration (default: 32)",
    )
    loop.add_argument(
        "--n-samples-per-prompt",
        type=int,
        default=4,
        help="Rollouts per prompt for group advantage (GRPO) (default: 4)",
    )
    loop.add_argument(
        "--global-batch-size",
        type=int,
        default=128,
        help="Training global batch size (default: 128)",
    )
    loop.add_argument(
        "--save-dir",
        default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# SLIME training step factory
# ---------------------------------------------------------------------------


def make_slime_train_fn(rollout_manager: Any, actor_model: Any) -> Any:
    """Return a synchronous training-step function for `OnlineRLLoop`.

    The returned callable is passed to `OnlineRLLoop(slime_train_fn=...)`.
    It calls SLIME's Ray actors synchronously (Ray futures block with
    `ray.get`).

    Args:
        rollout_manager: SLIME `RolloutManager` Ray remote actor.
        actor_model: SLIME actor `TrainRayActor` Ray remote actor.

    Returns:
        Callable ``(iteration: int, data_source: AGLDataSource) → None``.
    """
    import ray  # local import so the function can be defined before Ray is init'd

    def slime_train(iteration: int, data_source: AGLDataSource) -> None:
        # RolloutManager.generate() calls data_source.get_samples() internally
        # and runs the full generate+RM pipeline.  Since AGLDataSource skips
        # SGLang generation (the response is already in the Sample), you may
        # need a lightweight custom rollout function.  See README for details.
        rollout_data_ref = ray.get(rollout_manager.generate.remote(iteration))
        ray.get(actor_model.async_train.remote(iteration, rollout_data_ref))

    return slime_train


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --- Lazy imports ---
    import ray
    from transformers import AutoTokenizer

    # --- Init Ray ---
    if not ray.is_initialized():
        ray.init()
        logger.info("Ray initialised")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    logger.info("Loaded tokenizer from %s", args.model_path)

    # --- Agent Lightning store ---
    store = InMemoryLightningStore()

    # --- AGLDataSource (AGL → SLIME bridge) ---
    data_source = AGLDataSource(
        store=store,
        tokenizer=tokenizer,
        args=args,
        n_samples_per_prompt=args.n_samples_per_prompt,
    )

    # --- SLIME Ray actors ---
    # Replace the stubs below with your actual SLIME setup.
    # Refer to SLIME's train.py and placement_group.py for a full example.
    #
    #   from slime.ray.placement_group import create_placement_groups
    #   from slime.ray.rollout import RolloutManager
    #   from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    #
    #   placement_groups = create_placement_groups(args)
    #   rollout_manager = RolloutManager.options(...).remote(args, data_source)
    #   actor_model = MegatronTrainRayActor.options(...).remote(args)
    #   ray.get(actor_model.init_model.remote())

    from slime.ray.placement_group import create_placement_groups
    from slime.ray.rollout import RolloutManager

    placement_groups = create_placement_groups(args)
    rollout_pg = (
        placement_groups.get("rollout")
        if isinstance(placement_groups, dict)
        else placement_groups
    )
    if rollout_pg is None:
        rollout_manager = RolloutManager.remote(args, data_source)
    else:
        rollout_manager = RolloutManager.options(
            placement_group=rollout_pg
        ).remote(args, data_source)

    if args.train_backend == "megatron":
        from slime.backends.megatron_utils.actor import (
            MegatronTrainRayActor as TrainRayActor,
        )
    else:
        from slime.backends.fsdp.actor import FSDPTrainRayActor as TrainRayActor

    actor_pg = None
    if isinstance(placement_groups, dict):
        actor_pg = placement_groups.get("actor") or placement_groups.get("train")
    else:
        actor_pg = placement_groups

    if actor_pg is None:
        actor_model = TrainRayActor.remote(args)
    else:
        actor_model = TrainRayActor.options(placement_group=actor_pg).remote(args)

    ray.get(actor_model.init_model.remote())

    slime_train_fn = make_slime_train_fn(rollout_manager, actor_model)

    def weight_sync_fn() -> None:
        ray.get(actor_model.update_weights.remote())

    # --- Training dataset ---
    # Replace with your actual dataset.
    train_dataset = [
        {"prompt": "What is 2 + 2?", "label": "4"},
        {"prompt": "What is the capital of France?", "label": "Paris"},
        # Add more samples here...
    ]

    # --- Agent ---
    agent = MyAgent(
        sglang_url=args.sglang_url,
        tokenizer=tokenizer,
        sampling_params={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
    )

    # --- Run the online RL loop ---
    loop = OnlineRLLoop(
        agent=agent,
        store=store,
        data_source=data_source,
        slime_train_fn=slime_train_fn,
        weight_sync_fn=weight_sync_fn,
        n_runners=args.n_runners,
    )

    loop.run(
        train_dataset=train_dataset,
        num_iterations=args.num_iterations,
        rollout_batch_size=args.rollout_batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
    )


if __name__ == "__main__":
    main()
