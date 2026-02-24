# Agent Lightning × SLIME Online RL Integration

This example shows how to combine **Agent Lightning** (AGL) as the agent execution and data-collection layer with **SLIME** as the distributed PPO/GRPO policy-training backend, forming a closed online RL loop.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OnlineRLLoop (per iteration)                 │
│                                                                     │
│  1. COLLECT                                                         │
│     OnlineRLLoop enqueues tasks → LightningStore                   │
│     AGL runners call LitAgent.rollout_async()                       │
│       └─ SGLangLitAgent POSTs to SGLang router                      │
│       └─ emit_reward() records the reward span                      │
│     Completed rollouts land in LightningStore (status="succeeded")  │
│                                                                     │
│  2. TRAIN                                                           │
│     SLIME's RolloutManager calls data_source.get_samples()          │
│       └─ AGLDataSource polls LightningStore (background thread)     │
│       └─ converter.rollout_to_sample() extracts prompt/response/    │
│          reward/log-probs and tokenises                             │
│     SLIME actor_model.async_train() runs PPO/GRPO update            │
│                                                                     │
│  3. SYNC                                                            │
│     actor_model.update_weights() pushes new policy to SGLang        │
│     → next iteration uses the updated model                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 | |
| `agent-lightning` | `pip install agent-lightning` or `uv sync` from repo root |
| `slime` | Clone from <https://github.com/zai-org/slime> and install |
| `transformers` | `pip install transformers` |
| `aiohttp` | `pip install aiohttp` |
| SGLang server | Follow SLIME's quick-start to launch the router |
| Ray + training GPUs | For Megatron/FSDP backend |

## Quick Start (smoke test)

The following verifies that `AGLDataSource` converts a synthetic rollout correctly — no GPU or SGLang required:

```python
# smoke_test.py
import asyncio
from argparse import Namespace
from transformers import AutoTokenizer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types.core import EnqueueRolloutRequest

# Import the bridge modules
from agl_datasource import AGLDataSource

async def main():
    store = InMemoryLightningStore()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # any tokenizer
    args = Namespace(reward_key=None)

    ds = AGLDataSource(store=store, tokenizer=tokenizer, args=args, n_samples_per_prompt=1)

    # Enqueue one rollout
    [rollout] = await store.enqueue_many_rollouts([
        EnqueueRolloutRequest(input={"prompt": "Hello world", "label": "test"})
    ])

    # Simulate a runner claiming it, recording a reward span, and marking succeeded
    from agentlightning.types.tracer import Span
    from agentlightning.emitter.reward import emit_reward
    # ... (attach tracer, call emit_reward(1.0), finish rollout)

    # AGLDataSource.get_samples() would then return a Sample group
    ds.stop()

asyncio.run(main())
```

## Full Training Run

```bash
python train.py \
    --model-path /path/to/model \
    --sglang-url http://localhost:30000 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 2 \
    --num-iterations 50 \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 4
```

Before running, edit `train.py` to:

1. **Implement `MyAgent.compute_reward`** with your task-specific reward signal.
2. **Wire SLIME's Ray actors** (`RolloutManager`, `actor_model`) in the `main()` function — stubs with `TODO` comments mark the exact lines.

## Included Files

| File | Role |
|------|------|
| `converter.py` | Stateless conversion: `Rollout + Span[]` → SLIME `Sample`. Handles prompt extraction, response parsing (OpenAI OTEL + AGL native conventions), reward extraction via `find_final_reward`, tokenisation, and optional log-prob extraction. |
| `agl_datasource.py` | `AGLDataSource(DataSource)` — SLIME `DataSource` implementation. Runs a background asyncio thread that polls `LightningStore` for completed rollouts, groups them by `group_id`, and puts ready groups into a thread-safe queue consumed by `get_samples()`. |
| `sglang_lit_agent.py` | `SGLangLitAgent(LitAgent[dict])` — base class for user agents. Handles SGLang HTTP calls, chat-template application, and `emit_reward`. Subclass and implement `compute_reward` (and optionally `build_prompt`). |
| `online_rl_loop.py` | `OnlineRLLoop` — orchestrates the three-phase cycle. Starts AGL runners in a background thread, enqueues tasks with `group_id` metadata for grouping, calls the user-supplied `slime_train_fn` and `weight_sync_fn`. |
| `train.py` | End-to-end wiring example with `argparse`. Contains `MyAgent` stub, SLIME Ray actor setup placeholders, and the main training entry point. |
| `README.md` | This file. |

## Key Design Decisions

### Why a background thread for `AGLDataSource`?
SLIME's `DataSource.get_samples()` is synchronous, but `LightningStore` exposes an async API. A background daemon thread runs its own `asyncio` event loop, polling the store and converting rollouts without blocking SLIME's training code.

### How are rollouts grouped for GRPO?
`OnlineRLLoop._enqueue_rollouts()` assigns a shared `group_id = uuid4()` to all `n_samples_per_prompt` rollouts for the same prompt. `AGLDataSource` accumulates rollouts by `group_id` in `_pending_groups` and only enqueues a complete `list[Sample]` once all members arrive.

### Log probs are optional
When SGLang returns `output_token_logprobs`, they are stored in `Sample.rollout_log_probs` for PPO importance-sampling correction. If absent (e.g. when speculative decoding is enabled), `rollout_log_probs=None` — compatible with GRPO and REINFORCE which do not need old log-probs.

### No `--custom-generate-function-path` needed
SLIME's `generate_and_rm` (`slime/rollout/sglang_rollout.py:213`) has a built-in short-circuit:

```python
if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
    assert sample.response is not None
    if not args.group_rm:
        assert sample.reward is not None
    return sample  # skips SGLang entirely
```

`converter.rollout_to_sample()` always sets `status=COMPLETED` and a non-`None` `reward`,
so SLIME skips re-generation automatically. **Do not pass `--custom-generate-function-path`.**
