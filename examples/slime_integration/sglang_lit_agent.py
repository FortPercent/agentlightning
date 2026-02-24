"""Base Agent Lightning `LitAgent` that calls SLIME's SGLang router for inference.

Subclass `SGLangLitAgent` and implement `compute_reward` to build your own
task-specific agent.  Optionally override `build_prompt` to customise how the
task input is turned into a chat message list or plain string.

Example::

    class MathAgent(SGLangLitAgent):
        async def compute_reward(self, task: dict, response: str) -> float:
            return 1.0 if grade_math_answer(response, task["label"]) else 0.0

CLI usage: instantiate and pass to `OnlineRLLoop`; see `train.py`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import aiohttp

from agentlightning import LitAgent
from agentlightning.emitter.reward import emit_reward
from agentlightning.types.core import Rollout
from agentlightning.types.resources import NamedResources

try:
    from transformers import PreTrainedTokenizer
except ImportError as exc:
    raise ImportError("transformers package is required") from exc

logger = logging.getLogger(__name__)

__all__ = ["SGLangLitAgent"]

_DEFAULT_SAMPLING_PARAMS: dict[str, Any] = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
}


class SGLangLitAgent(LitAgent[dict]):
    """Agent Lightning `LitAgent` that uses SLIME's SGLang router for generation.

    This class handles the mechanics of calling SGLang and emitting the reward
    into the active trace.  Task-specific logic lives in the subclass.

    Args:
        sglang_url: Base URL of the SGLang router, e.g. ``http://localhost:30000``.
        tokenizer: HuggingFace tokenizer, used to apply the chat template.
        sampling_params: SGLang sampling parameters merged on top of the
            module-level defaults. Pass ``{"max_new_tokens": 1024}`` etc.
    """

    def __init__(
        self,
        sglang_url: str,
        tokenizer: PreTrainedTokenizer,
        sampling_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._sglang_url = sglang_url.rstrip("/")
        self._tokenizer = tokenizer
        self._sampling_params: dict[str, Any] = {
            **_DEFAULT_SAMPLING_PARAMS,
            **(sampling_params or {}),
        }

    # ------------------------------------------------------------------
    # Override point: prompt construction
    # ------------------------------------------------------------------

    def build_prompt(
        self, task: dict, resources: NamedResources
    ) -> str | list[dict]:
        """Construct the prompt for the LLM from a task input and resources.

        The default implementation:
        - Uses ``resources["system_prompt"]`` as the system message when present.
        - Uses ``task["prompt"]`` (or ``task["input"]``, or ``str(task)``) as the
          user message.

        Override this method to implement multi-turn dialogue, tool descriptions,
        few-shot examples, or any other prompt-engineering logic.

        Args:
            task: Dict from the training dataset, e.g. ``{"prompt": "...", "label": "..."}``.
            resources: Named resources from `LightningStore` — may carry a versioned
                ``"system_prompt"`` or other prompt components updated by the algorithm.

        Returns:
            Either a plain string or a list of chat-message dicts compatible with
            ``tokenizer.apply_chat_template``.
        """
        system_prompt: str = resources.get("system_prompt", "") if resources else ""
        user_input: str = task.get("prompt", task.get("input", str(task)))

        if system_prompt:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        return user_input

    # ------------------------------------------------------------------
    # Override point: reward computation
    # ------------------------------------------------------------------

    async def compute_reward(self, task: dict, response: str) -> float:
        """Compute the scalar reward for a (task, response) pair.

        This method **must** be overridden in subclasses.

        Args:
            task: The task dict from the training dataset.
            response: The generated text produced by SGLang.

        Returns:
            Scalar reward in any range (SLIME normalises within groups).
        """
        raise NotImplementedError(
            "Implement `compute_reward(task, response) -> float` in your SGLangLitAgent subclass."
        )

    # ------------------------------------------------------------------
    # LitAgent rollout
    # ------------------------------------------------------------------

    async def rollout_async(
        self,
        task: dict,
        resources: NamedResources,
        rollout: Rollout,
    ) -> None:
        """Execute one rollout: build prompt → call SGLang → emit reward.

        Agent Lightning's runner calls this method; the active tracer records
        the LLM call span automatically (when using the OtelTracer).  The
        reward is written into the trace via `emit_reward` so that
        `AGLDataSource` can extract it later.

        Args:
            task: Task input dict (``{"prompt": ..., "label": ...}``).
            resources: Versioned named resources from `LightningStore`.
            rollout: Rollout metadata (rollout_id, mode, etc.).
        """
        prompt = self.build_prompt(task, resources)
        logger.debug("rollout %s: calling SGLang", rollout.rollout_id)

        response, _ = await self._call_sglang(prompt)
        reward = await self.compute_reward(task, response)

        emit_reward(float(reward))
        logger.debug(
            "rollout %s: reward=%.4f, response_len=%d chars",
            rollout.rollout_id,
            reward,
            len(response),
        )

    # ------------------------------------------------------------------
    # Internal: SGLang HTTP call
    # ------------------------------------------------------------------

    async def _call_sglang(
        self,
        prompt: str | list[dict],
        extra_sampling_params: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Optional[list[float]]]:
        """POST a generation request to the SGLang router.

        Args:
            prompt: Plain string or chat-message list.
            extra_sampling_params: Per-call overrides merged on top of the
                instance-level defaults.

        Returns:
            ``(response_text, per_token_log_probs)``.  Log-probs are ``None``
            when SGLang does not return them (e.g. speculative decoding mode).
        """
        params = {**self._sampling_params, **(extra_sampling_params or {})}

        if isinstance(prompt, list):
            # Apply chat template to get token IDs for structured input.
            prompt_ids: list[int] = self._tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True
            )
            payload: dict[str, Any] = {
                "input_ids": prompt_ids,
                "sampling_params": params,
                "return_logprob": True,
                "logprob_start_len": len(prompt_ids),
            }
        else:
            payload = {
                "text": prompt,
                "sampling_params": params,
                "return_logprob": True,
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._sglang_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data: dict[str, Any] = await resp.json()

        response_text: str = data.get("text", "")

        # Extract per-token log-probs from SGLang's meta_info.
        log_probs: Optional[list[float]] = None
        raw_lp = (data.get("meta_info") or {}).get("output_token_logprobs")
        if raw_lp:
            log_probs = [
                item[0] if isinstance(item, (list, tuple)) else float(item)
                for item in raw_lp
            ]

        return response_text, log_probs
