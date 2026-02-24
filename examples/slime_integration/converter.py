"""Convert Agent Lightning rollouts and spans into SLIME Sample objects.

Entry point is `convert(args, rollout, spans, tokenizer)`, which dispatches
to a built-in converter or a user-supplied function — the same pattern as
SLIME's `slime/rollout/rm_hub/__init__.py`.

Built-in converter types (set via ``args.converter_type``):

- ``"single_turn"`` *(default)* — one LLM call per rollout.
  Looks for spans matching ``args.converter_llm_match`` (default:
  ``openai.chat.completion``), tokenises prompt + response, and builds a
  simple ``[0…0, 1…1]`` loss mask.

- ``"multistep"`` — multi-turn tool-loop agents (e.g. ``ClaudeCodeAgent``).
  Reads ``raw_gen_ai_request`` spans emitted by the LLM proxy, stitches all
  turns into one sequence, and sets ``loss_mask = 1`` only on model-generated
  tokens.

Custom converter (set via ``args.custom_converter_path``):
  Path in ``module.submodule.function`` format, loaded with
  ``importlib.import_module``. Signature::

      def my_converter(
          args: Namespace,
          rollout: Rollout,
          spans: Sequence[Span],
          tokenizer: PreTrainedTokenizer,
      ) -> Sample | None: ...

CLI usage: import and call `convert`; see `agl_datasource.py`.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from argparse import Namespace
from typing import Optional, Sequence

from transformers import PreTrainedTokenizer

from agentlightning.emitter.reward import find_final_reward
from agentlightning.types.core import Rollout
from agentlightning.types.tracer import Span

try:
    from slime.utils.types import Sample
except ImportError as exc:
    raise ImportError(
        "slime package not found. Install it from https://github.com/zai-org/slime"
    ) from exc

logger = logging.getLogger(__name__)

__all__ = ["convert"]

# ────────────────────────────────────────────────────────────────────────────
# Public dispatch — mirrors SLIME's async_rm()
# ────────────────────────────────────────────────────────────────────────────

def convert(
    args: Namespace,
    rollout: Rollout,
    spans: Sequence[Span],
    tokenizer: PreTrainedTokenizer,
) -> Optional[Sample]:
    """Dispatch entry point: convert one rollout's spans into a SLIME Sample.

    Resolution order (same as SLIME's ``async_rm``):
    1. ``args.custom_converter_path`` → load and call user function.
    2. ``args.converter_type`` → built-in implementation.

    Args:
        args: Argument namespace. Relevant fields:
            - ``converter_type``: ``"single_turn"`` (default) or ``"multistep"``.
            - ``custom_converter_path``: Dotted import path to a custom function.
            - ``converter_llm_match``: Regex for LLM span name (single_turn only).
        rollout: Completed rollout from ``LightningStore``.
        spans: All spans for the latest attempt, sorted by ``sequence_id``.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Populated ``Sample`` ready for SLIME training, or ``None`` on failure.
    """
    custom_path = getattr(args, "custom_converter_path", None)
    if custom_path:
        fn = _load_function(custom_path)
        return fn(args, rollout, spans, tokenizer)

    converter_type = (getattr(args, "converter_type", None) or "single_turn").strip()

    if converter_type == "single_turn":
        return _single_turn(args, rollout, spans, tokenizer)
    elif converter_type == "multistep":
        return _multistep(args, rollout, spans, tokenizer)
    else:
        raise NotImplementedError(
            f"converter_type={converter_type!r} is not implemented. "
            "Use 'single_turn', 'multistep', or set args.custom_converter_path."
        )


# ────────────────────────────────────────────────────────────────────────────
# Built-in: single_turn
# ────────────────────────────────────────────────────────────────────────────

def _single_turn(
    args: Namespace,
    rollout: Rollout,
    spans: Sequence[Span],
    tokenizer: PreTrainedTokenizer,
) -> Optional[Sample]:
    """One LLM call per rollout (default SGLangLitAgent path).

    Finds the last span whose name matches ``args.converter_llm_match``,
    extracts response text and optional log-probs, then tokenises
    ``prompt_text + response`` and builds a ``[0…0, 1…1]`` loss mask.
    """
    llm_match = getattr(args, "converter_llm_match", r"openai\.chat\.completion")

    # --- Prompt ---
    task_input = rollout.input
    prompt: str | list[dict]
    label: Optional[str] = None

    if isinstance(task_input, dict):
        prompt = task_input.get("messages", task_input.get("prompt", str(task_input)))
        label = task_input.get("label")
    else:
        prompt = str(task_input)

    # --- Last matching LLM span ---
    llm_spans = [s for s in spans if re.search(llm_match, s.name)]
    if not llm_spans:
        logger.warning(
            "rollout %s [single_turn]: no LLM spans matched %r",
            rollout.rollout_id, llm_match,
        )
        return None

    last_span = max(llm_spans, key=lambda s: s.sequence_id)
    response, log_probs = _response_and_logprobs_from_otel(last_span)
    if not response:
        logger.warning(
            "rollout %s [single_turn]: could not extract response text", rollout.rollout_id
        )
        return None

    # --- Reward ---
    reward = find_final_reward(list(spans))
    if reward is None:
        logger.warning("rollout %s [single_turn]: no reward span — defaulting to 0.0", rollout.rollout_id)
        reward = 0.0

    # --- Tokenise ---
    prompt_text = (
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        if isinstance(prompt, list)
        else str(prompt)
    )
    full_tokens = tokenizer.encode(prompt_text + response, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_length = len(full_tokens) - len(prompt_tokens)

    if response_length <= 0:
        logger.warning("rollout %s [single_turn]: empty response after tokenisation", rollout.rollout_id)
        return None

    loss_mask = [0] * len(prompt_tokens) + [1] * response_length

    return Sample(
        prompt=prompt,
        tokens=full_tokens,
        response=response,
        response_length=response_length,
        label=label,
        reward=float(reward),
        loss_mask=loss_mask,
        rollout_log_probs=log_probs,
        status=Sample.Status.COMPLETED,
        metadata={"rollout_id": rollout.rollout_id, **(rollout.metadata or {})},
    )


# ────────────────────────────────────────────────────────────────────────────
# Built-in: multistep
# ────────────────────────────────────────────────────────────────────────────

def _multistep(
    args: Namespace,
    rollout: Rollout,
    spans: Sequence[Span],
    tokenizer: PreTrainedTokenizer,  # noqa: ARG001 — kept for API uniformity
) -> Optional[Sample]:
    """Multi-turn tool-loop converter for ClaudeCodeAgent-style agents.

    Reads ``raw_gen_ai_request`` spans emitted by the AGL LLM proxy.  Each
    span carries the *full cumulative* ``prompt_token_ids`` and the
    model-only ``response_token_ids`` for that turn.

    The full trajectory is stitched as::

        tokens    = P1 | R1 | Δ12 | R2 | Δ23 | R3 | … | RN
        loss_mask = 0…  1…  0…    1…  0…    1…       1…

    where ``Δi(i+1) = P(i+1)[len(Pi) + len(Ri):]`` (the tool-result / env
    tokens appended between turns).

    Log-probs from all turns are concatenated in the same order as the
    response tokens and stored in ``Sample.rollout_log_probs``.
    """
    # Sort all spans deterministically.
    sorted_spans = sorted(spans, key=lambda s: (s.sequence_id, s.start_time or 0))

    # Collect raw_gen_ai_request spans in turn order.
    turns: list[dict] = []
    seen_request_ids: set[str] = set()

    for s in sorted_spans:
        if s.name != "raw_gen_ai_request":
            continue
        attrs = s.attributes or {}
        prompt_ids, resp_ids, logprobs = _tokens_from_llmproxy(attrs)

        if not prompt_ids or not resp_ids:
            logger.warning(
                "rollout %s [multistep]: span %s missing token IDs — skipping",
                rollout.rollout_id, s.span_id,
            )
            continue

        rid = _request_id(attrs)
        if rid and rid in seen_request_ids:
            continue
        if rid:
            seen_request_ids.add(rid)

        if logprobs and len(logprobs) != len(resp_ids):
            logger.warning(
                "rollout %s [multistep]: span %s logprob/token length mismatch (%d vs %d) — dropping logprobs",
                rollout.rollout_id, s.span_id, len(logprobs), len(resp_ids),
            )
            logprobs = []

        turns.append({"prompt_ids": prompt_ids, "resp_ids": resp_ids, "logprobs": logprobs})

    if not turns:
        logger.warning("rollout %s [multistep]: no valid raw_gen_ai_request spans", rollout.rollout_id)
        return None

    # --- Stitch trajectory ---
    all_tokens: list[int] = []
    loss_mask: list[int] = []
    all_logprobs: list[float] = []
    have_logprobs = all(t["logprobs"] for t in turns)

    for i, turn in enumerate(turns):
        p_ids = turn["prompt_ids"]
        r_ids = turn["resp_ids"]

        if i == 0:
            # First turn: emit the full prompt (masked) then the response.
            all_tokens.extend(p_ids)
            loss_mask.extend([0] * len(p_ids))
        else:
            # Subsequent turns: emit only the DELTA tokens (tool result /
            # environment observation) between the end of the previous
            # response and the start of this prompt.
            prev = turns[i - 1]
            prev_end = len(prev["prompt_ids"]) + len(prev["resp_ids"])
            delta = p_ids[prev_end:]
            all_tokens.extend(delta)
            loss_mask.extend([0] * len(delta))

        all_tokens.extend(r_ids)
        loss_mask.extend([1] * len(r_ids))
        if have_logprobs:
            all_logprobs.extend(turn["logprobs"])

    # --- Reward (trajectory-level) ---
    reward = find_final_reward(list(spans))
    if reward is None:
        logger.warning("rollout %s [multistep]: no reward span — defaulting to 0.0", rollout.rollout_id)
        reward = 0.0

    # Build a readable text representation of the full response for reference.
    full_response = "".join(
        tokenizer.decode(t["resp_ids"], skip_special_tokens=True) for t in turns
    )

    # prompt field: the first turn's messages (for logging / inspection).
    task_input = rollout.input
    prompt: str | list[dict] = (
        task_input.get("messages", task_input.get("prompt", str(task_input)))
        if isinstance(task_input, dict)
        else str(task_input)
    )
    label: Optional[str] = task_input.get("label") if isinstance(task_input, dict) else None

    return Sample(
        prompt=prompt,
        tokens=all_tokens,
        response=full_response,
        response_length=sum(len(t["resp_ids"]) for t in turns),
        label=label,
        reward=float(reward),
        loss_mask=loss_mask,
        rollout_log_probs=all_logprobs if have_logprobs else None,
        status=Sample.Status.COMPLETED,
        metadata={
            "rollout_id": rollout.rollout_id,
            "num_turns": len(turns),
            **(rollout.metadata or {}),
        },
    )


# ────────────────────────────────────────────────────────────────────────────
# Span attribute helpers
# ────────────────────────────────────────────────────────────────────────────

def _response_and_logprobs_from_otel(span: Span) -> tuple[Optional[str], Optional[list[float]]]:
    """Extract response text + log-probs from an OpenAI-OTEL convention span."""
    attrs = span.attributes or {}
    response: Optional[str] = None

    raw = attrs.get("gen_ai.completion")
    if raw is not None:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                response = first.get("text") or (first.get("message") or {}).get("content") or ""
            elif isinstance(parsed, str):
                response = parsed
        except (json.JSONDecodeError, AttributeError, TypeError):
            response = str(raw)

    if not response:
        choices_raw = attrs.get("agentlightning.operation.output.choices")
        if choices_raw is not None:
            try:
                choices = json.loads(choices_raw) if isinstance(choices_raw, str) else choices_raw
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    response = first.get("text") or (first.get("message") or {}).get("content") or ""
            except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
                pass

    log_probs: Optional[list[float]] = None
    lp_raw = attrs.get("logprobs.content")
    if lp_raw is not None:
        try:
            lp_data = json.loads(lp_raw) if isinstance(lp_raw, str) else lp_raw
            if isinstance(lp_data, list):
                log_probs = [item["logprob"] for item in lp_data if "logprob" in item]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return response, log_probs


def _tokens_from_llmproxy(attrs: dict) -> tuple[list[int], list[int], list[float]]:
    """Extract token IDs and log-probs from a ``raw_gen_ai_request`` span."""
    prompt_ids: list[int] = []
    resp_ids: list[int] = []
    logprobs: list[float] = []

    def _parse(val):  # noqa: ANN001, ANN202
        if isinstance(val, str):
            try:
                import ast
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return None
        return val

    p = _parse(attrs.get("llm.hosted_vllm.prompt_token_ids"))
    if isinstance(p, list) and all(isinstance(x, int) for x in p):
        prompt_ids = p

    choices = _parse(attrs.get("llm.hosted_vllm.choices"))
    if isinstance(choices, list) and choices:
        cand = choices[0]
        if isinstance(cand, dict):
            tids = cand.get("token_ids")
            if isinstance(tids, list) and all(isinstance(x, int) for x in tids):
                resp_ids = tids
            lp_dict = cand.get("logprobs")
            if isinstance(lp_dict, dict) and "content" in lp_dict:
                logprobs = [float(item["logprob"]) for item in lp_dict["content"] if "logprob" in item]

    return prompt_ids, resp_ids, logprobs


def _request_id(attrs: dict) -> Optional[str]:
    return attrs.get("llm.hosted_vllm.id") or attrs.get("gen_ai.response.id")


# ────────────────────────────────────────────────────────────────────────────
# Module loader (mirrors slime.utils.misc.load_function)
# ────────────────────────────────────────────────────────────────────────────

def _load_function(path: str):  # noqa: ANN202
    """Load ``module.submodule.function`` the same way SLIME does."""
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
