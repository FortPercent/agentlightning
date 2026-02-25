# Copyright (c) Microsoft. All rights reserved.

"""Controller module for managing Claude Code executions in containerized environments.

This module provides the ClaudeController class that manages the execution of Claude Code
within Docker containers. It handles container initialization, command execution, and
patch application for SWE-bench evaluation tasks.
"""

import datetime
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from typing import IO, Literal, Optional, TypedDict

import dotenv
from swebench.harness.constants import SWEbenchInstance
from swebench_utils.docker_runtime import Runtime
from swebench_utils.logging import log_for_evaluation
import json
import requests
import time
import shlex

SWEBENCH_EXTRA_SYSTEM_PROMPT = """
You are an expert software engineer solving swebench bug fixing tasks.

Tool calling guidelines:
- When multiple operations are INDEPENDENT (e.g., reading several unrelated files, running multiple unrelated checks), call those tools IN PARALLEL by including multiple tool calls in a single response.
- Only call tools sequentially when the output of one is needed as input to the next.
- Prefer parallel tool calls wherever possible to save time.
"""

PLANNING_SYSTEM_PROMPT = """You are a planning agent for software bug-fixing tasks.
Given a bug description and a total turn budget, output a JSON execution plan.

Critical output rules:
- Output ONLY valid JSON. No prose, no markdown, no prefixes.
- Do NOT output "Thinking Process", analysis, rationale, or any text outside JSON.
- Return exactly one JSON object with top-level key "steps".

Plan rules:
- Each step must have:
  - "name": short unique slug
  - "instruction": concrete executable task
  - "max_turns": integer >= 1
- Steps are executed sequentially.
- Typical workflow: reproduce, locate, fix, validate.
- Total budget constraint is strict:
  - sum(step.max_turns for step in steps) <= TOTAL_TURN_BUDGET
- Prefer 3-5 steps unless clearly necessary.

Output schema:
{
  "steps": [
    {"name": "reproduce", "instruction": "...", "max_turns": 1},
    {"name": "locate",    "instruction": "...", "max_turns": 1},
    {"name": "fix",       "instruction": "...", "max_turns": 2},
    {"name": "validate",  "instruction": "...", "max_turns": 1}
  ]
}
"""

REPLANNING_SYSTEM_PROMPT = """You are an adaptive bug-fixing planner.
You run in a loop: Plan -> Act -> Observe -> Replan.

Rules:
- Output ONLY valid JSON.
- Decide the single BEST next step from latest observations.
- Keep step focused and executable with tools in one short burst.
- Each step is exactly one turn. Always set `max_turns` to 1.
- If the task appears complete, set `stop` to true.

Output schema:
{
  "stop": false,
  "reason": "why this next step",
  "step": {
    "name": "short_slug",
    "instruction": "concrete instruction for the next action",
    "max_turns": 1
  }
}
"""

SWEBENCH_USER_PROMPT = """
You are given a code repository in the current directory (/testbed).
The bug description is:
{description}
=================================================
You task is to fix the bug with the following steps:
(1) write test cases to reproduce the bug.
(2) explore the source codes to locate the bug.
(3) edit the source codes to fix the bug.
(4) rerun your written test cases to validate that the bug is fixed. If not, go back to explore the source codes and fix the codes again.
(5) remember to delete the test cases you write at last.
Please do not commit your edits. We will do it later.
"""

logger = logging.getLogger("claude_code_agent")

import json
import re
import requests
from typing import Literal, TypedDict, Any, Dict, List, Optional, cast


class StreamLogger:
    """Writes structured JSONL event records to a file, one JSON object per line.

    Each `emit()` call appends one record with a UTC timestamp and any extra
    keyword arguments supplied by the caller.  Pass ``path=None`` to create a
    no-op logger that discards all events (useful when streaming is disabled).
    """

    def __init__(self, path: Optional[str]) -> None:
        self._file: Optional[IO[str]] = open(path, "a", encoding="utf-8") if path else None
        self._lock = threading.Lock()

    def emit(self, **kwargs: Any) -> None:
        if self._file is None:
            return
        kwargs.setdefault("time", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
        with self._lock:
            self._file.write(json.dumps(kwargs, ensure_ascii=False) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


def _default_list() -> List[str]:
    return []


@dataclass
class SubAgent:
    """Represents a sub-task in the multi-step pipeline.

    Each SubAgent has its own independent container and tool loop.
    """

    name: str
    instruction: str
    max_turns: int
    result: str = ""
    depends_on: List[str] = field(default_factory=_default_list)


class RunInstanceResult(TypedDict):
    instance_id: str
    model_patch: str
    model_name_or_path: str


class ClaudeController:
    """Manages the execution of Claude Code within a Docker runtime.

    This controller handles the lifecycle of a SWE-bench task execution, including
    environment setup, tool installation, agent execution (via CLI or Python SDK),
    and result extraction.

    Attributes:
        container: The active Docker runtime session.
    """

    def __init__(self, image: str, instance: SWEbenchInstance, run_id: str, endpoint: str, api_key: str) -> None:
        """Initialize the ClaudeController.

        Args:
            image: The Docker image tag.
            instance: The dataset instance containing the problem statement and ID.
            run_id: The identifier for the evaluation run.
            endpoint: The API endpoint URL.
            api_key: The API authentication key.
        """
        self.image = image
        self.instance = instance
        self.run_id = run_id
        self.endpoint = endpoint
        self.api_key = api_key
        self.container: Runtime = self.init_container(self.image, self.instance)
        self._log: StreamLogger = StreamLogger(None)  # replaced by run_instance when stream_path is set
        self._last_tool_loop_turns: int = 0

    def init_container(self, image: str, instance: SWEbenchInstance) -> Runtime:
        """Initializes the Docker container and sets up the Claude Code environment.

        This method starts the container session, installs the Claude CLI,
        configures environment variables for authentication and sandbox mode.

        Args:
            image: The Docker image tag to start.
            instance: The dataset instance to load into the environment.

        Returns:
            An initialized and configured Docker runtime object.
        """
        container = Runtime.start_session(
            image,
            instance,
            log_function=partial(log_for_evaluation, run_id=self.run_id, instance_id=instance["instance_id"]),
        )
        # Install Claude CLI
        # container.send_command("curl -fsSL https://claude.ai/install.sh | bash")
        # container.send_command('alias claude="$HOME/.local/bin/claude"')

        # Configure Environment
        dotenv.load_dotenv()
        container.send_command(f"export ANTHROPIC_BASE_URL={self.endpoint}")
        container.send_command(f"export ANTHROPIC_AUTH_TOKEN={self.api_key}")
        container.send_command("export IS_SANDBOX=1")

        return container

    def _run_cli(self, instance: SWEbenchInstance, max_turns: int, time_limit: int) -> None:
        """Executes Claude Code using the Command Line Interface.

        Constructs a safe heredoc for the prompt to avoid shell interpolation issues
        and executes the `claude` binary directly.

        Args:
            instance: The problem instance containing the problem statement.
            max_turns: The maximum number of interaction turns allowed.
            time_limit: The execution time limit in minutes.
        """
        # Prepare prompt safely: write it to a file inside the container using a single-quoted heredoc
        # directly applying prompt for heredoc may raise error for windows line ending \r\n
        prompt_text = SWEBENCH_USER_PROMPT.format(description=instance["problem_statement"].replace('"""', "'''"))

        # Choose a simple filename and a heredoc delimiter unlikely to collide
        heredoc_cmd = "cat > /tmp/cc_prompt.txt <<'CC_PROMPT'\n" + prompt_text + "\nCC_PROMPT\n"
        self.container.send_command(heredoc_cmd)

        # Run claude reading the prompt from the file
        claude_cmd = (
            f'claude -p "$(cat /tmp/cc_prompt.txt)" '
            f'--append-system-prompt "{SWEBENCH_EXTRA_SYSTEM_PROMPT}" '
            f"--max-turns {max_turns} "
            f"--dangerously-skip-permissions "
            f"--output-format json --verbose"
        )
        logger.info(f"Running Claude Code CLI command: {claude_cmd}")
        self.container.send_command(claude_cmd, time_limit * 60)
        logger.info(f"Claude Code CLI command completed")
    
    def _build_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_bash",
                    "description": "Run shell command in sandbox container",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read UTF-8 text file content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_bytes": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write UTF-8 text file content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "append": {"type": "boolean"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Apply unified diff patch in /testbed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patch": {"type": "string"},
                            "strip": {"type": "integer"},
                        },
                        "required": ["patch"],
                    },
                },
            },
        ]

    def _exec_container_tool(self, container: Runtime, name: str, args: Dict[str, Any], time_limit: int) -> str:
        """Execute one tool in container and return JSON string."""
        def _ok(content: str, **meta: Any) -> str:
            return json.dumps({"ok": True, "content": content, "metadata": meta}, ensure_ascii=False)

        def _err(content: str, **meta: Any) -> str:
            return json.dumps({"ok": False, "content": content, "metadata": meta}, ensure_ascii=False)

        def _tool_matches_env(var_name: str) -> bool:
            raw = os.getenv(var_name, "")
            if not raw:
                return False
            selected = {x.strip() for x in raw.split(",") if x.strip()}
            return "*" in selected or name in selected

        def _maybe_debug_breakpoint() -> None:
            # Example: CC_BREAK_ON_TOOL=run_bash (or "*") to break before execution.
            if _tool_matches_env("CC_BREAK_ON_TOOL"):
                logger.warning("[tool-loop] Breakpoint before tool execution: tool=%s args=%r", name, args)
                breakpoint()

        def _maybe_debug_raise() -> None:
            # Example: CC_RAISE_ON_TOOL=run_bash (or "*") to fail fast in non-interactive runs.
            if _tool_matches_env("CC_RAISE_ON_TOOL"):
                raise RuntimeError(f"[tool-loop] Debug stop before tool execution: tool={name} args={args!r}")

        try:
            timeout_default = min(120, time_limit * 60)
            _maybe_debug_raise()
            _maybe_debug_breakpoint()

            if name == "run_bash":
                cmd = args["cmd"]
                timeout = int(args.get("timeout", timeout_default))
                blocked = ["rm -rf /", ":(){:|:&};:", "shutdown", "reboot"]
                if any(x in cmd.lower() for x in blocked):
                    return _err(f"Blocked command: {cmd}", tool=name)
                full = f"cd /testbed && {cmd}"
                logger.info("[tool-loop] run_bash dispatch: cmd=%r timeout=%s", full, timeout)
                out = container.send_command(full, timeout)
                txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                logger.info("[tool-loop] run_bash completed: output_len=%d", len(txt))
                return _ok(txt, tool=name, cmd=full, timeout=timeout)

            if name == "read_file":
                path = args["path"]
                max_bytes = int(args.get("max_bytes", 60000))
                py = (
                    "import os\n"
                    f"p={path!r}\n"
                    "if not os.path.isabs(p): p=os.path.join('/testbed', p)\n"
                    f"b=open(p,'rb').read({max_bytes})\n"
                    "print(b.decode('utf-8', errors='replace'))\n"
                )
                cmd = f"cd /testbed && python - <<'PY'\n{py}PY"
                out = container.send_command(cmd, timeout_default)
                txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                return _ok(txt, tool=name, path=path, max_bytes=max_bytes)

            if name == "write_file":
                path = args["path"]
                content = args["content"]
                append = bool(args.get("append", False))
                mode = "a" if append else "w"
                py = (
                    "import os\n"
                    f"p={path!r}\n"
                    "if not os.path.isabs(p): p=os.path.join('/testbed', p)\n"
                    "os.makedirs(os.path.dirname(p), exist_ok=True)\n"
                    f"open(p,{mode!r},encoding='utf-8').write({content!r})\n"
                    "print('OK')\n"
                )
                cmd = f"cd /testbed && python - <<'PY'\n{py}PY"
                out = container.send_command(cmd, timeout_default)
                txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                return _ok(txt, tool=name, path=path, append=append)

            if name == "apply_patch":
                patch = args["patch"]
                strip = int(args.get("strip", 0))
                cmd = (
                    "cd /testbed && "
                    "cat > /tmp/agent.patch <<'PATCH'\n"
                    f"{patch}\n"
                    "PATCH\n"
                    f"(git apply --whitespace=nowarn -p{strip} /tmp/agent.patch || patch -p{strip} < /tmp/agent.patch)"
                )
                out = container.send_command(cmd, timeout_default)
                txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                return _ok(txt, tool=name, strip=strip)

            return _err(f"Unknown tool: {name}", tool=name, args=args)

        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", tool=name, args=args)

    def _tool_loop(
        self,
        container: Runtime,
        messages: List[Dict[str, Any]],
        max_turns: int,
        time_limit: int,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> str:
        """Generic multi-turn tool loop that executes in the specified container.

        Args:
            container: The Docker runtime container to execute tools in.
            messages: Initial conversation history.
            max_turns: Maximum number of LLM interaction turns.
            time_limit: Time limit for tool execution in minutes.
            model: The model name to use for LLM calls.

        Returns:
            The final text response from the LLM.
        """
        url = f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if getattr(self, "api_key", None):
            headers["Authorization"] = f"Bearer {self.api_key}"

        tools = self._build_tools_schema()
        final_text = ""

        def _extract_content_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join([x.get("text", "") for x in content if x.get("type") == "text"])
            return str(content) if content else ""

        def _tool_args_to_dict(args: Any) -> Dict[str, Any]:
            if isinstance(args, dict):
                return args
            if isinstance(args, str):
                try:
                    loaded = json.loads(args)
                    return loaded if isinstance(loaded, dict) else {}
                except Exception:
                    return {}
            return {}

        def _tool_args_to_string(args: Any) -> str:
            if isinstance(args, str):
                return args
            if isinstance(args, dict):
                return json.dumps(args, ensure_ascii=False)
            return "{}"

        def _normalize_messages_for_api(src: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            normalized: List[Dict[str, Any]] = []
            for m in src:
                role = m.get("role")
                if role == "assistant" and isinstance(m.get("tool_calls"), list):
                    tool_calls_out: List[Dict[str, Any]] = []
                    for tc in m.get("tool_calls", []):
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function") or {}
                        if not isinstance(fn, dict):
                            fn = {}
                        tool_calls_out.append(
                            {
                                "id": tc.get("id", "sf"),
                                "type": "function",
                                "function": {
                                    "name": fn.get("name"),
                                    "arguments": _tool_args_to_string(fn.get("arguments", "{}")),
                                },
                            }
                        )
                    normalized.append(
                        {
                            "role": "assistant",
                            "content": m.get("content", ""),
                            "tool_calls": tool_calls_out,
                        }
                    )
                elif role == "tool":
                    normalized.append(
                        {
                            "role": "tool",
                            "tool_call_id": m.get("tool_call_id"),
                            "name": m.get("name"),
                            "content": str(m.get("content", "")),
                        }
                    )
                else:
                    normalized.append(m)
            return normalized

        def _parse_qwen_xml_tool_calls(content: str) -> List[Dict[str, Any]]:
            """Parses Qwen specific XML format for tool calls."""
            tool_calls = []
            tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
            func_name_pattern = re.compile(r"<function=(.*?)>", re.DOTALL)
            param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)

            for match in tool_call_pattern.finditer(content):
                block = match.group(1)
                fn_match = func_name_pattern.search(block)
                if not fn_match:
                    continue
                func_name = fn_match.group(1).strip()

                arguments = {}
                for p_match in param_pattern.finditer(block):
                    key = p_match.group(1).strip()
                    val = p_match.group(2)
                    arguments[key] = val

                tool_calls.append({
                    "id": f"call_{len(tool_calls)}_{func_name}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False)
                    }
                })
            return tool_calls

        # Pre-flight check
        try:
            container.send_command("cd /testbed && git config --global --add safe.directory /testbed", timeout=10)
        except Exception:
            pass

        turns_used = 0
        for turn in range(max_turns):
            turns_used = turn + 1
            payload = {
                "model": model,
                "messages": _normalize_messages_for_api(messages),
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.0,
                "max_tokens": 4096,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }

            logger.info(f"[tool-loop] turn={turn+1} invoking model...")
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=time_limit * 60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"API Request failed: {e}")
                break

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content_raw = msg.get("content", "")
            content_text = _extract_content_text(content_raw)

            # Try native parsing first, fallback to XML
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls and "<tool_call>" in content_text:
                tool_calls = _parse_qwen_xml_tool_calls(content_text)

            logger.info(f"[tool-loop] turn={turn+1} content_len={len(content_text)} tool_calls={len(tool_calls)}")

            # Emit structured log event
            usage: Dict[str, Any] = data.get("usage") or {}
            tc_summary: List[Dict[str, Any]] = []
            for tc in tool_calls:
                tc_dict = cast(Dict[str, Any], tc)
                fn_info: Dict[str, Any] = cast(Dict[str, Any], tc_dict.get("function") or {})
                tc_summary.append({"name": fn_info.get("name"), "args": fn_info.get("arguments")})
            self._log.emit(
                type="llm_response",
                turn=turn + 1,
                content=content_text[:2000],
                tool_calls=tc_summary,
                finish_reason=choice.get("finish_reason"),
                usage=usage,
            )

            # No tool calls -> final answer
            if not tool_calls:
                final_text = content_text
                messages.append({"role": "assistant", "content": final_text})
                break

            # Sanitize tool calls
            sanitized_tool_calls_for_api: List[Dict[str, Any]] = []
            sanitized_tool_calls_for_exec: List[Dict[str, Any]] = []
            for tc in tool_calls:
                tc_dict = tc if isinstance(tc, dict) else {}
                fn = tc_dict.get("function") or {}
                if not isinstance(fn, dict):
                    fn = {}
                args_raw = fn.get("arguments", "{}")
                args_for_exec = _tool_args_to_dict(args_raw)
                args_for_api = _tool_args_to_string(args_raw)
                tool_call_id = tc_dict.get("id", "sf")
                fn_name = fn.get("name")

                sanitized_tool_calls_for_api.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": args_for_api
                    }
                })
                sanitized_tool_calls_for_exec.append({
                    "id": tool_call_id,
                    "function": {
                        "name": fn_name,
                        "arguments": args_for_exec
                    }
                })

            messages.append({
                "role": "assistant",
                "content": content_text,
                "tool_calls": sanitized_tool_calls_for_api
            })

            # Execute tools — run in parallel when multiple tool calls are returned.
            def _exec_one(tc: Dict[str, Any]) -> Dict[str, Any]:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                tcid = tc["id"]
                start_msg = f"[tool-loop] TOOL_START id={tcid} name={fn_name} args={fn_args}"
                print(start_msg)
                logger.info(start_msg)
                result_str = self._exec_container_tool(container, fn_name, fn_args, time_limit=time_limit)
                try:
                    parsed = json.loads(result_str)
                    ok = parsed.get("ok")
                    content = str(parsed.get("content", ""))
                    end_msg = (
                        f"[tool-loop] TOOL_END id={tcid} name={fn_name} ok={ok} "
                        f"content_preview={content[:200]!r}"
                    )
                except Exception:
                    end_msg = f"[tool-loop] TOOL_END id={tcid} name={fn_name} raw_preview={result_str[:200]!r}"
                print(end_msg)
                logger.info(end_msg)
                self._log.emit(type="tool_result", tool=fn_name, args=fn_args, result=result_str[:2000])
                return {"tool_call_id": tcid, "name": fn_name, "content": result_str}

            if len(sanitized_tool_calls_for_exec) == 1:
                tool_results = [_exec_one(sanitized_tool_calls_for_exec[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(sanitized_tool_calls_for_exec)) as _pool:
                    # Preserve original order so tool_call_id pairing stays correct.
                    tool_results = list(_pool.map(_exec_one, sanitized_tool_calls_for_exec))

            for tr in tool_results:
                messages.append({"role": "tool", **tr})

        self._last_tool_loop_turns = turns_used
        return final_text

    def _run_sub_agent(
        self,
        sub: SubAgent,
        container: Runtime,
        context: str,
        time_limit: int,
    ) -> SubAgent:
        """Execute a single sub-agent using the provided container.

        The caller is responsible for the container's lifecycle (creation and cleanup).
        For sequential pipelines pass `self.container`; for parallel execution pass
        a dedicated container per sub-agent.

        Args:
            sub: The SubAgent configuration.
            container: The Docker runtime in which tools will execute.
            context: Summary of results from previous steps.
            time_limit: Time limit for tool execution in minutes.

        Returns:
            The SubAgent with the `result` field filled in.
        """
        logger.info(f"[sub-agent] Starting sub-agent: {sub.name}")
        # Use per-step budget from planning output (or fallback to 1).
        effective_max_turns = max(1, int(sub.max_turns or 1))
        self._log.emit(type="step_start", step=sub.name, instruction=sub.instruction, max_turns=effective_max_turns)

        messages = self._build_sub_agent_messages(sub, context)
        sub.max_turns = effective_max_turns
        sub.result = self._tool_loop(container, messages, effective_max_turns, time_limit)

        logger.info(f"[sub-agent] Completed sub-agent: {sub.name}")
        self._log.emit(type="step_end", step=sub.name, result=sub.result[:1000])

        return sub

    def _build_sub_agent_messages(self, sub: SubAgent, context: str) -> List[Dict[str, Any]]:
        """Build initial chat messages for a sub-agent."""
        system_prompt = (
            SWEBENCH_EXTRA_SYSTEM_PROMPT
            + "\n\nYou are working on a specific step of a multi-step task."
            + "\nFocus only on your assigned task."
        )

        user_prompt = ""
        if context:
            user_prompt += f"Context from previous steps:\n{context}\n\n"
        user_prompt += f"Your current task:\n{sub.instruction}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _latest_assistant_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the latest assistant text from a message list."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
        return ""

    def _run_sub_agents_parallel(
        self,
        subs: List[SubAgent],
        instance: SWEbenchInstance,
        context: str,
        time_limit: int,
        max_workers: int = 4,
    ) -> List[SubAgent]:
        """Execute multiple sub-agents in parallel, each with its own container.

        Args:
            subs: List of SubAgent configurations to execute.
            instance: The SWE-bench instance.
            context: Shared context for all sub-agents.
            time_limit: Time limit for each sub-agent in minutes.
            max_workers: Maximum number of parallel workers.

        Returns:
            List of completed SubAgents with results.
        """
        logger.info(f"[parallel] Starting {len(subs)} sub-agents in parallel")

        # Each parallel sub-agent gets its own container for isolation.
        containers: List[Runtime] = [self.init_container(self.image, instance) for _ in subs]  # type: ignore[no-untyped-call]
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._run_sub_agent, sub, container, context, time_limit): sub
                    for sub, container in zip(subs, containers)
                }

                results: List[SubAgent] = []
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        sub = futures[future]
                        logger.error(f"[parallel] Sub-agent {sub.name} failed: {e}")
                        sub.result = f"ERROR: {e}"
                        results.append(sub)
        finally:
            for container in containers:
                container.cleanup()

        logger.info(f"[parallel] Completed {len(results)} sub-agents")
        return results

    def _get_default_plan(self, instance: SWEbenchInstance, max_turns: int) -> List[SubAgent]:
        """Return the hardcoded 4-step plan as a fallback.

        Args:
            instance: The SWE-bench instance.
            max_turns: Total turns to distribute evenly across steps.

        Returns:
            List of SubAgent objects for the default pipeline.
        """
        per_step = max(3, max_turns // 4)
        return [
            SubAgent(
                name="reproduce",
                instruction=(
                    "Step 1: Write test cases to reproduce the bug.\n"
                    f"Bug description:\n{instance['problem_statement']}\n\n"
                    "Tasks:\n"
                    "- Create a test file at /testbed/test_repro.py\n"
                    "- Run the test to confirm the bug exists\n"
                    "- Output the test failure"
                ),
                max_turns=per_step,
            ),
            SubAgent(
                name="locate",
                instruction=(
                    "Step 2: Explore the source code to find the root cause of the bug.\n\n"
                    "Tasks:\n"
                    "- Read relevant source files\n"
                    "- Identify the exact file(s) and line(s) that need to be changed\n"
                    "- Explain the root cause"
                ),
                max_turns=per_step,
            ),
            SubAgent(
                name="fix",
                instruction=(
                    "Step 3: Edit the source code to fix the bug.\n\n"
                    "Tasks:\n"
                    "- Apply minimal, targeted changes to fix the bug\n"
                    "- Do NOT modify test files\n"
                    "- Ensure the fix addresses the root cause identified earlier"
                ),
                max_turns=per_step,
            ),
            SubAgent(
                name="validate",
                instruction=(
                    "Step 4: Validate the fix by running tests.\n\n"
                    "Tasks:\n"
                    "- Run the reproduction test from step 1\n"
                    "- Confirm the test now passes\n"
                    "- Delete /testbed/test_repro.py after validation\n"
                    "- Summarize the fix"
                ),
                max_turns=per_step,
            ),
        ]

    def _generate_plan(
        self,
        instance: SWEbenchInstance,
        max_turns: int,
        time_limit: int,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> List[SubAgent]:
        """Ask the LLM to generate a dynamic execution plan for this bug.

        Calls the LLM once (no tools) with the bug description and expects a
        JSON response containing a list of steps. Falls back to the default
        4-step plan on any error.

        Args:
            instance: The SWE-bench instance.
            max_turns: Budget hint passed to the planner so it can distribute turns.
            time_limit: Per-request timeout in minutes.
            model: Model to use for the planning call.

        Returns:
            List of SubAgent objects derived from the LLM's plan.
        """
        url = f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        planning_user_prompt = (
            f"Bug description:\n{instance['problem_statement']}\n\n"
            f"Total turn budget: {max_turns}.\n"
            "Generate a focused execution plan. Output ONLY the JSON object."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": planning_user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        logger.info("[planner] Requesting dynamic plan from LLM...")
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=time_limit * 60)
            resp.raise_for_status()
            data = resp.json()

            choices: List[Any] = data.get("choices") or [{}]
            raw_content: str = str(choices[0].get("message", {}).get("content", "") or "")
            logger.info(f"[planner] Raw plan response: {raw_content[:500]}")

            # Strip markdown code fences if present
            cleaned: str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.MULTILINE).strip()
            plan_data: Dict[str, Any] = json.loads(cleaned)
            steps_raw: List[Dict[str, Any]] = plan_data.get("steps", [])

            if not steps_raw:
                raise ValueError("Plan contained no steps")

            sub_agents: List[SubAgent] = []
            for step in steps_raw:
                step_name: str = str(step.get("name", f"step_{len(sub_agents)}")).strip()
                instruction: str = str(step.get("instruction", "")).strip()
                mt: int = int(step.get("max_turns", max(3, max_turns // len(steps_raw))))
                if not instruction:
                    logger.warning(f"[planner] Step '{step_name}' has empty instruction, skipping")
                    continue
                sub_agents.append(SubAgent(name=step_name, instruction=instruction, max_turns=mt))

            logger.info(f"[planner] Generated plan with {len(sub_agents)} steps: {[s.name for s in sub_agents]}")
            return sub_agents

        except Exception as e:
            logger.warning(f"[planner] Plan generation failed ({e}), falling back to default plan")
            return self._get_default_plan(instance, max_turns)

    def _format_replan_history(self, history: List[Dict[str, Any]], max_items: int = 6) -> str:
        """Format recent Plan/Act/Observe records for replanning."""
        if not history:
            return "No previous cycles yet."
        blocks: List[str] = []
        for item in history[-max_items:]:
            blocks.append(
                f"[cycle {item.get('cycle', '?')}] step={item.get('name', 'unknown')} "
                f"turns={item.get('turns_used', '?')}\n"
                f"Result:\n{str(item.get('result', ''))[:800]}\n"
                f"Observation:\n{str(item.get('observation', ''))[:800]}"
            )
        return "\n\n".join(blocks)

    def _collect_replan_observation(self) -> str:
        """Collect lightweight repo feedback after a step execution."""
        snippets: List[str] = []
        try:
            diff_stat = self.container.send_command("cd /testbed && git --no-pager diff --stat", timeout=30).output.strip()
            snippets.append(f"git diff --stat:\n{diff_stat[:1200]}")
        except Exception as e:
            snippets.append(f"git diff --stat failed: {e}")
        try:
            diff_names = self.container.send_command("cd /testbed && git --no-pager diff --name-only", timeout=30).output.strip()
            snippets.append(f"modified files:\n{diff_names[:800]}")
        except Exception as e:
            snippets.append(f"git diff --name-only failed: {e}")
        return "\n\n".join(snippets)

    def _generate_next_step(
        self,
        instance: SWEbenchInstance,
        history: List[Dict[str, Any]],
        remaining_turns: int,
        time_limit: int,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> Dict[str, Any]:
        """Generate the next step using Plan->Act->Observe->Replan context."""
        url = f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        history_text = self._format_replan_history(history)
        planning_user_prompt = (
            f"Bug description:\n{instance['problem_statement']}\n\n"
            f"Remaining turn budget: {remaining_turns}\n\n"
            f"Recent execution history:\n{history_text}\n\n"
            "Choose the NEXT best step."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": REPLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": planning_user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 768,
        }

        try:
            logger.info("[planner] Requesting next adaptive step...")
            resp = requests.post(url, headers=headers, json=payload, timeout=time_limit * 60)
            resp.raise_for_status()
            data = resp.json()

            choices: List[Any] = data.get("choices") or [{}]
            raw_content: str = str(choices[0].get("message", {}).get("content", "") or "")
            logger.info(f"[planner] Raw adaptive response: {raw_content[:500]}")
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.MULTILINE).strip()
            plan_data: Dict[str, Any] = json.loads(cleaned)

            if bool(plan_data.get("stop", False)):
                return {"stop": True, "reason": str(plan_data.get("reason", "")).strip()}

            step_raw = cast(Dict[str, Any], plan_data.get("step") or {})
            step_name = str(step_raw.get("name", "next_step")).strip()
            instruction = str(step_raw.get("instruction", "")).strip()
            if not instruction:
                raise ValueError("Adaptive planner returned empty instruction")
            mt = 1
            return {
                "stop": False,
                "reason": str(plan_data.get("reason", "")).strip(),
                "step": SubAgent(name=step_name, instruction=instruction, max_turns=mt),
            }
        except Exception as e:
            logger.warning(f"[planner] Adaptive planning failed ({e}), using fallback next-step")
            if history:
                fallback = SubAgent(
                    name="fix_retry",
                    instruction=(
                        "Use the latest observations to make a targeted fix. "
                        "Run a focused test command that verifies the bug status, then update code."
                    ),
                    max_turns=1,
                )
            else:
                fallback = SubAgent(
                    name="reproduce",
                    instruction=(
                        "Reproduce the bug with a concrete test or command, capture exact failure output, "
                        "and identify likely source files to modify."
                    ),
                    max_turns=1,
                )
            return {"stop": False, "reason": "fallback", "step": fallback}

    def _run_multistep(self, instance: SWEbenchInstance, max_turns: int, time_limit: int) -> str:
        """Multi-step pipeline execution with a pre-generated N-step plan.

        Args:
            instance: The SWE-bench instance.
            max_turns: Total turn budget across all planned steps.
            time_limit: Time limit for each step in minutes.

        Returns:
            Summary of the final step.
        """
        logger.info("[multi-step] Starting planned multi-step pipeline")
        t0 = time.monotonic()
        instance_id: str = str(instance.get("instance_id", "unknown"))  # type: ignore[attr-defined]
        self._log.emit(type="start", instance_id=instance_id, max_turns=max_turns, time_limit=time_limit)
        steps = self._generate_plan(instance, max_turns, time_limit)  # type: ignore[no-untyped-call]
        for step in steps:
            # Keep planner-provided max_turns, but ensure it is at least 1.
            step.max_turns = max(1, int(step.max_turns or 1))
        self._log.emit(type="plan", steps=[{"name": s.name, "max_turns": s.max_turns} for s in steps], planning_mode="pre_generated")

        remaining_turns = max_turns
        step_summaries: List[str] = []
        completed: Dict[str, SubAgent] = {}

        for idx, step in enumerate(steps, start=1):
            if remaining_turns <= 0:
                logger.info("[multi-step] Turn budget exhausted, stopping early")
                break

            context = "\n".join(step_summaries)
            logger.info(
                "[multi-step] Executing planned step %s/%s: %s",
                idx,
                len(steps),
                step.name,
            )
            # Enforce global budget: a step may use at most remaining turns.
            step.max_turns = min(step.max_turns, remaining_turns)
            step = self._run_sub_agent(step, self.container, context, time_limit)
            completed[step.name] = step
            turns_used = max(1, int(getattr(self, "_last_tool_loop_turns", 1)))
            remaining_turns = max(0, remaining_turns - turns_used)

            observation = self._collect_replan_observation()
            step_summaries.append(
                f"[{step.name}] turns={turns_used}\nResult:\n{step.result[:500]}\nObservation:\n{observation[:500]}"
            )
            self._log.emit(
                type="step_observe",
                step=step.name,
                turns_used=turns_used,
                remaining_turns=remaining_turns,
                observation=observation[:2000],
            )

        elapsed = round(time.monotonic() - t0, 1)
        logger.info("[multi-step] Planned multi-step pipeline completed")
        self._log.emit(
            type="end",
            instance_id=instance_id,
            n_steps=len(completed),
            steps_completed=list(completed.keys()),
            turns_used=max_turns - remaining_turns,
            elapsed_seconds=elapsed,
        )
        return step_summaries[-1] if step_summaries else ""

    def _run_openai_tools(self, instance: SWEbenchInstance, max_turns: int, time_limit: int) -> str:
        """Multi-turn tool loop tailored for Qwen XML-style tool calling.

        This method constructs the initial messages and delegates to _tool_loop.
        """
        prompt_text = SWEBENCH_USER_PROMPT.format(
            description=instance["problem_statement"].replace('"""', "'''")
        )

        system_prompt = (
            SWEBENCH_EXTRA_SYSTEM_PROMPT
            + "\n\nYou are a code-fixing agent with tools in sandbox."
            + "\nRules:"
            + "\n1) If info is missing, call tools first."
            + "\n2) Do NOT provide only high-level suggestions."
            + "\n3) For code changes, use apply_patch/write_file with exact content."
            + "\n4) Before final answer, run validation commands via tools."
            + "\n5) Final answer must include changed files and exact patch summary."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        return self._tool_loop(self.container, messages, max_turns, time_limit)


    # _run_openai
    def _run_openai(self, instance: SWEbenchInstance, max_turns: int, time_limit: int) -> None:
        """
        关键版本：
        - 真正通过 Proxy 路由地址 self.endpoint 发 chat/completions
        - 不在容器里 curl 模型
        - 拿到回复后再让容器执行改文件
        """
        prompt_text = SWEBENCH_USER_PROMPT.format(
            description=instance["problem_statement"].replace('"""', "'''")
        )

        # 走 Proxy 的 OpenAI-compatible endpoint
        # self.endpoint 形如: http://host:12358/rollout/ro.../attempt/at.../v1
        url = f"{self.endpoint}/chat/completions"

        def _normalize_model_name(model: object) -> str:
            if isinstance(model, str):
                return model
            if isinstance(model, list):
                if not model:
                    raise ValueError("model list is empty")
                if not isinstance(model[0], str):
                    raise TypeError(f"model[0] must be str, got {type(model[0])}")
                return model[0]
            raise TypeError(f"model must be str or list[str], got {type(model)}")

        payload: Dict[str, Any] = {
            "model": "claude-sonnet-4-5-20250929",  # 必须是 llm_proxy 注册过的 frontend model 名
            "messages": [
                {"role": "system", "content": SWEBENCH_EXTRA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.2,
            # vllm 训练抽取常用
            "logprobs": True,
            "top_logprobs": 1,
        }

        logger.info("payload.model type=%s value=%r", type(payload["model"]).__name__, payload["model"])

        headers = {"Content-Type": "application/json"}
        # 一般 proxy 不需要 auth；若你环境要求可保留
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info("Calling traced LLM proxy: %s", url)
        resp = requests.post(url, headers=headers, json=payload, timeout=time_limit * 60)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {json.dumps(data, ensure_ascii=False)[:2000]}")

        msg = (choices[0].get("message") or {}).get("content", "")
        finish_reason = choices[0].get("finish_reason")
        usage = data.get("usage", {})

        logger.info("OPENAI_PARSE_OK")
        logger.info("FINISH_REASON: %s", finish_reason)
        logger.info("USAGE: %s", json.dumps(usage, ensure_ascii=False))
        logger.info("ASSISTANT_FULL:\n%s", msg)

        # （可选）把完整响应落盘到容器，便于排障
        escaped = json.dumps(data, ensure_ascii=False)
        self.container.send_command(
            "python - <<'PY'\n"
            "import json\n"
            f"obj = {escaped}\n"
            "with open('/tmp/cc_openai_resp.json','w',encoding='utf-8') as f:\n"
            "    json.dump(obj, f, ensure_ascii=False, indent=2)\n"
            "print('SAVE_RESP_OK')\n"
            "PY"
        )

        # 你的 demo 修改：写入文件开头
        self.container.send_command(
            "python - <<'PY'\n"
            "p='/testbed/astropy/modeling/separable.py'\n"
            "s=open(p,'r',encoding='utf-8').read()\n"
            "open(p,'w',encoding='utf-8').write('print(\"Demo test\")\\n'+s)\n"
            "print('PREPEND_OK')\n"
            "PY"
        )


    def _run_python_sdk(self, instance: SWEbenchInstance, max_turns: int, time_limit: int) -> None:
        """Executes Claude Code using the Python SDK wrapper.

        Installs the Python SDK if necessary, hydrates a template script with the
        problem prompt, and executes the generated Python script.

        Note:
            This path is still under development and not yet stable.

        Args:
            instance: The problem instance containing the problem statement.
            max_turns: The maximum number of interaction turns allowed.
            time_limit: The execution time limit in minutes.
        """
        # Ensure Python 3.12 is available
        self.container.send_command(
            f"""
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed. Installing Python 3.12..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3.12
else
    echo "Python is already installed."
fi
"""
        )
        self.container.send_command("python3 -m pip install claude-code-sdk")

        # Load and fill the execution template
        with open("src/agent/cc/claude_code_main.py.template") as f:
            entrance_template = f.read()

        script_content = (
            entrance_template.replace("SYS_PROMPT", SWEBENCH_EXTRA_SYSTEM_PROMPT)
            .replace(
                "PROMPT", SWEBENCH_USER_PROMPT.format(description=instance["problem_statement"].replace('"""', "'''"))
            )
            .replace("MAX_STEP", str(max_turns))
        )

        # Write the script to the container and execute
        self.container.send_command(f"cat > /tmp/claude_code_main.py <<'CC_MAIN'\n{script_content}\nCC_MAIN\n")
        self.container.send_command("python3 /tmp/claude_code_main.py", time_limit * 60)
        return

    def run_instance(
        self,
        instance: SWEbenchInstance,
        max_turns: int = 40,
        time_limit: int = 30,
        run_method: Literal["python", "cli", "openai_tools", "multistep"] = "openai_tools",
        stream_path: Optional[str] = None,
    ) -> RunInstanceResult:
        """Runs the agent on a specific SWE-bench instance.

        This method orchestrates the agent execution via the specified method.

        Args:
            instance: The dataset instance dictionary.
            max_turns: Maximum conversation turns allowed for the agent. Defaults to 40.
            time_limit: Time limit for the execution in minutes. Defaults to 30.
            run_method: The execution method. Options: "python", "cli", "openai_tools", "multistep".
            stream_path: Optional path to a JSONL file for structured event logging. Each line
                is a JSON object with a `type` field (e.g. `start`, `plan`, `step_start`,
                `llm_response`, `tool_result`, `step_end`, `end`). Pass `None` to disable.

        Returns:
            A dictionary containing the result:

            - instance_id: The ID of the processed instance.
            - model_patch: The git diff generated by the agent.
            - model_name_or_path: Hardcoded to "cc" (Claude Code).

        Raises:
            ValueError: If `run_method` is not one of the supported methods.
        """
        self._log = StreamLogger(stream_path)
        print(f"===run_method: {run_method}=====")
        try:
            if run_method == "python":
                logger.warning("Running Claude Code using Python SDK is still under development and not yet stable.")
                self._run_python_sdk(instance, max_turns, time_limit)  # type: ignore[no-untyped-call]
            elif run_method == "cli":
                self._run_openai(instance, max_turns, time_limit)  # type: ignore[no-untyped-call]
            elif run_method == "openai_tools":
                self._run_openai_tools(instance, max_turns, time_limit)  # type: ignore[no-untyped-call]
            else:
                self._run_multistep(instance, max_turns, time_limit)  # type: ignore[no-untyped-call]
        finally:
            self._log.close()

        result = self.container.send_command("git --no-pager diff HEAD")
        logger.info(f"====== Result: {result} ==========")
        git_diff = result.output.replace("git --no-pager diff HEAD\n", "")
        logger.info(f"====== git_diff: {git_diff} ==========")
        return {
            "instance_id": instance["instance_id"],
            "model_patch": git_diff,
            "model_name_or_path": "cc",
        }

    def __del__(self) -> None:
        """Destructor to ensure container resources are cleaned up."""
        if hasattr(self, "container"):
            self.container.cleanup()
