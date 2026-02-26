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
Given a bug description, repository exploration results, and a total turn budget, output a JSON execution plan.

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
- Typical workflow is reproduce -> locate -> fix -> validate, but adapt to this bug.
- Every instruction must reference at least one concrete bug artifact (function/class/file/test/symbol) from the bug description.
- Do not use boilerplate-only instructions such as "inspect code", "fix bug", or "run tests" without specific targets.
- Do not repeat semantically identical instructions across steps.
- Total budget constraint is strict:
  - sum(step.max_turns for step in steps) <= TOTAL_TURN_BUDGET
- Prefer 3-5 steps unless clearly necessary.

IMPORTANT - You are given REPO EXPLORATION RESULTS below.
Use them to write CONCRETE instructions with:
- Exact file paths (e.g. /testbed/astropy/modeling/separable.py)
- Exact function/class names found in the source
- Exact test file paths
- Specific line ranges or code snippets when relevant

IMPORTANT for fix/apply steps:
- The fix step instruction MUST specify the exact file path and function/class to modify.
- The fix step should describe WHAT needs to change, not just "fix the bug".
- Give the fix step at least 2 max_turns so the agent can read the file and then write the patch.
- Example good instruction: "Read /testbed/astropy/modeling/separable.py. In the _separable function (around line 85), the left and right separability matrices are combined using np.ones for the 'and' operator case. For nested CompoundModels where the sub-model already has a separability matrix, use the existing matrix instead of np.ones. Apply the fix using write_file."
- Example bad instruction: "Fix the separability computation logic"

Output schema:
{
  "steps": [
    {"name": "step_name", "instruction": "specific instruction with exact paths", "max_turns": 2}
  ]
}"""

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
                strip_arg = args.get("strip", None)
                strip_candidates = [int(strip_arg)] if strip_arg is not None else [1, 0]
                attempt_logs: List[str] = []

                for strip in strip_candidates:
                    cmd = (
                        "cd /testbed && "
                        "cat > /tmp/agent.patch <<'PATCH'\n"
                        f"{patch}\n"
                        "PATCH\n"
                        f"(git apply --whitespace=nowarn -p{strip} /tmp/agent.patch || patch -p{strip} < /tmp/agent.patch)\n"
                        "ec=$?\n"
                        "echo \"__AGL_APPLY_PATCH_EXIT_CODE__=$ec\"\n"
                        "exit $ec"
                    )
                    out = container.send_command(cmd, timeout_default)
                    txt = out if isinstance(out, str) else getattr(out, "output", str(out))

                    marker_match = re.search(r"__AGL_APPLY_PATCH_EXIT_CODE__=(\d+)", txt)
                    marker_exit: Optional[int] = int(marker_match.group(1)) if marker_match else None
                    meta = None if isinstance(out, str) else getattr(out, "metadata", None)
                    meta_exit = None if meta is None else getattr(meta, "exit_code", None)
                    exit_code: Optional[int]
                    if marker_exit is not None:
                        exit_code = marker_exit
                    elif meta_exit is not None:
                        exit_code = int(meta_exit)
                    else:
                        exit_code = None

                    cleaned_txt = re.sub(r"\n?__AGL_APPLY_PATCH_EXIT_CODE__=\d+\s*$", "", txt).strip()
                    if exit_code == 0:
                        verify_out = container.send_command(
                            "cd /testbed && git --no-pager diff --name-only",
                            timeout_default,
                        )
                        verify_txt = (
                            verify_out
                            if isinstance(verify_out, str)
                            else getattr(verify_out, "output", str(verify_out))
                        )
                        changed_files = [
                            ln.strip()
                            for ln in verify_txt.splitlines()
                            if ln.strip() and not ln.strip().startswith("cd /testbed &&")
                        ]
                        if changed_files:
                            return _ok(
                                cleaned_txt,
                                tool=name,
                                strip=strip,
                                tried_strips=strip_candidates,
                                changed_files=changed_files[:50],
                            )
                        attempt_logs.append(
                            f"[strip={strip} exit_code={exit_code} but no git diff]\n"
                            f"apply_output:\n{cleaned_txt if cleaned_txt else '(no stderr/stdout output)'}\n\n"
                            f"git diff --name-only output:\n{verify_txt.strip() if verify_txt.strip() else '(empty)'}"
                        )
                        continue

                    attempt_logs.append(
                        f"[strip={strip} exit_code={exit_code}]\n"
                        f"{cleaned_txt if cleaned_txt else '(no stderr/stdout output)'}"
                    )

                diag_txt = ""
                try:
                    diag_cmd = (
                        "cd /testbed && "
                        "echo '=== git status --short ===' && git status --short && "
                        "echo '=== git diff --stat ===' && git --no-pager diff --stat && "
                        "echo '=== patch preview (/tmp/agent.patch) ===' && sed -n '1,120p' /tmp/agent.patch"
                    )
                    diag_out = container.send_command(diag_cmd, timeout_default)
                    diag_txt = (
                        diag_out
                        if isinstance(diag_out, str)
                        else getattr(diag_out, "output", str(diag_out))
                    ).strip()
                except Exception as diag_err:
                    diag_txt = f"(failed to collect diagnostics: {type(diag_err).__name__}: {diag_err})"

                return _err(
                    (
                        ("\n\n".join(attempt_logs) if attempt_logs else "apply_patch failed with unknown error")
                        + "\n\n=== post-failure diagnostics ===\n"
                        + diag_txt
                    ),
                    tool=name,
                    strip=strip_arg,
                    tried_strips=strip_candidates,
                )

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
                    preview_limit = 200 if ok else 2000
                    end_msg = (
                        f"[tool-loop] TOOL_END id={tcid} name={fn_name} ok={ok} "
                        f"content_preview={content[:preview_limit]!r}"
                    )
                except Exception:
                    end_msg = f"[tool-loop] TOOL_END id={tcid} name={fn_name} raw_preview={result_str[:200]!r}"
                print(end_msg)
                if " ok=False " in end_msg:
                    logger.error(end_msg)
                else:
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
            + "\n\nBefore using any tool, first analyze:"
            + "\n1. What has been accomplished in previous steps?"
            + "\n2. What specific information from the context is relevant to your task?"
            + "\n3. What exact actions do you need to take?"
            + "\nThen execute your actions based on that analysis."
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

    def _explore_repo_for_planning(
        self,
        instance: SWEbenchInstance,
        time_limit: int,
    ) -> str:
        """Explore the repository to gather concrete context for planning.

        This is the key difference from blind planning: the orchestrator first
        gathers real information about the repo (file structure, source code of
        relevant modules, test files) so the planner can generate instructions
        with exact file paths, function names, and line numbers.

        Analogous to the "main agent" in the screenshot doing `ls -la` before
        spawning sub-agents with specific file paths.

        Args:
            instance: The SWE-bench instance containing the problem statement.
            time_limit: Timeout in minutes.

        Returns:
            A string containing structured exploration results.
        """
        logger.info("[orchestrator] Exploring repo before planning...")
        exploration_results: List[str] = []
        timeout = min(30, time_limit * 60)

        # --- Phase 1: Repo structure ---
        try:
            out = self.container.send_command(
                "cd /testbed && find . -maxdepth 2 -type f -name '*.py' | head -60",
                timeout,
            )
            txt = out if isinstance(out, str) else getattr(out, "output", str(out))
            exploration_results.append(f"## Repository Python files (top 2 levels):\n{txt[:3000]}")
        except Exception as e:
            exploration_results.append(f"## Repo structure: failed ({e})")

        # --- Phase 2: Extract keywords from bug description and grep ---
        problem = instance["problem_statement"]
        # Extract likely module/function/class names from the problem statement
        # Look for Python identifiers in backticks, import statements, etc.
        import_pattern = re.compile(r'(?:from|import)\s+([\w.]+)', re.MULTILINE)
        backtick_pattern = re.compile(r'`([\w.]+)`')
        candidates: List[str] = []
        for m in import_pattern.finditer(problem):
            candidates.append(m.group(1))
        for m in backtick_pattern.finditer(problem):
            candidates.append(m.group(1))
        # Deduplicate while preserving order
        seen: set = set()
        unique_candidates: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        if unique_candidates:
            exploration_results.append(f"## Keywords extracted from bug description: {unique_candidates}")

        # --- Phase 3: Find the relevant source files ---
        for keyword in unique_candidates[:5]:  # limit to top 5 keywords
            # Convert module path to file path (e.g., astropy.modeling.separable -> astropy/modeling/separable)
            file_path_guess = keyword.replace(".", "/")
            try:
                out = self.container.send_command(
                    f"cd /testbed && find . -path '*{file_path_guess}*' -type f 2>/dev/null | head -10",
                    timeout,
                )
                txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                if txt.strip():
                    exploration_results.append(f"## Files matching '{keyword}':\n{txt.strip()}")
            except Exception:
                pass

        # --- Phase 4: Read the most likely source file ---
        # Try to find and read the main source file mentioned in the bug
        for keyword in unique_candidates[:3]:
            file_path_guess = keyword.replace(".", "/")
            try:
                find_cmd = f"cd /testbed && find . -path '*{file_path_guess}.py' -type f 2>/dev/null | head -1"
                out = self.container.send_command(find_cmd, timeout)
                found_path = (out if isinstance(out, str) else getattr(out, "output", str(out))).strip()
                # Remove command echo if present
                lines = found_path.split("\n")
                found_path = next((l.strip() for l in lines if l.strip().startswith("./") or l.strip().startswith("/")), "")

                if found_path and found_path.endswith(".py"):
                    # Read the file outline (first 120 lines + function signatures)
                    read_cmd = (
                        f"cd /testbed && head -150 {found_path}"
                    )
                    out = self.container.send_command(read_cmd, timeout)
                    txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                    exploration_results.append(
                        f"## Source file content (first 150 lines of {found_path}):\n{txt[:4000]}"
                    )

                    # Also get function/class signatures
                    grep_cmd = (
                        f"cd /testbed && grep -n 'def \\|class ' {found_path} | head -30"
                    )
                    out = self.container.send_command(grep_cmd, timeout)
                    txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                    exploration_results.append(
                        f"## Function/class signatures in {found_path}:\n{txt[:2000]}"
                    )
                    break  # Found and read the main file, stop
            except Exception:
                pass

        # --- Phase 5: Find test files ---
        for keyword in unique_candidates[:3]:
            last_part = keyword.split(".")[-1]
            try:
                out = self.container.send_command(
                    f"cd /testbed && find . -name 'test_{last_part}*' -o -name '*{last_part}*test*' 2>/dev/null | head -5",
                    timeout,
                )
                txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                if txt.strip():
                    exploration_results.append(f"## Test files for '{last_part}':\n{txt.strip()}")
            except Exception:
                pass

        result = "\n\n".join(exploration_results)
        logger.info(f"[orchestrator] Exploration completed. Context length: {len(result)} chars")
        return result

    # ------------------------------------------------------------------ #
    #                    Plan Quality Validation                           #
    # ------------------------------------------------------------------ #

    # Words that signal a vague, boilerplate instruction when found WITHOUT
    # any concrete anchors (file paths, function names, etc.).
    _VAGUE_PHRASES: List[str] = [
        "fix the bug",
        "fix the issue",
        "fix the logic",
        "inspect code",
        "investigate the",
        "look into the",
        "explore the codebase",
        "run tests",
        "run the tests",
        "verify the fix",
    ]

    def _instruction_quality_score(self, instruction: str, step_name: str) -> Dict[str, Any]:
        """Score an instruction's specificity. Returns a dict with score and diagnostics.

        Scoring criteria:
        - Has file path (e.g. /testbed/..., ./..., *.py)         → +3
        - Has function/class name (def xxx, class xxx, `xxx`)    → +2
        - Has tool action verb (read_file, write_file, run_bash) → +1
        - Has specific line reference (line XX, around line)     → +1
        - Instruction length > 80 chars                          → +1
        - Contains vague-only phrases with no anchors            → -3

        A score < 3 for fix/edit steps is considered "vague".
        A score < 2 for other steps is considered "vague".
        """
        score = 0
        reasons: List[str] = []
        instr_lower = instruction.lower()

        # Check for file paths
        has_path = bool(re.search(r'(?:/testbed/|\./)[\w/.-]+\.py', instruction))
        if has_path:
            score += 3
            reasons.append("has_file_path")

        # Check for function/class names (backtick references or def/class keywords)
        has_symbol = bool(re.search(r'`\w+`|(?:function|method|class|def)\s+\w+', instruction, re.IGNORECASE))
        if has_symbol:
            score += 2
            reasons.append("has_symbol_ref")

        # Check for tool action verbs
        tool_verbs = ["read_file", "write_file", "apply_patch", "run_bash", "head ", "grep ", "cat "]
        if any(v in instr_lower for v in tool_verbs):
            score += 1
            reasons.append("has_tool_verb")

        # Check for line number references
        if re.search(r'line\s+\d+|around line|lines?\s+\d+-\d+', instr_lower):
            score += 1
            reasons.append("has_line_ref")

        # Length bonus
        if len(instruction) > 80:
            score += 1
            reasons.append("sufficient_length")

        # Vagueness penalty — only if there are no concrete anchors
        if not has_path and not has_symbol:
            for phrase in self._VAGUE_PHRASES:
                if phrase in instr_lower:
                    score -= 3
                    reasons.append(f"vague_phrase:{phrase}")
                    break

        # Determine threshold based on step type
        name_lower = step_name.lower()
        is_fix_step = any(kw in name_lower for kw in ["fix", "apply", "edit", "patch"])
        threshold = 3 if is_fix_step else 2

        return {
            "score": score,
            "threshold": threshold,
            "is_vague": score < threshold,
            "is_fix_step": is_fix_step,
            "reasons": reasons,
        }

    def _validate_plan_quality(self, sub_agents: List[SubAgent]) -> List[Dict[str, Any]]:
        """Validate each step's instruction quality. Returns list of diagnostics.

        This is the programmatic guardrail: instead of trusting the LLM to
        follow prompt instructions, we CHECK the output and flag vague steps.
        """
        diagnostics: List[Dict[str, Any]] = []
        for sub in sub_agents:
            diag = self._instruction_quality_score(sub.instruction, sub.name)
            diag["step_name"] = sub.name
            diag["instruction_preview"] = sub.instruction[:120]
            diagnostics.append(diag)
            if diag["is_vague"]:
                logger.warning(
                    f"[plan-validator] Step '{sub.name}' scored {diag['score']}/{diag['threshold']} "
                    f"(vague). Reasons: {diag['reasons']}. Preview: {sub.instruction[:80]!r}"
                )
            else:
                logger.info(
                    f"[plan-validator] Step '{sub.name}' scored {diag['score']}/{diag['threshold']} (OK)"
                )
        return diagnostics

    def _ensure_validate_step(self, sub_agents: List[SubAgent], total_budget: int) -> List[SubAgent]:
        """Ensure the execution plan includes a dedicated validation step.

        If the planner omits validate/test/verify, append a `validate_fix` step.
        Then rebalance max_turns so sum(max_turns) stays within `total_budget`
        whenever possible.
        """
        if not sub_agents:
            return sub_agents

        def _is_validate_like(sub: SubAgent) -> bool:
            text = f"{sub.name} {sub.instruction}".lower()
            tokens = ("validate", "validation", "verify", "pytest", "test_", "run tests", "regression")
            return any(tok in text for tok in tokens)

        has_validate = any(_is_validate_like(s) for s in sub_agents)
        if not has_validate:
            validate_instruction = (
                "Step: Validate the fix with a focused command. "
                "Use run_bash to execute: cd /testbed && python -m pytest "
                "astropy/modeling/tests/test_separable.py -k separable -q. "
                "Then run: cd /testbed && git --no-pager diff --stat and summarize whether the bug behavior is fixed."
            )
            sub_agents = list(sub_agents) + [SubAgent(name="validate_fix", instruction=validate_instruction, max_turns=1)]
            logger.info("[planner] Added missing validate step: validate_fix")

        # Rebalance turns to respect total budget.
        total_turns = sum(max(1, int(s.max_turns or 1)) for s in sub_agents)
        if total_turns <= total_budget:
            return sub_agents

        # Prefer shrinking non-validate steps first, then validate as last resort.
        def _sort_key(item: tuple[int, SubAgent]) -> tuple[int, int]:
            idx, step = item
            is_validate = 1 if _is_validate_like(step) else 0
            return (is_validate, -int(step.max_turns or 1))

        ordered = sorted(list(enumerate(sub_agents)), key=_sort_key)
        for idx, _step in ordered:
            while total_turns > total_budget and sub_agents[idx].max_turns > 1:
                sub_agents[idx].max_turns -= 1
                total_turns -= 1
            if total_turns <= total_budget:
                break

        if total_turns > total_budget:
            logger.warning(
                "[planner] Could not fully rebalance plan turns to budget: total_turns=%s budget=%s",
                total_turns,
                total_budget,
            )
        else:
            logger.info("[planner] Rebalanced plan turns to budget: total_turns=%s budget=%s", total_turns, total_budget)
        return sub_agents

    def _refine_vague_instructions(
        self,
        sub_agents: List[SubAgent],
        diagnostics: List[Dict[str, Any]],
        repo_context: str,
        instance: SWEbenchInstance,
        time_limit: int,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> List[SubAgent]:
        """Rewrite vague instructions using a focused refinement prompt.

        For each step flagged as "vague" by _validate_plan_quality, ask the LLM
        to rewrite it with concrete file paths, function names, and actions. This
        is a targeted, per-step call — cheaper and more effective than re-generating
        the whole plan.
        """
        url = f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        refined = list(sub_agents)  # shallow copy

        for i, (sub, diag) in enumerate(zip(sub_agents, diagnostics)):
            if not diag["is_vague"]:
                continue

            logger.info(f"[plan-refiner] Refining vague step '{sub.name}': {sub.instruction[:80]!r}")

            refine_prompt = (
                f"You must rewrite the following sub-agent instruction to be CONCRETE and ACTIONABLE.\n\n"
                f"Step name: {sub.name}\n"
                f"Original instruction: {sub.instruction}\n\n"
                f"Bug description:\n{instance['problem_statement'][:2000]}\n\n"
                f"Repository exploration results:\n{repo_context[:4000]}\n\n"
                f"Requirements for the rewritten instruction:\n"
                f"- MUST include at least one exact file path from the repo (e.g. /testbed/path/to/file.py)\n"
                f"- MUST name specific functions or classes to work with\n"
                f"- MUST specify what tool to use (read_file, write_file, apply_patch, run_bash)\n"
                f"- For fix steps: describe the SPECIFIC code change needed, not just 'fix the bug'\n"
                f"- Keep it concise but complete (2-5 sentences)\n\n"
                f"Output ONLY the rewritten instruction text. No JSON, no markdown, no explanation."
            )

            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": refine_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            }

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=time_limit * 60)
                resp.raise_for_status()
                data = resp.json()
                choices: List[Any] = data.get("choices") or [{}]
                new_instruction = str(choices[0].get("message", {}).get("content", "") or "").strip()

                if new_instruction and len(new_instruction) > 20:
                    # Validate the refined instruction is actually better
                    new_diag = self._instruction_quality_score(new_instruction, sub.name)
                    if new_diag["score"] > diag["score"]:
                        logger.info(
                            f"[plan-refiner] Improved '{sub.name}' from score {diag['score']} to {new_diag['score']}"
                        )
                        refined[i] = SubAgent(
                            name=sub.name,
                            instruction=new_instruction,
                            max_turns=sub.max_turns,
                        )
                    else:
                        logger.warning(
                            f"[plan-refiner] Refined instruction for '{sub.name}' not better "
                            f"(old={diag['score']}, new={new_diag['score']}), keeping original"
                        )
            except Exception as e:
                logger.warning(f"[plan-refiner] Failed to refine '{sub.name}': {e}")

        return refined

    # ------------------------------------------------------------------ #
    #                       Plan Generation                                #
    # ------------------------------------------------------------------ #

    def _generate_plan(
        self,
        instance: SWEbenchInstance,
        max_turns: int,
        time_limit: int,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> List[SubAgent]:
        """Three-phase plan generation: explore → plan → validate & refine.

        Phase 1: Run targeted exploration commands in the container to discover
                 repo structure, relevant source files, function signatures, and test files.
        Phase 2: Feed the exploration results + bug description to the LLM planner
                 so it can generate instructions with exact file paths and function names.
        Phase 3: Programmatically validate each instruction for specificity. If any step
                 is flagged as "vague", auto-refine it with a targeted LLM call.

        Args:
            instance: The SWE-bench instance.
            max_turns: Budget hint passed to the planner so it can distribute turns.
            time_limit: Per-request timeout in minutes.
            model: Model to use for the planning call.

        Returns:
            List of SubAgent objects derived from the LLM's plan, with vague
            instructions refined to include concrete file paths and function names.
        """
        # ===== Phase 1: Explore the repo to gather concrete context =====
        repo_context = self._explore_repo_for_planning(instance, time_limit)

        # ===== Phase 2: Generate plan with enriched context =====
        url = f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        planning_user_prompt = (
            f"Bug description:\n{instance['problem_statement']}\n\n"
            f"=== REPO EXPLORATION RESULTS ===\n{repo_context}\n\n"
            f"Total turn budget: {max_turns}.\n"
            "Generate a focused execution plan using the EXACT file paths and function names from the exploration above.\n"
            "Output ONLY the JSON object."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": planning_user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
        }

        logger.info("[planner] Requesting dynamic plan from LLM with enriched context...")
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

            # ===== Phase 3: Validate & refine =====
            diagnostics = self._validate_plan_quality(sub_agents)
            has_vague = any(d["is_vague"] for d in diagnostics)
            if has_vague:
                logger.info("[planner] Vague steps detected, running auto-refinement...")
                sub_agents = self._refine_vague_instructions(
                    sub_agents, diagnostics, repo_context, instance, time_limit, model
                )
                # Log final plan quality
                final_diags = self._validate_plan_quality(sub_agents)
                still_vague = sum(1 for d in final_diags if d["is_vague"])
                if still_vague:
                    logger.warning(f"[planner] {still_vague} steps still vague after refinement")
                else:
                    logger.info("[planner] All steps passed quality validation after refinement")
            else:
                logger.info("[planner] All steps passed quality validation on first pass")

            sub_agents = self._ensure_validate_step(sub_agents, max_turns)
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
                f"[{step.name}] turns={turns_used}\nResult:\n{step.result}\nObservation:\n{observation}"
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

        result = self.container.send_command("cd /testbed && git --no-pager diff HEAD")
        logger.info(f"====== Result: {result} ==========")
        git_diff = result.output.replace("cd /testbed && git --no-pager diff HEAD\n", "")
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
