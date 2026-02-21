# Copyright (c) Microsoft. All rights reserved.

"""Controller module for managing Claude Code executions in containerized environments.

This module provides the ClaudeController class that manages the execution of Claude Code
within Docker containers. It handles container initialization, command execution, and
patch application for SWE-bench evaluation tasks.
"""

import logging
from functools import partial
from typing import Literal, TypedDict

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
from typing import Literal, TypedDict, Any, Dict, List, Optional


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

    def _exec_container_tool(self, name: str, args: Dict[str, Any], time_limit: int) -> str:
        """Execute one tool in container and return JSON string."""
        def _ok(content: str, **meta: Any) -> str:
            return json.dumps({"ok": True, "content": content, "metadata": meta}, ensure_ascii=False)

        def _err(content: str, **meta: Any) -> str:
            return json.dumps({"ok": False, "content": content, "metadata": meta}, ensure_ascii=False)

        try:
            timeout_default = min(120, time_limit * 60)

            if name == "run_bash":
                cmd = args["cmd"]
                timeout = int(args.get("timeout", timeout_default))
                blocked = ["rm -rf /", ":(){:|:&};:", "shutdown", "reboot"]
                if any(x in cmd.lower() for x in blocked):
                    return _err(f"Blocked command: {cmd}", tool=name)
                full = f"cd /testbed && {cmd}"
                out = self.container.send_command(full, timeout)
                txt = out if isinstance(out, str) else getattr(out, "output", str(out))
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
                out = self.container.send_command(cmd, timeout_default)
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
                out = self.container.send_command(cmd, timeout_default)
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
                    f"(git apply -p{strip} /tmp/agent.patch || patch -p{strip} < /tmp/agent.patch)"
                )
                out = self.container.send_command(cmd, timeout_default)
                txt = out if isinstance(out, str) else getattr(out, "output", str(out))
                return _ok(txt, tool=name, strip=strip)

            return _err(f"Unknown tool: {name}", tool=name, args=args)

        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", tool=name, args=args)

    def _run_openai_tools(self, instance: SWEbenchInstance, max_turns: int, time_limit: int) -> str:
        """
        Multi-turn tool loop tailored for Qwen XML-style tool calling.
        Matches the tokenizer_config.json chat_template logic.
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

        url = f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if getattr(self, "api_key", None):
            headers["Authorization"] = f"Bearer {self.api_key}"

        tools = self._build_tools_schema()
        final_text = ""
        
        # Ensure model name matches your deployment
        model_name = "claude-sonnet-4-5-20250929" 

        def _extract_content_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Handle multi-modal content blocks if necessary
                return "".join([x.get("text", "") for x in content if x.get("type") == "text"])
            return str(content) if content else ""

        def _parse_qwen_xml_tool_calls(content: str) -> List[Dict[str, Any]]:
            """
            Parses Qwen specific XML format:
            <tool_call>
            <function=func_name>
            <parameter=param_name>value</parameter>
            </function>
            </tool_call>
            """
            tool_calls = []
            # 1. Find all <tool_call> blocks
            # Use DOTALL to match across newlines
            tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
            
            # 2. Inside tool_call, find function name
            func_name_pattern = re.compile(r"<function=(.*?)>", re.DOTALL)
            
            # 3. Find parameters
            # Note: Value might be multi-line
            param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)

            for match in tool_call_pattern.finditer(content):
                block = match.group(1)
                
                # Extract function name
                fn_match = func_name_pattern.search(block)
                if not fn_match:
                    continue
                func_name = fn_match.group(1).strip()
                
                # Extract arguments
                arguments = {}
                for p_match in param_pattern.finditer(block):
                    key = p_match.group(1).strip()
                    val = p_match.group(2) # Do not strip blindly, value might be code with indentation
                    # Try to unescape or clean generic JSON artifacts if strictly needed, 
                    # but usually Qwen outputs raw text in the XML.
                    arguments[key] = val

                tool_calls.append({
                    "id": f"call_{len(tool_calls)}_{func_name}", # Synthetic ID
                    "type": "function",
                    "function": {
                        "name": func_name,
                        # IMPORTANT: Store as Dict for the Jinja template to iterate with |items
                        "arguments": arguments 
                    }
                })
            return tool_calls

        # Pre-flight check
        try:
            self.container.send_command("cd /testbed && git config --global --add safe.directory /testbed", timeout=10)
        except Exception:
            pass

        for turn in range(max_turns):
            payload = {
                "model": model_name,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.0, # Zero temp for deterministic tool usage
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
            
            # 1. Try native parsing first (if vLLM/Server supports Qwen tool parsing)
            tool_calls = msg.get("tool_calls") or []
            
            # 2. Fallback to XML parsing if native calls are empty but text contains XML
            if not tool_calls and "<tool_call>" in content_text:
                tool_calls = _parse_qwen_xml_tool_calls(content_text)

            logger.info(f"[tool-loop] turn={turn+1} content_len={len(content_text)} tool_calls={len(tool_calls)}")

            # CASE A: No tool calls -> This is the final answer or a chat response
            if not tool_calls:
                final_text = content_text
                messages.append({"role": "assistant", "content": final_text})
                
                # --- Heuristic Check: Did it actually do anything? ---
                if "run_validation" not in final_text and turn < 2:
                     # Optional: Force it to continue if it quit too early
                     pass 
                break

            # CASE B: Tool calls detected
            # Add assistant message to history. 
            # CRITICAL: 'arguments' must be Dict to match your jinja template {% for k,v in args|items %}
            # If native API returned JSON string args, parse them to Dict.
            sanitized_tool_calls = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {} # Fail safe
                
                sanitized_tool_calls.append({
                    "id": tc.get("id", "sf"),
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "arguments": args # Ensure Dict
                    }
                })

            messages.append({
                "role": "assistant",
                "content": content_text, # Keep thought process
                "tool_calls": sanitized_tool_calls
            })

            # Execute Tools
            for tc in sanitized_tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                tcid = tc["id"]

                logger.info(f"[tool-loop] Executing {fn_name} args={fn_args}")
                
                # Execute inside Docker
                result_str = self._exec_container_tool(fn_name, fn_args, time_limit=time_limit)

                # Format Observation
                # The Jinja template handles role="tool" by wrapping in <tool_response>
                messages.append({
                    "role": "tool",
                    "tool_call_id": tcid,
                    "name": fn_name,
                    "content": result_str
                })

        return final_text


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
        run_method: Literal["python", "cli", "openai_tools"] = "openai_tools",
    ) -> RunInstanceResult:
        """Runs the agent on a specific SWE-bench instance.

        This method orchestrates the agent execution via the specified method (CLI or Python),
        and extracts the generated git diff (patch) upon completion.

        Args:
            instance: The dataset instance dictionary.
            max_turns: Maximum conversation turns allowed for the agent. Defaults to 40.
            time_limit: Time limit for the execution in minutes. Defaults to 30.
            run_method: The execution method, either "python" (SDK) or "cli". Defaults to "python".

        Returns:
            A dictionary containing the result:

            - instance_id: The ID of the processed instance.
            - model_patch: The git diff generated by the agent.
            - model_name_or_path: Hardcoded to "cc" (Claude Code).

        Raises:
            ValueError: If `run_method` is not "python" or "cli".
        """
        print(f"===run_method: {run_method}=====")
        # exit()
        if run_method == "python":
            logger.warning("Running Claude Code using Python SDK is still under development and not yet stable.")
            self._run_python_sdk(instance, max_turns, time_limit)
        elif run_method == "cli":
            # self._run_cli(instance, max_turns, time_limit)
            self._run_openai(instance, max_turns, time_limit)
        elif run_method == "openai_tools":
            self._run_openai_tools(instance, max_turns, time_limit)            
        else:
            raise ValueError(f"Wrong run_method '{run_method}', run_method should be in ['python', 'cli']")

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
