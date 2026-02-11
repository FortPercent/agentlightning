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
        run_method: Literal["python", "cli"] = "python",
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
        if run_method == "python":
            logger.warning("Running Claude Code using Python SDK is still under development and not yet stable.")
            self._run_python_sdk(instance, max_turns, time_limit)
        elif run_method == "cli":
            # self._run_cli(instance, max_turns, time_limit)
            self._run_openai(instance, max_turns, time_limit)
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
