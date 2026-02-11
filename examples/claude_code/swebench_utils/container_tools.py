# container_tools.py
from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable


@dataclass
class ToolResult:
    ok: bool
    tool: str
    result: str = ""
    error: str = ""
    exit_code: Optional[int] = None
    latency_sec: float = 0.0
    finished: bool = False
    summary: str = ""


class ContainerTools:
    """
    A white-box tool executor for container operations.

    Expected container interface:
        container.send_command(cmd: str, timeout_sec: int | None = None) -> Any
    """

    def __init__(
        self,
        container: Any,
        workspace_root: str = ".",
        max_output_chars: int = 30000,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.container = container
        self.workspace_root = workspace_root
        self.max_output_chars = max_output_chars
        self.logger = logger

    # ---------- public ----------
    @staticmethod
    def tool_specs() -> list[dict]:
        """Tool schemas for model prompting / function-calling style."""
        return [
            {
                "name": "run_shell",
                "description": "Run a shell command inside container.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "timeout_sec": {"type": "integer", "default": 120},
                    },
                    "required": ["cmd"],
                },
            },
            {
                "name": "read_file",
                "description": "Read file text content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_chars": {"type": "integer", "default": 12000},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write full content to file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "apply_patch",
                "description": "Apply unified diff patch with apply_patch helper.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string"},
                    },
                    "required": ["patch"],
                },
            },
            {
                "name": "run_tests",
                "description": "Run tests or script command for verification.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_cmd": {"type": "string"},
                        "timeout_sec": {"type": "integer", "default": 600},
                    },
                    "required": ["test_cmd"],
                },
            },
            {
                "name": "finish",
                "description": "Finish the task with a concise summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
        ]

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Main dispatcher."""
        start = time.time()
        try:
            if name == "run_shell":
                cmd = str(arguments["cmd"])
                timeout_sec = int(arguments.get("timeout_sec", 120))
                out = self._send(cmd, timeout_sec)
                return self._ok(name, out, latency=time.time() - start)

            if name == "read_file":
                path = str(arguments["path"])
                max_chars = int(arguments.get("max_chars", 12000))
                self._ensure_safe_path(path)
                cmd = (
                    "python - <<'PY'\n"
                    "from pathlib import Path\n"
                    f"p = Path({path!r})\n"
                    "if not p.exists():\n"
                    "    print('[ERROR] File not found:', p)\n"
                    "else:\n"
                    "    txt = p.read_text(encoding='utf-8', errors='ignore')\n"
                    f"    print(txt[:{max_chars}])\n"
                    "PY"
                )
                out = self._send(cmd, 60)
                return self._ok(name, out, latency=time.time() - start)

            if name == "write_file":
                path = str(arguments["path"])
                content = str(arguments["content"])
                self._ensure_safe_path(path)
                cmd = (
                    f"mkdir -p {shlex.quote(str(self._parent_dir(path)))}\n"
                    f"cat > {shlex.quote(path)} <<'EOF_WRITE'\n"
                    f"{content}\n"
                    "EOF_WRITE\n"
                )
                out = self._send(cmd, 60)
                return self._ok(name, out, latency=time.time() - start)

            if name == "apply_patch":
                patch = str(arguments["patch"])
                cmd = (
                    "apply_patch <<'EOF_PATCH'\n"
                    f"{patch}\n"
                    "EOF_PATCH\n"
                )
                out = self._send(cmd, 120)
                return self._ok(name, out, latency=time.time() - start)

            if name == "run_tests":
                test_cmd = str(arguments["test_cmd"])
                timeout_sec = int(arguments.get("timeout_sec", 600))
                out = self._send(test_cmd, timeout_sec)
                return self._ok(name, out, latency=time.time() - start)

            if name == "finish":
                summary = str(arguments["summary"])
                return ToolResult(
                    ok=True,
                    tool=name,
                    result="",
                    latency_sec=time.time() - start,
                    finished=True,
                    summary=summary,
                )

            return ToolResult(
                ok=False,
                tool=name,
                error=f"Unknown tool: {name}",
                latency_sec=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                ok=False,
                tool=name,
                error=f"{type(e).__name__}: {e}",
                latency_sec=time.time() - start,
            )

    # ---------- helpers ----------
    def _ok(self, tool: str, output: Any, latency: float) -> ToolResult:
        text = self._normalize_output(output)
        text = self._truncate(text)
        return ToolResult(ok=True, tool=tool, result=text, latency_sec=latency)

    def _send(self, cmd: str, timeout_sec: int) -> Any:
        self._log(f"[tool] exec timeout={timeout_sec}s\n{cmd}")
        try:
            return self.container.send_command(cmd, timeout_sec)
        except TypeError:
            # fallback for containers whose signature is send_command(cmd)
            return self.container.send_command(cmd)

    def _normalize_output(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        # try common dict keys
        if isinstance(output, dict):
            if "stdout" in output or "stderr" in output:
                stdout = str(output.get("stdout", ""))
                stderr = str(output.get("stderr", ""))
                code = output.get("exit_code", output.get("returncode", ""))
                return f"[exit_code={code}]\n[stdout]\n{stdout}\n[stderr]\n{stderr}"
            return json.dumps(output, ensure_ascii=False, indent=2)
        return str(output)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + "\n...[TRUNCATED]..."

    def _ensure_safe_path(self, path: str) -> None:
        bad_tokens = ["..", "~", "/etc/", "/root/.ssh", "/proc/", "/sys/"]
        if any(tok in path for tok in bad_tokens):
            raise ValueError(f"Unsafe path: {path}")

    def _parent_dir(self, path: str) -> str:
        from pathlib import Path
        return str(Path(path).parent)

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger(msg)
