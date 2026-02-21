# agent_runtime/container_tools.py
from __future__ import annotations
import json
import shlex
from dataclasses import dataclass
from typing import Any, Dict, Callable


@dataclass
class ToolResult:
    ok: bool
    content: str
    metadata: Dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "content": self.content, "metadata": self.metadata or {}},
            ensure_ascii=False
        )


class ContainerTools:
    """
    container 需要提供:
      send_command(cmd: str, timeout: int | None = None) -> str 或对象(含output字段)
    """
    def __init__(self, container: Any, workdir: str = "/testbed", default_timeout: int = 120):
        self.container = container
        self.workdir = workdir
        self.default_timeout = default_timeout

        self._dispatch: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
            "run_bash": self.run_bash,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "apply_patch": self.apply_patch,
        }

    @staticmethod
    def openai_tools_schema() -> list[dict]:
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
                            "timeout": {"type": "integer"}
                        },
                        "required": ["cmd"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read UTF-8 text file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_bytes": {"type": "integer"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write UTF-8 text to file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "append": {"type": "boolean"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Apply unified diff patch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patch": {"type": "string"},
                            "strip": {"type": "integer"}
                        },
                        "required": ["patch"]
                    }
                }
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name not in self._dispatch:
            return ToolResult(False, f"Unknown tool: {name}", {"tool": name, "args": args})
        try:
            return self._dispatch[name](args)
        except Exception as e:
            return ToolResult(False, f"{type(e).__name__}: {e}", {"tool": name, "args": args})

    def _as_text(self, out: Any) -> str:
        if isinstance(out, str):
            return out
        if hasattr(out, "output"):
            return str(out.output)
        return json.dumps(out, ensure_ascii=False)

    def run_bash(self, args: Dict[str, Any]) -> ToolResult:
        cmd = args["cmd"]
        timeout = int(args.get("timeout", self.default_timeout))

        blocked = ["rm -rf /", ":(){:|:&};:", "shutdown", "reboot"]
        if any(x in cmd.lower() for x in blocked):
            return ToolResult(False, f"Blocked command: {cmd}")

        full = f"cd {shlex.quote(self.workdir)} && {cmd}"
        out = self.container.send_command(full, timeout)
        return ToolResult(True, self._as_text(out), {"cmd": full, "timeout": timeout})

    def read_file(self, args: Dict[str, Any]) -> ToolResult:
        path = args["path"]
        max_bytes = int(args.get("max_bytes", 50000))
        cmd = (
            "python - <<'PY'\n"
            "import os\n"
            f"p={path!r}\n"
            f"base={self.workdir!r}\n"
            "if not os.path.isabs(p): p=os.path.join(base,p)\n"
            f"b=open(p,'rb').read({max_bytes})\n"
            "print(b.decode('utf-8', errors='replace'))\n"
            "PY"
        )
        out = self.container.send_command(f"cd {shlex.quote(self.workdir)} && {cmd}", self.default_timeout)
        return ToolResult(True, self._as_text(out), {"path": path, "max_bytes": max_bytes})

    def write_file(self, args: Dict[str, Any]) -> ToolResult:
        path = args["path"]
        content = args["content"]
        append = bool(args.get("append", False))
        mode = "a" if append else "w"

        cmd = (
            "python - <<'PY'\n"
            "import os\n"
            f"p={path!r}\n"
            f"base={self.workdir!r}\n"
            "if not os.path.isabs(p): p=os.path.join(base,p)\n"
            "os.makedirs(os.path.dirname(p), exist_ok=True)\n"
            f"open(p,{mode!r},encoding='utf-8').write({content!r})\n"
            "print('OK')\n"
            "PY"
        )
        out = self.container.send_command(f"cd {shlex.quote(self.workdir)} && {cmd}", self.default_timeout)
        return ToolResult(True, self._as_text(out), {"path": path, "append": append})

    def apply_patch(self, args: Dict[str, Any]) -> ToolResult:
        patch = args["patch"]
        strip = int(args.get("strip", 0))
        cmd = (
            f"cd {shlex.quote(self.workdir)} && "
            "cat > /tmp/agent.patch <<'PATCH'\n"
            f"{patch}\n"
            "PATCH\n"
            f"(git apply -p{strip} /tmp/agent.patch || patch -p{strip} < /tmp/agent.patch)"
        )
        out = self.container.send_command(cmd, self.default_timeout)
        return ToolResult(True, self._as_text(out), {"strip": strip})
