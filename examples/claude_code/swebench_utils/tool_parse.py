# agent_runtime/tool_parse.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FunctionCall:
    name: str
    arguments: str
    tool_call_id: Optional[str] = None

    def args_dict(self) -> Dict[str, Any]:
        try:
            obj = json.loads(self.arguments) if self.arguments else {}
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


class ToolParser:
    TAG_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

    def parse(self, assistant_message: Dict[str, Any]) -> List[FunctionCall]:
        # 1) native tool_calls
        tool_calls = assistant_message.get("tool_calls") or []
        out: List[FunctionCall] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments", "{}")
            tcid = tc.get("id")
            if name:
                out.append(FunctionCall(name=name, arguments=args or "{}", tool_call_id=tcid))
        if out:
            return out

        # 2) fallback parse from content text
        content = assistant_message.get("content") or ""
        if not isinstance(content, str):
            return out

        def load_obj(s: str) -> Optional[dict]:
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

        def obj_to_fc(obj: dict) -> Optional[FunctionCall]:
            name = obj.get("tool") or obj.get("name")
            args = obj.get("arguments", obj.get("args", {}))
            if not name:
                return None
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            return FunctionCall(name=name, arguments=args)

        for m in self.TAG_PATTERN.finditer(content):
            obj = load_obj(m.group(1))
            if obj:
                fc = obj_to_fc(obj)
                if fc:
                    out.append(fc)

        for m in self.JSON_BLOCK_PATTERN.finditer(content):
            obj = load_obj(m.group(1))
            if obj:
                fc = obj_to_fc(obj)
                if fc:
                    out.append(fc)

        if not out:
            obj = load_obj(content.strip())
            if obj:
                fc = obj_to_fc(obj)
                if fc:
                    out.append(fc)

        return out
