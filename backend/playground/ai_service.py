from __future__ import annotations

import json
import os
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field
from .config import ai_status


class DatasetInput(BaseModel):
    dataset_id: str
    version: int = Field(default=1, ge=1)


class AnalysisInput(DatasetInput):
    operation: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool:
    def __init__(self, name: str, description: str, schema: type[BaseModel], handler: Callable[[BaseModel], Any], mutating: bool = False):
        self.name, self.description, self.schema, self.handler, self.mutating = name, description, schema, handler, mutating

    def invoke(self, payload: dict[str, Any]):
        return self.handler(self.schema.model_validate(payload))


def provider_config() -> dict[str, Any]:
    return ai_status()


def require_provider() -> dict[str, Any]:
    config = provider_config()
    if not config["configured"]: raise RuntimeError("AI provider is not configured. Set GEMINI_API_KEY in backend/.env.")
    return config


def _gemini_client(config: dict[str, Any]):
    """Create the official Gemini client without ever returning/logging credentials."""
    from google import genai
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""),
                        http_options={"timeout": int(config["timeout"] * 1000)})


def _generate(prompt: str, config: dict[str, Any]) -> str:
    """Bounded Gemini request with retry and a deliberately small surface area."""
    last: Exception | None = None
    client = _gemini_client(config)
    for attempt in range(int(config["max_retries"]) + 1):
        try:
            response = client.models.generate_content(
                model=config["model"], contents=prompt,
                config={"response_mime_type": "text/plain"},
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            return str(text)
        except Exception as exc:  # provider errors are normalized at the API boundary
            last = exc
            if attempt < int(config["max_retries"]):
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"Gemini request failed after retries: {type(last).__name__}") from last


def run_conversation(*, mode: str, message: str, context: dict[str, Any] | None,
                     tools: dict[str, Tool]) -> dict[str, Any]:
    """Run a bounded Ask/Agent turn. Agent planning is JSON-only and tool calls use shared handlers."""
    config = require_provider()
    safe_context = context or {}
    if mode == "ask":
        prompt = ("You are the read-only Data Analysis Playground analyst. Answer using only "
                  "the supplied authoritative context; do not invent metrics.\nCONTEXT:\n" +
                  json.dumps(safe_context, default=str) + "\nQUESTION:\n" + message)
        return {"response": _generate(prompt, config), "tools": []}
    plan_prompt = ("You are a bounded data analyst planner. Return JSON only with a `calls` array, "
                   "each item containing `tool` and `arguments`. Select at most " +
                   str(config["max_tool_calls"]) + " tools from this registry. "
                   "You may call generate_visualization to create an actual chart artifact; this is a non-mutating operation. "
                   "When columns are not specified, call inspect_schema first, then generate the chart using compatible columns. "
                   "Never propose or execute transformations. Dataset ID and version must exactly match authoritative context.\nREGISTRY:\n" +
                   json.dumps(tool_specs(tools), default=str) + "\nCONTEXT:\n" +
                   json.dumps(safe_context, default=str) + "\nREQUEST:\n" + message)
    raw = _generate(plan_prompt, config)
    try:
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            plan = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise
            plan = json.loads(candidate[start:end + 1])
        calls = plan.get("calls", [])
    except Exception as exc:
        raise RuntimeError("Gemini returned an invalid execution plan") from exc
    events: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for call in calls[: int(config["max_tool_calls"])]:
        name = call.get("tool")
        if name not in tools or tools[name].mutating:
            events.append({"tool": name, "status": "failed", "error": "Tool is not allowlisted for automatic execution"})
            continue
        try:
            result = tools[name].invoke(call.get("arguments", {}))
            events.append({"tool": name, "status": "completed"})
            results.append({"tool": name, "result": result})
        except Exception as exc:
            events.append({"tool": name, "status": "failed", "error": str(exc)})
    final_prompt = ("Summarize these authoritative tool results for the user. Do not invent values. "
                    "Mention failed tools clearly and recommend approval before any mutation.\n" +
                    json.dumps(results, default=str) + "\nUSER REQUEST:\n" + message)
    return {"response": _generate(final_prompt, config), "tools": events}


def tool_specs(tools: dict[str, Tool]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.schema.model_json_schema(), "mutating": t.mutating} for t in tools.values()]
