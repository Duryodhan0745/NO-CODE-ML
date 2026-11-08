from __future__ import annotations

import os

def setting(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def ai_status() -> dict[str, object]:
    provider = setting("AI_PROVIDER", "")
    key = setting("GEMINI_API_KEY") if provider == "google" else setting("AI_API_KEY")
    return {"provider": provider, "model": setting("GEMINI_MODEL", "gemini-2.5-flash") if provider == "google" else setting("AI_MODEL"), "configured": bool(key), "timeout": int(setting("AI_TIMEOUT", "60")), "max_retries": int(setting("AI_MAX_RETRIES", "2")), "max_tool_calls": int(setting("AI_MAX_TOOL_CALLS", "10")), "max_iterations": int(setting("AI_MAX_ITERATIONS", "10"))}
