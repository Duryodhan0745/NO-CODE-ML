"""Local, credential-safe Gemini smoke test. Run from backend with network access."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai_service import _generate
from app.config import ai_status

status = ai_status()
print({"provider": status["provider"], "model": status["model"], "configured": status["configured"]})
if not status["configured"]:
    raise SystemExit("GEMINI_API_KEY is not configured")
reply = _generate("Reply with exactly the word READY.", status)
print({"request_succeeded": True, "response_length": len(reply), "ready": reply.strip().upper() == "READY"})
