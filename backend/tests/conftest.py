import sys
import os
from pathlib import Path

# This is a standalone live-server smoke script, not a pytest module. It is
# invoked after Uvicorn starts in the end-to-end workflow.
collect_ignore = ["e2e_test.py"]

# Tests must not inherit local convenience mode or production request ceilings
# from backend/.env. Individual authentication tests supply their own tokens.
os.environ["APP_ENV"] = "testing"
os.environ["OPEN_ACCESS_MODE"] = "false"
os.environ["AI_RATE_LIMIT_UPLOAD"] = "10000"
os.environ["AI_RATE_LIMIT_CONVERSATION"] = "10000"
os.environ["AI_RATE_LIMIT_TOOL"] = "10000"

# Add backend dir to path so tests can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
