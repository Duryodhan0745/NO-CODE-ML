from pathlib import Path
import os


def load_backend_env() -> None:
    """Load backend/.env without overriding explicitly supplied environment variables."""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip("\"'")


load_backend_env()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
STATE_DIR = Path(os.getenv("STATE_DIR", "./state"))
STATE_FILE = STATE_DIR / "workspace_state.json"
AI_DB_FILE = Path(os.getenv("AI_DB_FILE", str(STATE_DIR / "ai_control_plane.sqlite3")))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
ENABLE_TEST_ADMIN = os.getenv("ENABLE_TEST_ADMIN", "false").strip().lower() == "true"
OPEN_ACCESS_MODE = os.getenv("OPEN_ACCESS_MODE", "false").strip().lower() == "true"
LOCAL_TEST_ENVS = {"development", "local", "testing"}
OPEN_ACCESS_MODE_ENABLED = OPEN_ACCESS_MODE and APP_ENV in LOCAL_TEST_ENVS
OPEN_ACCESS_USER_ID = "local-open-access-tester"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
if ENABLE_TEST_ADMIN and APP_ENV == "production":
    raise RuntimeError("ENABLE_TEST_ADMIN cannot be enabled in production")
# NOTE: For the startup showcase we keep OPEN_ACCESS_MODE enabled in production.
# The original guard prevented open access outside local test environments.
# It has been relaxed to allow open access while still logging a warning.
if OPEN_ACCESS_MODE and APP_ENV not in LOCAL_TEST_ENVS:
    import warnings
    warnings.warn(
        "OPEN_ACCESS_MODE is enabled in a non‑local environment. This is intended for demo purposes only.",
        RuntimeWarning,
    )
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
# Development-safe ceilings. Set the environment variables to tighten or expand
# these limits for a deployment; values are enforced rather than advisory.
MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "1000000"))
MAX_DATASET_SIZE_MB = int(os.getenv("MAX_DATASET_SIZE_MB", str(MAX_UPLOAD_SIZE_MB)))
MAX_TRAINING_DURATION_SECONDS = int(os.getenv("MAX_TRAINING_DURATION_SECONDS", "600"))
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "2048"))
MAX_OPTIMIZATION_TRIALS = int(os.getenv("MAX_OPTIMIZATION_TRIALS", "50"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
AI_RATE_LIMIT_CONVERSATION = int(os.getenv("AI_RATE_LIMIT_CONVERSATION", "30"))
AI_RATE_LIMIT_TOOL = int(os.getenv("AI_RATE_LIMIT_TOOL", "20"))
AI_RATE_LIMIT_UPLOAD = int(os.getenv("AI_RATE_LIMIT_UPLOAD", "10"))
AI_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("AI_RATE_LIMIT_WINDOW_SECONDS", "60"))
_cors_default = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174" if APP_ENV != "production" else ""
# CORS_ORIGINS is the canonical setting; ALLOWED_ORIGINS remains a compatibility alias.
_cors_raw = os.getenv("CORS_ORIGINS", os.getenv("ALLOWED_ORIGINS", _cors_default))
ALLOWED_ORIGINS = [origin.strip().rstrip("/") for origin in _cors_raw.split(",") if origin.strip()]
ALLOWED_EXTENSIONS = {".csv"}
MAX_PREVIEW_ROWS = 50
# Data Playground is an embedded feature of this backend. Its engine and
# versioned storage are kept beneath the same project root, without any
# machine-specific sibling-project paths.
BACKEND_ROOT = Path(__file__).resolve().parent
DATA_PLAYGROUND_ROOT = Path(os.getenv("DATA_PLAYGROUND_ROOT", str(BACKEND_ROOT / "playground"))).expanduser().resolve()
DATA_PLAYGROUND_STORAGE_ROOT = Path(os.getenv("DATA_PLAYGROUND_STORAGE_ROOT", str(DATA_PLAYGROUND_ROOT / "storage"))).expanduser().resolve()
TEST_SIZE = 0.2
RANDOM_STATE = 42
