"""Conservative extraction of explicit, durable ML working preferences."""
from __future__ import annotations

import re


_PREFERENCE_CUE = re.compile(r"\b(?:i prefer|i usually use|my preferred|my default(?: is)?|always use|from now on|remember that)\b", re.I)
_PROJECT_OBJECTIVE = re.compile(r"\b(?:this project(?:'s)? (?:goal|objective) is|our project (?:goal|objective) is|project objective:)\s*(.+)", re.I)
_PROJECT_DECISION = re.compile(r"\b(?:we decided to|our decision is to|for this project, use)\s+(.+)", re.I)
_SENSITIVE = re.compile(r"\b(?:password|api[_ -]?key|bearer\s+\S+|secret|private key)\b", re.I)


def extract_durable_memories(text: str) -> list[dict]:
    """Return only explicit durable preferences and clearly labeled project facts."""
    clean = " ".join(text.strip().split())
    if not clean or len(clean) > 1200 or _SENSITIVE.search(clean):
        return []
    if re.search(r"\bfor this (?:one|run|session|request|task)\b", clean, re.I):
        return []
    candidates = []
    if _PREFERENCE_CUE.search(clean):
        lowered = clean.casefold()
        if any(name in lowered for name in ("random forest", "logistic regression", "decision tree", "gradient boosting", "linear regression", "svm", "xgboost")):
            category = "preferred_model"
            key = "classification" if "classif" in lowered else "regression" if "regress" in lowered else "general"
        elif any(name in lowered for name in ("imputation", "missing value", "median", "mean", "mode", "scaling", "standard scaler", "minmax", "one-hot")):
            category, key = "preferred_preprocessing", "default"
        elif any(name in lowered for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "r2", "r²", "rmse", "mae")):
            category, key = "preferred_metric", "default"
        elif any(name in lowered for name in ("concise", "detailed", "depth", "technical", "simple")):
            category, key = "explanation_style", "default"
        elif any(name in lowered for name in ("chart", "plot", "visualization", "visualisation")):
            category, key = "visualization_style", "default"
        else:
            category, key = "working_preference", "default"
        candidates.append({"scope_type": "user", "category": category, "memory_key": key,
                           "content": clean[:800], "confidence": 0.92, "importance": 4,
                           "review_policy": "replace_on_explicit_correction"})
    objective = _PROJECT_OBJECTIVE.search(clean)
    if objective:
        candidates.append({"scope_type": "project", "category": "project_objective", "memory_key": "objective",
                           "content": objective.group(1).strip()[:800], "confidence": 0.95,
                           "importance": 5, "review_policy": "until_project_deleted"})
    decision = _PROJECT_DECISION.search(clean)
    if decision:
        chosen = decision.group(1).strip()[:800]
        candidates.append({"scope_type": "project", "category": "project_decision",
                           "memory_key": re.sub(r"\W+", "_", chosen.casefold()).strip("_")[:100],
                           "content": chosen, "confidence": 0.88, "importance": 4,
                           "review_policy": "until_project_deleted"})
    return candidates
