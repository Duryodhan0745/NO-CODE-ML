"""Optuna hyperparameter optimization engine for No-Code ML Platform."""

import time
import logging
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from ml import MODEL_REGISTRY, create_model, build_preprocessing_pipeline, compute_metrics

log = logging.getLogger(__name__)

# Reduce Optuna log verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

METRIC_DIRECTIONS = {
    "accuracy": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "f1": "maximize",
    "roc_auc": "maximize",
    "r2": "maximize",
    "mae": "minimize",
    "mse": "minimize",
    "rmse": "minimize",
}

def run_sklearn_search(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str,
                       preprocessing_config: dict, model_id: str, strategy: str,
                       param_grid: dict | None = None, n_iter: int = 10,
                       cv_folds: int = 3, random_state: int = 42) -> dict:
    """Leakage-safe GridSearchCV/RandomizedSearchCV over the training partition."""
    if model_id not in MODEL_REGISTRY or MODEL_REGISTRY[model_id]["problem_type"] != problem_type:
        raise ValueError(f"Model '{model_id}' does not match problem type '{problem_type}'")
    if strategy not in ("grid", "random"):
        raise ValueError("strategy must be 'grid' or 'random'")
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, preprocessing_config)
    estimator = create_model(model_id, {})
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)]) if preprocessor is not None else Pipeline([("model", estimator)])
    defaults = {}
    for p in MODEL_REGISTRY[model_id]["parameters"]:
        value = p.get("default")
        if value is not None and p.get("options"):
            defaults[f"model__{p['name']}"] = p["options"]
        elif value is not None and p["type"] in ("int", "float"):
            low, high = p.get("min"), p.get("max")
            if low is not None and high is not None:
                defaults[f"model__{p['name']}"] = [low, value, high]
    supplied = {f"model__{k}": v for k, v in (param_grid or {}).items()}
    search_space = supplied or defaults
    scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"
    Search = GridSearchCV if strategy == "grid" else RandomizedSearchCV
    kwargs = dict(estimator=pipeline, param_distributions=search_space if strategy == "random" else search_space,
                  scoring=scoring, cv=cv_folds, n_jobs=1, refit=True, random_state=random_state) if strategy == "random" else dict(
                      estimator=pipeline, param_grid=search_space, scoring=scoring, cv=cv_folds, n_jobs=1, refit=True)
    if strategy == "random":
        kwargs["n_iter"] = max(1, int(n_iter))
    search = Search(**kwargs)
    search.fit(X_train, y_train)
    return {"strategy": strategy, "model_id": model_id, "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "best_cv_score": float(search.best_score_), "cv_results": [{"params": {k.replace("model__", ""): v for k, v in row.items()}, "score": float(score)} for row, score in zip(search.cv_results_["params"][:50], search.cv_results_["mean_test_score"][:50])], "status": "COMPLETED"}


def _sample_params_for_model(trial: optuna.Trial, model_id: str) -> dict:
    """Sample hyperparameters for a model based on its registry optuna_space."""
    entry = MODEL_REGISTRY[model_id]
    space = entry.get("optuna_space", {})
    params = {}

    for param_name, spec in space.items():
        trial_key = f"{model_id}__{param_name}"
        stype = spec.get("type")
        if stype == "int":
            params[param_name] = trial.suggest_int(trial_key, spec["low"], spec["high"])
        elif stype == "float":
            params[param_name] = trial.suggest_float(
                trial_key, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif stype == "categorical":
            params[param_name] = trial.suggest_categorical(trial_key, spec["choices"])

    return params


def run_optuna_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    preprocessing_config: dict,
    candidate_models: list[str],
    metric: str = "accuracy",
    n_trials: int = 15,
    cv_folds: int = 3,
    timeout: int | None = 60,
    random_state: int = 42,
) -> dict:
    """Run Optuna cross-validated hyperparameter optimization on training data.

    ZERO LEAKAGE GUARANTEE:
    - Only receives training data (X_train, y_train).
    - In each CV fold, preprocessing and SMOTE are fitted strictly on the fold training split.
    - Test set is never touched.
    """
    if not candidate_models:
        raise ValueError("At least one candidate model must be specified for optimization")

    for mid in candidate_models:
        if mid not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model in candidate models: {mid}")
        if MODEL_REGISTRY[mid]["problem_type"] != problem_type:
            raise ValueError(f"Model '{mid}' does not match problem type '{problem_type}'")

    direction = METRIC_DIRECTIONS.get(metric, "maximize")
    start_time = time.time()
    trials_history = []

    # Configure CV splitter
    if problem_type == "classification":
        # Ensure n_splits does not exceed smallest class count
        min_class_count = y_train.value_counts().min()
        actual_cv = max(2, min(cv_folds, min_class_count))
        cv = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=random_state)
    else:
        actual_cv = max(2, min(cv_folds, len(X_train)))
        cv = KFold(n_splits=actual_cv, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        # 1. Select model
        if len(candidate_models) == 1:
            chosen_model_id = candidate_models[0]
        else:
            chosen_model_id = trial.suggest_categorical("model_id", candidate_models)

        # 2. Sample hyperparameters
        params = _sample_params_for_model(trial, chosen_model_id)

        fold_scores = []

        try:
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_f_tr = X_train.iloc[train_idx]
                X_f_val = X_train.iloc[val_idx]
                y_f_tr = y_train.iloc[train_idx]
                y_f_val = y_train.iloc[val_idx]

                num_cols = X_f_tr.select_dtypes(include=["number"]).columns.tolist()
                cat_cols = X_f_tr.select_dtypes(include=["object", "category"]).columns.tolist()

                # Build & fit preprocessing pipeline inside fold
                preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, preprocessing_config)
                if preprocessor is not None:
                    X_f_tr_proc = preprocessor.fit_transform(X_f_tr)
                    X_f_val_proc = preprocessor.transform(X_f_val)
                else:
                    X_f_tr_proc = X_f_tr.values
                    X_f_val_proc = X_f_val.values

                # Fold-level SMOTE if requested
                if preprocessing_config.get("smote") and problem_type == "classification":
                    try:
                        smote = SMOTE(random_state=random_state)
                        X_f_tr_proc, y_f_tr = smote.fit_resample(X_f_tr_proc, y_f_tr)
                    except Exception as e:
                        log.debug("Fold SMOTE skipped: %s", e)

                # Train model
                model = create_model(chosen_model_id, params)
                model.fit(X_f_tr_proc, y_f_tr)

                # Predict & evaluate
                y_val_pred = model.predict(X_f_val_proc)
                y_val_proba = None
                if metric == "roc_auc" and hasattr(model, "predict_proba"):
                    try:
                        y_val_proba = model.predict_proba(X_f_val_proc)
                    except Exception:
                        pass

                fold_metrics = compute_metrics(y_f_val, y_val_pred, problem_type, y_proba=y_val_proba)
                score = fold_metrics.get(metric, 0.0)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            trial_record = {
                "trial_number": trial.number,
                "model_id": chosen_model_id,
                "params": params,
                "score": round(mean_score, 4),
                "state": "COMPLETE",
            }
            trials_history.append(trial_record)
            return mean_score

        except Exception as e:
            log.warning("Trial %d failed: %s", trial.number, e)
            trial_record = {
                "trial_number": trial.number,
                "model_id": chosen_model_id,
                "params": params,
                "score": None,
                "state": "FAILED",
                "error": str(e),
            }
            trials_history.append(trial_record)
            return float("-inf") if direction == "maximize" else float("inf")

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))

    duration = round(time.time() - start_time, 2)

    # Determine best trial
    if len(study.trials) == 0 or study.best_trial is None:
        raise ValueError("Optuna optimization failed to complete any trials successfully")

    best_trial = study.best_trial

    # Extract best model_id and best params
    if len(candidate_models) == 1:
        best_model_id = candidate_models[0]
    else:
        best_model_id = best_trial.params.get("model_id", candidate_models[0])

    prefix = f"{best_model_id}__"
    best_params = {}
    for k, v in best_trial.params.items():
        if k.startswith(prefix):
            best_params[k[len(prefix):]] = v

    log.info(
        "Optuna optimization complete: best_model=%s, best_%s=%.4f (duration=%.1fs)",
        best_model_id, metric, best_trial.value, duration
    )

    return {
        "best_model_id": best_model_id,
        "best_params": best_params,
        "best_cv_score": round(float(best_trial.value), 4),
        "metric": metric,
        "direction": direction,
        "n_trials": len(study.trials),
        "trials": trials_history,
        "duration_seconds": duration,
        "status": "COMPLETED",
    }


def run_bayesian_optimization(*args, **kwargs) -> dict:
    """Run Bayesian hyperparameter optimization using Optuna's TPE sampler.

    TPE models promising regions from completed trials and samples the next
    configuration from that posterior-like density. It is the Bayesian search
    strategy exposed by the product; the CV and leakage guarantees are the
    same as the Optuna path.
    """
    result = run_optuna_optimization(*args, **kwargs)
    result["strategy"] = "bayesian"
    result["optimizer"] = "optuna_tpe"
    return result
