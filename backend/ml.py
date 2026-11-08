"""ML engine: model registry, factory, preprocessing pipeline, evaluation metrics."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)


# ──────────────────────────── Model Registry ────────────────────────────

MODEL_REGISTRY = {
    # ── Classification ──
    "logistic_regression": {
        "model_id": "logistic_regression",
        "name": "Logistic Regression",
        "problem_type": "classification",
        "description": "Linear classifier using logistic function. Good baseline for binary and multiclass problems.",
        "cls": LogisticRegression,
        "needs_scaling": True,
        "parameters": [
            {"name": "C", "type": "float", "default": 1.0,
             "description": "Inverse regularization strength", "min": 0.001, "max": 1000.0},
            {"name": "max_iter", "type": "int", "default": 200,
             "description": "Maximum solver iterations", "min": 10, "max": 10000},
        ],
        "optuna_space": {
            "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 500},
        },
    },
    "decision_tree_classifier": {
        "model_id": "decision_tree_classifier",
        "name": "Decision Tree Classifier",
        "problem_type": "classification",
        "description": "Tree-based classifier that learns decision rules from features.",
        "cls": DecisionTreeClassifier,
        "needs_scaling": False,
        "parameters": [
            {"name": "max_depth", "type": "int", "default": None, "nullable": True,
             "description": "Maximum tree depth (None = unlimited)", "min": 1, "max": 100},
            {"name": "min_samples_split", "type": "int", "default": 2,
             "description": "Minimum samples to split a node", "min": 2, "max": 100},
            {"name": "min_samples_leaf", "type": "int", "default": 1,
             "description": "Minimum samples in a leaf", "min": 1, "max": 100},
        ],
        "optuna_space": {
            "max_depth": {"type": "int", "low": 2, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
    },
    "random_forest_classifier": {
        "model_id": "random_forest_classifier",
        "name": "Random Forest Classifier",
        "problem_type": "classification",
        "description": "Ensemble of decision trees with bagging. Robust and accurate.",
        "cls": RandomForestClassifier,
        "needs_scaling": False,
        "parameters": [
            {"name": "n_estimators", "type": "int", "default": 100,
             "description": "Number of trees", "min": 10, "max": 1000},
            {"name": "max_depth", "type": "int", "default": None, "nullable": True,
             "description": "Maximum tree depth (None = unlimited)", "min": 1, "max": 100},
            {"name": "min_samples_split", "type": "int", "default": 2,
             "description": "Minimum samples to split a node", "min": 2, "max": 100},
            {"name": "min_samples_leaf", "type": "int", "default": 1,
             "description": "Minimum samples in a leaf", "min": 1, "max": 100},
        ],
        "optuna_space": {
            "n_estimators": {"type": "int", "low": 20, "high": 200},
            "max_depth": {"type": "int", "low": 2, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
    },
    "svm_classifier": {
        "model_id": "svm_classifier",
        "name": "Support Vector Machine",
        "problem_type": "classification",
        "description": "Maximizes the decision boundary margin. Highly effective in high-dimensional feature spaces.",
        "cls": SVC,
        "needs_scaling": True,
        "parameters": [
            {"name": "C", "type": "float", "default": 1.0,
             "description": "Regularization parameter", "min": 0.001, "max": 1000.0},
            {"name": "kernel", "type": "string", "default": "rbf",
             "description": "Kernel type", "options": ["rbf", "linear", "poly", "sigmoid"]},
        ],
        "optuna_space": {
            "C": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear"]},
        },
    },
    "gradient_boosting_classifier": {
        "model_id": "gradient_boosting_classifier",
        "name": "Gradient Boosting Classifier",
        "problem_type": "classification",
        "description": "Sequential boosting ensemble that minimizes loss functions iteratively.",
        "cls": GradientBoostingClassifier,
        "needs_scaling": False,
        "parameters": [
            {"name": "n_estimators", "type": "int", "default": 100,
             "description": "Number of boosting stages", "min": 10, "max": 1000},
            {"name": "learning_rate", "type": "float", "default": 0.1,
             "description": "Shrinkage factor of contribution", "min": 0.001, "max": 1.0},
            {"name": "max_depth", "type": "int", "default": 3,
             "description": "Maximum depth of the individual regression estimators", "min": 1, "max": 20},
            {"name": "subsample", "type": "float", "default": 1.0,
             "description": "Fraction of samples used for fitting individual base learners", "min": 0.1, "max": 1.0},
        ],
        "optuna_space": {
            "n_estimators": {"type": "int", "low": 20, "high": 200},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        },
    },
    "knn_classifier": {
        "model_id": "knn_classifier", "name": "K-Nearest Neighbors", "problem_type": "classification",
        "description": "Classifies observations from the labels of nearby training examples.",
        "cls": KNeighborsClassifier, "needs_scaling": True,
        "parameters": [
            {"name": "n_neighbors", "type": "int", "default": 5, "description": "Number of neighbors voting for a class", "min": 1, "max": 100},
            {"name": "weights", "type": "string", "default": "uniform", "description": "How neighbor votes are weighted", "options": ["uniform", "distance"]},
            {"name": "p", "type": "int", "default": 2, "description": "Minkowski distance power: 1 Manhattan, 2 Euclidean", "min": 1, "max": 10},
        ],
        "optuna_space": {"n_neighbors": {"type": "int", "low": 1, "high": 30}, "weights": {"type": "categorical", "choices": ["uniform", "distance"]}},
    },
    "sgd_classifier": {
        "model_id": "sgd_classifier", "name": "SGD Classifier", "problem_type": "classification",
        "description": "Linear classifier trained with stochastic gradient descent; useful for large datasets.",
        "cls": SGDClassifier, "needs_scaling": True,
        "parameters": [
            {"name": "loss", "type": "string", "default": "hinge", "description": "Loss function", "options": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]},
            {"name": "penalty", "type": "string", "default": "l2", "description": "Regularization penalty", "options": ["l2", "l1", "elasticnet"]},
            {"name": "alpha", "type": "float", "default": 0.0001, "description": "Regularization strength", "min": 0.000001, "max": 1.0},
            {"name": "max_iter", "type": "int", "default": 1000, "description": "Maximum passes over the training data", "min": 10, "max": 10000},
        ],
        "optuna_space": {"alpha": {"type": "float", "low": 0.00001, "high": 0.01, "log": True}},
    },
    "xgboost_classifier": {
        "model_id": "xgboost_classifier",
        "name": "XGBoost Classifier",
        "problem_type": "classification",
        "description": "Extreme Gradient Boosting. High-performance, scalable distributed gradient-boosted decision trees.",
        "cls": XGBClassifier,
        "needs_scaling": False,
        "parameters": [
            {"name": "n_estimators", "type": "int", "default": 100,
             "description": "Number of gradient boosted trees", "min": 10, "max": 1000},
            {"name": "max_depth", "type": "int", "default": 6,
             "description": "Maximum tree depth for base learners", "min": 1, "max": 20},
            {"name": "learning_rate", "type": "float", "default": 0.1,
             "description": "Boosting learning rate (eta)", "min": 0.001, "max": 1.0},
            {"name": "subsample", "type": "float", "default": 1.0,
             "description": "Subsample ratio of the training instances", "min": 0.1, "max": 1.0},
        ],
        "optuna_space": {
            "n_estimators": {"type": "int", "low": 20, "high": 200},
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        },
    },

    # ── Regression ──
    "linear_regression": {
        "model_id": "linear_regression",
        "name": "Linear Regression",
        "problem_type": "regression",
        "description": "Ordinary least squares linear regression.",
        "cls": LinearRegression,
        "needs_scaling": True,
        "parameters": [
            {"name": "fit_intercept", "type": "bool", "default": True,
             "description": "Whether to calculate the intercept"},
        ],
        "optuna_space": {
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
        },
    },
    "ridge_regression": {
        "model_id": "ridge_regression", "name": "Ridge Regression", "problem_type": "regression",
        "description": "Linear regression with L2 regularization to reduce coefficient variance.", "cls": Ridge, "needs_scaling": True,
        "parameters": [
            {"name": "alpha", "type": "float", "default": 1.0, "description": "Regularization strength", "min": 0.0, "max": 1000.0},
            {"name": "fit_intercept", "type": "bool", "default": True, "description": "Fit an intercept"},
        ], "optuna_space": {"alpha": {"type": "float", "low": 0.0001, "high": 100.0, "log": True}},
    },
    "lasso_regression": {
        "model_id": "lasso_regression", "name": "Lasso Regression", "problem_type": "regression",
        "description": "Linear regression with L1 regularization that can shrink some coefficients to zero.", "cls": Lasso, "needs_scaling": True,
        "parameters": [
            {"name": "alpha", "type": "float", "default": 1.0, "description": "Regularization strength", "min": 0.0, "max": 1000.0},
            {"name": "max_iter", "type": "int", "default": 1000, "description": "Maximum coordinate-descent iterations", "min": 10, "max": 10000},
        ], "optuna_space": {"alpha": {"type": "float", "low": 0.0001, "high": 100.0, "log": True}},
    },
    "elasticnet_regression": {
        "model_id": "elasticnet_regression", "name": "ElasticNet", "problem_type": "regression",
        "description": "Linear regression combining L1 and L2 regularization.", "cls": ElasticNet, "needs_scaling": True,
        "parameters": [
            {"name": "alpha", "type": "float", "default": 1.0, "description": "Overall regularization strength", "min": 0.0, "max": 1000.0},
            {"name": "l1_ratio", "type": "float", "default": 0.5, "description": "Balance between L1 and L2 penalties", "min": 0.0, "max": 1.0},
            {"name": "max_iter", "type": "int", "default": 1000, "description": "Maximum coordinate-descent iterations", "min": 10, "max": 10000},
        ], "optuna_space": {"alpha": {"type": "float", "low": 0.0001, "high": 100.0, "log": True}, "l1_ratio": {"type": "float", "low": 0.05, "high": 0.95}},
    },
    "sgd_regressor": {
        "model_id": "sgd_regressor", "name": "SGD Regressor", "problem_type": "regression",
        "description": "Linear regressor trained with stochastic gradient descent for large datasets.", "cls": SGDRegressor, "needs_scaling": True,
        "parameters": [
            {"name": "loss", "type": "string", "default": "squared_error", "description": "Regression loss", "options": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]},
            {"name": "penalty", "type": "string", "default": "l2", "description": "Regularization penalty", "options": ["l2", "l1", "elasticnet"]},
            {"name": "alpha", "type": "float", "default": 0.0001, "description": "Regularization strength", "min": 0.000001, "max": 1.0},
            {"name": "max_iter", "type": "int", "default": 1000, "description": "Maximum passes over the training data", "min": 10, "max": 10000},
        ], "optuna_space": {"alpha": {"type": "float", "low": 0.00001, "high": 0.01, "log": True}},
    },
    "decision_tree_regressor": {
        "model_id": "decision_tree_regressor",
        "name": "Decision Tree Regressor",
        "problem_type": "regression",
        "description": "Tree-based regressor that learns decision rules from features.",
        "cls": DecisionTreeRegressor,
        "needs_scaling": False,
        "parameters": [
            {"name": "max_depth", "type": "int", "default": None, "nullable": True,
             "description": "Maximum tree depth (None = unlimited)", "min": 1, "max": 100},
            {"name": "min_samples_split", "type": "int", "default": 2,
             "description": "Minimum samples to split a node", "min": 2, "max": 100},
            {"name": "min_samples_leaf", "type": "int", "default": 1,
             "description": "Minimum samples in a leaf", "min": 1, "max": 100},
        ],
        "optuna_space": {
            "max_depth": {"type": "int", "low": 2, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
    },
    "random_forest_regressor": {
        "model_id": "random_forest_regressor",
        "name": "Random Forest Regressor",
        "problem_type": "regression",
        "description": "Ensemble of decision trees for regression. Robust and accurate.",
        "cls": RandomForestRegressor,
        "needs_scaling": False,
        "parameters": [
            {"name": "n_estimators", "type": "int", "default": 100,
             "description": "Number of trees", "min": 10, "max": 1000},
            {"name": "max_depth", "type": "int", "default": None, "nullable": True,
             "description": "Maximum tree depth (None = unlimited)", "min": 1, "max": 100},
            {"name": "min_samples_split", "type": "int", "default": 2,
             "description": "Minimum samples to split a node", "min": 2, "max": 100},
            {"name": "min_samples_leaf", "type": "int", "default": 1,
             "description": "Minimum samples in a leaf", "min": 1, "max": 100},
        ],
        "optuna_space": {
            "n_estimators": {"type": "int", "low": 20, "high": 200},
            "max_depth": {"type": "int", "low": 2, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
    },
    "svm_regressor": {
        "model_id": "svm_regressor",
        "name": "Support Vector Regression",
        "problem_type": "regression",
        "description": "Finds a hyperplane in an epsilon-tube around continuous targets.",
        "cls": SVR,
        "needs_scaling": True,
        "parameters": [
            {"name": "C", "type": "float", "default": 1.0,
             "description": "Regularization parameter", "min": 0.001, "max": 1000.0},
            {"name": "kernel", "type": "string", "default": "rbf",
             "description": "Kernel type", "options": ["rbf", "linear", "poly", "sigmoid"]},
            {"name": "epsilon", "type": "float", "default": 0.1,
             "description": "Epsilon in epsilon-SVR model", "min": 0.0, "max": 10.0},
        ],
        "optuna_space": {
            "C": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear"]},
            "epsilon": {"type": "float", "low": 0.01, "high": 1.0},
        },
    },
    "gradient_boosting_regressor": {
        "model_id": "gradient_boosting_regressor",
        "name": "Gradient Boosting Regressor",
        "problem_type": "regression",
        "description": "Additive regression ensemble building weak decision trees stage by stage.",
        "cls": GradientBoostingRegressor,
        "needs_scaling": False,
        "parameters": [
            {"name": "n_estimators", "type": "int", "default": 100,
             "description": "Number of boosting stages", "min": 10, "max": 1000},
            {"name": "learning_rate", "type": "float", "default": 0.1,
             "description": "Shrinkage factor of contribution", "min": 0.001, "max": 1.0},
            {"name": "max_depth", "type": "int", "default": 3,
             "description": "Maximum depth of the individual regression estimators", "min": 1, "max": 20},
            {"name": "subsample", "type": "float", "default": 1.0,
             "description": "Fraction of samples used for fitting individual base learners", "min": 0.1, "max": 1.0},
        ],
        "optuna_space": {
            "n_estimators": {"type": "int", "low": 20, "high": 200},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        },
    },
    "xgboost_regressor": {
        "model_id": "xgboost_regressor",
        "name": "XGBoost Regressor",
        "problem_type": "regression",
        "description": "Extreme Gradient Boosting regression engine optimized for speed and accuracy.",
        "cls": XGBRegressor,
        "needs_scaling": False,
        "parameters": [
            {"name": "n_estimators", "type": "int", "default": 100,
             "description": "Number of gradient boosted trees", "min": 10, "max": 1000},
            {"name": "max_depth", "type": "int", "default": 6,
             "description": "Maximum tree depth for base learners", "min": 1, "max": 20},
            {"name": "learning_rate", "type": "float", "default": 0.1,
             "description": "Boosting learning rate (eta)", "min": 0.001, "max": 1.0},
            {"name": "subsample", "type": "float", "default": 1.0,
             "description": "Subsample ratio of the training instances", "min": 0.1, "max": 1.0},
        ],
        "optuna_space": {
            "n_estimators": {"type": "int", "low": 20, "high": 200},
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        },
    },
}


# ──────────────────────────── Model Factory ────────────────────────────

def _validate_param(name: str, value, pdef: dict):
    if value is None:
        if pdef.get("nullable"):
            return None
        raise ValueError(f"Parameter '{name}' cannot be None")

    if pdef["type"] == "int":
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"Parameter '{name}' must be an integer")
    elif pdef["type"] == "float":
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Parameter '{name}' must be a number")
    elif pdef["type"] == "bool":
        if isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")
        value = bool(value)
    elif pdef["type"] == "string":
        value = str(value)
        if "options" in pdef and value not in pdef["options"]:
            raise ValueError(f"Parameter '{name}' must be one of {pdef['options']}")

    if pdef["type"] in ("int", "float") and value is not None:
        if "min" in pdef and pdef["min"] is not None and value < pdef["min"]:
            raise ValueError(f"Parameter '{name}' must be >= {pdef['min']}")
        if "max" in pdef and pdef["max"] is not None and value > pdef["max"]:
            raise ValueError(f"Parameter '{name}' must be <= {pdef['max']}")

    return value


def create_model(model_id: str, params: dict):
    """Validate params and instantiate an estimator from the registry."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")

    entry = MODEL_REGISTRY[model_id]
    param_defs = {p["name"]: p for p in entry["parameters"]}
    validated = {}

    for name, value in params.items():
        if name not in param_defs:
            raise ValueError(f"Unknown parameter '{name}' for model '{model_id}'")
        validated[name] = _validate_param(name, value, param_defs[name])

    for pdef in entry["parameters"]:
        if pdef["name"] not in validated:
            validated[pdef["name"]] = pdef["default"]

    extra_kwargs = {}
    cls_name = entry["cls"].__name__
    if cls_name == "SVC":
        extra_kwargs["probability"] = True
        extra_kwargs["random_state"] = 42
    elif "XGB" in cls_name:
        extra_kwargs["random_state"] = 42
        extra_kwargs["eval_metric"] = "logloss" if entry["problem_type"] == "classification" else "rmse"
    elif cls_name not in ("LinearRegression", "SVR", "Ridge", "Lasso", "ElasticNet", "KNeighborsClassifier"):
        extra_kwargs["random_state"] = 42

    return entry["cls"](**validated, **extra_kwargs)


def get_model_info(model_id: str) -> dict:
    """Return frontend-safe model info (no cls)."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    e = MODEL_REGISTRY[model_id]
    return {k: v for k, v in e.items() if k != "cls"}


def list_models(problem_type: str | None = None) -> list[dict]:
    out = []
    for e in MODEL_REGISTRY.values():
        if problem_type and e["problem_type"] != problem_type:
            continue
        out.append({k: v for k, v in e.items() if k != "cls"})
    return out


# ──────────────────────────── Preprocessing ────────────────────────────

class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips numerical features to [Q1-1.5*IQR, Q3+1.5*IQR]. Fit on train only."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(X, 25, axis=0)
        self.q3_ = np.nanpercentile(X, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - 1.5 * iqr
        self.upper_ = self.q3_ + 1.5 * iqr
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


def build_preprocessing_pipeline(numerical_cols, categorical_cols, config: dict):
    """Build a ColumnTransformer or Pipeline with PCA from user preprocessing config.

    config keys: missing_values (mean|median|mode|drop), scaling (standard|minmax|None),
                 outlier_handling (bool), pca (bool), pca_components (int|None).
    """
    transformers = []

    if numerical_cols:
        num_steps = []
        strategy = config.get("missing_values", "mean")
        if strategy == "mode":
            strategy = "most_frequent"
        if strategy != "drop":
            num_steps.append(("imputer", SimpleImputer(strategy=strategy)))
        if config.get("outlier_handling"):
            num_steps.append(("outlier", IQRClipper()))
        scaling = config.get("scaling")
        if scaling == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaling == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        transformers.append(
            ("num", Pipeline(num_steps) if num_steps else "passthrough", numerical_cols)
        )

    if categorical_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    if not transformers:
        return None

    col_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    if config.get("pca"):
        pca_kwargs = {}
        if config.get("pca_components"):
            pca_kwargs["n_components"] = int(config["pca_components"])
        return Pipeline([
            ("features", col_transformer),
            ("pca", PCA(**pca_kwargs)),
        ])

    return col_transformer


# ──────────────────────────── Evaluation ────────────────────────────

def compute_metrics(y_true, y_pred, problem_type: str, y_proba=None) -> dict:
    if problem_type == "classification":
        labels = sorted(set(y_true) | set(y_pred), key=str)
        avg = "weighted" if len(labels) > 2 else "binary"
        pos = labels[1] if len(labels) == 2 else None
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0, pos_label=pos)),
            "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0, pos_label=pos)),
            "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0, pos_label=pos)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "class_labels": [str(l) for l in labels],
        }
        if y_proba is not None:
            try:
                if len(labels) == 2:
                    proba_col = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 else y_proba
                    metrics["roc_auc"] = float(roc_auc_score(y_true, proba_col))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
            except Exception:
                pass
        return metrics

    if problem_type == "regression":
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    raise ValueError(f"Unknown problem type: {problem_type}")
