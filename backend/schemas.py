from pydantic import BaseModel, Field
from typing import Any


class CreateWorkspaceRequest(BaseModel):
    dataset_id: str


class PreprocessingConfig(BaseModel):
    missing_values: str = "mean"
    scaling: str | None = None
    outlier_handling: bool = False
    smote: bool = False
    pca: bool = False
    pca_components: int | None = None


class ConfigureRequest(BaseModel):
    target: str
    problem_type: str
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    model_id: str
    model_params: dict[str, Any] = {}


class OptimizeRequest(BaseModel):
    mode: str = "selected_model"  # "selected_model" or "multi_model"
    candidate_models: list[str] = Field(default_factory=list)
    metric: str = "accuracy"
    n_trials: int = 15
    cv_folds: int = 3
    timeout: int | None = 60
    random_state: int = 42
    strategy: str = "optuna"  # optuna, bayesian, grid, random
    param_grid: dict[str, Any] = Field(default_factory=dict)
    n_iter: int = 10
