# No-Code ML Platform

A unified No-Code Machine Learning application with an integrated Data Playground. Upload CSV, XLSX, or Parquet data into the Playground to explore, transform, version, analyze, export, and hand off a prepared dataset to the ML workflow. The ML workspace supports target selection, preprocessing, model training, evaluation, and artifact downloads.

---

## Architecture

```
User (Browser)
    │
    ▼
Frontend (React 18 + Vite + Tailwind CSS)  [http://localhost:5173]
    │  /api proxy
    ▼
Backend (FastAPI + Pydantic v2 + Uvicorn)   [http://localhost:8000]
    │
    ├── Services (Dataset, Workspace, Training, Prediction, Artifacts)
    │       │
    │       ▼
    ├── Data Playground engine (versioned Parquet data, SQLite metadata,
    │   analysis, visualizations, transformations, export)
    └── ML Engine (scikit-learn + NumPy + Pandas)
            ├── Model Registry (classification and regression estimators, including SVM, XGBoost, SGD, KNN, and regularized linear models)
            ├── Parameter Factory (type-safe validation, range checks)
            ├── Preprocessing Pipeline (ColumnTransformer, split-leakage safe)
            ├── Evaluation Engine (accuracy, precision, recall, f1, MAE, MSE, RMSE, R²)
            └── Artifact Store (joblib serialization, CSV outputs)
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Start Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Backend health check: `http://localhost:8000/api/health`
Interactive Swagger API docs: `http://localhost:8000/docs`

### 2. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### 3. Run Tests

```bash
cd backend
python -m pytest tests/ -v
python tests/e2e_test.py
```

The reproducible backend verification harnesses are in `scripts/verification/`.
The repository map and extension conventions are documented in
[`docs/architecture/REPOSITORY_STRUCTURE.md`](docs/architecture/REPOSITORY_STRUCTURE.md).

### AI assistant sign-in

Google sign-in is optional and is used only for the authenticated AI assistant. Set
`GOOGLE_CLIENT_ID` in `backend/.env` to the public OAuth Web client ID, and add the
frontend origin (for example, `http://localhost:5173` or `http://127.0.0.1:5173`,
matching the URL used in the browser) to that OAuth client's authorized JavaScript
origins and to `ALLOWED_ORIGINS` when overriding its development default. The
backend verifies Google ID tokens against that audience; tokens are kept in browser
session storage. The regular ML workflow remains available without sign-in.

Gemini requests use the backend-only `AI_API_KEY`, `AI_BASE_URL`, and `AI_MODEL`
settings. Never place the API key in frontend code or expose it through the auth
configuration endpoint.

## Data Playground storage and migration

The embedded Data Playground engine lives in `backend/playground/`; its immutable
dataset versions and metadata live in `backend/playground/storage/`. These paths
are project-relative defaults and can be overridden with `DATA_PLAYGROUND_ROOT`
and `DATA_PLAYGROUND_STORAGE_ROOT` for a managed deployment. Do not delete this
storage directory: it contains the Playground SQLite metadata database and the
original and transformed dataset versions.

The prior standalone `data play1/` directory is retained as a recovery source
while consolidation is verified. Its former storage directory has been moved into
the embedded engine, so only the primary frontend and primary backend should be
started for normal development.

---

## Supported Models

### Classification
- **Logistic Regression**: Regularized linear classifier (`C`, `max_iter`)
- **Decision Tree Classifier**: Non-parametric tree (`max_depth`, `min_samples_split`, `min_samples_leaf`)
- **Random Forest Classifier**: Ensemble bagging trees (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`)
- **Support Vector Classifier**: Margin-based classification (`C`, `kernel`)
- **Gradient Boosting Classifier**: Additive tree boosting (`n_estimators`, `learning_rate`, `max_depth`, `subsample`)
- **XGBoost Classifier**: XGBoost tree boosting (`n_estimators`, `max_depth`, `learning_rate`, `subsample`)
- **K-Nearest Neighbors**: Neighbor voting (`n_neighbors`, `weights`, `p`)
- **SGD Classifier**: Stochastic-gradient linear classifier (`loss`, `penalty`, `alpha`, `max_iter`)

### Regression
- **Linear Regression**: Ordinary least squares (`fit_intercept`)
- **Decision Tree Regressor**: Tree-based regression (`max_depth`, `min_samples_split`, `min_samples_leaf`)
- **Random Forest Regressor**: Ensemble regression (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`)
- **Ridge Regression**: L2-regularized linear regression (`alpha`, `fit_intercept`)
- **Lasso Regression**: L1-regularized linear regression (`alpha`, `max_iter`)
- **ElasticNet**: Combined L1/L2 linear regression (`alpha`, `l1_ratio`, `max_iter`)
- **Support Vector Regressor**: Epsilon-tube regression (`C`, `kernel`, `epsilon`)
- **Gradient Boosting Regressor**: Additive tree boosting (`n_estimators`, `learning_rate`, `max_depth`, `subsample`)
- **XGBoost Regressor**: XGBoost tree boosting (`n_estimators`, `max_depth`, `learning_rate`, `subsample`)
- **SGD Regressor**: Stochastic-gradient linear regression (`loss`, `penalty`, `alpha`, `max_iter`)

---

## Preprocessing & Data Safety

- **Train/Test Splitting First**: Train/test split occurs before fitting any preprocessor. Test data is strictly transformed via `transform()` and never `fit_transform()`.
- **Missing Value Imputation**: Mean / Median (numerical), Most-frequent (categorical), or row dropping with high-missingness threshold warnings.
- **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fitted on train features.
- **Feature Scaling**: `StandardScaler` or `MinMaxScaler`. Tree models clearly flagged as not requiring scaling.
- **Outlier Handling**: Custom `IQRClipper` transformer clipping features to `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`, fitted strictly on training data.
- **Class Balancing (SMOTE)**: Synthetic Minority Oversampling Technique via `imbalanced-learn`, applied strictly to the training fold of classification datasets.
- **Multicollinearity & VIF Analysis**: Variance Inflation Factor and correlation matrices via `statsmodels` to pinpoint collinear predictors.
- **Automated Profiling**: `ydata-profiling` HTML report generation and in-browser inspection.
- **Artifact Traceability**: Trained models, preprocessor pipelines, training metadata, `config.json`, and test predictions saved in per-run artifact directories.
- **Multi-Run History**: Track, inspect, and benchmark multiple training runs within a workspace.
- **Inference Integrity**: Prediction uses the exact saved `preprocessor.joblib` and `model.joblib`. Zero preprocessing re-fitting occurs on new input data, with schema and data type validation.

Scaling guidance is surfaced from the backend model registry (`needs_scaling`) and is advisory. The current configurable scalers are StandardScaler, MinMaxScaler, or none. GridSearchCV and RandomizedSearchCV are available through the same optimization endpoint as Optuna (`strategy: "grid"` or `"random"`). Bayesian optimization is exposed as `strategy: "bayesian"` and uses Optuna's TPE sampler over the registry search spaces. All four search paths keep the held-out test partition out of selection. Workspaces and run history are persisted to `STATE_DIR/workspace_state.json`, while uploaded datasets and generated artifacts are stored locally. Clustering, time-aware splitting, configurable encoders, and advanced feature selection remain outside the current implementation.

---

## 10-Step ML Workspace Workflow

1. **Upload Dataset**: CSV file upload with size/type validation.
2. **Preview**: Tabular inspection of top rows and summary metrics.
3. **Analysis**: Detailed column-level statistics, missing% warnings, correlation heatmap, VIF scores, and full YData HTML report.
4. **Target & Problem Selection**: Choose target column and problem type (Classification / Regression).
5. **Preprocessing**: Configure missing values strategy, scaling, outlier clipping, and SMOTE oversampling.
6. **Model Selection**: Select from compatible registry estimators with model-aware guidance.
7. **Parameters**: Configure hyperparameters with real-time range and type validation.
8. **Train**: Real scikit-learn training execution on train split with full reproducibility.
9. **Evaluation**: Real metrics, confusion matrix heatmaps (classification), residual metrics (regression), and multi-run history log.
10. **Predict & Download**: Upload unlabelled/new data to produce predictions and download full artifacts (`model.joblib`, `preprocessor.joblib`, `config.json`, `predictions.csv`).
# Test Super Admin (non-production only)

To provision a dedicated testing administrator, configure the backend environment without committing credentials:

```text
APP_ENV=test
ENABLE_TEST_ADMIN=true
ADMIN_EMAIL=<testing admin email>
ADMIN_PASSWORD=<strong unique password, at least 12 characters>
```

The backend creates the account at startup, stores only a salted scrypt password hash in the AI SQLite database, and issues a signed `super_admin` bearer token through `POST /api/auth/test-admin/login`. Test-admin provisioning is rejected when `APP_ENV=production`. Never use production credentials or enable this flag in production.

## Open Access Testing Mode

For isolated development exploration only, set `APP_ENV=test` and `OPEN_ACCESS_MODE=true`. The backend assigns unauthenticated local test requests a synthetic `super_admin` testing principal, bypasses workspace ownership and training approval gates, and exposes an explicit banner in the AI workspace. Input validation, allowlisted tools, real ML execution, and secret protection remain active. The backend refuses to start when `OPEN_ACCESS_MODE=true` with `APP_ENV=production`. Disable it by removing the variable or setting `OPEN_ACCESS_MODE=false`.
# No-Code ML

The project is organized into a React/Vite frontend and a FastAPI/scikit-learn backend. See [the repository structure guide](docs/architecture/REPOSITORY_STRUCTURE.md) for module boundaries and extension conventions.
