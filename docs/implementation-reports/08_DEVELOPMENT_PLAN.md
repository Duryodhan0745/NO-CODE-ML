# No-Code ML Platform — Development Plan

**Document:** Development Plan  
**Version:** 1.0  
**Status:** V1 Development Plan

## 1. Objective

Build a functional No-Code Machine Learning platform that allows a user to:

1. Upload a dataset.
2. Inspect the dataset.
3. Select a target column.
4. Select a problem type.
5. Configure preprocessing.
6. Select an ML model.
7. Configure important model parameters.
8. Train the model.
9. Evaluate the model.
10. Generate predictions.
11. Download generated artifacts.

AI-powered functionality is explicitly outside the V1 implementation.

## 2. Development Philosophy

Build incrementally:

```text
Foundation
    ↓
Dataset Engine
    ↓
Data Analysis
    ↓
Preprocessing Engine
    ↓
Model Engine
    ↓
Training/Evaluation
    ↓
Prediction & Artifacts
    ↓
Backend API
    ↓
Frontend
    ↓
Integration
    ↓
Testing
    ↓
3D Landing Page
    ↓
Optimization
```

## 3. Development Phases

### Phase 0 — Project Foundation
Create the backend/frontend structure, environment configuration, health endpoint, and development setup.

### Phase 1 — Dataset Engine
Implement CSV upload, parsing, validation, previews, row/column detection, data types, missing values, and duplicates.

### Phase 2 — Data Analysis
Implement dataset statistics and profiling for numerical and categorical columns.

### Phase 3 — Preprocessing Engine
Implement missing-value handling, encoding, scaling, outlier handling, feature selection, dimensionality reduction, and class balancing according to `PREPROCESSING_SPEC.md`.

Critical rule:

```text
Dataset
 ↓
Train/Test Split
 ↓
Fit preprocessing on Train
 ↓
Transform Train/Test
```

### Phase 4 — Model Engine
Create a centralized model registry and model factory. Expose important parameters only by default.

### Phase 5 — Training Engine
Build training services, persist trained models, and store training metadata.

### Phase 6 — Evaluation Engine
Implement classification and regression metrics and evaluation result structures.

### Phase 7 — Prediction & Artifacts
Reuse the saved preprocessing pipeline and model for prediction. Generate downloadable prediction artifacts.

### Phase 8 — Backend API
Expose the ML workflow through documented API endpoints. Keep routes thin and business logic in services.

### Phase 9 — Frontend Foundation
Build routing, layouts, API client, reusable components, loading states, and error states.

### Phase 10 — ML Workspace UI
Implement:

```text
Dataset Upload
 → Dataset Preview
 → Analysis
 → Target
 → Problem Type
 → Preprocessing
 → Model
 → Parameters
 → Training
 → Evaluation
 → Prediction
 → Artifacts
```

### Phase 11 — Integration
Connect the real frontend to the backend and verify the complete workflow.

### Phase 12 — Landing Page + 3D
Build the public landing page only after the core workspace is functional.

### Phase 13 — Testing
Run unit, ML, API, frontend, integration, and end-to-end tests.

### Phase 14 — Optimization
Optimize only after correctness is established.

### Phase 15 — V1 Release
Verify the complete workflow and documentation.

## 4. Initial Model Scope

### Classification

- Logistic Regression
- Decision Tree
- Random Forest

### Regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Expand only after the architecture is stable.

## 5. Initial Preprocessing Scope

Start with:

- Missing Value Handling
- Categorical Encoding
- Feature Scaling
- Basic Outlier Handling

Then expand to:

- Feature Selection
- PCA
- SMOTE
- Advanced preprocessing

## 6. MVP Definition

```text
CSV
 ↓
Analysis
 ↓
Target Selection
 ↓
Classification / Regression
 ↓
Basic Preprocessing
 ↓
Model Selection
 ↓
Important Parameters
 ↓
Training
 ↓
Evaluation
 ↓
Prediction
```

## 7. Coding Agent Rules

- Do not implement future features unless requested.
- Do not duplicate business logic.
- Keep API routes thin.
- Keep ML logic independent from frontend code.
- Centralize model and preprocessing metadata.
- Do not move to the next phase until acceptance criteria are met.
- Write tests alongside important functionality.

## 8. Final Architecture

```text
USER
  ↓
FRONTEND
  ↓
API
  ↓
APPLICATION SERVICES
  ↓
ML ENGINE
  ├── Dataset
  ├── Preprocessing
  ├── Models
  ├── Training
  ├── Evaluation
  └── Prediction
  ↓
ARTIFACTS
```

The core ML workflow takes priority over AI, 3D, animations, and visual polish.
