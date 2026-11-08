# No-Code ML Platform — Coding Agent Rules

**Document:** Coding Agent Rules  
**Version:** 1.0  
**Status:** Mandatory Development Rules

## 1. Purpose

This document defines how Codex, Antigravity, and other coding agents must work on the No-Code ML Platform.

Priorities:

1. Correctness
2. Existing architecture
3. Maintainability
4. Testability
5. Security
6. Reproducibility
7. Minimal unnecessary changes

## 2. Source of Truth

Use the project specifications as the primary source of truth:

```text
PROJECT_SPEC.md
ARCHITECTURE.md
ML_PIPELINE.md
PREPROCESSING_SPEC.md
MODEL_SPEC.md
API_SPEC.md
UI_SPEC.md
DEVELOPMENT_PLAN.md
TESTING_SPEC.md
```

If specifications conflict, do not silently invent a resolution. Identify the conflict and follow the more specific specification where clearly applicable.

## 3. Inspect Before Editing

Always:

```text
Inspect repository
 ↓
Understand structure
 ↓
Locate relevant files
 ↓
Read existing implementation
 ↓
Read relevant specification
 ↓
Plan
 ↓
Implement
 ↓
Test
```

Never immediately rewrite a module.

## 4. Existing Code Has Priority

If functionality already exists, modify it rather than creating duplicate implementations.

Avoid multiple overlapping services such as:

```text
DatasetService
DatasetManager
DatasetHandler
DatasetProcessor
```

unless their responsibilities are genuinely different.

## 5. Minimal Change Principle

Change only what is required.

Do not unnecessarily:

- Rename unrelated files
- Reformat the project
- Replace frameworks
- Rewrite working modules
- Change API contracts
- Change architecture

## 6. No Unrequested Features

Do not implement future features simply because they appear useful.

Unless explicitly requested, do not add:

- AI assistant
- AutoML
- LLM integration
- Natural-language ML
- Automatic model recommendation
- Hyperparameter optimization
- Billing
- Collaboration
- Advanced deployment

## 7. Development Order

```text
ML Core
 ↓
Backend API
 ↓
Workspace UI
 ↓
Integration
 ↓
Testing
 ↓
Landing Page
 ↓
3D
 ↓
Advanced Features
```

## 8. Backend Architecture

Keep:

```text
API
 ↓
Service
 ↓
ML Engine
 ↓
Utilities
```

API routes must remain thin.

Business logic belongs in services.

## 9. ML Independence

The ML engine must not depend on frontend components.

Correct:

```text
Frontend
 ↓
API
 ↓
Services
 ↓
ML Engine
```

Never put sklearn/model-training logic directly into frontend code.

## 10. Model Registry

Maintain one centralized model registry.

It should define:

```text
model_id
name
problem_type
description
parameters
factory
```

Do not hard-code model lists independently in frontend, API, and model factory.

## 11. Preprocessing Registry

Maintain centralized preprocessing metadata.

The frontend should consume backend metadata rather than maintaining a conflicting hard-coded list.

## 12. Model Parameters

Expose important parameters by default.

Backend validation is authoritative.

Both frontend and backend may validate, but the backend must enforce the rules.

## 13. Data Leakage Prevention

Never fit preprocessing before the train/test split.

Correct:

```text
Dataset
 ↓
Train/Test Split
 ├── Train → fit/transform
 └── Test  → transform
```

## 14. SMOTE Rule

SMOTE operates only on training data.

Never:

```text
Dataset
 ↓
SMOTE
 ↓
Split
```

## 15. Target Separation

Explicitly separate target from features.

```python
X = data.drop(columns=[target])
y = data[target]
```

## 16. Prediction Pipeline

Training:

```text
Raw Train Data
 ↓
Fit Preprocessor
 ↓
Transform
 ↓
Train Model
```

Prediction:

```text
New Raw Data
 ↓
Saved Preprocessor
 ↓
Transform
 ↓
Saved Model
 ↓
Prediction
```

Never refit preprocessing during prediction.

## 17. Artifact Rule

Training should produce traceable artifacts such as:

```text
model
preprocessing pipeline
metadata
evaluation results
prediction output
```

## 18. Reproducibility

Use controlled random states where supported.

Do not add unsupported parameters blindly.

## 19. Error Handling

Errors must be:

- Predictable
- Structured
- User-understandable
- Safe

Never expose internal stack traces, filesystem paths, secrets, or environment variables to users.

Avoid:

```python
except Exception:
    pass
```

Catch specific exceptions where practical.

## 20. Logging

Log important backend operations such as:

```text
Dataset uploaded
Dataset validated
Training started
Training completed
Training failed
Prediction started
Prediction completed
```

Never log secrets or sensitive information.

## 21. File Uploads

Validate uploaded files server-side.

Do not trust filename, extension, or MIME type alone.

Respect configured upload limits and safely manage temporary files.

## 22. Session State

Use explicit workflow/session state.

Avoid global mutable variables such as:

```python
CURRENT_DATASET
CURRENT_MODEL
CURRENT_PREPROCESSOR
```

## 23. Frontend State

Represent workflow state clearly:

```text
dataset
analysis
target
problemType
preprocessing
model
parameters
training
evaluation
prediction
```

Avoid unnecessary duplicated state.

## 24. API Contract

The frontend consumes API contracts.

Do not reproduce backend model/preprocessing logic in the frontend.

Once an API contract is established, do not casually change it.

If a breaking change is necessary, update:

```text
API_SPEC.md
Backend
Frontend
Tests
```

## 25. UI States

Every asynchronous operation should define:

```text
Loading
Success
Error
```

Major empty states should also be handled gracefully.

## 26. Component Reuse

Before creating a component:

```text
Search existing components
 ↓
Reuse
 ↓
Extend
 ↓
Create new only if necessary
```

## 27. 3D Isolation

3D must remain isolated from ML functionality.

A 3D failure must not break:

- Dataset upload
- Training
- Evaluation
- Prediction

## 28. Performance

Do not optimize prematurely.

First:

```text
Correctness
 ↓
Performance
```

Measure before making architectural performance changes.

## 29. Dependencies

Before adding a dependency, check:

- Is it necessary?
- Does an existing package already solve the problem?
- Is it maintained?
- Does it increase complexity?

Do not add packages without reason.

## 30. Version Compatibility

Respect project dependency versions.

Do not upgrade major dependencies during unrelated feature work.

If an upgrade is necessary:

```text
Identify reason
 ↓
Check compatibility
 ↓
Update
 ↓
Run tests
```

## 31. Notebook Rule

Notebooks are for:

- Experimentation
- Exploration
- Learning
- Prototyping

Production logic should live in reusable Python modules.

## 32. Code Quality

Prefer code that is:

- Readable
- Modular
- Testable
- Maintainable

Avoid unnecessary abstraction.

## 33. Naming

Use descriptive names:

```text
DatasetService
TrainingService
ModelFactory
EvaluationService
ArtifactManager
```

Avoid vague names such as:

```text
Manager2
Helper
Utils2
NewService
Temp
```

## 34. Comments

Comments should explain why something is done when that reason is not obvious.

Do not add comments that merely repeat the code.

## 35. Configuration

Centralize:

- Upload limits
- Artifact paths
- Environment settings
- API configuration
- Allowed file types

Do not scatter configuration values throughout the codebase.

## 36. Environment Variables

Never hard-code secrets.

Use `.env` locally where appropriate and commit `.env.example`.

Never commit secrets.

## 37. Git

Before committing:

```text
Run tests
 ↓
Review changed files
 ↓
Review diff
 ↓
Remove debugging code
```

Meaningful commit examples:

```text
feat: add dataset validation service
feat: implement model registry
feat: add training endpoint
fix: prevent preprocessing leakage
test: add prediction pipeline tests
refactor: separate evaluation service
```

## 38. No Fake Implementations

Do not fabricate ML results.

Do not return:

```python
{"accuracy": 0.95}
```

as if real training occurred unless the code is explicitly a mock.

## 39. Mock Data

Mocks are allowed only for:

- Frontend development
- UI prototyping
- API contract testing

Clearly separate mock and production behavior.

## 40. No Silent Fallbacks

Do not silently replace invalid configuration with another model or preprocessing method.

Example:

```text
Invalid model
 ↓
Automatically use Random Forest
```

is not acceptable unless explicitly specified.

## 41. No Silent Data Modification

Do not silently:

- Drop columns
- Drop rows
- Change target
- Remove outliers
- Encode categories
- Scale features

unless explicitly configured.

## 42. Traceability

User selections should remain traceable.

Example:

```text
Scaler = StandardScaler
Model = RandomForest
n_estimators = 200
```

should be represented in the training configuration/artifacts.

## 43. Testing Critical ML Logic

Prioritize tests for:

- Preprocessing
- Leakage prevention
- Parameter validation
- Prediction pipeline
- Artifact loading

## 44. Debugging Procedure

```text
Reproduce
 ↓
Identify layer
 ↓
Inspect error
 ↓
Find root cause
 ↓
Smallest correct fix
 ↓
Regression test
 ↓
Run affected tests
 ↓
Run broader tests
```

Do not immediately rewrite the entire module.

## 45. Requirements Ambiguity

For architecture-changing ambiguity, ask for clarification.

For small implementation details, follow established project conventions.

## 46. Before Starting a Task

Verify:

```text
[ ] What feature is requested?
[ ] Which specification describes it?
[ ] Does it already exist?
[ ] Which files are involved?
[ ] What dependencies are required?
[ ] What tests are required?
[ ] Does this change the API?
[ ] Does this change architecture?
```

## 47. Before Finishing

Verify:

```text
[ ] Implementation complete
[ ] No duplicate logic
[ ] Tests added/updated
[ ] Tests pass
[ ] Errors handled
[ ] Documentation updated if needed
[ ] No secrets
[ ] No debugging code
[ ] Diff reviewed
```

## 48. Completion Report

Use:

```text
### Implemented
- Feature 1
- Feature 2

### Files Changed
- file/path
- file/path

### Tests
- Test 1
- Test 2

### Result
PASS / PARTIAL / BLOCKED

### Notes
- Limitations
- Required follow-up
```

Do not claim success if tests were not run.

## 49. If Blocked

Report:

```text
BLOCKED

Reason:
...

Required information:
...

Files affected:
...
```

Do not invent missing requirements.

## 50. Core ML Safety Checklist

```text
[ ] Correct target
[ ] Correct feature set
[ ] Correct train/test split
[ ] No preprocessing leakage
[ ] Correct preprocessing order
[ ] Correct model
[ ] Valid parameters
[ ] Correct evaluation set
[ ] Correct metrics
[ ] Saved preprocessing
[ ] Saved model
[ ] Prediction pipeline works
[ ] Feature representation is consistent
[ ] Artifacts can be reloaded
```

## 51. V1 Scope Protection

Unless explicitly requested, keep these outside V1:

```text
AI assistant
LLM
RAG
Agents
AutoML
Automatic model selection
Hyperparameter search
Advanced deployment
Multi-user collaboration
Billing
Enterprise authentication
```

## 52. Final Rule

```text
DO NOT GUESS
DO NOT DUPLICATE
DO NOT OVERENGINEER
DO NOT SKIP TESTS
DO NOT BREAK EXISTING CONTRACTS
DO NOT IMPLEMENT UNREQUESTED FEATURES
```

Instead:

```text
READ
 ↓
UNDERSTAND
 ↓
PLAN
 ↓
IMPLEMENT
 ↓
TEST
 ↓
VERIFY
 ↓
DOCUMENT
```

The objective is not to generate the most code. The objective is to build the correct No-Code ML platform incrementally and reliably.
