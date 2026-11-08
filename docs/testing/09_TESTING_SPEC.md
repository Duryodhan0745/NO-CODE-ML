# No-Code ML Platform — Testing Specification

**Document:** Testing Specification  
**Version:** 1.0  
**Status:** V1 Development Specification

## 1. Purpose

Testing must verify:

- Software correctness
- ML pipeline correctness
- API correctness
- Frontend behavior
- Workflow state management
- Data validation
- Preprocessing correctness
- Model configuration
- Training correctness
- Evaluation correctness
- Prediction consistency
- Artifact generation
- Error handling

The goal is not simply that the application runs. The goal is that it produces a technically valid ML workflow.

## 2. Testing Layers

```text
Testing
 ├── Unit
 ├── ML Component
 ├── API
 ├── Frontend
 ├── Integration
 ├── End-to-End
 └── Regression
```

## 3. Test Directory

```text
tests/
├── unit/
├── api/
├── integration/
├── fixtures/
└── conftest.py
```

Recommended fixtures:

```text
classification.csv
regression.csv
missing_values.csv
categorical.csv
imbalanced.csv
```

## 4. Dataset Tests

Test:

- Valid CSV
- Empty file
- Malformed CSV
- Unsupported file
- Missing rows/columns
- Row and column counts
- Column names
- Data types
- Missing values
- Duplicate rows

## 5. Target and Problem-Type Tests

Valid targets must be accepted.

Invalid targets must be rejected.

Supported V1 problem types:

```text
classification
regression
```

Invalid/unsupported problem types must produce controlled errors.

## 6. Preprocessing Tests

Test independently:

- Missing-value imputation
- Encoding
- Scaling
- Outlier handling
- Feature selection
- PCA
- SMOTE

### Critical leakage test

Never:

```text
Dataset
 ↓
Fit preprocessing
 ↓
Train/Test Split
```

Correct:

```text
Dataset
 ↓
Train/Test Split
 ├── Train → fit/transform
 └── Test  → transform
```

### Critical SMOTE test

Never apply SMOTE before splitting.

Correct:

```text
Dataset
 ↓
Split
 ├── Train → SMOTE
 └── Test  → unchanged
```

## 7. Model Tests

Every registered model must have:

- Model ID
- Name
- Problem type
- Parameter metadata
- Factory mapping

Test:

- Valid parameters
- Invalid parameters
- Invalid parameter types
- Default parameters
- Model factory output

## 8. Training Tests

For every supported model:

```text
Dataset
 ↓
Preprocessing
 ↓
Model
 ↓
fit()
```

must complete successfully using the appropriate fixture.

Also test invalid target, invalid parameters, empty training data, and incompatible features.

## 9. Evaluation Tests

### Classification

Test:

- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix

### Regression

Test:

- MAE
- MSE
- RMSE
- R²

Use small manually verifiable examples for metric correctness.

## 10. Prediction Tests

Prediction must use:

```text
Saved preprocessing pipeline
+
Saved trained model
```

Never refit preprocessing on prediction data.

Test:

- Valid prediction data
- Missing feature
- Extra feature
- Different column order
- Unknown category
- Missing model
- Missing preprocessing artifact

## 11. Artifact Tests

Verify generation and reload of:

- Model
- Preprocessing pipeline
- Metadata
- Evaluation results
- Prediction CSV

Also test missing/corrupted artifacts.

## 12. Session and Workflow Tests

Verify valid workflow:

```text
Created
 ↓
Dataset uploaded
 ↓
Configured
 ↓
Trained
 ↓
Evaluated
 ↓
Prediction available
```

Reject invalid transitions, such as training before a dataset exists.

If preprocessing/model configuration changes after training, stale artifacts must not silently be reused.

## 13. API Tests

Every endpoint should test:

- Valid request
- Invalid request
- Missing fields
- Invalid session
- Invalid configuration
- Expected response
- Error response

Errors should follow the API contract.

## 14. Frontend Tests

Test:

- Navigation
- Upload UI
- Forms
- Loading states
- Error states
- Model selection
- Parameter controls
- Evaluation display
- Prediction display
- Reusable components

## 15. Integration Tests

Classification:

```text
Load dataset
 ↓
Select target
 ↓
Split
 ↓
Preprocess
 ↓
Train Random Forest
 ↓
Evaluate
 ↓
Predict
```

Regression should have an equivalent complete test.

## 16. End-to-End Test

Simulate:

```text
Open application
 ↓
Create workspace
 ↓
Upload CSV
 ↓
View analysis
 ↓
Select target
 ↓
Select problem type
 ↓
Configure preprocessing
 ↓
Select model
 ↓
Configure parameters
 ↓
Train
 ↓
View evaluation
 ↓
Upload prediction dataset
 ↓
Generate prediction
 ↓
Download CSV
```

## 17. Performance Tests

After correctness is established, measure:

- Upload time
- Analysis time
- Training time
- Prediction time
- API response time
- Memory usage
- Frontend rendering

Large datasets should use appropriate pagination/sampling/virtualized rendering where needed.

## 18. Security Tests

Test:

- Invalid uploads
- Malformed requests
- Path traversal attempts
- Unauthorized access where applicable
- Secret leakage
- Internal stack-trace exposure

## 19. Regression Testing

Every fixed bug should ideally produce a regression test.

## 20. Test Naming

Prefer:

```python
test_random_forest_rejects_negative_n_estimators()
test_preprocessor_fits_only_on_training_data()
test_prediction_uses_saved_preprocessing_pipeline()
```

Avoid vague names such as `test1()`.

## 21. Minimum V1 Requirement

```text
✓ Dataset tests
✓ Preprocessing tests
✓ Model tests
✓ Training tests
✓ Evaluation tests
✓ Prediction tests
✓ API tests
✓ Frontend tests
✓ Integration tests
✓ E2E tests
✓ Leakage tests
✓ Error handling tests
```

## 22. Final ML Validation Checklist

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

The platform should not be considered correct merely because no exception occurred. It must demonstrate that the ML workflow itself is valid.
