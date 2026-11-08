"""End-to-end classification + regression workflow test via HTTP."""
import urllib.request, json

BASE = "http://localhost:8000/api"

def post_json(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def post_file(path, filename, content):
    boundary = "---boundary123"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req).read())

def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


print("=" * 60)
print("CLASSIFICATION WORKFLOW")
print("=" * 60)

csv_clf = (
    b"age,income,edu,target\n"
    b"25,30000,12,0\n30,45000,16,1\n35,50000,14,0\n40,60000,18,1\n"
    b"45,70000,16,1\n22,25000,12,0\n28,35000,14,0\n50,80000,20,1\n"
    b"33,48000,16,1\n27,32000,12,0\n38,55000,14,1\n42,65000,18,1\n"
    b"55,90000,20,1\n23,28000,12,0\n31,42000,16,0\n36,52000,14,1\n"
    b"29,38000,14,0\n48,75000,18,1\n26,31000,12,0\n44,68000,16,1\n"
)

ds = post_file("/datasets/upload", "test_clf.csv", csv_clf)
print(f"1. Upload: {ds['rows']} rows, {ds['num_columns']} cols")

prev = get(f"/datasets/{ds['id']}/preview?rows=5")
print(f"2. Preview: {len(prev['data'])} rows shown")

an = get(f"/datasets/{ds['id']}/analysis")
print(f"3. Analysis: {len(an['numerical_columns'])} numerical, {len(an['categorical_columns'])} categorical")

models = get("/models?problem_type=classification")
print(f"4. Models: {[m['name'] for m in models['models']]}")

ws = post_json("/workspaces", {"dataset_id": ds["id"]})
print(f"5. Workspace: {ws['id']}")

cfg = post_json(f"/workspaces/{ws['id']}/configure", {
    "target": "target", "problem_type": "classification",
    "preprocessing": {"missing_values": "mean", "scaling": "standard"},
    "model_id": "random_forest_classifier", "model_params": {"n_estimators": 50},
})
print(f"6. Configured: {cfg['status']}")

tr = post_json(f"/workspaces/{ws['id']}/train", {})
print(f"7. Trained: {tr['status']}, accuracy={tr['metrics']['accuracy']:.4f}")

ev = get(f"/workspaces/{ws['id']}/evaluation")
m = ev["metrics"]
print(f"8. Evaluation: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")
print(f"   Confusion matrix: {m['confusion_matrix']}")

pred_csv = b"age,income,edu\n30,45000,16\n40,60000,18\n55,90000,20\n"
pr = post_file(f"/workspaces/{ws['id']}/predict", "pred.csv", pred_csv)
print(f"9. Predictions: {pr['predictions']}")

arts = get(f"/workspaces/{ws['id']}/artifacts")
print(f"10. Artifacts: {[a['filename'] for a in arts['artifacts']]}")

print("\n=== CLASSIFICATION E2E: PASS ===\n")

print("=" * 60)
print("REGRESSION WORKFLOW")
print("=" * 60)

csv_reg = (
    b"size,bedrooms,age,location,price\n"
    b"1500,3,10,urban,300000\n2000,4,5,suburban,450000\n1200,2,20,urban,250000\n"
    b"1800,3,8,rural,280000\n2500,5,3,suburban,550000\n1000,1,15,urban,200000\n"
    b"1600,3,12,suburban,320000\n2200,4,7,rural,400000\n1400,2,18,urban,270000\n"
    b"1900,3,6,suburban,380000\n2800,5,2,urban,600000\n1100,1,25,rural,180000\n"
    b"1700,3,9,suburban,350000\n2100,4,4,urban,480000\n1300,2,22,rural,230000\n"
    b"2400,5,1,suburban,520000\n1550,3,14,urban,310000\n2050,4,6,rural,420000\n"
    b"1150,2,19,urban,240000\n1850,3,11,suburban,370000\n"
)

ds2 = post_file("/datasets/upload", "test_reg.csv", csv_reg)
print(f"1. Upload: {ds2['rows']} rows, {ds2['num_columns']} cols")

an2 = get(f"/datasets/{ds2['id']}/analysis")
print(f"2. Analysis: {len(an2['numerical_columns'])} numerical, {len(an2['categorical_columns'])} categorical")

ws2 = post_json("/workspaces", {"dataset_id": ds2["id"]})
print(f"3. Workspace: {ws2['id']}")

cfg2 = post_json(f"/workspaces/{ws2['id']}/configure", {
    "target": "price", "problem_type": "regression",
    "preprocessing": {"missing_values": "mean", "scaling": "standard"},
    "model_id": "random_forest_regressor", "model_params": {"n_estimators": 50},
})
print(f"4. Configured: {cfg2['status']}")

tr2 = post_json(f"/workspaces/{ws2['id']}/train", {})
print(f"5. Trained: {tr2['status']}, r2={tr2['metrics']['r2']:.4f}")

ev2 = get(f"/workspaces/{ws2['id']}/evaluation")
m2 = ev2["metrics"]
print(f"6. Evaluation: MAE={m2['mae']:.2f} RMSE={m2['rmse']:.2f} R2={m2['r2']:.4f}")

pred_reg = b"size,bedrooms,age,location\n1500,3,10,urban\n2500,5,3,suburban\n"
pr2 = post_file(f"/workspaces/{ws2['id']}/predict", "pred_reg.csv", pred_reg)
print(f"7. Predictions: {[round(p, 2) for p in pr2['predictions']]}")

arts2 = get(f"/workspaces/{ws2['id']}/artifacts")
print(f"8. Artifacts: {[a['filename'] for a in arts2['artifacts']]}")

print("\n=== REGRESSION E2E: PASS ===\n")

print("=" * 60)
print("OPTUNA AUTOML WORKFLOW")
print("=" * 60)

ws_opt = post_json("/workspaces", {"dataset_id": ds["id"]})
print(f"1. Workspace for Optuna: {ws_opt['id']}")

post_json(f"/workspaces/{ws_opt['id']}/configure", {
    "target": "target", "problem_type": "classification",
    "preprocessing": {"missing_values": "mean", "scaling": "standard"},
    "model_id": "logistic_regression", "model_params": {},
})

opt_res = post_json(f"/workspaces/{ws_opt['id']}/optimize", {
    "candidate_models": ["logistic_regression", "decision_tree_classifier", "xgboost_classifier"],
    "metric": "accuracy",
    "n_trials": 4,
    "cv_folds": 2,
})
best = opt_res["optimization"]
print(f"2. Optuna Winner: {best['best_model_id']}, CV Score={best['best_cv_score']}, Trials={best['n_trials']}")
print(f"   Final Test Accuracy: {opt_res['metrics']['accuracy']:.4f}")

opt_details = get(f"/workspaces/{ws_opt['id']}/optimization")
assert opt_details["optimization"] is not None
print(f"3. Optuna Details Verified: {len(opt_details['optimization']['trials'])} trials recorded")

print("\n=== OPTUNA E2E: PASS ===")
print("\n" + "=" * 60)
print("ALL E2E TESTS (CLASSIFICATION, REGRESSION, OPTUNA) PASSED")
print("=" * 60)
