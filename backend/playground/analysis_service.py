from __future__ import annotations

import math
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def clean(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, float): return None if not math.isfinite(value) else value
    if pd.isna(value): return None
    return value.item() if hasattr(value, "item") else value


def envelope(dataset_id: str, version: int, operation: str, columns: list[str], parameters: dict[str, Any], fn):
    started = time.perf_counter(); warnings: list[str] = []
    try: result = fn(warnings); status = "completed"
    except ValueError as exc: raise
    duration = round(time.perf_counter() - started, 6)
    return clean({"operation_id": str(uuid.uuid4()), "operation_type": operation, "dataset_id": dataset_id, "dataset_version": version, "selected_columns": columns, "parameters": parameters, "result": result, "warnings": warnings, "status": status, "execution_duration": duration})


def run(frame: pd.DataFrame, dataset_id: str, version: int, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
    columns = parameters.get("columns") or ([] if not parameters.get("column") else [parameters["column"]])
    if any(c not in frame.columns for c in columns): raise ValueError("Unknown analysis column.")
    def calc(warnings):
        if operation == "descriptive":
            output = {}
            for col in columns:
                s = frame[col]; numeric = pd.api.types.is_numeric_dtype(s); nonnull = s.dropna(); item = {"count": int(s.count()), "missing_count": int(s.isna().sum()), "missing_percent": float(s.isna().mean() * 100), "unique_count": int(s.nunique(dropna=True))}
                if numeric and len(nonnull):
                    q = nonnull.quantile([.25, .5, .75]); item.update({"mean": nonnull.mean(), "median": nonnull.median(), "mode": nonnull.mode().tolist(), "min": nonnull.min(), "max": nonnull.max(), "range": nonnull.max() - nonnull.min(), "sum": nonnull.sum(), "variance": nonnull.var(), "std": nonnull.std(), "quantiles": {str(k): v for k, v in q.items()}, "iqr": q.iloc[2] - q.iloc[0], "skewness": nonnull.skew(), "kurtosis": nonnull.kurtosis()})
                elif not numeric:
                    item.update({"mode": nonnull.mode().astype(str).tolist(), "frequencies": {str(k): int(v) for k, v in nonnull.astype(str).value_counts().head(100).items()}})
                output[col] = item
            return output
        if operation in {"correlation", "covariance"}:
            numeric = frame[columns].select_dtypes(include="number")
            if len(numeric.columns) < 2: raise ValueError("At least two numeric columns are required.")
            method = parameters.get("method", "pearson") if operation == "correlation" else None
            return {"method": method or "covariance", "matrix": numeric.corr(method=method).to_dict() if method else numeric.cov().to_dict()}
        if operation == "distribution":
            col = columns[0]; s = frame[col].dropna(); result = {"column": col, "count": len(s), "frequencies": {str(k): int(v) for k, v in s.astype(str).value_counts().head(100).items()}}
            if pd.api.types.is_numeric_dtype(s):
                result["summary"] = {"mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max()}
                if len(s) >= 8: result["normality"] = {"statistic": scipy_stats.normaltest(s).statistic, "p_value": scipy_stats.normaltest(s).pvalue}
                else: warnings.append("Normality test requires at least 8 observations.")
            return result
        if operation == "outliers":
            col = columns[0]; s = pd.to_numeric(frame[col], errors="coerce"); method = parameters.get("method", "iqr"); valid = s.dropna()
            if not len(valid): raise ValueError("No numeric observations available.")
            if method == "iqr":
                q1, q3 = valid.quantile([.25, .75]); spread = q3 - q1; low, high = q1 - 1.5 * spread, q3 + 1.5 * spread; mask = (s < low) | (s > high); thresholds = {"low": low, "high": high}
            elif method == "zscore":
                threshold = float(parameters.get("threshold", 3)); mean, std = valid.mean(), valid.std(); low, high = mean - threshold * std, mean + threshold * std; mask = (s < low) | (s > high); thresholds = {"low": low, "high": high, "threshold": threshold}
            elif method == "modified_zscore":
                median = valid.median(); mad = (valid - median).abs().median(); score = .6745 * (s - median) / mad if mad else pd.Series(0, index=s.index); threshold = float(parameters.get("threshold", 3.5)); mask = score.abs() > threshold; thresholds = {"median": median, "mad": mad, "threshold": threshold}
            else: raise ValueError("Unsupported outlier method.")
            indices = frame.index[mask.fillna(False)].tolist()[:1000]
            if int(mask.sum()) > 1000: warnings.append("Outlier row references were capped at 1000.")
            return {"column": col, "method": method, "outlier_count": int(mask.sum()), "outlier_percent": float(mask.mean() * 100), "row_indices": indices, "thresholds": thresholds}
        if operation == "groupby":
            groups = parameters.get("group_columns") or columns[:1]; value = parameters.get("value_column"); aggregations = parameters.get("aggregations") or ["count"]
            if any(c not in frame.columns for c in groups + ([value] if value else [])): raise ValueError("Unknown grouping column.")
            grouped = frame.groupby(groups, dropna=False)
            result = {}
            for agg in aggregations:
                if agg == "count": result[agg] = grouped.size()
                elif not value or value not in frame.select_dtypes(include="number").columns: raise ValueError("Numeric value_column required for this aggregation.")
                else: result[agg] = getattr(grouped[value], agg)()
            return {"groups": pd.DataFrame(result).reset_index().head(1000).to_dict(orient="records"), "limited": len(pd.DataFrame(result)) > 1000}
        if operation in {"independent_t_test", "paired_t_test", "chi_square", "anova", "normality"}:
            if operation == "normality":
                s = pd.to_numeric(frame[columns[0]], errors="coerce").dropna()
                if len(s) < 3: warnings.append("Normality tests require at least 3 observations."); return {"test": "shapiro", "sample_size": len(s), "statistic": None, "p_value": None}
                test = scipy_stats.shapiro(s); return {"test": "shapiro", "sample_size": len(s), "statistic": test.statistic, "p_value": test.pvalue, "interpretation": "Evidence against normality" if test.pvalue < .05 else "No strong evidence against normality"}
            groups = parameters.get("groups") or columns
            if operation == "independent_t_test":
                if len(groups) != 2: raise ValueError("Two numeric groups are required.")
                a = pd.to_numeric(frame[groups[0]], errors="coerce").dropna(); b = pd.to_numeric(frame[groups[1]], errors="coerce").dropna();
                if len(a) < 2 or len(b) < 2: raise ValueError("Each sample requires at least two observations.")
                test = scipy_stats.ttest_ind(a, b, equal_var=False); return {"test": "independent_t_test", "statistic": test.statistic, "p_value": test.pvalue, "sample_sizes": [len(a), len(b)], "degrees_of_freedom": len(a) + len(b) - 2, "warnings": warnings}
            if operation == "paired_t_test":
                if len(groups) != 2: raise ValueError("Two paired numeric columns are required.")
                pair = frame[groups].apply(pd.to_numeric, errors="coerce").dropna();
                if len(pair) < 2: raise ValueError("At least two complete pairs are required.")
                test = scipy_stats.ttest_rel(pair.iloc[:, 0], pair.iloc[:, 1]); return {"test": "paired_t_test", "statistic": test.statistic, "p_value": test.pvalue, "sample_size": len(pair), "degrees_of_freedom": len(pair) - 1}
            if operation == "chi_square":
                if len(groups) != 2: raise ValueError("Two categorical columns are required.")
                table = pd.crosstab(frame[groups[0]], frame[groups[1]]); test = scipy_stats.chi2_contingency(table); return {"test": "chi_square", "statistic": test[0], "p_value": test[1], "degrees_of_freedom": test[2], "sample_size": int(table.values.sum())}
            samples = [pd.to_numeric(frame[g], errors="coerce").dropna() for g in groups]
            if len(samples) < 2 or any(len(s) < 2 for s in samples): raise ValueError("ANOVA requires at least two groups with two observations each.")
            test = scipy_stats.f_oneway(*samples); return {"test": "one_way_anova", "statistic": test.statistic, "p_value": test.pvalue, "sample_sizes": [len(s) for s in samples]}
        raise ValueError("Unsupported analysis operation.")
    return envelope(dataset_id, version, operation, columns, parameters, calc)
