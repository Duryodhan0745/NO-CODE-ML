from __future__ import annotations

from typing import Any
import ast
import math
import numpy as np
import pandas as pd


ALLOWED_FILTERS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "not_contains", "is_null", "not_null", "in", "not_in"}

def calculated(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    expression = str(params.get("expression", "")); target = params.get("target_column")
    if not target or len(expression) > 500: raise ValueError("A bounded target column and expression are required.")
    tree = ast.parse(expression, mode="eval")
    functions = {"ABS": abs, "ROUND": round, "MIN": min, "MAX": max, "SQRT": math.sqrt, "LOG": math.log, "LOWER": lambda x: str(x).lower(), "UPPER": lambda x: str(x).upper(), "TRIM": lambda x: str(x).strip(), "LENGTH": lambda x: len(str(x)), "COALESCE": lambda x, y: y if pd.isna(x) else x}
    def visit(node, row):
        if isinstance(node, ast.Expression): return visit(node.body, row)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)): return node.value
        if isinstance(node, ast.Name) and node.id in row: return row[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)): return {ast.USub: lambda x: -x, ast.UAdd: lambda x: +x, ast.Not: lambda x: not x}[type(node.op)](visit(node.operand, row))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)): return {ast.Add: lambda a,b:a+b, ast.Sub: lambda a,b:a-b, ast.Mult: lambda a,b:a*b, ast.Div: lambda a,b:a/b, ast.Mod: lambda a,b:a%b}[type(node.op)](visit(node.left,row), visit(node.right,row))
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [bool(visit(v,row)) for v in node.values]; return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a,b=visit(node.left,row),visit(node.comparators[0],row); op=node.ops[0]; return a == b if isinstance(op, ast.Eq) else a != b if isinstance(op, ast.NotEq) else a > b if isinstance(op, ast.Gt) else a < b if isinstance(op, ast.Lt) else a >= b if isinstance(op, ast.GtE) else a <= b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords: return functions[node.func.id](*(visit(a,row) for a in node.args))
        raise ValueError("Expression contains an unsupported construct.")
    if any(isinstance(n, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Import)) for n in ast.walk(tree)): raise ValueError("Attribute, subscript, lambda, and import expressions are forbidden.")
    try: return frame.apply(lambda row: visit(tree, row), axis=1)
    except Exception as exc: raise ValueError(f"Calculated expression failed: {exc}")


def apply(frame: pd.DataFrame, operation: str, params: dict[str, Any], preview: bool = False) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    out = frame.copy(deep=True); warnings: list[str] = []; affected_rows = 0; affected_columns: list[str] = []
    cols = params.get("columns") or ([params["column"]] if params.get("column") else [])
    if any(c not in out.columns for c in cols): raise ValueError("Unknown transformation column.")
    if operation in {"fill", "drop_missing"}:
        if operation == "drop_missing":
            before = len(out); out = out.dropna(subset=cols or None); affected_rows = before - len(out); affected_columns = cols or list(frame.columns)
        else:
            strategy = params.get("strategy", "mean"); affected_columns = cols or list(out.columns)
            for col in affected_columns:
                s = out[col]
                if not s.isna().any(): continue
                if strategy == "constant": value = params.get("value")
                elif strategy == "mean" and pd.api.types.is_numeric_dtype(s): value = s.mean()
                elif strategy == "median" and pd.api.types.is_numeric_dtype(s): value = s.median()
                elif strategy == "mode": value = s.mode().iloc[0] if not s.mode().empty else None
                else: raise ValueError(f"Strategy {strategy} is incompatible with column {col}.")
                affected_rows += int(s.isna().sum()); out[col] = s.fillna(value)
    elif operation == "drop_duplicates":
        subset = cols or None; before = len(out); out = out.drop_duplicates(subset=subset); affected_rows = before - len(out); affected_columns = subset or list(out.columns)
    elif operation == "rename":
        mapping = params.get("mapping", {}); 
        if any(c not in out.columns for c in mapping): raise ValueError("Unknown rename column.")
        if len(set(mapping.values())) != len(mapping.values()) or any(v in out.columns and v not in mapping for v in mapping.values()): raise ValueError("Renamed columns must be unique.")
        out = out.rename(columns=mapping); affected_columns = list(mapping)
    elif operation in {"remove", "select", "reorder"}:
        selected = params.get("columns", [])
        if any(c not in out.columns for c in selected): raise ValueError("Unknown transformation column.")
        if operation == "remove": out = out.drop(columns=selected)
        else: out = out[selected]
        affected_columns = selected
    elif operation == "convert":
        target = params.get("dtype", "string"); affected_columns = cols
        for col in cols:
            try:
                if target in {"numeric", "float", "integer"}: out[col] = pd.to_numeric(out[col], errors="raise").astype(float if target == "float" else int if target == "integer" else float)
                elif target == "datetime": out[col] = pd.to_datetime(out[col], errors="raise")
                elif target == "boolean": out[col] = out[col].astype(bool)
                elif target == "category": out[col] = out[col].astype("category")
                else: out[col] = out[col].astype(str)
            except Exception as exc: raise ValueError(f"Could not convert {col}: {exc}")
    elif operation in {"standardize", "normalize", "log"}:
        affected_columns = cols
        for col in cols:
            if not pd.api.types.is_numeric_dtype(out[col]): raise ValueError(f"Numeric column required: {col}")
            if operation == "standardize":
                std = out[col].std(); out[col] = (out[col] - out[col].mean()) / std if std else 0
            elif operation == "normalize":
                lo, hi = out[col].min(), out[col].max(); out[col] = (out[col] - lo) / (hi - lo) if hi != lo else 0
            else:
                if (out[col] <= 0).any(): raise ValueError(f"Log transform requires positive values: {col}")
                out[col] = np.log(out[col])
    elif operation == "filter":
        col, op, value = params.get("column"), params.get("operator"), params.get("value")
        if col not in out.columns or op not in ALLOWED_FILTERS: raise ValueError("Invalid filter transformation.")
        s = out[col]
        if op == "eq": mask = s == value
        elif op == "ne": mask = s != value
        elif op in {"gt", "gte", "lt", "lte"}: mask = {"gt": s.gt, "gte": s.ge, "lt": s.lt, "lte": s.le}[op](value)
        elif op == "contains": mask = s.astype(str).str.contains(str(value), regex=False, na=False)
        elif op == "not_contains": mask = ~s.astype(str).str.contains(str(value), regex=False, na=False)
        elif op == "is_null": mask = s.isna()
        elif op == "not_null": mask = s.notna()
        elif op == "in": mask = s.isin(value if isinstance(value, list) else [value])
        else: mask = ~s.isin(value if isinstance(value, list) else [value])
        affected_rows = int((~mask).sum()); affected_columns = [col]; out = out[mask.fillna(False)]
    elif operation == "one_hot":
        if not cols: raise ValueError("Select at least one column.")
        out = pd.get_dummies(out, columns=cols, dummy_na=True, dtype=int); affected_columns = cols
    elif operation == "label_encode":
        for col in cols:
            out[col] = pd.Categorical(out[col]).codes; affected_columns.append(col)
    elif operation == "calculated":
        target = params.get("target_column")
        if target in out.columns: raise ValueError("Calculated target column already exists.")
        out[target] = calculated(out, params); affected_columns = [target]; affected_rows = len(out)
    else: raise ValueError("Unsupported transformation operation.")
    return out, {"affected_rows": affected_rows, "affected_columns": affected_columns, "before_shape": [len(frame), len(frame.columns)], "after_shape": [len(out), len(out.columns)], "sample_before": frame.head(5).to_dict(orient="records"), "sample_after": out.head(5).to_dict(orient="records")}, warnings
