"""Validated Python visualization registry and artifact generation."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CHARTS = [
    ("histogram", "Histogram", "Distribution", "Frequency distribution of a numeric column."),
    ("kde", "KDE / Density", "Distribution", "Smoothed probability density."),
    ("boxplot", "Box Plot", "Distribution", "Quartiles and outliers by group."),
    ("violin", "Violin Plot", "Distribution", "Distribution shape by group."),
    ("ecdf", "ECDF Plot", "Distribution", "Cumulative distribution of numeric values."),
    ("strip", "Strip Plot", "Distribution", "Individual observations by category."),
    ("swarm", "Swarm Plot", "Distribution", "Non-overlapping observations by category."),
    ("bar", "Bar Chart", "Comparison", "Aggregate values by category."),
    ("grouped_bar", "Grouped Bar Chart", "Comparison", "Compare categories across groups."),
    ("stacked_bar", "Stacked Bar Chart", "Comparison", "Compare category totals by stacked group."),
    ("countplot", "Count Plot", "Comparison", "Counts in each category."),
    ("pie", "Pie Chart", "Comparison", "Proportions across categories."),
    ("donut", "Donut Chart", "Comparison", "Proportions with a central hole."),
    ("treemap", "Treemap", "Comparison", "Hierarchical area comparison."),
    ("scatter", "Scatter Plot", "Relationship", "Relationship between two numeric columns."),
    ("bubble", "Bubble Chart", "Relationship", "Scatter plot with size encoded by a numeric column."),
    ("line", "Line Chart", "Time Series", "Values across an ordered axis."),
    ("area", "Area Chart", "Time Series", "Magnitude across an ordered axis."),
    ("heatmap", "Heatmap", "Relationship", "Aggregated values across two categories."),
    ("correlation_heatmap", "Correlation Heatmap", "Relationship", "Pairwise numeric correlations."),
    ("pairplot", "Pair Plot", "Relationship", "Pairwise relationships among numeric columns."),
    ("jointplot", "Joint Plot", "Relationship", "Two numeric variables with marginal distributions."),
    ("hexbin", "Hexbin Plot", "Relationship", "Dense bivariate relationship using hexagonal bins."),
    ("qq", "Q-Q Plot", "Statistical", "Compare a numeric sample with a normal distribution."),
    ("regression", "Regression Plot", "Statistical", "Numeric relationship with fitted regression."),
]
REGISTRY = {x[0]: {"id": x[0], "name": x[1], "category": x[2], "description": x[3]} for x in CHARTS}
MAX_ROWS = 100_000
MAX_POINTS = 10_000


def _require(p: dict, *names: str) -> None:
    for name in names:
        if not isinstance(p.get(name), str) or not p[name]:
            raise ValueError(f"Parameter '{name}' is required.")


def _numeric(frame: pd.DataFrame, *names: str) -> None:
    for name in names:
        if name not in frame or not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"Column '{name}' must be numeric.")


def generate(frame: pd.DataFrame, chart: str, parameters: dict[str, Any], export: str, artifact_root: Path, dataset_id: str, version: int) -> dict:
    if chart not in REGISTRY:
        raise ValueError("Unsupported chart type.")
    p = dict(parameters)
    if len(frame) == 0: raise ValueError("Cannot visualize an empty dataset.")
    if len(frame) > MAX_ROWS: raise ValueError(f"Dataset exceeds visualization limit of {MAX_ROWS} rows.")
    known = set(frame.columns)
    for k in ("x", "y", "color", "group", "size", "value"):
        if p.get(k) is not None and p[k] not in known: raise ValueError(f"Unknown {k} column '{p[k]}'.")
    chart_columns = {
        "histogram": ("x",), "kde": ("x",), "boxplot": ("x",), "violin": ("x",), "ecdf": ("x",), "strip": ("x",), "swarm": ("x",), "frequency": ("x",),
        "bar": ("x",), "grouped_bar": ("x", "group"), "stacked_bar": ("x", "group"), "countplot": ("x",), "pie": ("x",), "donut": ("x",), "treemap": ("x",),
        "scatter": ("x", "y"), "bubble": ("x", "y", "size"), "line": ("x", "y"), "area": ("x", "y"), "heatmap": ("x", "y", "value"), "jointplot": ("x", "y"), "hexbin": ("x", "y"), "qq": ("x",), "regression": ("x", "y")}
    if chart in chart_columns: _require(p, *chart_columns[chart])
    numeric = {"histogram": ("x",), "kde": ("x",), "ecdf": ("x",), "scatter": ("x", "y"), "bubble": ("x", "y", "size"), "line": ("y",), "area": ("y",), "jointplot": ("x", "y"), "hexbin": ("x", "y"), "qq": ("x",), "regression": ("x", "y")}
    if chart in numeric:
        numeric_cols = [p[key] for key in numeric[chart]]
        _numeric(frame, *numeric_cols)
    if chart in {"bar", "grouped_bar", "stacked_bar", "pie", "donut", "treemap"} and p.get("y"): _numeric(frame, p["y"])
    if chart == "heatmap": _numeric(frame, p["value"])
    if chart == "correlation_heatmap" and frame.select_dtypes(include="number").shape[1] < 2: raise ValueError("Correlation heatmap needs at least two numeric columns.")
    if chart == "pairplot":
        selected = p.get("columns") or list(frame.select_dtypes(include="number").columns[:5])
        if len(selected) < 2: raise ValueError("Pair plot needs at least two numeric columns.")
        _numeric(frame, *selected); p["columns"] = selected
    if chart in {"kde", "qq", "regression", "jointplot"} and len(frame.dropna(subset=[c for c in (p.get("x"), p.get("y")) if c])) < 2: raise ValueError("At least two complete observations are required.")
    if chart == "swarm" and len(frame) > 3000: raise ValueError("Swarm plot is limited to 3,000 observations; choose strip plot or sample the data.")
    if export not in {"png", "svg", "html"}: raise ValueError("Export must be png, svg or html.")
    sampled = len(frame) > MAX_POINTS
    work = frame.sample(MAX_POINTS, random_state=0) if sampled else frame
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = p.get("x"), p.get("y")
    title = p.get("title", REGISTRY[chart]["name"])
    if chart == "histogram": ax.hist(work[x].dropna(), bins=max(1, min(200, int(p.get("bins", 20)))), color=p.get("color", "#4f75d2"), alpha=.8, orientation=p.get("orientation", "vertical")); ax.set_xlabel(x)
    elif chart == "kde": sns.kdeplot(data=work, x=x, ax=ax, fill=bool(p.get("fill", False)))
    elif chart in {"boxplot", "violin", "strip", "swarm"}:
        cat = p.get("group") or p.get("color")
        fn = {"boxplot": sns.boxplot, "violin": sns.violinplot, "strip": sns.stripplot, "swarm": sns.swarmplot}[chart]
        kwargs = {"data": work, "x": cat if cat else None, "y": x, "ax": ax}
        if p.get("color"): kwargs["hue"] = p["color"]
        if chart == "boxplot": kwargs["showfliers"] = bool(p.get("show_outliers", True))
        fn(**kwargs)
    elif chart == "ecdf": sns.ecdfplot(data=work, x=x, ax=ax)
    elif chart in {"bar", "countplot", "pie", "donut", "treemap"}:
        if chart in {"pie", "donut", "treemap"}:
            values = work.groupby(x, dropna=False)[y].sum() if y else work[x].value_counts().head(50)
            values = values.sort_values(ascending=False).head(50)
            if chart in {"pie", "donut"}: ax.pie(values.values, labels=values.index.astype(str), autopct="%1.1f%%" if p.get("show_percent", True) else None)
            if chart == "donut": ax.add_artist(plt.Circle((0, 0), float(p.get("hole", .45)), color="white"))
            if chart == "treemap":
                ax.clear(); _treemap(ax, values.values.astype(float), values.index.astype(str).tolist()); ax.axis("off")
        elif chart == "countplot": sns.countplot(data=work, x=x, ax=ax, order=work[x].value_counts().head(50).index)
        else:
            vals = work.groupby(x)[y].agg(p.get("aggregation", "sum")) if y else work[x].value_counts()
            vals.head(50).plot(kind="bar", ax=ax, color=p.get("color", "#4f75d2"))
    elif chart in {"grouped_bar", "stacked_bar"}:
        table = pd.crosstab(work[x], work[p["group"]], values=work[y] if y else None, aggfunc=p.get("aggregation", "sum")).fillna(0) if y else pd.crosstab(work[x], work[p["group"]])
        table.head(50).plot(kind="bar", stacked=chart == "stacked_bar", ax=ax)
    elif chart in {"scatter", "bubble"}: ax.scatter(work[x], work[y], s=np.clip(work[p["size"]].fillna(0), 5, 200) if chart == "bubble" else float(p.get("point_size", 24)), c=p.get("color") if p.get("color") in work and pd.api.types.is_numeric_dtype(work[p["color"]]) else None, alpha=float(p.get("opacity", .7))); ax.set_xlabel(x); ax.set_ylabel(y)
    elif chart in {"line", "area"}:
        data = work.sort_values(x) if p.get("sort", True) else work
        if chart == "line": sns.lineplot(data=data, x=x, y=y, hue=p.get("group"), estimator=p.get("aggregation", "mean"), ax=ax)
        else: sns.lineplot(data=data, x=x, y=y, hue=p.get("group"), estimator=p.get("aggregation", "mean"), ax=ax); ax.fill_between(data[x], data[y], alpha=.25)
    elif chart == "heatmap":
        table = work.pivot_table(index=p["y"], columns=x, values=p["value"], aggfunc=p.get("aggregation", "mean")); sns.heatmap(table, cmap=p.get("color_scale", "viridis"), annot=bool(p.get("annotate", False)), ax=ax)
    elif chart == "correlation_heatmap": sns.heatmap(work.select_dtypes(include="number").corr(), vmin=-1, vmax=1, cmap="coolwarm", annot=bool(p.get("annotate", True)), ax=ax)
    elif chart == "pairplot":
        plt.close(fig); grid = sns.pairplot(work[p["columns"]].dropna().sample(min(len(work.dropna(subset=p["columns"])), 2000), random_state=0)); fig = grid.fig
    elif chart == "jointplot":
        plt.close(fig); grid = sns.jointplot(data=work, x=x, y=y); fig = grid.fig
    elif chart == "hexbin": ax.hexbin(work[x], work[y], gridsize=int(p.get("gridsize", 35)), cmap=p.get("color_scale", "viridis")); ax.set_xlabel(x); ax.set_ylabel(y); fig.colorbar(ax.collections[0], ax=ax)
    elif chart == "qq":
        from scipy import stats
        stats.probplot(work[x].dropna(), dist="norm", plot=ax)
    elif chart == "regression": sns.regplot(data=work, x=x, y=y, ax=ax, scatter_kws={"alpha": float(p.get("opacity", .65))})
    fig.suptitle(title); fig.tight_layout()
    artifact_root.mkdir(parents=True, exist_ok=True); artifact_id = str(uuid.uuid4()); path = artifact_root / f"{artifact_id}.{export}"
    if export == "html":
        try:
            import plotly.express as px
            if chart == "histogram": pf = px.histogram(work, x=x, nbins=int(p.get("bins", 20)), title=title)
            elif chart in {"scatter", "bubble", "regression"}: pf = px.scatter(work, x=x, y=y, size=p.get("size") if chart == "bubble" else None, color=p.get("color") if p.get("color") in work else None, trendline="ols" if chart == "regression" or p.get("trendline") else None, title=title)
            elif chart in {"line", "area"}: pf = px.area(work, x=x, y=y, color=p.get("group"), title=title) if chart == "area" else px.line(work, x=x, y=y, color=p.get("group"), title=title)
            elif chart in {"pie", "donut"}: pf = px.pie(work, names=x, values=y, hole=float(p.get("hole", .45)) if chart == "donut" else 0, title=title)
            elif chart in {"bar", "grouped_bar", "stacked_bar", "countplot"}: pf = px.bar(work, x=x, y=y, color=p.get("group"), title=title, barmode="stack" if chart == "stacked_bar" else "group")
            elif chart in {"heatmap", "correlation_heatmap"}: pf = px.imshow(work.select_dtypes(include="number").corr() if chart == "correlation_heatmap" else work.pivot_table(index=p["y"], columns=x, values=p["value"], aggfunc=p.get("aggregation", "mean")), title=title, aspect="auto")
            elif chart == "ecdf": pf = px.ecdf(work, x=x, title=title)
            elif chart in {"boxplot", "violin", "strip", "swarm"}: pf = {"boxplot": px.box, "violin": px.violin, "strip": px.strip, "swarm": px.strip}[chart](work, x=p.get("group"), y=x, color=p.get("color"), title=title)
            elif chart == "treemap": pf = px.treemap(work, path=[x], values=y, title=title)
            elif chart == "hexbin": pf = px.density_heatmap(work, x=x, y=y, title=title)
            elif chart == "jointplot": pf = px.scatter(work, x=x, y=y, marginal_x="histogram", marginal_y="histogram", title=title)
            elif chart == "qq":
                from scipy import stats as scipy_stats
                vals = np.sort(work[x].dropna().to_numpy()); theoretical = scipy_stats.norm.ppf((np.arange(vals.shape[0]) + .5) / vals.shape[0]); pf = px.scatter(x=theoretical, y=vals, title=title, labels={"x": "Theoretical quantile", "y": x})
            elif chart == "pairplot":
                from pandas.plotting import scatter_matrix
                # Interactive pairplot uses Plotly's scatter matrix built from the validated numeric columns.
                pf = px.scatter_matrix(work[p["columns"]], dimensions=p["columns"], title=title)
            elif chart == "histogram" and p.get("kde"): pf = px.histogram(work, x=x, nbins=int(p.get("bins", 20)), histnorm="probability density", title=title, marginal="rug")
            else: pf = None
            if pf is not None: pf.write_html(path, include_plotlyjs="cdn"); plt.close(fig)
            else: raise ValueError(f"Interactive HTML is not implemented for {chart}.")
        except ImportError:
            fig.savefig(path.with_suffix(".png"), format="png"); path.write_text(f"<html><body><h1>{title}</h1><img src='{artifact_id}.png' alt='Static PNG preview'></body></html>", encoding="utf-8")
    else: fig.savefig(path, format=export)
    plt.close("all")
    png_path = artifact_root / f"{artifact_id}.png"
    if export != "png" and not png_path.exists(): fig.savefig(png_path, format="png")
    return {"artifact_id": artifact_id, "dataset_id": dataset_id, "dataset_version": version, "chart_type": chart, "title": title, "parameters": p, "columns_used": [c for c in p.values() if isinstance(c, str) and c in known], "render_format": export, "artifact_url": f"/api/v1/artifacts/{artifact_id}.{export}", "preview_url": f"/api/v1/artifacts/{artifact_id}.png", "download_url": f"/api/v1/artifacts/{artifact_id}.{export}", "sampled": sampled, "rows_processed": len(work)}


def _treemap(ax, sizes: np.ndarray, labels: list[str]) -> None:
    """Simple recursive slice-and-dice treemap, avoiding optional chart packages."""
    sizes = np.maximum(sizes, 0)
    total = float(sizes.sum())
    if total <= 0: raise ValueError("Treemap values must contain a positive total.")
    def draw(indices: list[int], x: float, y: float, w: float, h: float, horizontal: bool) -> None:
        if not indices: return
        if len(indices) == 1:
            i = indices[0]
            import matplotlib.pyplot as plt
            ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=plt.cm.tab20(i % 20), edgecolor="white"))
            ax.text(x + w / 2, y + h / 2, labels[i][:18], ha="center", va="center", fontsize=8, clip_on=True)
            return
        subtotal = sum(sizes[i] for i in indices)
        split = subtotal / 2
        acc = 0.0; mid = 1
        for pos, idx in enumerate(indices[:-1], start=1):
            acc += sizes[idx]
            mid = pos
            if acc >= split: break
        first, second = indices[:mid], indices[mid:]
        share = sum(sizes[i] for i in first) / subtotal if subtotal else .5
        if horizontal:
            draw(first, x, y, w * share, h, not horizontal); draw(second, x + w * share, y, w * (1-share), h, not horizontal)
        else:
            draw(first, x, y, w, h * share, not horizontal); draw(second, x, y + h * share, w, h * (1-share), not horizontal)
    draw(list(np.argsort(sizes)[::-1]), 0, 0, 1, 1, True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
