import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.visualizations import generate

frame = pd.DataFrame({"n": pd.Series([1., 2, 3, 4, 5, 6], dtype="float64"), "m": pd.Series([4., 2, 3, 7, 5, 8], dtype="float64"), "c": ["a", "b", "a", "b", "c", "a"], "g": ["u", "u", "v", "v", "u", "v"]})
cases = {
    "histogram": {"x": "n"}, "kde": {"x": "n"}, "boxplot": {"x": "n", "group": "c"}, "violin": {"x": "n", "group": "c"},
    "ecdf": {"x": "n"}, "strip": {"x": "n", "group": "c"}, "swarm": {"x": "n", "group": "c"}, "bar": {"x": "c"},
    "grouped_bar": {"x": "c", "group": "g"}, "stacked_bar": {"x": "c", "group": "g"}, "countplot": {"x": "c"}, "pie": {"x": "c"},
    "donut": {"x": "c"}, "treemap": {"x": "c"}, "scatter": {"x": "n", "y": "m"}, "bubble": {"x": "n", "y": "m", "size": "m"},
    "line": {"x": "n", "y": "m"}, "area": {"x": "n", "y": "m"}, "heatmap": {"x": "c", "y": "g", "value": "n"},
    "correlation_heatmap": {}, "pairplot": {}, "jointplot": {"x": "n", "y": "m"}, "hexbin": {"x": "n", "y": "m"},
    "qq": {"x": "n"}, "regression": {"x": "n", "y": "m"}, "frequency": {"x": "c"},
}
errors = []
for chart, params in cases.items():
    try:
        generate(frame, chart, params, "png", Path(tempfile.gettempdir()) / "visualization-smoke", "dataset", 1)
    except Exception as exc:
        errors.append((chart, repr(exc), frame.dtypes.astype(str).to_dict()))
print(errors)
