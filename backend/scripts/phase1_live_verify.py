"""Run real Phase 1 HTTP workflows against an already-running local server."""

from __future__ import annotations

import io
import json
import os

import httpx
import pandas as pd


BASE_URL = os.getenv("PLAYGROUND_API_URL", "http://127.0.0.1:8765/api/v1")


def request(method: str, path: str, **kwargs):
    response = httpx.request(method, f"{BASE_URL}{path}", timeout=30, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path}: {response.status_code} {response.text}")
    return response


def main() -> None:
    assert request("GET", "/health").json() == {"status": "ok"}

    csv = b"name,value,group\nalpha,1,A\nbeta,2,B\nalpha,3,A\n,4,B\n"
    csv_item = request("POST", "/datasets", files={"file": ("workflow.csv", csv, "text/csv")}).json()
    assert csv_item["rows"] == 4 and csv_item["cols"] == 3

    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"score": [2, 4], "team": ["x", "y"]}).to_excel(writer, index=False, sheet_name="Data")
        pd.DataFrame({"not_loaded": [99]}).to_excel(writer, index=False, sheet_name="Extra")
    xlsx_item = request("POST", "/datasets", files={"file": ("workflow.xlsx", workbook.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}).json()
    assert (xlsx_item["rows"], xlsx_item["cols"]) == (2, 2)

    dataset_id = csv_item["id"]
    schema = request("GET", f"/datasets/{dataset_id}/schema").json()
    filtered = request("GET", f"/datasets/{dataset_id}/rows", params=[("filters", json.dumps({"column": "group", "operator": "eq", "value": "A"})), ("sort_by", "value"), ("sort_dir", "desc"), ("limit", "1")]).json()
    distinct = request("GET", f"/datasets/{dataset_id}/columns/group/distinct", params={"limit": 1}).json()
    assert schema["row_count"] == 4
    assert filtered["total"] == 2 and filtered["rows"][0]["value"] == 3
    assert distinct["truncated"] and distinct["values"][0]["count"] == 2

    assert request("POST", f"/datasets/{dataset_id}/archive").json()["archived"]
    assert not any(item["id"] == dataset_id for item in request("GET", "/datasets").json()["items"])
    archived = request("GET", "/datasets", params={"archived": "true"}).json()["items"]
    assert any(item["id"] == dataset_id for item in archived)
    assert request("POST", f"/datasets/{dataset_id}/restore").json()["id"] == dataset_id

    delete_id = xlsx_item["id"]
    failed_delete = httpx.delete(f"{BASE_URL}/datasets/{delete_id}", timeout=30)
    assert failed_delete.status_code == 400
    assert request("DELETE", f"/datasets/{delete_id}", params={"confirm": "true"}).json()["deleted"]

    print(json.dumps({"persistence_dataset_id": dataset_id, "version": csv_item["current_version"], "csv_rows": csv_item["rows"], "xlsx_shape": [xlsx_item["rows"], xlsx_item["cols"]], "filter_total": filtered["total"], "distinct_truncated": distinct["truncated"], "archive_restore": True, "confirmed_delete": True}, sort_keys=True))


if __name__ == "__main__":
    main()
