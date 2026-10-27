"""JSON artifact renderer with deterministic formatting and provenance envelope."""

import json
from decimal import Decimal
from typing import Any
import uuid

from rag_app.reports.base import BaseArtifactRenderer
from rag_app.schemas.artifacts import DownloadableArtifactFormat, TabularDataset, VerificationStatus


class _EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, uuid.UUID):
            return str(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if hasattr(o, "model_dump"):
            return o.model_dump()
        return super().default(o)


class JSONRenderer(BaseArtifactRenderer):
    format = DownloadableArtifactFormat.JSON
    mime_type = "application/json; charset=utf-8"
    file_extension = "json"

    def render(
        self,
        dataset: TabularDataset,
        title: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        records = []
        for idx, row in enumerate(dataset.rows):
            row_id = f"row_{idx}"
            prov = dataset.row_provenance.get(row_id)
            records.append({
                "row_id": row_id,
                "data": row,
                "provenance": prov.model_dump() if prov else None,
            })

        payload = {
            "title": title or dataset.title,
            "description": dataset.description or summary,
            "columns": [col.model_dump() for col in dataset.columns],
            "total_verified_records": len(dataset.rows),
            "records": records,
            "summary_metrics": dataset.summary_metrics,
            "coverage": dataset.coverage.model_dump() if dataset.coverage else None,
            "metadata": metadata or {},
        }

        json_str = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            cls=_EnhancedJSONEncoder,
            sort_keys=False,
        )
        return json_str.encode("utf-8")

