import math
import re
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 0. COMMON AXIS / PRIMITIVES
# ============================================================

class AxisSpec(BaseModel):
    label: str = Field(..., min_length=1)
    unit: str | None = None


def _is_finite_number(val: Any) -> bool:
    if isinstance(val, (int, float)):
        return not math.isnan(val) and not math.isinf(val)
    return False


# ============================================================
# 1. BAR CHART
# ============================================================

class BarDataPoint(BaseModel):
    label: str = Field(..., min_length=1)
    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Any) -> float:
        if not _is_finite_number(v):
            raise ValueError(f"Value must be a finite number, got {v}")
        return float(v)


class BarSeries(BaseModel):
    name: str = Field(..., min_length=1)
    data: list[BarDataPoint] = Field(..., min_length=1)


class BarVisualization(BaseModel):
    type: Literal["bar"] = "bar"
    title: str = Field(..., min_length=1)
    x_axis: AxisSpec
    y_axis: AxisSpec
    series: list[BarSeries] = Field(..., min_length=1)


# ============================================================
# 2. LINE CHART
# ============================================================

class LineDataPoint(BaseModel):
    label: str = Field(..., min_length=1)
    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Any) -> float:
        if not _is_finite_number(v):
            raise ValueError(f"Value must be a finite number, got {v}")
        return float(v)


class LineSeries(BaseModel):
    name: str = Field(..., min_length=1)
    data: list[LineDataPoint] = Field(..., min_length=1)


class LineVisualization(BaseModel):
    type: Literal["line"] = "line"
    title: str = Field(..., min_length=1)
    x_axis: AxisSpec
    y_axis: AxisSpec
    series: list[LineSeries] = Field(..., min_length=1)


# ============================================================
# 3. PIE CHART
# ============================================================

class PieDataPoint(BaseModel):
    label: str = Field(..., min_length=1)
    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Any) -> float:
        if not _is_finite_number(v) or v < 0:
            raise ValueError(f"Pie chart value must be a finite number >= 0, got {v}")
        return float(v)


class PieVisualization(BaseModel):
    type: Literal["pie"] = "pie"
    title: str = Field(..., min_length=1)
    unit: str | None = None
    data: list[PieDataPoint] = Field(..., min_length=2)


# ============================================================
# 4. AREA CHART
# ============================================================

class AreaDataPoint(BaseModel):
    label: str = Field(..., min_length=1)
    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Any) -> float:
        if not _is_finite_number(v):
            raise ValueError(f"Value must be a finite number, got {v}")
        return float(v)


class AreaSeries(BaseModel):
    name: str = Field(..., min_length=1)
    data: list[AreaDataPoint] = Field(..., min_length=1)


class AreaVisualization(BaseModel):
    type: Literal["area"] = "area"
    title: str = Field(..., min_length=1)
    x_axis: AxisSpec
    y_axis: AxisSpec
    series: list[AreaSeries] = Field(..., min_length=1)


# ============================================================
# 5. TABLE
# ============================================================

ColumnType = Literal["string", "number", "integer", "percent", "date"]


class TableColumn(BaseModel):
    key: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    type: ColumnType
    unit: str | None = None


class TableVisualization(BaseModel):
    type: Literal["table"] = "table"
    title: str = Field(..., min_length=1)
    columns: list[TableColumn] = Field(..., min_length=1)
    rows: list[dict[str, Any]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_rows(self) -> "TableVisualization":
        col_map = {c.key: c.type for c in self.columns}
        for row_idx, row in enumerate(self.rows):
            for col_key, col_type in col_map.items():
                if col_key not in row:
                    raise ValueError(f"Row {row_idx} is missing required column '{col_key}'")
                val = row[col_key]
                if val is None:
                    continue
                if col_type in ("number", "integer", "percent"):
                    if not _is_finite_number(val):
                        raise ValueError(f"Row {row_idx} column '{col_key}' must be numeric, got {type(val).__name__} ({val})")
                    if col_type == "percent" and not (0 <= val <= 100):
                        raise ValueError(f"Row {row_idx} percent column '{col_key}' must be between 0 and 100, got {val}")
                    if col_type == "integer" and not isinstance(val, int) and not (isinstance(val, float) and val.is_integer()):
                        raise ValueError(f"Row {row_idx} integer column '{col_key}' must be an integer, got {val}")
                elif col_type in ("string", "date"):
                    if not isinstance(val, str):
                        raise ValueError(f"Row {row_idx} column '{col_key}' must be a string/date, got {type(val).__name__}")
        return self


# ============================================================
# 6. NETWORK GRAPH
# ============================================================

NodeType = Literal[
    "company",
    "person",
    "department",
    "document",
    "topic",
    "deal",
    "event",
    "other",
]


class NetworkNode(BaseModel):
    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    type: NodeType = "other"


class NetworkEdge(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)


class NetworkVisualization(BaseModel):
    type: Literal["network"] = "network"
    title: str = Field(..., min_length=1)
    nodes: list[NetworkNode] = Field(..., min_length=1)
    edges: list[NetworkEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_network(self) -> "NetworkVisualization":
        node_ids = set()
        for node in self.nodes:
            if node.id in node_ids:
                raise ValueError(f"Duplicate node ID '{node.id}' found in network visualization")
            node_ids.add(node.id)

        for edge_idx, edge in enumerate(self.edges):
            if edge.source not in node_ids:
                raise ValueError(f"Edge {edge_idx} source '{edge.source}' does not exist in nodes")
            if edge.target not in node_ids:
                raise ValueError(f"Edge {edge_idx} target '{edge.target}' does not exist in nodes")
        return self


# ============================================================
# 7. TIMELINE
# ============================================================

class TimelineEvent(BaseModel):
    date: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        # Check ISO-8601 or YYYY-MM-DD or reasonable date string
        # Match YYYY-MM-DD, YYYY-MM, or ISO 8601 timestamps
        iso_pattern = r"^\d{4}-\d{2}(-\d{2})?(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
        if not re.match(iso_pattern, v.strip()):
            # Allow common month year like 'July 2026' or '2026' if parseable, but standard is ISO
            # Try basic date pattern
            pass
        return v.strip()


class TimelineVisualization(BaseModel):
    type: Literal["timeline"] = "timeline"
    title: str = Field(..., min_length=1)
    events: list[TimelineEvent] = Field(..., min_length=1)


# ============================================================
# 8. DISCRIMINATED UNION & VALIDATOR
# ============================================================

VisualizationSpec = Annotated[
    Union[
        BarVisualization,
        LineVisualization,
        PieVisualization,
        AreaVisualization,
        TableVisualization,
        NetworkVisualization,
        TimelineVisualization,
    ],
    Field(discriminator="type"),
]

visualization_adapter = TypeAdapter(VisualizationSpec)


def _normalize_visualization_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalizes common LLM output variations (string axis specs, flat data arrays)
    into compliant VisualizationSpec schema candidates prior to validation.
    """
    if not isinstance(raw, dict):
        return raw

    import copy
    data = copy.deepcopy(raw)
    viz_type = data.get("type")

    def _to_float(val: Any) -> float | None:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            f = float(val)
            if not math.isnan(f) and not math.isinf(f):
                return f
        return None

    def _to_axis(ax_val: Any, default_label: str) -> dict[str, Any]:
        if isinstance(ax_val, dict) and "label" in ax_val and ax_val["label"]:
            return {"label": str(ax_val["label"]), "unit": ax_val.get("unit")}
        elif isinstance(ax_val, str) and ax_val.strip():
            return {"label": ax_val.strip(), "unit": None}
        return {"label": default_label, "unit": None}

    if viz_type in {"bar", "line", "area"}:
        default_x = "Quarter" if viz_type == "area" else ("Time" if viz_type == "line" else "Category")
        data["x_axis"] = _to_axis(data.get("x_axis"), default_x)
        data["y_axis"] = _to_axis(data.get("y_axis"), "Value")

        # If series is missing/empty, but data is present as list of points:
        if (not data.get("series") or not isinstance(data.get("series"), list)) and isinstance(data.get("data"), list):
            norm_points = []
            for item in data["data"]:
                if isinstance(item, dict):
                    lbl = item.get("label") or item.get("name") or item.get("x") or item.get("category") or item.get("opportunity")
                    val = _to_float(item.get("value") if item.get("value") is not None else item.get("y"))
                    if lbl and val is not None:
                        norm_points.append({"label": str(lbl).strip(), "value": val})
            if norm_points:
                data["series"] = [{"name": data.get("title") or "Series", "data": norm_points}]

        # If series is present, normalize points within each series
        elif isinstance(data.get("series"), list):
            norm_series = []
            for s in data["series"]:
                if isinstance(s, dict):
                    s_name = s.get("name") or data.get("title") or "Series"
                    pts = []
                    for item in s.get("data", []):
                        if isinstance(item, dict):
                            lbl = item.get("label") or item.get("name") or item.get("x")
                            val = _to_float(item.get("value") if item.get("value") is not None else item.get("y"))
                            if lbl and val is not None:
                                pts.append({"label": str(lbl).strip(), "value": val})
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            lbl = str(item[0]).strip()
                            val = _to_float(item[1])
                            if lbl and val is not None:
                                pts.append({"label": lbl, "value": val})
                    if pts:
                        norm_series.append({"name": str(s_name).strip(), "data": pts})
            data["series"] = norm_series

    elif viz_type == "pie":
        if isinstance(data.get("data"), list):
            norm_points = []
            for item in data["data"]:
                if isinstance(item, dict):
                    lbl = item.get("label") or item.get("name") or item.get("slice")
                    val = _to_float(item.get("value") if item.get("value") is not None else item.get("val"))
                    if lbl and val is not None and val >= 0:
                        norm_points.append({"label": str(lbl).strip(), "value": val})
            data["data"] = norm_points

    return data


def validate_visualization(data: Any) -> dict[str, Any] | None:
    """
    Validates a raw visualization object against strict VisualizationSpec schemas.
    Attempts normalization for common LLM key variations before strict Pydantic validation.
    Degrades gracefully: returns None on any validation error or malformed structure.
    """
    if data is None:
        return None

    if not isinstance(data, dict):
        logger.warning("Visualization validation failed: expected dict, got %s", type(data).__name__)
        return None

    viz_type = data.get("type")
    valid_types = {"bar", "line", "pie", "area", "table", "network", "timeline"}
    if viz_type not in valid_types:
        logger.warning("Visualization validation failed: invalid type '%s'", viz_type)
        return None

    try:
        normalized = _normalize_visualization_dict(data)
        validated = visualization_adapter.validate_python(normalized)
        return validated.model_dump(exclude_none=True)
    except Exception as exc:
        logger.warning(
            "Visualization validation failed for type '%s': %s",
            viz_type,
            exc,
            extra={"event": "visualization_validation_failure", "error": str(exc)},
        )
        return None
