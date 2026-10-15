"""Evidence-preserving audit of raw OOI ERDDAP acquisition chunks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UTC = UTC


class AuditError(RuntimeError):
    """Raised when raw-data integrity is not sufficient for a scientific audit."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    return value


def read_erddap_csv(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read an ERDDAP CSV while retaining its second-row unit declarations."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    unit_row = pd.read_csv(path, skiprows=1, nrows=1, header=None, names=header)
    if unit_row.empty:
        raise AuditError(f"Missing ERDDAP units row: {path}")
    units = {
        column: "" if pd.isna(unit_row.iloc[0][column]) else str(unit_row.iloc[0][column])
        for column in header
    }
    frame = pd.read_csv(path, skiprows=[1], low_memory=False)
    if frame.columns.tolist() != header:
        raise AuditError(f"Header changed while reading {path}")
    return frame, units


@dataclass
class StreamAccumulator:
    site: str
    stream: str
    dataset_id: str
    files: int = 0
    rows: int = 0
    timestamp_missing: int = 0
    timestamp_invalid: int = 0
    timestamps: list[pd.DatetimeIndex] = field(default_factory=list)
    columns: set[str] = field(default_factory=set)
    units: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    variable_values: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    variable_total: Counter[str] = field(default_factory=Counter)
    variable_nonfinite: Counter[str] = field(default_factory=Counter)
    qc_counts: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    input_records: list[dict[str, Any]] = field(default_factory=list)


def _load_provenance(csv_path: Path, verify_sha256: bool) -> dict[str, Any]:
    sidecar = csv_path.with_suffix(".provenance.json")
    if not sidecar.exists():
        raise AuditError(f"Missing provenance sidecar: {sidecar}")
    with sidecar.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    required = {"site", "stream", "dataset_id", "sha256", "data_rows"}
    missing = required.difference(record)
    if missing:
        raise AuditError(f"Incomplete provenance {sidecar}: {sorted(missing)}")
    observed_sha = sha256_file(csv_path)
    if verify_sha256 and observed_sha != record["sha256"]:
        raise AuditError(f"SHA-256 mismatch: {csv_path}")
    record["audited_sha256"] = observed_sha
    record["relative_file"] = csv_path.as_posix()
    return record


def _numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _summarize_values(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {key: None for key in ("minimum", "q01", "median", "q99", "maximum")}
    q01, median, q99 = np.quantile(values, [0.01, 0.50, 0.99])
    return {
        "minimum": _json_scalar(values.min()),
        "q01": _json_scalar(q01),
        "median": _json_scalar(median),
        "q99": _json_scalar(q99),
        "maximum": _json_scalar(values.max()),
    }


def _collect_chunk(
    accumulator: StreamAccumulator,
    frame: pd.DataFrame,
    units: dict[str, str],
    record: dict[str, Any],
) -> None:
    if len(frame) != int(record["data_rows"]):
        raise AuditError(
            f"Row-count mismatch for {record['relative_file']}: "
            f"sidecar={record['data_rows']} observed={len(frame)}"
        )
    if "time" not in frame:
        raise AuditError(f"Missing time column: {record['relative_file']}")

    parsed = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    accumulator.files += 1
    accumulator.rows += len(frame)
    accumulator.timestamp_missing += int(frame["time"].isna().sum())
    accumulator.timestamp_invalid += int(parsed.isna().sum() - frame["time"].isna().sum())
    accumulator.timestamps.append(pd.DatetimeIndex(parsed.dropna()))
    accumulator.columns.update(frame.columns)
    accumulator.input_records.append(record)

    for column in frame.columns:
        accumulator.units[column].add(units.get(column, ""))
        if column == "time":
            continue
        values = _numeric_values(frame[column])
        if column.endswith("_qc_agg"):
            accumulator.qc_counts[column].update(int(x) for x in values)
        elif column not in {"latitude", "longitude", "z"}:
            accumulator.variable_values[column].append(values)
            accumulator.variable_total[column] += len(frame[column])
            accumulator.variable_nonfinite[column] += len(frame[column]) - len(values)


def _stream_summary(
    accumulator: StreamAccumulator,
    gap_factors: Iterable[int],
    top_gaps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    timestamps = (
        pd.DatetimeIndex(
            pd.to_datetime(
                np.concatenate([index.asi8 for index in accumulator.timestamps]), utc=True
            )
        )
        if accumulator.timestamps
        else pd.DatetimeIndex([])
    )
    timestamps = timestamps.sort_values()
    duplicate_timestamps = int(timestamps.duplicated().sum())
    unique = timestamps.drop_duplicates()
    deltas = np.diff(unique.asi8) / 60_000_000_000 if len(unique) > 1 else np.array([])
    positive = deltas[deltas > 0]
    nominal = float(np.median(positive)) if positive.size else None

    gap_rows: list[dict[str, Any]] = []
    if nominal and top_gaps > 0:
        largest_positions = np.argsort(deltas)[::-1][:top_gaps]
        for rank, position in enumerate(largest_positions, start=1):
            gap_rows.append(
                {
                    "site": accumulator.site,
                    "stream": accumulator.stream,
                    "rank": rank,
                    "gap_start": unique[position].isoformat(),
                    "gap_end": unique[position + 1].isoformat(),
                    "gap_minutes": float(deltas[position]),
                    "nominal_cadence_multiple": float(deltas[position] / nominal),
                }
            )

    summary = {
        "site": accumulator.site,
        "stream": accumulator.stream,
        "dataset_id": accumulator.dataset_id,
        "files": accumulator.files,
        "rows": accumulator.rows,
        "unique_timestamps": int(len(unique)),
        "duplicate_timestamps": duplicate_timestamps,
        "timestamp_missing": accumulator.timestamp_missing,
        "timestamp_invalid": accumulator.timestamp_invalid,
        "first_time": unique.min().isoformat() if len(unique) else None,
        "last_time": unique.max().isoformat() if len(unique) else None,
        "nominal_cadence_minutes": nominal,
    }
    for factor in gap_factors:
        summary[f"gaps_over_{factor}x"] = int(np.sum(positive > factor * nominal)) if nominal else 0

    variable_rows: list[dict[str, Any]] = []
    for column, arrays in sorted(accumulator.variable_values.items()):
        values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
        variable_rows.append(
            {
                "site": accumulator.site,
                "stream": accumulator.stream,
                "variable": column,
                "units": "|".join(sorted(accumulator.units[column])),
                "total_values": accumulator.variable_total[column],
                "finite_values": int(values.size),
                "missing_or_nonfinite": accumulator.variable_nonfinite[column],
                "missing_or_nonfinite_fraction": (
                    accumulator.variable_nonfinite[column] / accumulator.variable_total[column]
                    if accumulator.variable_total[column]
                    else None
                ),
                **_summarize_values(values),
            }
        )

    qc_rows: list[dict[str, Any]] = []
    for column, counts in sorted(accumulator.qc_counts.items()):
        total = sum(counts.values())
        for flag, count in sorted(counts.items()):
            qc_rows.append(
                {
                    "site": accumulator.site,
                    "stream": accumulator.stream,
                    "qc_variable": column,
                    "flag": flag,
                    "count": count,
                    "fraction": count / total if total else None,
                }
            )
    summary["_unique_index"] = unique
    return summary, variable_rows, qc_rows, gap_rows


def _add_physical_range_checks(
    variable_rows: list[dict[str, Any]],
    accumulators: dict[tuple[str, str], StreamAccumulator],
    ranges: dict[str, dict[str, Any]],
) -> None:
    lookup = {(row["site"], row["stream"], row["variable"]): row for row in variable_rows}
    for (site, stream), accumulator in accumulators.items():
        for variable, spec in ranges.items():
            arrays = accumulator.variable_values.get(variable, [])
            row = lookup.get((site, stream, variable))
            if row is None:
                continue
            values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
            low, high = float(spec["minimum"]), float(spec["maximum"])
            observed_units = accumulator.units.get(variable, {""})
            expected_units = str(spec.get("expected_units", ""))
            units_match = expected_units in observed_units
            outside = int(np.sum((values < low) | (values > high))) if units_match else None
            row["physical_minimum"] = low
            row["physical_maximum"] = high
            row["outside_physical_range"] = outside
            row["outside_physical_fraction"] = (
                outside / values.size if outside is not None and values.size else None
            )
            row["expected_units"] = expected_units
            row["units_match"] = units_match


def _overlap_rows(
    summaries: list[dict[str, Any]], candidate_minutes: Iterable[int]
) -> list[dict[str, Any]]:
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_site[summary["site"]].append(summary)
    output: list[dict[str, Any]] = []
    for site, streams in sorted(by_site.items()):
        if len(streams) < 2 or any(len(item["_unique_index"]) == 0 for item in streams):
            continue
        start = max(item["_unique_index"].min() for item in streams)
        end = min(item["_unique_index"].max() for item in streams)
        for minutes in candidate_minutes:
            frequency = f"{int(minutes)}min"
            full = pd.date_range(
                start.ceil(frequency), end.floor(frequency), freq=frequency, tz="UTC"
            )
            observed = [
                item["_unique_index"].floor(frequency).drop_duplicates() for item in streams
            ]
            joint = observed[0]
            for index in observed[1:]:
                joint = joint.intersection(index)
            row: dict[str, Any] = {
                "site": site,
                "grid_minutes": int(minutes),
                "overlap_start": start.isoformat(),
                "overlap_end": end.isoformat(),
                "possible_bins": int(len(full)),
                "joint_observed_bins": int(len(joint.intersection(full))),
                "joint_coverage": len(joint.intersection(full)) / len(full) if len(full) else None,
            }
            for item, index in zip(streams, observed, strict=True):
                row[f"{item['stream']}_coverage"] = (
                    len(index.intersection(full)) / len(full) if len(full) else None
                )
            output.append(row)
    return output


def _availability_window_rows(
    summaries: list[dict[str, Any]],
    candidate_minutes: Iterable[int],
    window_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure coverage in fixed windows using timestamps only, never sensor values."""
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_site[summary["site"]].append(summary)

    window_days = int(window_config["window_days"])
    minimum_window_days = int(window_config["minimum_window_days"])
    primary_threshold = float(window_config["minimum_joint_coverage"])
    thresholds = sorted({float(x) for x in window_config["coverage_thresholds"]})
    rows: list[dict[str, Any]] = []

    for site, streams in sorted(by_site.items()):
        if len(streams) < 2 or any(len(item["_unique_index"]) == 0 for item in streams):
            continue
        for minutes in candidate_minutes:
            frequency = f"{int(minutes)}min"
            global_start = max(item["_unique_index"].min() for item in streams).ceil(frequency)
            global_end = min(item["_unique_index"].max() for item in streams).floor(
                frequency
            ) + pd.Timedelta(minutes=int(minutes))
            observed = {
                item["stream"]: item["_unique_index"].floor(frequency).drop_duplicates()
                for item in streams
            }
            joint = next(iter(observed.values()))
            for index in list(observed.values())[1:]:
                joint = joint.intersection(index)

            cursor = global_start
            window_id = 0
            while cursor < global_end:
                stop = min(cursor + pd.Timedelta(days=window_days), global_end)
                duration_days = (stop - cursor).total_seconds() / 86_400
                if duration_days < minimum_window_days:
                    break
                full = pd.date_range(cursor, stop, freq=frequency, inclusive="left", tz="UTC")
                joint_count = int(((joint >= cursor) & (joint < stop)).sum())
                row: dict[str, Any] = {
                    "site": site,
                    "grid_minutes": int(minutes),
                    "window_id": window_id,
                    "window_start": cursor.isoformat(),
                    "window_end": stop.isoformat(),
                    "window_days": duration_days,
                    "possible_bins": int(len(full)),
                    "joint_observed_bins": joint_count,
                    "joint_coverage": joint_count / len(full) if len(full) else None,
                }
                for stream, index in sorted(observed.items()):
                    count = int(((index >= cursor) & (index < stop)).sum())
                    row[f"{stream}_coverage"] = count / len(full) if len(full) else None
                row["eligible_primary"] = bool(
                    row["joint_coverage"] is not None and row["joint_coverage"] >= primary_threshold
                )
                for threshold in thresholds:
                    label = str(threshold).replace(".", "p")
                    row[f"eligible_{label}"] = bool(
                        row["joint_coverage"] is not None and row["joint_coverage"] >= threshold
                    )
                rows.append(row)
                window_id += 1
                cursor = stop

    retention: list[dict[str, Any]] = []
    if rows:
        frame = pd.DataFrame(rows)
        for (site, grid), group in frame.groupby(["site", "grid_minutes"], sort=True):
            for threshold in thresholds:
                column = f"eligible_{str(threshold).replace('.', 'p')}"
                selected = group[group[column]]
                retention.append(
                    {
                        "site": site,
                        "grid_minutes": int(grid),
                        "coverage_threshold": threshold,
                        "windows_total": int(len(group)),
                        "windows_eligible": int(len(selected)),
                        "eligible_window_fraction": (
                            len(selected) / len(group) if len(group) else None
                        ),
                        "eligible_days": float(selected["window_days"].sum()),
                        "possible_bins_in_eligible_windows": int(selected["possible_bins"].sum()),
                        "joint_bins_in_eligible_windows": int(
                            selected["joint_observed_bins"].sum()
                        ),
                        "coverage_within_eligible_windows": (
                            float(
                                selected["joint_observed_bins"].sum()
                                / selected["possible_bins"].sum()
                            )
                            if selected["possible_bins"].sum()
                            else None
                        ),
                    }
                )
    return rows, retention


def _schema_compatibility(accumulators: dict[tuple[str, str], StreamAccumulator]) -> dict[str, Any]:
    by_stream: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for (site, stream), accumulator in accumulators.items():
        by_stream[stream][site] = accumulator.columns
    result: dict[str, Any] = {}
    for stream, sites in sorted(by_stream.items()):
        schemas = list(sites.values())
        common = set.intersection(*schemas) if schemas else set()
        union = set.union(*schemas) if schemas else set()
        result[stream] = {
            "compatible": (
                all(columns == schemas[0] for columns in schemas[1:]) if schemas else True
            ),
            "common_columns": sorted(common),
            "union_columns": sorted(union),
            "site_missing_from_union": {
                site: sorted(union.difference(columns)) for site, columns in sorted(sites.items())
            },
        }
    return result


def audit_ooi(raw_root: Path, output_root: Path, audit_config: dict[str, Any]) -> dict[str, Any]:
    """Audit all acquired OOI chunks and write deterministic evidence tables."""
    csv_paths = sorted(raw_root.glob("*/*/*.csv"))
    if not csv_paths:
        raise AuditError(f"No OOI CSV chunks found under {raw_root}")

    accumulators: dict[tuple[str, str], StreamAccumulator] = {}
    input_records: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        record = _load_provenance(csv_path, bool(audit_config["verify_sha256"]))
        key = (str(record["site"]), str(record["stream"]))
        accumulator = accumulators.setdefault(
            key,
            StreamAccumulator(key[0], key[1], str(record["dataset_id"])),
        )
        if accumulator.dataset_id != record["dataset_id"]:
            raise AuditError(f"Multiple dataset IDs encountered for {key}")
        frame, units = read_erddap_csv(csv_path)
        _collect_chunk(accumulator, frame, units, record)
        input_records.append(record)

    summaries: list[dict[str, Any]] = []
    variables: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for accumulator in accumulators.values():
        summary, variable_rows, stream_qc, stream_gaps = _stream_summary(
            accumulator,
            audit_config["gap_factors"],
            int(audit_config["top_gaps_per_stream"]),
        )
        summaries.append(summary)
        variables.extend(variable_rows)
        qc_rows.extend(stream_qc)
        gaps.extend(stream_gaps)

    _add_physical_range_checks(variables, accumulators, audit_config["physical_ranges"])
    overlaps = _overlap_rows(summaries, audit_config["candidate_grid_minutes"])
    availability_windows, retention = _availability_window_rows(
        summaries,
        audit_config["candidate_grid_minutes"],
        audit_config["availability_windows"],
    )
    schemas = _schema_compatibility(accumulators)

    serializable_summaries = []
    for summary in summaries:
        cleaned = {key: value for key, value in summary.items() if key != "_unique_index"}
        serializable_summaries.append(cleaned)
    serializable_summaries.sort(key=lambda row: (row["site"], row["stream"]))

    _atomic_csv(output_root / "stream_audit.csv", pd.DataFrame(serializable_summaries))
    _atomic_csv(output_root / "variable_audit.csv", pd.DataFrame(variables))
    _atomic_csv(output_root / "qc_audit.csv", pd.DataFrame(qc_rows))
    _atomic_csv(output_root / "gap_audit.csv", pd.DataFrame(gaps))
    _atomic_csv(output_root / "overlap_audit.csv", pd.DataFrame(overlaps))
    _atomic_csv(output_root / "availability_window_audit.csv", pd.DataFrame(availability_windows))
    _atomic_csv(output_root / "coverage_retention_summary.csv", pd.DataFrame(retention))
    _atomic_json(output_root / "schema_compatibility.json", schemas)

    insufficient = [
        row
        for row in overlaps
        if row["joint_coverage"] is not None
        and row["joint_coverage"] < float(audit_config["minimum_joint_coverage"])
    ]
    config_bytes = json.dumps(audit_config, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "audit_version": "1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "raw_root": str(raw_root),
        "files_audited": len(csv_paths),
        "streams_audited": len(serializable_summaries),
        "rows_audited": sum(row["rows"] for row in serializable_summaries),
        "integrity_failures": 0,
        "schema_incompatibilities": [
            stream for stream, value in schemas.items() if not value["compatible"]
        ],
        "candidate_grids_below_joint_coverage_target": len(insufficient),
        "availability_windows_evaluated": len(availability_windows),
        "availability_selection_uses_measurement_values": bool(
            audit_config["availability_windows"]["selection_uses_measurement_values"]
        ),
        "network_outages_injected_after_availability_selection": bool(
            audit_config["availability_windows"]["network_outages_injected_after_selection"]
        ),
        "outputs": [
            "stream_audit.csv",
            "variable_audit.csv",
            "qc_audit.csv",
            "gap_audit.csv",
            "overlap_audit.csv",
            "availability_window_audit.csv",
            "coverage_retention_summary.csv",
            "schema_compatibility.json",
            "input_manifest.json",
        ],
    }
    _atomic_json(output_root / "input_manifest.json", input_records)
    _atomic_json(output_root / "audit_summary.json", result)
    return result
