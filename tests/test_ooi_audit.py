import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from safe_silence.data.audit import audit_ooi


def _write_chunk(path: Path, times: list[str], temperatures: list[float]) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "sea_water_temperature", "sea_water_temperature_qc_agg"])
        writer.writerow(["UTC", "degree_C", "1"])
        for time, temperature in zip(times, temperatures, strict=True):
            writer.writerow([time, temperature, 1])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {
        "site": "TEST",
        "stream": "ctd",
        "dataset_id": "synthetic-ctd",
        "sha256": digest,
        "data_rows": len(times),
    }
    path.with_suffix(".provenance.json").write_text(json.dumps(sidecar), encoding="utf-8")


def test_audit_reports_duplicates_gaps_and_ranges(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "audit"
    chunk = raw / "TEST" / "ctd" / "chunk.csv"
    _write_chunk(
        chunk,
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:02:00Z",
            "2026-01-01T00:10:00Z",
        ],
        [10.0, 11.0, 11.0, 12.0, 50.0],
    )
    config = {
        "verify_sha256": True,
        "gap_factors": [2, 4],
        "candidate_grid_minutes": [1, 5],
        "minimum_joint_coverage": 0.8,
        "top_gaps_per_stream": 5,
        "physical_ranges": {
            "sea_water_temperature": {
                "expected_units": "degree_C",
                "minimum": -2.5,
                "maximum": 40.0,
            }
        },
        "availability_windows": {
            "window_days": 1,
            "minimum_window_days": 1,
            "minimum_joint_coverage": 0.8,
            "coverage_thresholds": [0.7, 0.8, 0.9],
            "selection_uses_measurement_values": False,
            "network_outages_injected_after_selection": True,
        },
    }
    summary = audit_ooi(raw, output, config)
    stream = pd.read_csv(output / "stream_audit.csv").iloc[0]
    variable = pd.read_csv(output / "variable_audit.csv").iloc[0]
    assert summary["files_audited"] == 1
    assert stream["duplicate_timestamps"] == 1
    assert stream["gaps_over_4x"] == 1
    assert variable["outside_physical_range"] == 1
    assert (output / "input_manifest.json").exists()
    assert (output / "availability_window_audit.csv").exists()
    assert (output / "coverage_retention_summary.csv").exists()
