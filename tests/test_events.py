import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from safe_silence.config import load_config
from safe_silence.data.events import (
    fit_protocol_thresholds,
    label_events,
    label_protocol_frame,
)
from safe_silence.data.preprocess import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["temperature", "salinity", "oxygen", "chlorophyll"]


def _event_config() -> dict:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")["events"]
    config = deepcopy(config)
    config["calibration"]["minimum_samples"] = 2
    return config


def _frame() -> pd.DataFrame:
    records = []
    values = {
        "temperature": [10.0, 10.1, 10.0, 10.2, 10.1, 10.4],
        "salinity": [32.0, 32.1, 32.0, 32.2, 32.1, 32.3],
        "oxygen": [250.0, 245.0, 240.0, 235.0, 230.0, 225.0],
        "chlorophyll": [1.0, 1.1, 1.2, 1.1, 1.3, 1.4],
    }
    for window_index in range(2):
        start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=30 * window_index)
        for step in range(6):
            row = {
                "protocol": "primary_15min",
                "site": "CE02SHSM",
                "window_key": f"w{window_index}",
                "timestamp": start + pd.Timedelta(minutes=15 * step),
                "split": "train",
            }
            for channel in CHANNELS:
                row[channel] = values[channel][step] + window_index * 0.01
                row[f"observed_{channel}"] = 1
                row[f"imputed_{channel}"] = 0
            records.append(row)
    return pd.DataFrame(records)


def test_thresholds_ignore_nontraining_and_external_values() -> None:
    config = _event_config()
    development = _frame()
    first = fit_protocol_thresholds(development, "primary_15min", 15, config)

    contamination = development.copy()
    contamination["site"] = "CE07SHSM"
    contamination["split"] = "external_test"
    contamination["oxygen"] = -999.0
    contamination["temperature"] = 999.0
    second = fit_protocol_thresholds(
        pd.concat([development, contamination], ignore_index=True),
        "primary_15min",
        15,
        config,
    )
    assert first == second


def test_imputed_value_never_creates_event_truth() -> None:
    config = _event_config()
    frame = _frame()
    thresholds = fit_protocol_thresholds(frame, "primary_15min", 15, config)
    thresholds["thresholds"]["low_oxygen_anomaly_umol_kg"] = 300.0
    frame.loc[1, "observed_oxygen"] = 0
    frame.loc[1, "imputed_oxygen"] = 1
    frame.loc[1, "oxygen"] = 1.0

    labeled = label_protocol_frame(frame, thresholds, config)
    assert labeled.loc[1, "valid_low_oxygen_anomaly"] == 0
    assert labeled.loc[1, "label_low_oxygen_anomaly"] == 0


def test_persistence_confirmation_is_causal_not_retroactive() -> None:
    config = _event_config()
    frame = _frame()
    thresholds = fit_protocol_thresholds(frame, "primary_15min", 15, config)
    thresholds["thresholds"]["hypoxia_absolute_umol_kg"] = 300.0
    labeled = label_protocol_frame(frame, thresholds, config)

    first_window = labeled[labeled["window_key"] == "w0"].reset_index(drop=True)
    assert first_window.loc[0, "label_hypoxia_absolute"] == 0
    assert first_window.loc[1, "label_hypoxia_absolute"] == 1


def test_causal_derivative_resets_at_window_boundary() -> None:
    config = _event_config()
    frame = _frame()
    thresholds = fit_protocol_thresholds(frame, "primary_15min", 15, config)
    thresholds["thresholds"]["rapid_temperature_change_c_per_hour"] = 0.0
    labeled = label_protocol_frame(frame, thresholds, config)
    first_rows = labeled.groupby("window_key", sort=False).head(1)
    assert (first_rows["valid_rapid_temperature_change"] == 0).all()
    assert first_rows["score_temperature_rate_c_per_hour"].isna().all()


def test_oxygen_sparse_site_cannot_claim_valid_critical_ground_truth() -> None:
    config = _event_config()
    training = _frame()
    thresholds = fit_protocol_thresholds(training, "primary_15min", 15, config)
    sparse = training.copy()
    sparse["site"] = "CE01ISSM"
    sparse["split"] = "external_test"
    sparse["oxygen"] = np.nan
    sparse["observed_oxygen"] = 0
    labeled = label_protocol_frame(sparse, thresholds, config)
    assert labeled["valid_hypoxia_absolute"].sum() == 0
    assert labeled["valid_low_oxygen_anomaly"].sum() == 0
    assert labeled["valid_critical_event"].sum() == 0


def test_task_h_end_to_end_writes_frozen_auditable_artifacts(tmp_path: Path) -> None:
    events_config = _event_config()
    input_root = tmp_path / "data" / "processed" / "task_g"
    output_root = tmp_path / "data" / "processed" / "task_h"
    protocol = "primary_15min"
    artifacts = []
    for site, split in (("CE02SHSM", "train"), ("CE07SHSM", "external_test")):
        frame = _frame()
        frame["site"] = site
        frame["split"] = split
        frame["window_key"] = site + "__" + frame["window_key"]
        path = input_root / protocol / f"{site}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        artifacts.append(
            {
                "path": str(path.relative_to(tmp_path / "data")),
                "sha256": sha256_file(path),
            }
        )
    summary = {"task": "G", "artifacts": artifacts}
    (input_root / "preprocessing_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    preprocessing = {
        "protocols": {
            protocol: {
                "grid_minutes": 15,
                "development_site": "CE02SHSM",
                "external_sites": ["CE07SHSM"],
            }
        }
    }

    result = label_events(input_root, output_root, events_config, preprocessing)
    assert result["external_data_used_for_thresholds"] is False
    assert result["imputed_values_used_as_event_truth"] is False
    assert (output_root / "frozen_event_thresholds.json").exists()
    assert (output_root / "event_counts.csv").exists()
    assert (output_root / protocol / "CE07SHSM.parquet").exists()
