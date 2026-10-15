from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.models.predictor import (
    WindowSequenceDataset,
    _build_tcn,
    assign_model_roles,
    conformal_quantile,
    fit_observed_scaler,
    worst_window_conformal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _role_frame() -> pd.DataFrame:
    rows = []
    for window in range(6):
        for step in range(2):
            rows.append(
                {
                    "window_key": f"w{window}",
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC")
                    + pd.Timedelta(days=30 * window, minutes=15 * step),
                    "split": "validation",
                    "arbitrary_measurement": 1000 - window,
                }
            )
    return pd.DataFrame(rows)


def test_validation_roles_are_disjoint_chronological_and_value_independent() -> None:
    frame = _role_frame()
    first = assign_model_roles(frame, 0.5)
    changed = frame.copy()
    changed["arbitrary_measurement"] = np.arange(len(changed)) ** 2
    second = assign_model_roles(changed, 0.5)
    assert first["model_role"].tolist() == second["model_role"].tolist()
    tuning = set(first.loc[first["model_role"] == "tuning", "window_key"])
    calibration = set(first.loc[first["model_role"] == "calibration", "window_key"])
    assert tuning == {"w0", "w1", "w2"}
    assert calibration == {"w3", "w4", "w5"}
    assert tuning.isdisjoint(calibration)


def test_scaler_uses_only_observed_training_values() -> None:
    rows = []
    for site, split, values, observed in (
        ("CE02SHSM", "train", [1.0, 3.0, 999.0], [1, 1, 0]),
        ("CE02SHSM", "validation", [-999.0], [1]),
        ("CE07SHSM", "external_test", [9999.0], [1]),
    ):
        for value, flag in zip(values, observed, strict=True):
            row = {"site": site, "split": split}
            for channel in ["temperature", "salinity", "oxygen", "chlorophyll"]:
                row[channel] = value
                row[f"observed_{channel}"] = flag
            rows.append(row)
    scaler = fit_observed_scaler(
        pd.DataFrame(rows),
        ["temperature", "salinity", "oxygen", "chlorophyll"],
        "CE02SHSM",
        "train",
    )
    assert scaler["channels"]["temperature"]["mean"] == 2.0
    assert scaler["channels"]["temperature"]["scale"] == 1.0
    assert scaler["channels"]["temperature"]["observations"] == 2


def test_conformal_quantile_uses_finite_sample_conservative_rank() -> None:
    value, rank, level = conformal_quantile(np.arange(1.0, 11.0), alpha=0.2)
    assert rank == 9
    assert level == 0.9
    assert value == 10.0


def test_worst_window_conformal_uses_most_conservative_window() -> None:
    scores = np.array([1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 1.3, 1.4, 1.5])
    windows = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    result = worst_window_conformal(
        scores, windows, np.ones(len(scores), dtype=bool), alpha=0.2, minimum_windows=3
    )
    assert result["worst_window"] == "b"
    assert result["multiplier"] == 2.2
    assert result["calibration_windows"] == 3


def test_canonical_task_i_config_is_locked() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    assert config["model"]["scaling"]["fit_split"] == "train"
    assert config["model"]["validation_roles"]["disjoint_tuning_and_calibration"]
    assert not config["model"]["evaluation"]["pool_temporal_protocols"]
    assert config["model"]["input_steps_by_protocol"]["primary_15min"] == 32
    assert config["model"]["input_steps_by_protocol"]["secondary_60min"] == 8
    assert config["model"]["predictor"] == "persistence_residual_tcn"


def test_sequence_dataset_never_crosses_window_boundary() -> None:
    pytest.importorskip("torch")
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    channels = config["model"]["channels"]
    rows = []
    for window, base in (("w0", 0.0), ("w1", 100.0)):
        for step in range(6):
            row = {
                "protocol": "primary_15min",
                "site": "CE02SHSM",
                "split": "train",
                "model_role": "train",
                "window_key": window,
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC")
                + pd.Timedelta(days=30 if window == "w1" else 0, minutes=15 * step),
            }
            for channel in channels:
                row[channel] = base + step
                row[f"observed_{channel}"] = 1
                row[f"imputed_{channel}"] = 0
                row[f"age_steps_{channel}"] = 0
            rows.append(row)
    scaler = {"channels": {channel: {"mean": 0.0, "scale": 1.0} for channel in channels}}
    dataset = WindowSequenceDataset(
        pd.DataFrame(rows), channels, scaler, input_steps=4, age_cap_steps=96
    )
    assert len(dataset) == 4
    first_x, _, _, _ = dataset[0]
    third_x, _, _, _ = dataset[2]
    assert float(first_x[0].max()) < 10.0
    assert float(third_x[0].min()) >= 100.0


def test_tcn_output_shape_and_parameter_ceiling() -> None:
    torch = pytest.importorskip("torch")
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")["model"]
    model = _build_tcn(config, input_features=16, outputs=4)
    mean, log_variance = model(torch.zeros(3, 16, 32))
    assert mean.shape == (3, 4)
    assert log_variance.shape == (3, 4)
    assert sum(parameter.numel() for parameter in model.parameters()) < config["maximum_parameters"]
