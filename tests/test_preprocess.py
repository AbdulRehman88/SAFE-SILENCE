from copy import deepcopy

import numpy as np
import pandas as pd

from safe_silence.data.preprocess import _extract_windows, assign_splits, causal_fill


def test_causal_fill_never_uses_future_value() -> None:
    frame = pd.DataFrame(
        {
            "temperature": [np.nan, 10.0, np.nan, np.nan, 20.0],
            "observed_temperature": [0, 1, 0, 0, 1],
        }
    )
    result = causal_fill(frame, ["temperature"], maximum_steps=1, age_cap_steps=10)
    assert np.isnan(result.loc[0, "temperature"])
    assert result.loc[2, "temperature"] == 10.0
    assert np.isnan(result.loc[3, "temperature"])
    assert result.loc[2, "imputed_temperature"] == 1
    assert result.loc[4, "temperature"] == 20.0


def test_external_sites_are_test_only_and_embargoes_separate_splits() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "site": "DEV",
                "window_key": f"dev-{index}",
                "window_start": f"2024-{index + 1:02d}-01T00:00:00+00:00",
                "accepted_post_qc": True,
            }
        )
    for index in range(3):
        rows.append(
            {
                "site": "EXT",
                "window_key": f"ext-{index}",
                "window_start": f"2024-{index + 1:02d}-01T00:00:00+00:00",
                "accepted_post_qc": True,
            }
        )
    config = {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "internal_test_fraction": 0.2,
        "embargo_windows": 1,
    }
    result = assign_splits(pd.DataFrame(rows), "DEV", ["EXT"], config)
    assert set(result.loc[result["site"] == "EXT", "split"]) == {"external_test"}
    assert (result["split"] == "embargo_train_validation").sum() == 1
    assert (result["split"] == "embargo_validation_test").sum() == 1


def test_split_assignment_does_not_depend_on_measurements() -> None:
    registry = pd.DataFrame(
        {
            "site": ["DEV"] * 12,
            "window_key": [f"w-{index}" for index in range(12)],
            "window_start": pd.date_range("2020-01-01", periods=12, freq="30D", tz="UTC"),
            "accepted_post_qc": [True] * 12,
            "arbitrary_measurement": np.arange(12),
        }
    )
    config = {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "internal_test_fraction": 0.2,
        "embargo_windows": 1,
    }
    first = assign_splits(registry, "DEV", [], config)["split"].tolist()
    shuffled_values = deepcopy(registry)
    shuffled_values["arbitrary_measurement"] = np.arange(12)[::-1]
    second = assign_splits(shuffled_values, "DEV", [], config)["split"].tolist()
    assert first == second


def test_sparse_oxygen_does_not_reject_support_complete_external_window() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="60min", tz="UTC")
    harmonized = pd.DataFrame(index=index)
    for channel in ["temperature", "salinity", "chlorophyll"]:
        harmonized[channel] = 1.0
        harmonized[f"observed_{channel}"] = 1
    harmonized["oxygen"] = np.nan
    harmonized["observed_oxygen"] = 0
    windows = pd.DataFrame(
        {
            "window_id": [0],
            "window_start": [index[0]],
            "window_end": [index[-1] + pd.Timedelta(hours=1)],
        }
    )
    dataset, registry = _extract_windows(
        harmonized,
        windows,
        "EXT",
        "secondary_60min",
        60,
        ["temperature", "salinity", "oxygen", "chlorophyll"],
        1,
        96,
        0.8,
        ["temperature", "salinity", "chlorophyll"],
        ["oxygen"],
    )
    assert registry.loc[0, "accepted_post_qc"]
    assert registry.loc[0, "coverage_oxygen"] == 0.0
    assert not dataset.empty
