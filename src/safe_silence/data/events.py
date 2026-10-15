"""Leakage-safe, observation-grounded critical-event labeling for Task H."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_silence.data.preprocess import sha256_file

UTC = UTC
REQUIRED_CHANNELS = ("temperature", "salinity", "oxygen", "chlorophyll")


class EventLabelingError(RuntimeError):
    """Raised when Task H cannot preserve its scientific contract."""


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


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def verify_task_g_inputs(input_root: Path) -> tuple[dict[str, Any], int]:
    """Verify Task G hashes before consuming any processed measurements."""
    summary_path = input_root / "preprocessing_summary.json"
    if not summary_path.exists():
        raise EventLabelingError(f"Missing Task G summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != "G":
        raise EventLabelingError("Input directory is not a Task G artifact")

    data_root = input_root.parent.parent
    verified = 0
    for artifact in summary.get("artifacts", []):
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not relative or not expected:
            raise EventLabelingError("Task G artifact lacks path or SHA-256")
        path = data_root / str(relative)
        if not path.exists() or sha256_file(path) != str(expected):
            raise EventLabelingError(f"Task G artifact changed or is missing: {path}")
        verified += 1
    if verified == 0:
        raise EventLabelingError("Task G summary contains no verifiable artifacts")
    return summary, verified


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"protocol", "site", "window_key", "timestamp", "split"}
    for channel in REQUIRED_CHANNELS:
        required.update({channel, f"observed_{channel}"})
    missing = required.difference(frame.columns)
    if missing:
        raise EventLabelingError(f"Task G dataset missing columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise EventLabelingError("Task G dataset contains invalid timestamps")
    result = result.sort_values(
        ["protocol", "site", "window_key", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    duplicate = result.duplicated(["protocol", "site", "window_key", "timestamp"])
    if duplicate.any():
        raise EventLabelingError("Duplicate timestamp within a Task G window")
    return result


def _observed(frame: pd.DataFrame, channel: str) -> pd.Series:
    return frame[f"observed_{channel}"].fillna(0).astype(bool) & frame[channel].notna()


def _causal_difference(
    frame: pd.DataFrame, channel: str, grid_minutes: int
) -> tuple[pd.Series, pd.Series]:
    grouped = frame.groupby(["protocol", "site", "window_key"], sort=False)
    previous_value = grouped[channel].shift(1)
    previous_observed = grouped[f"observed_{channel}"].shift(1).fillna(0).astype(bool)
    previous_time = grouped["timestamp"].shift(1)
    expected_delta = pd.Timedelta(minutes=grid_minutes)
    contiguous = (frame["timestamp"] - previous_time) == expected_delta
    valid = _observed(frame, channel) & previous_observed & contiguous
    difference = (frame[channel] - previous_value).where(valid)
    return difference.astype(float), valid


def _minimum_run_filter(
    candidate: pd.Series,
    valid: pd.Series,
    windows: pd.Series,
    minimum_steps: int,
) -> pd.Series:
    """Confirm persistence using trailing evidence only; never back-label an onset."""
    if minimum_steps <= 1:
        return (candidate & valid).astype(bool)
    output = np.zeros(len(candidate), dtype=bool)
    candidate_array = (candidate & valid).to_numpy(dtype=bool)
    grouped_positions = (
        pd.Series(np.arange(len(windows))).groupby(windows, sort=False).groups.values()
    )
    for positions in grouped_positions:
        indices = np.asarray(list(positions), dtype=int)
        values = candidate_array[indices]
        start = 0
        while start < len(values):
            if not values[start]:
                start += 1
                continue
            end = start + 1
            while end < len(values) and values[end]:
                end += 1
            if end - start >= minimum_steps:
                output[indices[start + minimum_steps - 1 : end]] = True
            start = end
    return pd.Series(output, index=candidate.index)


def _require_samples(values: pd.Series, minimum: int, name: str) -> pd.Series:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    finite = finite[np.isfinite(finite)]
    if len(finite) < minimum:
        raise EventLabelingError(
            f"Insufficient observed training samples for {name}: {len(finite)} < {minimum}"
        )
    return finite


def fit_protocol_thresholds(
    frame: pd.DataFrame,
    protocol: str,
    grid_minutes: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit every statistical threshold using CE02SHSM training observations only."""
    data = _validate_frame(frame)
    calibration_site = str(config["calibration"]["site"])
    calibration_split = str(config["split"]["calibration_split"])
    train = data[
        (data["protocol"] == protocol)
        & (data["site"] == calibration_site)
        & (data["split"] == calibration_split)
    ].copy()
    if train.empty:
        raise EventLabelingError(f"No calibration rows for {protocol}")
    if set(train["site"]) != {calibration_site} or set(train["split"]) != {calibration_split}:
        raise EventLabelingError("Calibration data escaped the locked site/split")

    minimum = int(config["calibration"]["minimum_samples"])
    low_cfg = config["labels"]["low_oxygen_anomaly"]
    oxygen = _require_samples(
        train.loc[_observed(train, "oxygen"), "oxygen"], minimum, "oxygen level"
    )
    low_threshold = float(np.quantile(oxygen, float(low_cfg["training_quantile"])))

    chlor_cfg = config["labels"]["high_chlorophyll_anomaly"]
    chlorophyll = _require_samples(
        train.loc[_observed(train, "chlorophyll"), "chlorophyll"],
        minimum,
        "chlorophyll level",
    )
    chlor_threshold = float(np.quantile(chlorophyll, float(chlor_cfg["training_quantile"])))

    temperature_difference, temperature_valid = _causal_difference(
        train, "temperature", grid_minutes
    )
    temperature_rate = temperature_difference.abs() / (grid_minutes / 60.0)
    temperature_rate = _require_samples(
        temperature_rate[temperature_valid], minimum, "temperature rate"
    )
    rapid_cfg = config["labels"]["rapid_temperature_change"]
    rapid_threshold = float(np.quantile(temperature_rate, float(rapid_cfg["training_quantile"])))

    multivariate_cfg = config["labels"]["multivariate_innovation_anomaly"]
    multivariate_channels = [str(value) for value in multivariate_cfg["channels"]]
    differences: dict[str, pd.Series] = {}
    valid_all = pd.Series(True, index=train.index)
    for channel in multivariate_channels:
        difference, valid = _causal_difference(train, channel, grid_minutes)
        differences[channel] = difference
        valid_all &= valid
    difference_frame = pd.DataFrame(differences).loc[valid_all]
    if len(difference_frame) < minimum:
        raise EventLabelingError(
            "Insufficient complete observed training innovations for multivariate score: "
            f"{len(difference_frame)} < {minimum}"
        )
    center = difference_frame.median()
    mad = (difference_frame - center).abs().median()
    consistency = float(config["robust_score"]["mad_consistency_factor"])
    minimum_scale = float(config["robust_score"]["minimum_scale"])
    scale = (mad * consistency).clip(lower=minimum_scale)
    robust_score = np.sqrt(np.mean(np.square((difference_frame - center) / scale), axis=1))
    score_threshold = float(np.quantile(robust_score, float(multivariate_cfg["training_quantile"])))

    hypoxia = config["ecological_reference"]["hypoxia"]
    return {
        "protocol": protocol,
        "grid_minutes": int(grid_minutes),
        "calibration_site": calibration_site,
        "calibration_split": calibration_split,
        "observed_measurements_only": True,
        "calibration_rows": int(len(train)),
        "thresholds": {
            "hypoxia_absolute_umol_kg": float(hypoxia["threshold_umol_kg"]),
            "low_oxygen_anomaly_umol_kg": low_threshold,
            "rapid_temperature_change_c_per_hour": rapid_threshold,
            "high_chlorophyll_anomaly_microg_l": chlor_threshold,
            "multivariate_innovation_score": score_threshold,
        },
        "training_sample_counts": {
            "oxygen_level": int(len(oxygen)),
            "temperature_rate": int(len(temperature_rate)),
            "chlorophyll_level": int(len(chlorophyll)),
            "complete_multivariate_innovation": int(len(difference_frame)),
        },
        "robust_multivariate": {
            "channels": multivariate_channels,
            "center": {key: float(value) for key, value in center.items()},
            "scale": {key: float(value) for key, value in scale.items()},
            "definition": "sqrt(mean(((causal_difference-center)/scale)^2))",
        },
    }


def label_protocol_frame(
    frame: pd.DataFrame,
    thresholds: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply frozen thresholds; imputed measurements never define event truth."""
    data = _validate_frame(frame)
    grid_minutes = int(thresholds["grid_minutes"])
    values = thresholds["thresholds"]

    oxygen_valid = _observed(data, "oxygen")
    chlorophyll_valid = _observed(data, "chlorophyll")
    temperature_difference, temperature_valid = _causal_difference(
        data, "temperature", grid_minutes
    )
    temperature_rate = temperature_difference.abs() / (grid_minutes / 60.0)
    data["score_temperature_rate_c_per_hour"] = temperature_rate.astype("float32")

    candidates: dict[str, pd.Series] = {
        "hypoxia_absolute": oxygen_valid
        & (data["oxygen"] <= float(values["hypoxia_absolute_umol_kg"])),
        "low_oxygen_anomaly": oxygen_valid
        & (data["oxygen"] <= float(values["low_oxygen_anomaly_umol_kg"])),
        "rapid_temperature_change": temperature_valid
        & (temperature_rate >= float(values["rapid_temperature_change_c_per_hour"])),
        "high_chlorophyll_anomaly": chlorophyll_valid
        & (data["chlorophyll"] >= float(values["high_chlorophyll_anomaly_microg_l"])),
    }
    validities: dict[str, pd.Series] = {
        "hypoxia_absolute": oxygen_valid,
        "low_oxygen_anomaly": oxygen_valid,
        "rapid_temperature_change": temperature_valid,
        "high_chlorophyll_anomaly": chlorophyll_valid,
    }

    multivariate = thresholds["robust_multivariate"]
    multivariate_channels = list(multivariate["channels"])
    difference_values: dict[str, pd.Series] = {}
    multivariate_valid = pd.Series(True, index=data.index)
    for channel in multivariate_channels:
        difference, valid = _causal_difference(data, channel, grid_minutes)
        difference_values[channel] = difference
        multivariate_valid &= valid
    standardized = pd.DataFrame(
        {
            channel: (difference_values[channel] - float(multivariate["center"][channel]))
            / float(multivariate["scale"][channel])
            for channel in multivariate_channels
        }
    )
    multivariate_score = pd.Series(
        np.sqrt(np.mean(np.square(standardized), axis=1)), index=data.index
    ).where(multivariate_valid)
    data["score_multivariate_innovation"] = multivariate_score.astype("float32")
    candidates["multivariate_innovation_anomaly"] = multivariate_valid & (
        multivariate_score >= float(values["multivariate_innovation_score"])
    )
    validities["multivariate_innovation_anomaly"] = multivariate_valid

    for event_name, candidate in candidates.items():
        validity = validities[event_name]
        minimum_steps = int(config["labels"][event_name]["minimum_consecutive_steps"])
        label = _minimum_run_filter(
            candidate.fillna(False), validity, data["window_key"], minimum_steps
        )
        data[f"valid_{event_name}"] = validity.astype(np.uint8)
        data[f"label_{event_name}"] = label.astype(np.uint8)

    critical = [str(value) for value in config["critical_union"]]
    label_columns = [f"label_{name}" for name in critical]
    valid_columns = [f"valid_{name}" for name in critical]
    data["label_critical_event"] = data[label_columns].any(axis=1).astype(np.uint8)
    data["valid_critical_event"] = data[valid_columns].all(axis=1).astype(np.uint8)
    data["label_available_event"] = data[label_columns].any(axis=1).astype(np.uint8)
    data["valid_available_event"] = data[valid_columns].any(axis=1).astype(np.uint8)
    return data


def extract_episodes(
    labeled: pd.DataFrame,
    event_names: list[str],
    grid_minutes: int,
    merge_gap_steps: int,
) -> pd.DataFrame:
    """Create an episode registry without changing pointwise ground-truth labels."""
    records: list[dict[str, Any]] = []
    group_columns = ["protocol", "site", "split", "window_key"]
    for keys, group in labeled.groupby(group_columns, sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for event_name in event_names:
            positive = np.flatnonzero(group[f"label_{event_name}"].to_numpy(dtype=bool))
            if len(positive) == 0:
                continue
            episode_start = 0
            episode_index = 0
            for cursor in range(1, len(positive) + 1):
                boundary = cursor == len(positive) or (
                    positive[cursor] - positive[cursor - 1] > merge_gap_steps + 1
                )
                if not boundary:
                    continue
                positions = positive[episode_start:cursor]
                first = int(positions[0])
                last = int(positions[-1])
                records.append(
                    {
                        "protocol": keys[0],
                        "site": keys[1],
                        "split": keys[2],
                        "window_key": keys[3],
                        "event_type": event_name,
                        "episode_index": episode_index,
                        "start": group.loc[first, "timestamp"].isoformat(),
                        "end": group.loc[last, "timestamp"].isoformat(),
                        "positive_steps": int(len(positions)),
                        "span_steps": int(last - first + 1),
                        "span_hours": float((last - first + 1) * grid_minutes / 60.0),
                    }
                )
                episode_index += 1
                episode_start = cursor
    columns = [
        "protocol",
        "site",
        "split",
        "window_key",
        "event_type",
        "episode_index",
        "start",
        "end",
        "positive_steps",
        "span_steps",
        "span_hours",
    ]
    return pd.DataFrame(records, columns=columns)


def _event_counts(
    labeled_frames: list[pd.DataFrame], episodes: pd.DataFrame, event_names: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in labeled_frames:
        for keys, group in frame.groupby(["protocol", "site", "split"], sort=False):
            for event_name in event_names:
                episode_count = 0
                if not episodes.empty:
                    episode_count = int(
                        (
                            (episodes["protocol"] == keys[0])
                            & (episodes["site"] == keys[1])
                            & (episodes["split"] == keys[2])
                            & (episodes["event_type"] == event_name)
                        ).sum()
                    )
                rows.append(
                    {
                        "protocol": keys[0],
                        "site": keys[1],
                        "split": keys[2],
                        "event_type": event_name,
                        "rows": int(len(group)),
                        "valid_rows": int(group[f"valid_{event_name}"].sum()),
                        "event_rows": int(group[f"label_{event_name}"].sum()),
                        "episodes": episode_count,
                    }
                )
    return pd.DataFrame(rows)


def label_events(
    input_root: Path,
    output_root: Path,
    events_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute Task H and write frozen thresholds, labels, episodes, and audit counts."""
    _, verified = verify_task_g_inputs(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    thresholds_by_protocol: dict[str, Any] = {}
    source_frames: dict[tuple[str, str], pd.DataFrame] = {}

    for protocol, protocol_config in preprocessing_config["protocols"].items():
        sites = [protocol_config["development_site"], *protocol_config["external_sites"]]
        for site in sites:
            path = input_root / protocol / f"{site}.parquet"
            if not path.exists():
                raise EventLabelingError(f"Missing Task G dataset: {path}")
            source_frames[(protocol, site)] = pd.read_parquet(path)
        development = source_frames[(protocol, protocol_config["development_site"])]
        thresholds_by_protocol[protocol] = fit_protocol_thresholds(
            development,
            protocol,
            int(protocol_config["grid_minutes"]),
            events_config,
        )

    threshold_document = {
        "task": "H",
        "labeling_version": str(events_config["labeling_version"]),
        "fitted_from_site": events_config["calibration"]["site"],
        "fitted_from_split": events_config["split"]["calibration_split"],
        "external_data_used_for_fitting": False,
        "validation_or_test_data_used_for_fitting": False,
        "ecological_reference": events_config["ecological_reference"],
        "protocols": thresholds_by_protocol,
    }
    thresholds_path = output_root / "frozen_event_thresholds.json"
    _atomic_json(thresholds_path, threshold_document)

    labeled_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    artifacts: list[dict[str, Any]] = []
    event_names = [str(value) for value in events_config["critical_union"]]
    for (protocol, site), source in source_frames.items():
        labeled = label_protocol_frame(source, thresholds_by_protocol[protocol], events_config)
        path = output_root / protocol / f"{site}.parquet"
        _atomic_parquet(path, labeled)
        labeled_frames.append(labeled)
        grid = int(preprocessing_config["protocols"][protocol]["grid_minutes"])
        episode_frames.append(
            extract_episodes(
                labeled,
                event_names,
                grid,
                int(events_config["episode_merge_gap_steps"]),
            )
        )
        artifacts.append(
            {
                "protocol": protocol,
                "site": site,
                "path": str(path.relative_to(output_root.parent.parent)),
                "rows": int(len(labeled)),
                "sha256": sha256_file(path),
            }
        )

    episodes = pd.concat(episode_frames, ignore_index=True)
    counts = _event_counts(labeled_frames, episodes, event_names)
    episodes_path = output_root / "event_episode_registry.csv"
    counts_path = output_root / "event_counts.csv"
    _atomic_csv(episodes_path, episodes)
    _atomic_csv(counts_path, counts)
    artifacts.extend(
        [
            {
                "path": str(thresholds_path.relative_to(output_root.parent.parent)),
                "sha256": sha256_file(thresholds_path),
            },
            {
                "path": str(episodes_path.relative_to(output_root.parent.parent)),
                "sha256": sha256_file(episodes_path),
            },
            {
                "path": str(counts_path.relative_to(output_root.parent.parent)),
                "sha256": sha256_file(counts_path),
            },
        ]
    )
    config_hash = hashlib.sha256(
        json.dumps(events_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "H",
        "labeling_version": str(events_config["labeling_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_g_artifacts_reverified": verified,
        "threshold_calibration_site": events_config["calibration"]["site"],
        "threshold_calibration_split": events_config["split"]["calibration_split"],
        "external_data_used_for_thresholds": False,
        "validation_or_test_data_used_for_thresholds": False,
        "imputed_values_used_as_event_truth": False,
        "oxygen_sparse_sites_support_hypoxia_validation": False,
        "temporal_protocols_may_be_pooled": False,
        "episode_total_is_not_a_unique_ecological_event_count": True,
        "zero_support_event_classes_are_scored": False,
        "event_types": event_names,
        "episodes": int(len(episodes)),
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "labeling_summary.json", summary)
    return summary
