"""Leakage-safe OOI harmonization and immutable split construction."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UTC = UTC


class PreprocessingError(RuntimeError):
    """Raised when Task G cannot preserve its scientific invariants."""


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


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
    except ImportError as exc:
        raise PreprocessingError(
            "Parquet support is unavailable; reinstall the project so pyarrow is present"
        ) from exc
    os.replace(temporary, path)


def _parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def verify_task_f_inputs(raw_root: Path, audit_root: Path) -> int:
    """Reverify every Task F input so preprocessing cannot consume mutated raw bytes."""
    manifest_path = audit_root / "input_manifest.json"
    if not manifest_path.exists():
        raise PreprocessingError(f"Missing Task F input manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise PreprocessingError("Task F input manifest is empty or invalid")
    for record in records:
        path = raw_root / str(record["site"]) / str(record["stream"]) / str(record["file_name"])
        if not path.exists():
            raise PreprocessingError(f"Task F audited input is missing: {path}")
        expected = str(record.get("audited_sha256") or record.get("sha256") or "")
        if not expected or sha256_file(path) != expected:
            raise PreprocessingError(f"Raw input changed after Task F audit: {path}")
    return len(records)


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    stream: str
    variable: str
    qc_variable: str
    expected_units: str
    minimum: float
    maximum: float


def _read_units(path: Path) -> dict[str, str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    unit_row = pd.read_csv(path, skiprows=1, nrows=1, header=None, names=header)
    if unit_row.empty:
        raise PreprocessingError(f"Missing ERDDAP units row: {path}")
    return {
        column: "" if pd.isna(unit_row.iloc[0][column]) else str(unit_row.iloc[0][column])
        for column in header
    }


def _load_stream_channels(
    raw_root: Path,
    site: str,
    stream: str,
    specs: list[ChannelSpec],
    grid_minutes: int,
    accepted_flags: set[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = sorted((raw_root / site / stream).glob("*.csv"))
    if not paths:
        raise PreprocessingError(f"No raw chunks for {site}/{stream}")

    usecols = ["time"]
    for spec in specs:
        usecols.extend([spec.variable, spec.qc_variable])
    usecols = list(dict.fromkeys(usecols))
    chunks: list[pd.DataFrame] = []
    qc_report: dict[str, dict[str, int]] = {
        spec.name: {
            "raw_rows": 0,
            "finite_rows": 0,
            "physically_valid_rows": 0,
            "accepted_rows": 0,
        }
        for spec in specs
    }

    for path in paths:
        units = _read_units(path)
        for spec in specs:
            actual_unit = units.get(spec.variable, "")
            if actual_unit != spec.expected_units:
                raise PreprocessingError(
                    f"Unit mismatch for {site}/{spec.name} in {path.name}: "
                    f"expected={spec.expected_units!r} actual={actual_unit!r}"
                )
        frame = pd.read_csv(path, skiprows=[1], usecols=usecols, low_memory=False)
        time = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        valid_time = time.notna()
        prepared = pd.DataFrame({"timestamp": time[valid_time]})
        for spec in specs:
            values = pd.to_numeric(frame.loc[valid_time, spec.variable], errors="coerce")
            flags = pd.to_numeric(frame.loc[valid_time, spec.qc_variable], errors="coerce")
            in_range = values.between(spec.minimum, spec.maximum, inclusive="both")
            accepted = values.notna() & flags.isin(accepted_flags) & in_range
            prepared[spec.name] = values.where(accepted).to_numpy()
            prepared[f"observed_{spec.name}"] = accepted.astype(np.uint8).to_numpy()
            qc_report[spec.name]["raw_rows"] += int(valid_time.sum())
            qc_report[spec.name]["finite_rows"] += int(values.notna().sum())
            qc_report[spec.name]["physically_valid_rows"] += int((values.notna() & in_range).sum())
            qc_report[spec.name]["accepted_rows"] += int(accepted.sum())
        chunks.append(prepared)

    combined = pd.concat(chunks, ignore_index=True)
    combined["timestamp"] = combined["timestamp"].dt.floor(f"{grid_minutes}min")
    aggregations: dict[str, str] = {}
    for spec in specs:
        aggregations[spec.name] = "median"
        aggregations[f"observed_{spec.name}"] = "max"
    binned = combined.groupby("timestamp", as_index=False, sort=True).agg(aggregations)
    for spec in specs:
        binned[f"observed_{spec.name}"] = binned[f"observed_{spec.name}"].astype(np.uint8)
    return binned, qc_report


def harmonize_site(
    raw_root: Path,
    site: str,
    grid_minutes: int,
    channel_specs: list[ChannelSpec],
    accepted_flags: set[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate canonical channels without filling or learning from values."""
    by_stream: dict[str, list[ChannelSpec]] = {}
    for spec in channel_specs:
        by_stream.setdefault(spec.stream, []).append(spec)

    frames: list[pd.DataFrame] = []
    report: dict[str, Any] = {}
    for stream, specs in sorted(by_stream.items()):
        frame, stream_report = _load_stream_channels(
            raw_root, site, stream, specs, grid_minutes, accepted_flags
        )
        frames.append(frame.set_index("timestamp"))
        report[stream] = stream_report
    merged = pd.concat(frames, axis=1, join="outer").sort_index()
    if not merged.index.is_unique:
        raise PreprocessingError(f"Non-unique harmonized timestamps for {site}/{grid_minutes}")
    return merged, report


def causal_fill(
    frame: pd.DataFrame,
    channels: list[str],
    maximum_steps: int,
    age_cap_steps: int,
) -> pd.DataFrame:
    """Forward-fill only from the past and retain explicit observation evidence."""
    output = frame.copy()
    positions = np.arange(len(output), dtype=np.int64)
    for channel in channels:
        observed_column = f"observed_{channel}"
        observed = output[observed_column].fillna(0).astype(bool).to_numpy()
        original = pd.to_numeric(output[channel], errors="coerce")
        filled = original.ffill(limit=maximum_steps)
        imputed = original.isna() & filled.notna()

        last_observed = np.maximum.accumulate(np.where(observed, positions, -(10**9)))
        ages = positions - last_observed
        ages[last_observed < 0] = age_cap_steps + 1
        ages = np.minimum(ages, age_cap_steps + 1)

        output[channel] = filled
        output[observed_column] = observed.astype(np.uint8)
        output[f"imputed_{channel}"] = imputed.astype(np.uint8)
        output[f"age_steps_{channel}"] = ages.astype(np.int16)
    return output


def _eligible_windows(
    audit_path: Path,
    site: str,
    grid_minutes: int,
) -> pd.DataFrame:
    windows = pd.read_csv(audit_path)
    required = {
        "site",
        "grid_minutes",
        "window_id",
        "window_start",
        "window_end",
        "eligible_primary",
    }
    missing = required.difference(windows.columns)
    if missing:
        raise PreprocessingError(f"Availability audit missing columns: {sorted(missing)}")
    mask = (
        (windows["site"] == site)
        & (pd.to_numeric(windows["grid_minutes"]) == grid_minutes)
        & _parse_bool(windows["eligible_primary"])
    )
    selected = windows.loc[mask].copy()
    selected["window_start"] = pd.to_datetime(selected["window_start"], utc=True)
    selected["window_end"] = pd.to_datetime(selected["window_end"], utc=True)
    return selected.sort_values("window_start").reset_index(drop=True)


def _extract_windows(
    harmonized: pd.DataFrame,
    windows: pd.DataFrame,
    site: str,
    protocol: str,
    grid_minutes: int,
    channels: list[str],
    maximum_fill_steps: int,
    age_cap_steps: int,
    post_qc_threshold: float,
    quality_gate_channels: list[str],
    sparse_channels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []
    frequency = f"{grid_minutes}min"

    for row in windows.itertuples(index=False):
        start = pd.Timestamp(row.window_start)
        end = pd.Timestamp(row.window_end)
        index = pd.date_range(start, end, freq=frequency, inclusive="left", tz="UTC")
        window = harmonized.reindex(index)
        for channel in channels:
            observed_column = f"observed_{channel}"
            window[observed_column] = window[observed_column].fillna(0).astype(np.uint8)
        gate_observed = np.logical_and.reduce(
            [
                window[f"observed_{channel}"].astype(bool).to_numpy()
                for channel in quality_gate_channels
            ]
        )
        post_qc_coverage = float(gate_observed.mean()) if len(window) else 0.0
        accepted = post_qc_coverage >= post_qc_threshold
        key = f"{protocol}__{site}__w{int(row.window_id):04d}"
        registry_row = {
            "protocol": protocol,
            "site": site,
            "grid_minutes": grid_minutes,
            "source_window_id": int(row.window_id),
            "window_key": key,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "rows": len(window),
            "post_qc_gate_coverage": post_qc_coverage,
            "quality_gate_channels": "|".join(quality_gate_channels),
            "sparse_channels": "|".join(sparse_channels),
            "accepted_post_qc": accepted,
        }
        for channel in channels:
            registry_row[f"coverage_{channel}"] = float(window[f"observed_{channel}"].mean())
        registry.append(registry_row)
        if not accepted:
            continue
        window = causal_fill(
            window, channels, maximum_steps=maximum_fill_steps, age_cap_steps=age_cap_steps
        )
        window.insert(0, "timestamp", window.index)
        window.insert(0, "window_key", key)
        window.insert(0, "site", site)
        window.insert(0, "protocol", protocol)
        datasets.append(window.reset_index(drop=True))

    dataset = pd.concat(datasets, ignore_index=True) if datasets else pd.DataFrame()
    return dataset, pd.DataFrame(registry)


def assign_splits(
    registry: pd.DataFrame,
    development_site: str,
    external_sites: list[str],
    split_config: dict[str, Any],
) -> pd.DataFrame:
    """Assign windows chronologically; external sites are immutable test-only data."""
    result = registry.copy()
    result["split"] = "rejected_post_qc"
    accepted = result["accepted_post_qc"].astype(bool)
    result.loc[accepted & result["site"].isin(external_sites), "split"] = "external_test"

    dev_indices = result.index[accepted & (result["site"] == development_site)].tolist()
    dev_indices.sort(key=lambda index: pd.Timestamp(result.loc[index, "window_start"]))
    embargo = int(split_config["embargo_windows"])
    usable = len(dev_indices) - 2 * embargo
    if usable < 5:
        raise PreprocessingError(
            f"Too few accepted development windows after embargo: {len(dev_indices)}"
        )
    train_count = max(1, int(np.floor(usable * float(split_config["train_fraction"]))))
    validation_count = max(1, int(np.floor(usable * float(split_config["validation_fraction"]))))
    internal_count = usable - train_count - validation_count
    if internal_count < 1:
        raise PreprocessingError("Split fractions leave no internal-test windows")

    cursor = 0
    train_indices = dev_indices[cursor : cursor + train_count]
    cursor += train_count
    embargo_one = dev_indices[cursor : cursor + embargo]
    cursor += embargo
    validation_indices = dev_indices[cursor : cursor + validation_count]
    cursor += validation_count
    embargo_two = dev_indices[cursor : cursor + embargo]
    cursor += embargo
    test_indices = dev_indices[cursor:]

    result.loc[train_indices, "split"] = "train"
    result.loc[embargo_one, "split"] = "embargo_train_validation"
    result.loc[validation_indices, "split"] = "validation"
    result.loc[embargo_two, "split"] = "embargo_validation_test"
    result.loc[test_indices, "split"] = "internal_test"
    return result


def _validate_split_registry(
    registry: pd.DataFrame, development_site: str, external_sites: list[str]
) -> None:
    if registry["window_key"].duplicated().any():
        raise PreprocessingError("Duplicate window keys in split registry")
    forbidden = registry[
        registry["site"].isin(external_sites)
        & registry["split"].isin({"train", "validation", "internal_test"})
    ]
    if not forbidden.empty:
        raise PreprocessingError("External-site leakage into development splits")
    internal = registry[registry["site"] == development_site]
    ordered = internal.sort_values("window_start")
    rank = {
        "train": 0,
        "embargo_train_validation": 1,
        "validation": 2,
        "embargo_validation_test": 3,
        "internal_test": 4,
        "rejected_post_qc": -1,
    }
    sequence = [rank.get(value, -1) for value in ordered["split"] if rank.get(value, -1) >= 0]
    if sequence != sorted(sequence):
        raise PreprocessingError("Development splits are not chronologically ordered")


def preprocess_ooi(
    raw_root: Path,
    audit_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute all locked Task G protocols and write immutable artifacts."""
    channels = [ChannelSpec(name=name, **values) for name, values in config["channels"].items()]
    channel_names = [spec.name for spec in channels]
    accepted_flags = {int(value) for value in config["accepted_qartod_flags"]}
    availability_path = audit_root / "availability_window_audit.csv"
    if not availability_path.exists():
        raise PreprocessingError(f"Missing Task F availability audit: {availability_path}")
    verified_inputs = verify_task_f_inputs(raw_root, audit_root)

    output_root.mkdir(parents=True, exist_ok=True)
    all_registries: list[pd.DataFrame] = []
    qc_reports: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []

    for protocol_name, protocol in config["protocols"].items():
        grid = int(protocol["grid_minutes"])
        development_site = str(protocol["development_site"])
        external_sites = [str(site) for site in protocol["external_sites"]]
        sites = [development_site, *external_sites]
        protocol_datasets: dict[str, pd.DataFrame] = {}
        protocol_registries: list[pd.DataFrame] = []

        for site in sites:
            override = protocol.get("site_overrides", {}).get(site, {})
            gate_channels = [
                str(channel)
                for channel in override.get(
                    "quality_gate_channels", protocol["quality_gate_channels"]
                )
            ]
            sparse_channels = [str(channel) for channel in override.get("sparse_channels", [])]
            unknown_channels = set(gate_channels + sparse_channels).difference(channel_names)
            if unknown_channels:
                raise PreprocessingError(
                    f"Unknown protocol channels for {protocol_name}/{site}: "
                    f"{sorted(unknown_channels)}"
                )
            harmonized, qc_report = harmonize_site(raw_root, site, grid, channels, accepted_flags)
            windows = _eligible_windows(availability_path, site, grid)
            if windows.empty:
                raise PreprocessingError(
                    f"No Task F eligible windows for {protocol_name}/{site}/{grid}min"
                )
            dataset, registry = _extract_windows(
                harmonized,
                windows,
                site,
                protocol_name,
                grid,
                channel_names,
                int(config["maximum_causal_fill_steps"]),
                int(config["age_cap_steps"]),
                float(config["availability_threshold"]),
                gate_channels,
                sparse_channels,
            )
            protocol_datasets[site] = dataset
            protocol_registries.append(registry)
            qc_reports[f"{protocol_name}/{site}"] = qc_report

        protocol_registry = pd.concat(protocol_registries, ignore_index=True)
        protocol_registry = assign_splits(
            protocol_registry, development_site, external_sites, config["split"]
        )
        _validate_split_registry(protocol_registry, development_site, external_sites)
        split_lookup = protocol_registry.set_index("window_key")["split"]

        for site, dataset in protocol_datasets.items():
            if dataset.empty:
                raise PreprocessingError(f"No post-QC rows for {protocol_name}/{site}")
            dataset["split"] = dataset["window_key"].map(split_lookup)
            dataset = dataset[~dataset["split"].str.startswith("embargo")].copy()
            path = output_root / protocol_name / f"{site}.parquet"
            _atomic_parquet(path, dataset)
            artifacts.append(
                {
                    "protocol": protocol_name,
                    "site": site,
                    "path": str(path.relative_to(output_root.parent.parent)),
                    "rows": len(dataset),
                    "sha256": sha256_file(path),
                }
            )
        all_registries.append(protocol_registry)

    registry = pd.concat(all_registries, ignore_index=True)
    registry_path = output_root / "window_split_registry.csv"
    _atomic_csv(registry_path, registry)
    qc_path = output_root / "qartod_retention.json"
    _atomic_json(qc_path, qc_reports)

    split_counts = (
        registry.groupby(["protocol", "site", "split"]).size().rename("windows").reset_index()
    )
    split_counts_path = output_root / "split_counts.csv"
    _atomic_csv(split_counts_path, split_counts)
    artifacts.extend(
        [
            {
                "path": str(registry_path.relative_to(output_root.parent.parent)),
                "sha256": sha256_file(registry_path),
            },
            {
                "path": str(qc_path.relative_to(output_root.parent.parent)),
                "sha256": sha256_file(qc_path),
            },
            {
                "path": str(split_counts_path.relative_to(output_root.parent.parent)),
                "sha256": sha256_file(split_counts_path),
            },
        ]
    )
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "G",
        "preprocessing_version": "1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "accepted_qartod_flags": sorted(accepted_flags),
        "channels": channel_names,
        "protocols": list(config["protocols"]),
        "protocol_roles": {
            name: {
                "grid_minutes": values["grid_minutes"],
                "development_site": values["development_site"],
                "external_sites": values["external_sites"],
                "purpose": values["purpose"],
                "site_overrides": values.get("site_overrides", {}),
            }
            for name, values in config["protocols"].items()
        },
        "scalers_fitted": False,
        "measurement_values_used_for_availability_selection": False,
        "post_qc_gate_uses_qartod_finiteness_and_physical_validity": True,
        "raw_files_reverified_against_task_f": verified_inputs,
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "preprocessing_summary.json", summary)
    return summary
