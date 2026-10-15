"""Task L receiver-side dual belief and bounded-silence interpretation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_silence.data.preprocess import sha256_file


class ReceiverInferenceError(RuntimeError):
    """Raised when Task L cannot preserve its receiver-evidence contract."""


PROTECTIVE_STATES = {
    "ambiguous",
    "failure_suspected",
    "recovery_pending",
    "stale_unsafe",
}


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


def _resolve_task_k_artifact(record: dict[str, Any], root: Path) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    if all(record.get(key) for key in ("protocol", "site", "scenario")):
        return (
            root
            / "transport"
            / str(record["protocol"])
            / str(record["site"])
            / f"{record['scenario']}.parquet"
        )
    return root / declared.name


def verify_task_k_inputs(root: Path) -> tuple[dict[str, Any], int]:
    """Reverify all frozen Task K artifacts, including after project relocation."""
    summary_path = root / "task_k_summary.json"
    if not summary_path.exists():
        raise ReceiverInferenceError(f"Missing Task K summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != "K" or summary.get("network_version") != "1.0":
        raise ReceiverInferenceError("Task L requires the frozen Task K network contract v1.0")
    verified = 0
    for record in summary.get("artifacts", []):
        path = _resolve_task_k_artifact(record, root)
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise ReceiverInferenceError(f"Task K artifact changed or is missing: {path}")
        verified += 1
    if verified == 0:
        raise ReceiverInferenceError("Task K summary contains no verifiable artifacts")
    return summary, verified


def make_receiver_observations(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Mask source-side fields unless their packet was timely delivered."""
    required = {
        "timestamp",
        "window_key",
        "repetition",
        "packet_delivered_timely",
        "node_action",
        "semantic_risk",
        "validity_horizon_steps",
        "silence_contract_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ReceiverInferenceError(f"Task K transport lacks fields: {sorted(missing)}")
    delivered = frame["packet_delivered_timely"].fillna(0).to_numpy(bool)
    result = frame[["timestamp", "window_key", "repetition"]].copy()
    result["packet_arrived"] = delivered.astype(np.uint8)
    result["received_action"] = np.where(delivered, frame["node_action"], None)
    result["received_semantic_risk"] = np.where(
        delivered, pd.to_numeric(frame["semantic_risk"], errors="coerce"), np.nan
    )
    result["received_validity_horizon_steps"] = np.where(
        delivered,
        pd.to_numeric(frame["validity_horizon_steps"], errors="coerce"),
        np.nan,
    )
    result["received_contract_id"] = np.where(
        delivered,
        pd.to_numeric(frame["silence_contract_id"], errors="coerce"),
        np.nan,
    )
    allowed = set(config["receiver_observable_packet_fields"])
    expected = {
        "node_action",
        "semantic_risk",
        "validity_horizon_steps",
        "silence_contract_id",
    }
    if allowed != expected:
        raise ReceiverInferenceError("Receiver-observable packet-field contract changed")
    return result


def _posterior(prior: float, likelihood_reachable: float, likelihood_unreachable: float) -> float:
    numerator = prior * likelihood_reachable
    denominator = numerator + (1.0 - prior) * likelihood_unreachable
    if denominator <= 0:
        raise ReceiverInferenceError("Connectivity posterior has zero evidence mass")
    return float(numerator / denominator)


def _environmental_uncertainty(last_risk: float, age: int, growth: float) -> float:
    risk = float(np.clip(last_risk, 0.0, 1.0))
    return float(1.0 - (1.0 - risk) * math.exp(-growth * max(age, 0)))


def interpret_receiver_group(observations: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Interpret one chronological window without hidden state, truth, or future data."""
    forbidden_prefixes = tuple(str(value) for value in config["forbidden_inference_prefixes"])
    forbidden_fields = set(str(value) for value in config["forbidden_inference_fields"])
    forbidden_present = [
        column
        for column in observations.columns
        if column in forbidden_fields or column.startswith(forbidden_prefixes)
    ]
    if forbidden_present:
        raise ReceiverInferenceError(
            f"Forbidden receiver evidence supplied: {sorted(forbidden_present)}"
        )
    required = {
        "timestamp",
        "packet_arrived",
        "received_action",
        "received_semantic_risk",
        "received_validity_horizon_steps",
        "received_contract_id",
    }
    missing = required.difference(observations.columns)
    if missing:
        raise ReceiverInferenceError(f"Receiver observation lacks: {sorted(missing)}")

    frame = observations.sort_values("timestamp").reset_index(drop=True).copy()
    filter_config = config["connectivity_filter"]
    state_config = config["state_policy"]
    growth = float(config["environmental_belief"]["age_growth_per_step"])
    connectivity = float(filter_config["prior_reachable"])
    last_received_step: int | None = None
    last_risk = float(config["environmental_belief"]["missing_contract_uncertainty"])
    last_horizon = 0
    active_contract: int | None = None

    states: list[str] = []
    connectivity_beliefs: list[float] = []
    environmental_beliefs: list[float] = []
    contract_ages: list[int] = []
    overdue_steps_values: list[int] = []
    valid_contract_values: list[int] = []
    contact_due_values: list[int] = []
    active_contract_values: list[float] = []

    for step, row in enumerate(frame.itertuples(index=False)):
        connectivity = connectivity * float(filter_config["reachable_to_reachable"]) + (
            1.0 - connectivity
        ) * float(filter_config["unreachable_to_reachable"])
        arrived = bool(row.packet_arrived)
        if arrived:
            connectivity = _posterior(
                connectivity,
                float(filter_config["packet_arrival_given_reachable"]),
                float(filter_config["packet_arrival_given_unreachable"]),
            )
            if not np.isfinite(float(row.received_semantic_risk)):
                raise ReceiverInferenceError("Delivered packet lacks semantic risk")
            if not np.isfinite(float(row.received_validity_horizon_steps)):
                raise ReceiverInferenceError("Delivered packet lacks validity horizon")
            if not np.isfinite(float(row.received_contract_id)):
                raise ReceiverInferenceError("Delivered packet lacks contract identifier")
            last_received_step = step
            last_risk = float(np.clip(float(row.received_semantic_risk), 0.0, 1.0))
            last_horizon = max(0, int(row.received_validity_horizon_steps))
            active_contract = int(row.received_contract_id)
            age = 0
            overdue = 0
            valid_contract = True
            contact_due = False
            environmental = last_risk
            receiver_state = "updated"
        else:
            if last_received_step is None:
                age = step + 1
                valid_contract = False
                overdue = age
                environmental = float(
                    config["environmental_belief"]["missing_contract_uncertainty"]
                )
            else:
                age = step - last_received_step
                valid_contract = age < last_horizon
                first_due_age = max(last_horizon, 1)
                overdue = max(0, age - first_due_age + 1)
                environmental = _environmental_uncertainty(last_risk, age, growth)
            contact_due = not valid_contract
            if contact_due:
                connectivity = _posterior(
                    connectivity,
                    float(filter_config["due_packet_absence_given_reachable"]),
                    float(filter_config["due_packet_absence_given_unreachable"]),
                )

            if valid_contract:
                trusted = connectivity >= float(
                    state_config["trust_connectivity_threshold"]
                ) and environmental <= float(
                    state_config["trusted_environmental_uncertainty_maximum"]
                )
                if trusted:
                    receiver_state = "trusted_silence"
                elif connectivity < float(state_config["failure_suspected_threshold"]):
                    receiver_state = "failure_suspected"
                else:
                    receiver_state = "ambiguous"
            elif overdue >= int(state_config["stale_unsafe_after_overdue_steps"]):
                receiver_state = "stale_unsafe"
            elif connectivity < float(state_config["failure_suspected_threshold"]):
                receiver_state = "failure_suspected"
            else:
                receiver_state = "ambiguous"

        states.append(receiver_state)
        connectivity_beliefs.append(float(np.clip(connectivity, 0.0, 1.0)))
        environmental_beliefs.append(float(np.clip(environmental, 0.0, 1.0)))
        contract_ages.append(age)
        overdue_steps_values.append(overdue)
        valid_contract_values.append(int(valid_contract))
        contact_due_values.append(int(contact_due))
        active_contract_values.append(
            float(active_contract) if active_contract is not None else np.nan
        )

    frame["receiver_state"] = states
    frame["connectivity_belief"] = np.asarray(connectivity_beliefs, dtype=np.float32)
    frame["environmental_uncertainty"] = np.asarray(environmental_beliefs, dtype=np.float32)
    frame["receiver_contract_age_steps"] = np.asarray(contract_ages, dtype=np.int32)
    frame["receiver_overdue_steps"] = np.asarray(overdue_steps_values, dtype=np.int32)
    frame["receiver_contract_valid"] = np.asarray(valid_contract_values, dtype=np.uint8)
    frame["receiver_contact_due"] = np.asarray(contact_due_values, dtype=np.uint8)
    frame["receiver_active_contract_id"] = active_contract_values
    return frame


def interpret_receiver(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply the fixed receiver independently to each repetition and immutable window."""
    observations = make_receiver_observations(frame, config)
    pieces: list[pd.DataFrame] = []
    for _, group in observations.groupby(["repetition", "window_key"], sort=False):
        pieces.append(interpret_receiver_group(group, config))
    receiver = pd.concat(pieces, ignore_index=True)
    keys = ["repetition", "window_key", "timestamp"]
    truth = frame.copy()
    truth["timestamp"] = pd.to_datetime(truth["timestamp"], utc=True)
    receiver["timestamp"] = pd.to_datetime(receiver["timestamp"], utc=True)
    truth_columns = keys + [
        "node_action",
        "causal_critical_evidence",
        "packet_delivered_timely",
        "_oracle_link_state",
    ]
    if set(truth_columns).difference(truth.columns):
        raise ReceiverInferenceError("Task K lacks evaluation-only receiver truth")
    result = receiver.merge(truth[truth_columns], on=keys, how="left", validate="one_to_one")
    result = result.rename(
        columns={
            "node_action": "evaluation_only_source_action",
            "causal_critical_evidence": "evaluation_only_critical_event",
            "packet_delivered_timely": "evaluation_only_packet_delivered_timely",
            "_oracle_link_state": "evaluation_only_link_state",
        }
    )
    no_arrival = ~result["packet_arrived"].to_numpy(bool)
    source_non_silent = result["evaluation_only_source_action"].to_numpy() != "silent"
    contract_invalid = ~result["receiver_contract_valid"].to_numpy(bool)
    unsafe = no_arrival & (source_non_silent | contract_invalid)
    trusted = result["receiver_state"].to_numpy() == "trusted_silence"
    result["evaluation_only_unsafe_silence"] = unsafe.astype(np.uint8)
    result["evaluation_only_false_trust"] = (unsafe & trusted).astype(np.uint8)
    result["evaluation_only_reachable_truth"] = (
        result["evaluation_only_link_state"].to_numpy() != "outage"
    ).astype(np.uint8)
    return result


def _ece(probability: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probability)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1.0:
            selected = (probability >= lower) & (probability <= upper)
        else:
            selected = (probability >= lower) & (probability < upper)
        if selected.any():
            error += selected.mean() * abs(probability[selected].mean() - truth[selected].mean())
    return float(error if total else np.nan)


def _episode_delays(unsafe: np.ndarray, states: np.ndarray, target: set[str]) -> list[float]:
    delays: list[float] = []
    index = 0
    while index < len(unsafe):
        if not unsafe[index]:
            index += 1
            continue
        stop = index + 1
        while stop < len(unsafe) and unsafe[stop]:
            stop += 1
        hit = next((offset for offset in range(index, stop) if states[offset] in target), None)
        delays.append(float(hit - index) if hit is not None else np.nan)
        index = stop
    return delays


def receiver_metrics(
    frame: pd.DataFrame,
    protocol: str,
    site: str,
    model_role: str,
    scenario: str,
    repetition: int,
) -> dict[str, Any]:
    critical = frame["evaluation_only_critical_event"].to_numpy(bool)
    unsafe = frame["evaluation_only_unsafe_silence"].to_numpy(bool)
    false_trust = frame["evaluation_only_false_trust"].to_numpy(bool)
    states = frame["receiver_state"].to_numpy(str)
    trusted = states == "trusted_silence"
    protective = np.isin(states, list(PROTECTIVE_STATES))
    arrived = frame["packet_arrived"].to_numpy(bool)
    connectivity = frame["connectivity_belief"].to_numpy(float)
    reachable = frame["evaluation_only_reachable_truth"].to_numpy(float)
    ambiguity_delays = _episode_delays(unsafe, states, PROTECTIVE_STATES)
    stale_delays = _episode_delays(unsafe, states, {"stale_unsafe"})
    ambiguity_array = np.asarray(ambiguity_delays, dtype=float)
    stale_array = np.asarray(stale_delays, dtype=float)
    return {
        "protocol": protocol,
        "site": site,
        "model_role": model_role,
        "scenario": scenario,
        "repetition": repetition,
        "rows": int(len(frame)),
        "available_critical_rows": int(critical.sum()),
        "critical_false_trust_count": int((critical & false_trust).sum()),
        "critical_false_trust_rate": (
            float(false_trust[critical].mean()) if critical.any() else np.nan
        ),
        "overall_false_trust_rate": float(false_trust[unsafe].mean()) if unsafe.any() else 0.0,
        "unsafe_silence_rows": int(unsafe.sum()),
        "unsafe_silence_detection_recall": (
            float(protective[unsafe].mean()) if unsafe.any() else np.nan
        ),
        "critical_protective_state_recall": (
            float(protective[critical].mean()) if critical.any() else np.nan
        ),
        "critical_safe_handling_recall": (
            float((arrived | protective)[critical].mean()) if critical.any() else np.nan
        ),
        "trusted_silence_fraction": float(trusted.mean()),
        "ambiguous_fraction": float((states == "ambiguous").mean()),
        "failure_suspected_fraction": float((states == "failure_suspected").mean()),
        "stale_unsafe_fraction": float((states == "stale_unsafe").mean()),
        "connectivity_brier_score": float(np.mean((connectivity - reachable) ** 2)),
        "connectivity_ece": _ece(connectivity, reachable),
        "unsafe_episodes": int(len(ambiguity_array)),
        "ambiguity_episode_detection_fraction": (
            float(np.isfinite(ambiguity_array).mean()) if len(ambiguity_array) else np.nan
        ),
        "mean_time_to_ambiguity_steps": (
            float(np.nanmean(ambiguity_array)) if np.isfinite(ambiguity_array).any() else np.nan
        ),
        "stale_unsafe_episode_fraction": (
            float(np.isfinite(stale_array).mean()) if len(stale_array) else np.nan
        ),
        "mean_time_to_stale_unsafe_steps": (
            float(np.nanmean(stale_array)) if np.isfinite(stale_array).any() else np.nan
        ),
    }


def build_receiver_beliefs(
    input_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute Task L on every frozen Task K transport trace."""
    task_k, verified = verify_task_k_inputs(input_root)
    metrics: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    trace_records = [
        record
        for record in task_k["artifacts"]
        if all(record.get(key) for key in ("protocol", "site", "scenario"))
    ]
    for record in trace_records:
        input_path = _resolve_task_k_artifact(record, input_root)
        transport = pd.read_parquet(input_path)
        protocol = str(record["protocol"])
        site = str(record["site"])
        scenario = str(record["scenario"])
        model_role = str(record["model_role"])
        interpreted = interpret_receiver(transport, config)
        interpreted.insert(0, "receiver_version", str(config["receiver_version"]))
        interpreted.insert(1, "protocol", protocol)
        interpreted.insert(2, "site", site)
        interpreted.insert(3, "model_role", model_role)
        interpreted.insert(4, "scenario", scenario)
        for repetition, group in interpreted.groupby("repetition", sort=False):
            metrics.append(
                receiver_metrics(
                    group,
                    protocol,
                    site,
                    model_role,
                    scenario,
                    int(repetition),
                )
            )
        output_path = output_root / "beliefs" / protocol / site / f"{scenario}.parquet"
        _atomic_parquet(output_path, interpreted)
        artifacts.append(
            {
                "protocol": protocol,
                "site": site,
                "model_role": model_role,
                "scenario": scenario,
                "path": str(output_path),
                "rows": int(len(interpreted)),
                "sha256": sha256_file(output_path),
            }
        )

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = output_root / "receiver_metrics.csv"
    _atomic_csv(metrics_path, metrics_frame)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "L",
        "receiver_version": str(config["receiver_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_k_artifacts_reverified": verified,
        "task_k_network_version": task_k["network_version"],
        "receiver_uses_hidden_oracle_state": False,
        "receiver_uses_event_truth": False,
        "receiver_uses_source_action_when_packet_missing": False,
        "receiver_uses_future_measurements": False,
        "parameters_tuned_on_test_or_external_data": False,
        "algorithm_shared_across_sites_and_scenarios": True,
        "no_normal_fallback_after_contract_expiry": True,
        "unknown_ambiguity_retained": True,
        "primary_endpoint": "critical_false_trust_rate",
        "metrics_rows": int(len(metrics_frame)),
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "task_l_summary.json", summary)
    return summary
