"""Task K matched acoustic-network realization and transport simulator.

The simulator stores oracle state and per-attempt random numbers for exact replay, but
those columns are explicitly evaluation-only and may never be used by a receiver model.
"""

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


class NetworkSimulationError(RuntimeError):
    """Raised when Task K cannot preserve its matched-realization contract."""


STATES = ("good", "degraded", "outage")


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


def _resolve_task_j_artifact(record: dict[str, Any], root: Path) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    protocol = record.get("protocol")
    site = record.get("site")
    if protocol and site:
        return root / "evidence" / str(protocol) / f"{site}.parquet"
    return root / declared.name


def verify_task_j_inputs(root: Path) -> tuple[dict[str, Any], int]:
    """Reverify frozen Task J artifacts, including after project relocation."""
    summary_path = root / "task_j_summary.json"
    if not summary_path.exists():
        raise NetworkSimulationError(f"Missing Task J summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != "J" or summary.get("semantic_version") != "1.0":
        raise NetworkSimulationError("Task K requires the frozen Task J contract v1.0")
    verified = 0
    for record in summary.get("artifacts", []):
        path = _resolve_task_j_artifact(record, root)
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise NetworkSimulationError(f"Task J artifact changed or is missing: {path}")
        verified += 1
    if verified == 0:
        raise NetworkSimulationError("Task J summary contains no verifiable artifacts")
    return summary, verified


def _derived_seed(base_seed: int, *parts: object) -> int:
    material = "|".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63 - 1)


def _draw_dwell(rng: np.random.Generator, mean: float, shape: float) -> int:
    if mean <= 0 or shape <= 0:
        raise NetworkSimulationError("Semi-Markov dwell mean and shape must be positive")
    return max(1, int(round(rng.gamma(shape=shape, scale=mean / shape))))


def generate_oracle_realization(
    rows: int,
    scenario: dict[str, Any],
    rng: np.random.Generator,
    maximum_attempts: int,
    critical: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate hidden state and counterfactual random draws for one replayable run."""
    if rows < 1 or maximum_attempts < 1:
        raise NetworkSimulationError("A realization requires rows and attempts")
    state_names = np.empty(rows, dtype=object)
    state = str(scenario["start_state"])
    if state not in STATES:
        raise NetworkSimulationError(f"Unknown initial acoustic state: {state}")
    cursor = 0
    while cursor < rows:
        dwell = _draw_dwell(
            rng,
            float(scenario["dwell_mean_steps"][state]),
            float(scenario["dwell_shape"][state]),
        )
        stop = min(rows, cursor + dwell)
        state_names[cursor:stop] = state
        cursor = stop
        if cursor >= rows:
            break
        transitions = scenario["transition"][state]
        candidates = list(transitions)
        probabilities = np.asarray([float(transitions[name]) for name in candidates], dtype=float)
        if not np.isclose(probabilities.sum(), 1.0):
            raise NetworkSimulationError(f"Transition probabilities from {state} do not sum to one")
        state = str(rng.choice(candidates, p=probabilities))

    if "forced_partition_fraction" in scenario:
        duration = max(1, int(round(rows * float(scenario["forced_partition_fraction"]))))
        start = int(rng.integers(0, max(1, rows - duration + 1)))
        state_names[start : start + duration] = "outage"

    if bool(scenario.get("event_coupled", False)):
        if critical is None or len(critical) != rows:
            raise NetworkSimulationError("Event-coupled stress requires aligned critical truth")
        force = rng.random(rows) < float(scenario["critical_outage_probability"])
        state_names[np.asarray(critical, dtype=bool) & force] = "outage"

    output: dict[str, Any] = {"_oracle_link_state": state_names}
    for attempt in range(maximum_attempts):
        output[f"_oracle_data_u{attempt}"] = rng.random(rows).astype(np.float32)
        output[f"_oracle_ack_u{attempt}"] = rng.random(rows).astype(np.float32)
        output[f"_oracle_jitter_u{attempt}"] = rng.random(rows).astype(np.float32)
    return pd.DataFrame(output)


def apply_transport(
    evidence: pd.DataFrame,
    oracle: pd.DataFrame,
    scenario: dict[str, Any],
    config: dict[str, Any],
    grid_minutes: int,
) -> pd.DataFrame:
    """Replay Task J packets through a fixed oracle realization."""
    if len(evidence) != len(oracle):
        raise NetworkSimulationError("Evidence and oracle realization lengths differ")
    required = {
        "timestamp",
        "window_key",
        "model_role",
        "node_action",
        "semantic_payload_bytes",
        "causal_critical_evidence",
    }
    missing = required.difference(evidence.columns)
    if missing:
        raise NetworkSimulationError(f"Task J network input lacks: {sorted(missing)}")

    rows = len(evidence)
    maximum_attempts = int(config["maximum_attempts"])
    hops = int(config["hops"])
    bitrate = float(config["bitrate_bps"])
    header = int(config["packet_header_bytes"])
    ack_bytes = int(config["ack_bytes"])
    propagation = (
        hops * float(config["nominal_distance_per_hop_m"]) / float(config["propagation_speed_m_s"])
    )
    retry_backoff = float(config["retry_backoff_seconds"])
    payload = pd.to_numeric(evidence["semantic_payload_bytes"], errors="raise").to_numpy(int)
    sent = evidence["node_action"].to_numpy() != "silent"
    states = oracle["_oracle_link_state"].astype(str).to_numpy()

    attempts = np.zeros(rows, dtype=np.int8)
    data_successes = np.zeros(rows, dtype=np.int8)
    delivered = np.zeros(rows, dtype=bool)
    acknowledged = np.zeros(rows, dtype=bool)
    first_delivery_delay = np.full(rows, np.nan, dtype=np.float32)
    channel_bytes = np.zeros(rows, dtype=np.int32)
    energy_j = np.zeros(rows, dtype=np.float32)
    active = sent.copy()

    for attempt in range(maximum_attempts):
        attempted = active.copy()
        attempts[attempted] += 1
        packet_wire_bytes = payload + header
        data_airtime = packet_wire_bytes * 8.0 / bitrate
        channel_bytes[attempted] += packet_wire_bytes[attempted]
        energy_j[attempted] += (
            hops
            * data_airtime[attempted]
            * (float(config["energy"]["tx_watts"]) + float(config["energy"]["rx_watts"]))
        ).astype(np.float32)

        per_link_data = np.asarray([float(scenario["packet_success"][state]) for state in states])
        p_data = np.power(per_link_data, hops)
        data_ok = attempted & (oracle[f"_oracle_data_u{attempt}"].to_numpy(float) < p_data)
        data_successes[data_ok] += 1
        newly_delivered = data_ok & ~delivered
        jitter = oracle[f"_oracle_jitter_u{attempt}"].to_numpy(float)
        attempt_delay = (
            (attempt + 1) * propagation
            + (attempt + 1) * data_airtime
            + attempt * retry_backoff
            + jitter * 0.25 * (propagation + data_airtime)
        )
        first_delivery_delay[newly_delivered] = attempt_delay[newly_delivered].astype(np.float32)
        delivered |= data_ok

        ack_wire_bytes = ack_bytes + header
        ack_airtime = ack_wire_bytes * 8.0 / bitrate
        channel_bytes[data_ok] += ack_wire_bytes
        energy_j[data_ok] += np.float32(
            hops
            * ack_airtime
            * (float(config["energy"]["tx_watts"]) + float(config["energy"]["rx_watts"]))
        )
        per_link_ack = np.asarray([float(scenario["ack_success"][state]) for state in states])
        p_ack = np.power(per_link_ack, hops)
        ack_ok = data_ok & (oracle[f"_oracle_ack_u{attempt}"].to_numpy(float) < p_ack)
        acknowledged |= ack_ok
        active = attempted & ~ack_ok

    deadline_seconds = float(grid_minutes * 60)
    timely = delivered & (first_delivery_delay <= deadline_seconds)
    duplicate_deliveries = np.maximum(data_successes.astype(int) - delivered.astype(int), 0)
    cause = np.full(rows, "intentional_silence", dtype=object)
    cause[sent & ~delivered] = "acoustic_retry_exhausted"
    cause[delivered & ~acknowledged] = "ack_loss_after_delivery"
    cause[acknowledged] = "none"

    keep = [
        "timestamp",
        "window_key",
        "model_role",
        "node_action",
        "decision_reason",
        "semantic_payload_bytes",
        "validity_horizon_steps",
        "silence_contract_id",
        "contract_age_steps",
        "causal_critical_evidence",
        "semantic_risk",
    ]
    result = evidence[[column for column in keep if column in evidence.columns]].reset_index(
        drop=True
    )
    result = pd.concat([result, oracle.reset_index(drop=True)], axis=1)
    result["packet_attempted"] = sent.astype(np.uint8)
    result["transmission_attempts"] = attempts
    result["packet_delivered"] = delivered.astype(np.uint8)
    result["packet_delivered_timely"] = timely.astype(np.uint8)
    result["acknowledged"] = acknowledged.astype(np.uint8)
    result["duplicate_deliveries"] = duplicate_deliveries.astype(np.int8)
    result["first_delivery_delay_seconds"] = first_delivery_delay
    result["network_bytes"] = channel_bytes
    result["incremental_communication_energy_j"] = energy_j
    result["_oracle_immediate_outcome"] = cause
    return result


def _transport_metrics(
    frame: pd.DataFrame,
    protocol: str,
    site: str,
    scenario_name: str,
    scenario_role: str,
    repetition: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    sent = frame["packet_attempted"].to_numpy(bool)
    timely = frame["packet_delivered_timely"].to_numpy(bool)
    critical = frame["causal_critical_evidence"].to_numpy(bool)
    acked = frame["acknowledged"].to_numpy(bool)
    delay = frame["first_delivery_delay_seconds"].to_numpy(float)
    periodic_wire = len(frame) * (int(config["packet_header_bytes"]) + 24)
    return {
        "protocol": protocol,
        "site": site,
        "model_role": str(frame["model_role"].iloc[0]),
        "scenario": scenario_name,
        "scenario_role": scenario_role,
        "repetition": repetition,
        "rows": int(len(frame)),
        "attempted_packets": int(sent.sum()),
        "timely_delivered_packets": int(timely.sum()),
        "offered_transmission_fraction": float(sent.mean()),
        "packet_delivery_ratio_given_attempt": float(timely[sent].mean()) if sent.any() else np.nan,
        "acknowledgment_ratio_given_attempt": float(acked[sent].mean()) if sent.any() else np.nan,
        "available_critical_rows": int(critical.sum()),
        "network_critical_delivery_recall": (
            float(timely[critical].mean()) if critical.any() else np.nan
        ),
        "network_critical_miss_probability": (
            float((~timely[critical]).mean()) if critical.any() else np.nan
        ),
        "mean_delivered_delay_seconds": (
            float(np.nanmean(delay[timely])) if timely.any() else np.nan
        ),
        "p95_delivered_delay_seconds": (
            float(np.nanquantile(delay[timely], 0.95)) if timely.any() else np.nan
        ),
        "network_bytes": int(frame["network_bytes"].sum()),
        "ideal_periodic_raw_wire_bytes": int(periodic_wire),
        "wire_byte_reduction_fraction": float(1.0 - frame["network_bytes"].sum() / periodic_wire),
        "incremental_communication_energy_j": float(
            frame["incremental_communication_energy_j"].sum()
        ),
        "duplicate_deliveries": int(frame["duplicate_deliveries"].sum()),
        "unacknowledged_deliveries": int(
            ((frame["packet_delivered"] == 1) & (frame["acknowledged"] == 0)).sum()
        ),
    }


def simulate_network(
    input_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate matched network traces and evaluate frozen Task J packet actions."""
    task_j, verified = verify_task_j_inputs(input_root)
    artifacts: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    maximum_attempts = int(config["maximum_attempts"])
    repetitions = int(config["repetitions"])
    roles = set(str(role) for role in config["evaluation_roles"])

    for protocol, protocol_summary in task_j["protocols"].items():
        grid_minutes = int(protocol_summary["grid_minutes"])
        for site in protocol_summary["sites"]:
            source_path = input_root / "evidence" / protocol / f"{site}.parquet"
            evidence = pd.read_parquet(source_path)
            evidence = evidence[evidence["model_role"].isin(roles)].copy()
            if evidence.empty:
                continue
            evidence["timestamp"] = pd.to_datetime(evidence["timestamp"], utc=True)
            for model_role, role_frame in evidence.groupby("model_role", sort=False):
                role_frame = role_frame.sort_values(["window_key", "timestamp"]).reset_index(
                    drop=True
                )
                critical = role_frame["causal_critical_evidence"].to_numpy(bool)
                for scenario_name, scenario in config["scenarios"].items():
                    runs: list[pd.DataFrame] = []
                    for repetition in range(repetitions):
                        seed = _derived_seed(
                            int(config["base_seed"]),
                            protocol,
                            site,
                            model_role,
                            scenario_name,
                            repetition,
                        )
                        oracle = generate_oracle_realization(
                            len(role_frame),
                            scenario,
                            np.random.default_rng(seed),
                            maximum_attempts,
                            critical if scenario.get("event_coupled") else None,
                        )
                        transported = apply_transport(
                            role_frame, oracle, scenario, config, grid_minutes
                        )
                        transported.insert(0, "network_version", str(config["network_version"]))
                        transported.insert(1, "scenario", scenario_name)
                        transported.insert(2, "scenario_role", str(scenario["role"]))
                        transported.insert(3, "repetition", repetition)
                        transported.insert(4, "realization_seed", seed)
                        metrics.append(
                            _transport_metrics(
                                transported,
                                protocol,
                                site,
                                scenario_name,
                                str(scenario["role"]),
                                repetition,
                                config,
                            )
                        )
                        runs.append(transported)
                    combined = pd.concat(runs, ignore_index=True)
                    output_path = (
                        output_root / "transport" / protocol / site / f"{scenario_name}.parquet"
                    )
                    _atomic_parquet(output_path, combined)
                    artifacts.append(
                        {
                            "protocol": protocol,
                            "site": site,
                            "model_role": str(model_role),
                            "scenario": scenario_name,
                            "path": str(output_path),
                            "rows": int(len(combined)),
                            "sha256": sha256_file(output_path),
                        }
                    )

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = output_root / "network_transport_metrics.csv"
    _atomic_csv(metrics_path, metrics_frame)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "K",
        "network_version": str(config["network_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_j_artifacts_reverified": verified,
        "task_j_semantic_version": task_j["semantic_version"],
        "primary_outages_use_event_labels": False,
        "event_coupling_is_declared_stress_only": True,
        "matched_counterfactual_randomness_stored": bool(config["store_counterfactual_randomness"]),
        "hidden_oracle_state_is_receiver_input": False,
        "ack_loss_and_duplicates_are_explicit": True,
        "energy_is_modeled_not_hardware_measured": True,
        "energy_accounting": str(config["energy"]["accounting"]),
        "repetitions": repetitions,
        "scenarios": {
            name: {"role": value["role"], "event_coupled": bool(value["event_coupled"])}
            for name, value in config["scenarios"].items()
        },
        "metrics_rows": int(len(metrics_frame)),
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "task_k_summary.json", summary)
    return summary
