"""Task Q matched-realization relay-to-buoy delivery integration.

Policy decisions are completed before stored Task K oracle draws are attached.  The oracle
columns are used only by the physical replay and post-hoc scoring.
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
from safe_silence.protocol.scheduler import (
    PROPOSED_POLICY,
    REFERENCE_POLICY,
    schedule_opportunity,
    upper_cvar,
)


class IntegrationError(RuntimeError):
    """Raised when Task Q cannot preserve its matched-replay contract."""


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


def _resolve_artifact(record: dict[str, Any], root: Path, task: str) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    if task == "P" and all(record.get(key) for key in ("protocol", "site", "scenario")):
        return (
            root
            / "schedules"
            / str(record["protocol"])
            / str(record["site"])
            / f"{record['scenario']}.parquet"
        )
    if task == "O" and all(record.get(key) for key in ("protocol", "site", "artifact_kind")):
        folder = "membership" if record["artifact_kind"] == "membership" else "summaries"
        return root / folder / str(record["protocol"]) / f"{record['site']}.parquet"
    if task == "K" and all(record.get(key) for key in ("protocol", "site", "scenario")):
        return (
            root
            / "transport"
            / str(record["protocol"])
            / str(record["site"])
            / f"{record['scenario']}.parquet"
        )
    return root / declared.name


def _verify_summary(root: Path, task: str, version_key: str) -> tuple[dict[str, Any], int]:
    summary_path = root / f"task_{task.lower()}_summary.json"
    if not summary_path.exists():
        raise IntegrationError(f"Task Q requires {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != task or summary.get(version_key) != "1.0":
        raise IntegrationError(f"Task Q requires frozen Task {task} contract v1.0")
    verified = 0
    for record in summary.get("artifacts", []):
        path = _resolve_artifact(record, root, task)
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise IntegrationError(f"Task {task} artifact changed or is missing: {path}")
        verified += 1
    if not verified:
        raise IntegrationError(f"Task {task} contains no verifiable artifacts")
    return summary, verified


def verify_inputs(
    scheduler_root: Path, fusion_root: Path, network_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, int]]:
    """Reverify every frozen Task P, O, and K artifact."""
    scheduler, count_p = _verify_summary(scheduler_root, "P", "scheduler_version")
    fusion, count_o = _verify_summary(fusion_root, "O", "fusion_version")
    network, count_k = _verify_summary(network_root, "K", "network_version")
    if not scheduler.get("component_evaluation_only"):
        raise IntegrationError("Task P claim boundary is missing")
    if not network.get("matched_counterfactual_randomness_stored"):
        raise IntegrationError("Task K did not preserve matched counterfactual randomness")
    return scheduler, fusion, network, {"P": count_p, "O": count_o, "K": count_k}


def _membership_metadata(membership: pd.DataFrame) -> pd.DataFrame:
    required = {
        "batch_id",
        "sensing_timestamp",
        "critical_evidence_score",
        "residual_magnitude",
        "retained",
    }
    missing = required.difference(membership.columns)
    if missing:
        raise IntegrationError(f"Task O membership lacks: {sorted(missing)}")
    frame = membership.copy()
    frame["sensing_timestamp"] = pd.to_datetime(frame["sensing_timestamp"], utc=True)
    frame["critical"] = frame["critical_evidence_score"].astype(float) >= 0.5
    frame["retained_critical"] = frame["critical"] & frame["retained"].astype(bool)
    rows: list[dict[str, Any]] = []
    for batch_id, group in frame.groupby("batch_id", sort=False):
        critical = group.loc[group["critical"]]
        rows.append(
            {
                "batch_id": str(batch_id),
                "critical_items": int(group["critical"].sum()),
                "retained_critical_items": int(group["retained_critical"].sum()),
                "earliest_critical_timestamp": (
                    critical["sensing_timestamp"].min() if len(critical) else pd.NaT
                ),
                "residual_magnitude_max": float(group["residual_magnitude"].max()),
            }
        )
    return pd.DataFrame(rows)


def _feasible(
    totals: dict[str, float], candidate: dict[str, Any], scheduler_config: dict[str, Any]
) -> bool:
    budgets = scheduler_config["budgets"]
    queue = scheduler_config["queue_stress"]
    return (
        totals["packets"] + 1 <= int(queue["maximum_selected_packets"])
        and totals["bytes"] + float(candidate["expected_wire_bytes"])
        <= float(budgets["expected_wire_bytes_per_opportunity"]) + 1e-9
        and totals["energy"] + float(candidate["modeled_energy_j"])
        <= float(budgets["modeled_energy_j_per_opportunity"]) + 1e-9
        and totals["airtime"] + float(candidate["expected_airtime_seconds"])
        <= float(budgets["expected_airtime_seconds_per_opportunity"]) + 1e-9
    )


def _baseline_score(frame: pd.DataFrame, policy: str) -> np.ndarray:
    denominator = np.maximum(frame["cost_denominator"].to_numpy(float), 1e-12)
    flush = pd.to_datetime(frame["relay_flush_timestamp"], utc=True)
    earliest = pd.to_datetime(frame["earliest_sensing_timestamp"], utc=True)
    age = np.maximum((flush - earliest).dt.total_seconds().to_numpy(float), 1.0)
    if policy == "fixed_threshold":
        return (
            2.0 * frame["mandatory_critical"].to_numpy(float)
            + frame["mandatory_deadline"].to_numpy(float)
            + 1e-9 / (1.0 + np.arange(len(frame)))
        )
    if policy == "periodic_fifo":
        return 1.0 / (1.0 + np.arange(len(frame)))
    if policy == "age_of_information":
        return age / denominator
    if policy == "age_of_incorrect_information":
        return age * frame["residual_magnitude_max"].to_numpy(float) / denominator
    if policy == "prediction_residual":
        return frame["residual_magnitude_max"].to_numpy(float) / denominator
    if policy == "uncertainty_only":
        return frame["environmental_uncertainty_max"].to_numpy(float) / denominator
    if policy == "semantic_without_link_belief":
        benefit = np.maximum(
            frame["tail_loss_before"].to_numpy(float)
            - frame["tail_loss_after_success"].to_numpy(float),
            0.0,
        )
        return benefit / denominator
    raise IntegrationError(f"Unknown Task Q reference policy: {policy}")


def schedule_reference(
    opportunity: pd.DataFrame, policy: str, scheduler_config: dict[str, Any]
) -> pd.DataFrame:
    """Select a reference policy under exactly the Task P opportunity constraints."""
    if policy == REFERENCE_POLICY:
        return schedule_opportunity(
            opportunity.to_dict("records"), REFERENCE_POLICY, scheduler_config
        )
    frame = opportunity.copy().reset_index(drop=True)
    frame["policy_utility"] = _baseline_score(frame, policy)
    order = sorted(
        range(len(frame)),
        key=lambda index: (
            -float(frame.at[index, "policy_utility"]),
            pd.Timestamp(frame.at[index, "relay_flush_timestamp"]),
            str(frame.at[index, "batch_id"]),
        ),
    )
    totals = {"packets": 0.0, "bytes": 0.0, "energy": 0.0, "airtime": 0.0}
    selected: set[int] = set()
    for index in order:
        candidate = frame.iloc[index].to_dict()
        if _feasible(totals, candidate, scheduler_config):
            selected.add(index)
            totals["packets"] += 1
            totals["bytes"] += float(candidate["expected_wire_bytes"])
            totals["energy"] += float(candidate["modeled_energy_j"])
            totals["airtime"] += float(candidate["expected_airtime_seconds"])
    frame["policy"] = policy
    frame["selected"] = np.asarray(
        [int(index in selected) for index in range(len(frame))], dtype=np.uint8
    )
    frame["selection_reason"] = np.where(
        frame["selected"].astype(bool), "matched_budget_rank", "budget_or_rank_excluded"
    )
    frame["opportunity_selected_packets"] = int(totals["packets"])
    frame["opportunity_expected_wire_bytes"] = totals["bytes"]
    frame["opportunity_modeled_energy_j"] = totals["energy"]
    frame["opportunity_expected_airtime_seconds"] = totals["airtime"]
    return frame


def replay_selected(
    frame: pd.DataFrame,
    scenario: dict[str, Any],
    network_config: dict[str, Any],
    hops: int,
    deadline_seconds: float,
) -> pd.DataFrame:
    """Replay selected fused packets with stored Task K random draws."""
    if hops != 1:
        raise IntegrationError("Task Q currently defines the relay-to-buoy path as one hop")
    required = {"selected", "encoded_wire_bytes", "_oracle_link_state"}
    attempts_limit = int(network_config["maximum_attempts"])
    required.update(
        f"_oracle_{kind}_u{attempt}"
        for attempt in range(attempts_limit)
        for kind in ("data", "ack", "jitter")
    )
    missing = required.difference(frame.columns)
    if missing:
        raise IntegrationError(f"Matched replay lacks: {sorted(missing)}")

    output = frame.copy()
    rows = len(output)
    selected = output["selected"].to_numpy(bool)
    states = output["_oracle_link_state"].astype(str).to_numpy()
    wire = output["encoded_wire_bytes"].to_numpy(int)
    delivered = np.zeros(rows, dtype=bool)
    acknowledged = np.zeros(rows, dtype=bool)
    attempts = np.zeros(rows, dtype=np.int8)
    data_successes = np.zeros(rows, dtype=np.int8)
    first_delay = np.full(rows, np.nan, dtype=np.float32)
    actual_bytes = np.zeros(rows, dtype=np.int32)
    actual_energy = np.zeros(rows, dtype=np.float32)
    active = selected.copy()
    bitrate = float(network_config["bitrate_bps"])
    propagation = (
        hops
        * float(network_config["nominal_distance_per_hop_m"])
        / float(network_config["propagation_speed_m_s"])
    )
    power = float(network_config["energy"]["tx_watts"]) + float(
        network_config["energy"]["rx_watts"]
    )
    retry_backoff = float(network_config["retry_backoff_seconds"])
    ack_wire = int(network_config["packet_header_bytes"]) + int(network_config["ack_bytes"])
    p_data = np.asarray([float(scenario["packet_success"][state]) for state in states])
    p_ack = np.asarray([float(scenario["ack_success"][state]) for state in states])

    for attempt in range(attempts_limit):
        attempted = active.copy()
        attempts[attempted] += 1
        actual_bytes[attempted] += wire[attempted]
        data_airtime = wire * 8.0 / bitrate
        actual_energy[attempted] += (hops * data_airtime[attempted] * power).astype(np.float32)
        data_ok = attempted & (
            output[f"_oracle_data_u{attempt}"].to_numpy(float) < np.power(p_data, hops)
        )
        data_successes[data_ok] += 1
        newly_delivered = data_ok & ~delivered
        jitter = output[f"_oracle_jitter_u{attempt}"].to_numpy(float)
        attempt_delay = (
            (attempt + 1) * propagation
            + (attempt + 1) * data_airtime
            + attempt * retry_backoff
            + jitter * 0.25 * (propagation + data_airtime)
        )
        first_delay[newly_delivered] = attempt_delay[newly_delivered].astype(np.float32)
        delivered |= data_ok
        actual_bytes[data_ok] += ack_wire
        actual_energy[data_ok] += np.float32(hops * ack_wire * 8.0 / bitrate * power)
        ack_ok = data_ok & (
            output[f"_oracle_ack_u{attempt}"].to_numpy(float) < np.power(p_ack, hops)
        )
        acknowledged |= ack_ok
        active = attempted & ~ack_ok

    output["actual_attempts"] = attempts
    output["actual_packet_delivered"] = delivered.astype(np.uint8)
    output["actual_packet_delivered_timely"] = (
        delivered & (first_delay <= deadline_seconds)
    ).astype(np.uint8)
    output["actual_acknowledged"] = acknowledged.astype(np.uint8)
    output["actual_duplicate_deliveries"] = np.maximum(
        data_successes.astype(int) - delivered.astype(int), 0
    ).astype(np.int8)
    output["actual_first_delivery_delay_seconds"] = first_delay
    output["actual_network_bytes"] = actual_bytes
    output["actual_modeled_energy_j"] = actual_energy
    return output


def integration_metrics(
    frame: pd.DataFrame,
    protocol: str,
    site: str,
    model_role: str,
    scenario: str,
    scenario_role: str,
    repetition: int,
    policy: str,
    scheduler_config: dict[str, Any],
) -> dict[str, Any]:
    selected = frame["selected"].to_numpy(bool)
    timely = frame["actual_packet_delivered_timely"].to_numpy(bool)
    critical = frame["mandatory_critical"].to_numpy(bool)
    selected_rows = frame.loc[selected]
    critical_items = int(frame["critical_items"].sum())
    exact_delivered = int(frame.loc[timely, "retained_critical_items"].sum())
    critical_delay = (
        pd.to_datetime(frame["relay_flush_timestamp"], utc=True)
        - pd.to_datetime(frame["earliest_critical_timestamp"], utc=True)
    ).dt.total_seconds() + frame["actual_first_delivery_delay_seconds"]
    delivered_critical_delays = critical_delay.loc[timely & critical].dropna().to_numpy(float)
    realized_tail = np.where(
        timely,
        frame["tail_loss_after_success"].to_numpy(float),
        frame["tail_loss_before"].to_numpy(float),
    )
    realized_mean = np.where(
        timely,
        frame["expected_loss_after_success"].to_numpy(float),
        frame["expected_loss_before"].to_numpy(float),
    )
    opportunity = frame.drop_duplicates("opportunity_id")
    budgets = scheduler_config["budgets"]
    queue = scheduler_config["queue_stress"]
    violations = (
        (opportunity["opportunity_selected_packets"] > int(queue["maximum_selected_packets"]))
        | (
            opportunity["opportunity_expected_wire_bytes"]
            > float(budgets["expected_wire_bytes_per_opportunity"]) + 1e-9
        )
        | (
            opportunity["opportunity_modeled_energy_j"]
            > float(budgets["modeled_energy_j_per_opportunity"]) + 1e-9
        )
        | (
            opportunity["opportunity_expected_airtime_seconds"]
            > float(budgets["expected_airtime_seconds_per_opportunity"]) + 1e-9
        )
    )
    actual_energy = float(selected_rows["actual_modeled_energy_j"].sum())
    tail_reduction = float(np.sum(frame["tail_loss_before"].to_numpy(float) - realized_tail))
    return {
        "protocol": protocol,
        "site": site,
        "model_role": model_role,
        "scenario": scenario,
        "scenario_role": scenario_role,
        "repetition": repetition,
        "loss_profile": "primary",
        "policy": policy,
        "opportunities": int(frame["opportunity_id"].nunique()),
        "candidates": int(len(frame)),
        "selected_candidates": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "critical_batches": int(critical.sum()),
        "critical_batch_selection_recall": (
            float(selected[critical].mean()) if critical.any() else np.nan
        ),
        "critical_batch_timely_delivery_recall": (
            float(timely[critical].mean()) if critical.any() else np.nan
        ),
        "critical_batch_miss_probability": (
            float((~timely[critical]).mean()) if critical.any() else np.nan
        ),
        "critical_items": critical_items,
        "exact_retained_critical_item_timely_recall": (
            float(exact_delivered / critical_items) if critical_items else np.nan
        ),
        "mean_delivered_critical_batch_notification_delay_seconds": (
            float(np.mean(delivered_critical_delays)) if len(delivered_critical_delays) else np.nan
        ),
        "p95_delivered_critical_batch_notification_delay_seconds": (
            float(np.quantile(delivered_critical_delays, 0.95))
            if len(delivered_critical_delays)
            else np.nan
        ),
        "realized_tail_proxy_reduction": tail_reduction,
        "realized_expected_proxy_reduction": float(
            np.sum(frame["expected_loss_before"].to_numpy(float) - realized_mean)
        ),
        "realized_upper_cvar_of_batch_tail_loss": float(upper_cvar(realized_tail, 0.95)),
        "actual_network_bytes": int(selected_rows["actual_network_bytes"].sum()),
        "actual_modeled_energy_j": actual_energy,
        "tail_proxy_reduction_per_actual_modeled_joule": (
            tail_reduction / actual_energy if actual_energy > 0 else np.nan
        ),
        "actual_duplicate_deliveries": int(selected_rows["actual_duplicate_deliveries"].sum()),
        "matched_budget_violations": int(violations.sum()),
        "oracle_used_for_policy_decision": False,
    }


def _network_record(
    summary: dict[str, Any], protocol: str, site: str, scenario: str
) -> dict[str, Any]:
    matches = [
        record
        for record in summary["artifacts"]
        if record.get("protocol") == protocol
        and record.get("site") == site
        and record.get("scenario") == scenario
    ]
    if len(matches) != 1:
        raise IntegrationError(
            f"Expected one Task K artifact for {protocol}/{site}/{scenario}; found {len(matches)}"
        )
    return matches[0]


def run_integration(
    scheduler_root: Path,
    fusion_root: Path,
    network_root: Path,
    output_root: Path,
    integration_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    network_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute Task Q matched-policy replay over the frozen relay path."""
    scheduler_summary, fusion_summary, network_summary, verified = verify_inputs(
        scheduler_root, fusion_root, network_root
    )
    policies = [PROPOSED_POLICY, *integration_config["reference_policies"]]
    if len(policies) != len(set(policies)):
        raise IntegrationError("Task Q policy names must be unique")
    metrics: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    proposed_trace_rows = 0
    membership_cache: dict[tuple[str, str], pd.DataFrame] = {}

    schedule_records = [
        record
        for record in scheduler_summary["artifacts"]
        if all(record.get(key) for key in ("protocol", "site", "scenario"))
    ]
    for record in schedule_records:
        protocol = str(record["protocol"])
        site = str(record["site"])
        scenario_name = str(record["scenario"])
        model_role = str(record["model_role"])
        scenario = network_config["scenarios"][scenario_name]
        schedule = pd.read_parquet(_resolve_artifact(record, scheduler_root, "P"))
        schedule["relay_flush_timestamp"] = pd.to_datetime(
            schedule["relay_flush_timestamp"], utc=True
        )
        cache_key = (protocol, site)
        if cache_key not in membership_cache:
            membership_record = next(
                (
                    item
                    for item in fusion_summary["artifacts"]
                    if item.get("protocol") == protocol
                    and item.get("site") == site
                    and item.get("artifact_kind") == "membership"
                ),
                None,
            )
            if membership_record is None:
                raise IntegrationError(f"Missing Task O membership for {protocol}/{site}")
            membership = pd.read_parquet(_resolve_artifact(membership_record, fusion_root, "O"))
            membership_cache[cache_key] = _membership_metadata(membership)
        metadata_columns = [
            "critical_items",
            "retained_critical_items",
            "earliest_critical_timestamp",
            "residual_magnitude_max",
        ]
        schedule = schedule.drop(
            columns=[column for column in metadata_columns if column in schedule.columns]
        ).merge(membership_cache[cache_key], on="batch_id", how="left", validate="many_to_one")
        if schedule["critical_items"].isna().any():
            raise IntegrationError(f"Task P/Task O batch mismatch for {protocol}/{site}")

        network_record = _network_record(network_summary, protocol, site, scenario_name)
        transport = pd.read_parquet(_resolve_artifact(network_record, network_root, "K"))
        transport["timestamp"] = pd.to_datetime(transport["timestamp"], utc=True)
        oracle_columns = [
            "timestamp",
            "window_key",
            "repetition",
            "_oracle_link_state",
            *[
                f"_oracle_{kind}_u{attempt}"
                for attempt in range(int(network_config["maximum_attempts"]))
                for kind in ("data", "ack", "jitter")
            ],
        ]
        oracle = transport[oracle_columns].copy()
        if oracle.duplicated(["window_key", "timestamp", "repetition"]).any():
            raise IntegrationError(
                f"Task K oracle keys are not unique for {protocol}/{site}/{scenario_name}"
            )
        grid_minutes = int(preprocessing_config["protocols"][protocol]["grid_minutes"])
        proposed_parts: list[pd.DataFrame] = []
        for repetition, repetition_frame in schedule.groupby("repetition", sort=False):
            repetition_frame = repetition_frame.copy()
            policy_frames: dict[str, pd.DataFrame] = {PROPOSED_POLICY: repetition_frame.copy()}
            for policy in integration_config["reference_policies"]:
                parts = [
                    schedule_reference(group, str(policy), scheduler_config)
                    for _, group in repetition_frame.groupby("opportunity_id", sort=False)
                ]
                policy_frames[str(policy)] = pd.concat(parts, ignore_index=True)
            oracle_repetition = oracle.loc[oracle["repetition"] == int(repetition)]
            for policy, policy_frame in policy_frames.items():
                decision_columns = [
                    column for column in policy_frame.columns if not column.startswith("_oracle_")
                ]
                policy_frame = policy_frame[decision_columns].merge(
                    oracle_repetition,
                    left_on=["window_key", "relay_flush_timestamp", "repetition"],
                    right_on=["window_key", "timestamp", "repetition"],
                    how="left",
                    validate="many_to_one",
                )
                if policy_frame["_oracle_link_state"].isna().any():
                    raise IntegrationError(
                        f"Task P candidates do not align with Task K oracle for "
                        f"{protocol}/{site}/{scenario_name}/{repetition}"
                    )
                replayed = replay_selected(
                    policy_frame,
                    scenario,
                    network_config,
                    int(integration_config["relay_to_buoy_hops"]),
                    grid_minutes * 60.0,
                )
                metrics.append(
                    integration_metrics(
                        replayed,
                        protocol,
                        site,
                        model_role,
                        scenario_name,
                        str(scenario["role"]),
                        int(repetition),
                        policy,
                        scheduler_config,
                    )
                )
                if policy == PROPOSED_POLICY and bool(
                    integration_config["reporting"]["store_proposed_delivery_trace"]
                ):
                    keep = [
                        "batch_id",
                        "opportunity_id",
                        "window_key",
                        "relay_flush_timestamp",
                        "repetition",
                        "selected",
                        "selection_reason",
                        "mandatory_critical",
                        "critical_items",
                        "retained_critical_items",
                        "actual_attempts",
                        "actual_packet_delivered",
                        "actual_packet_delivered_timely",
                        "actual_acknowledged",
                        "actual_first_delivery_delay_seconds",
                        "actual_network_bytes",
                        "actual_modeled_energy_j",
                    ]
                    proposed_parts.append(replayed[keep].copy())
        if proposed_parts:
            proposed_trace = pd.concat(proposed_parts, ignore_index=True)
            proposed_trace.insert(0, "integration_version", "1.0")
            proposed_trace.insert(1, "policy", PROPOSED_POLICY)
            proposed_trace.insert(2, "scenario", scenario_name)
            trace_path = output_root / "delivery" / protocol / site / f"{scenario_name}.parquet"
            _atomic_parquet(trace_path, proposed_trace)
            artifacts.append(
                {
                    "protocol": protocol,
                    "site": site,
                    "model_role": model_role,
                    "scenario": scenario_name,
                    "path": str(trace_path),
                    "rows": int(len(proposed_trace)),
                    "sha256": sha256_file(trace_path),
                }
            )
            proposed_trace_rows += len(proposed_trace)

    metrics_frame = pd.DataFrame(metrics)
    keys = ["protocol", "site", "scenario", "repetition"]
    proposed_cost = metrics_frame.loc[
        metrics_frame["policy"] == PROPOSED_POLICY,
        [*keys, "actual_network_bytes", "actual_modeled_energy_j"],
    ].rename(
        columns={
            "actual_network_bytes": "proposed_actual_network_bytes",
            "actual_modeled_energy_j": "proposed_actual_modeled_energy_j",
        }
    )
    metrics_frame = metrics_frame.merge(proposed_cost, on=keys, how="left", validate="many_to_one")
    metrics_frame["actual_byte_ratio_to_proposed"] = metrics_frame[
        "actual_network_bytes"
    ] / metrics_frame["proposed_actual_network_bytes"].replace(0, np.nan)
    metrics_frame["actual_energy_ratio_to_proposed"] = metrics_frame[
        "actual_modeled_energy_j"
    ] / metrics_frame["proposed_actual_modeled_energy_j"].replace(0, np.nan)
    if int(metrics_frame["matched_budget_violations"].sum()) != 0:
        raise IntegrationError("Task Q produced a matched opportunity-budget violation")
    metrics_path = output_root / "end_to_end_metrics.csv"
    _atomic_csv(metrics_path, metrics_frame)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})

    config_hash = hashlib.sha256(
        json.dumps(integration_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "Q",
        "integration_version": str(integration_config["integration_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "upstream_artifacts_reverified": verified,
        "task_p_scheduler_version": scheduler_summary["scheduler_version"],
        "task_o_fusion_version": fusion_summary["fusion_version"],
        "task_k_network_version": network_summary["network_version"],
        "proposed_policy": PROPOSED_POLICY,
        "reference_policies": list(integration_config["reference_policies"]),
        "loss_profile": str(integration_config["loss_profile"]),
        "matched_counterfactual_randomness": True,
        "matched_opportunity_constraints": True,
        "relay_to_buoy_hops": int(integration_config["relay_to_buoy_hops"]),
        "oracle_used_for_policy_decisions": False,
        "event_truth_used_for_policy_decisions": False,
        "future_measurements_used": False,
        "parameters_tuned_on_test_or_external_data": False,
        "temporal_protocols_pooled": False,
        "actual_acoustic_delivery_replayed": True,
        "energy_is_modeled_not_hardware_measured": True,
        "complete_path_scope": "semantic_packet_to_relay_to_buoy",
        "receiver_recovery_reintegration_deferred": True,
        "calibration_not_scored_without_probability_forecast": True,
        "matched_budget_violations": 0,
        "metrics_rows": int(len(metrics_frame)),
        "proposed_delivery_trace_rows": int(proposed_trace_rows),
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "task_q_summary.json", summary)
    return summary
