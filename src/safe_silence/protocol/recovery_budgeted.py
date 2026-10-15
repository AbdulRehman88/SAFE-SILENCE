"""Common-budget recovery replay with bounded response admission.

Admission reserves a worst-case request plus bounded response before transmission.
Accounting charges only actual transmissions. Planned requests do not remove false trust.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_silence.data.preprocess import sha256_file


class RecoveryError(RuntimeError):
    """Raised when Task N cannot preserve its recovery contract."""


DEPLOYABLE_POLICIES = {"safe_silence_bounded", "no_recovery", "fixed_probe", "periodic_retry"}
REFERENCE_POLICIES = {"oracle_cause_reference", "unbounded_retry_ceiling"}
ALL_POLICIES = (*sorted(DEPLOYABLE_POLICIES), *sorted(REFERENCE_POLICIES))
PROTECTIVE_STATES = {"ambiguous", "failure_suspected", "recovery_pending", "stale_unsafe"}


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


def _resolve_artifact(record: dict[str, Any], root: Path, folder: str) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    if all(record.get(key) for key in ("protocol", "site", "scenario")):
        return (
            root
            / folder
            / str(record["protocol"])
            / str(record["site"])
            / (f"{record['scenario']}.parquet")
        )
    return root / declared.name


def verify_inputs(
    task_m_root: Path, task_k_root: Path
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Reverify every frozen Task M and Task K artifact before recovery replay."""
    m_path = task_m_root / "task_m_summary.json"
    k_path = task_k_root / "task_k_summary.json"
    if not m_path.exists() or not k_path.exists():
        raise RecoveryError("Task N requires Task M and Task K summaries")
    with m_path.open("r", encoding="utf-8") as handle:
        task_m = json.load(handle)
    with k_path.open("r", encoding="utf-8") as handle:
        task_k = json.load(handle)
    if task_m.get("task") != "M" or task_m.get("cause_version") != "1.1":
        raise RecoveryError("Task N requires frozen Task M cause contract v1.1")
    if task_k.get("task") != "K" or task_k.get("network_version") != "1.0":
        raise RecoveryError("Task N requires frozen Task K network contract v1.0")
    verified_m = 0
    for record in task_m.get("artifacts", []):
        path = _resolve_artifact(record, task_m_root, "posteriors")
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise RecoveryError(f"Task M artifact changed or is missing: {path}")
        verified_m += 1
    verified_k = 0
    for record in task_k.get("artifacts", []):
        path = _resolve_artifact(record, task_k_root, "transport")
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise RecoveryError(f"Task K artifact changed or is missing: {path}")
        verified_k += 1
    if not verified_m or not verified_k:
        raise RecoveryError("Task N found no verifiable upstream artifacts")
    return task_m, task_k, verified_m, verified_k


def make_recovery_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the complete and exclusive deployable-policy evidence contract."""
    columns = [
        "timestamp",
        "window_key",
        "repetition",
        "packet_arrived",
        "receiver_state",
        "connectivity_belief",
        "environmental_uncertainty",
        "receiver_contract_age_steps",
        "receiver_overdue_steps",
        "receiver_contract_valid",
        "receiver_contact_due",
        "cause_decision",
        "valid_suppression_admissible",
    ]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise RecoveryError(f"Task M trace lacks receiver evidence: {sorted(missing)}")
    return frame[columns].copy()


def decide_recovery_actions(
    observations: pd.DataFrame, policy: str, config: dict[str, Any]
) -> pd.DataFrame:
    """Choose deployable recovery actions without oracle state, truth, or future data."""
    if policy not in DEPLOYABLE_POLICIES:
        raise RecoveryError(f"Policy is not deployable: {policy}")
    forbidden = [
        column
        for column in observations.columns
        if column.startswith(("_oracle_", "evaluation_only_"))
    ]
    if forbidden:
        raise RecoveryError(f"Forbidden recovery evidence supplied: {sorted(forbidden)}")
    required = set(make_recovery_observations(observations).columns)
    missing = required.difference(observations.columns)
    if missing:
        raise RecoveryError(f"Recovery observation lacks: {sorted(missing)}")

    output = (
        observations.sort_values(["repetition", "window_key", "timestamp"])
        .reset_index(drop=True)
        .copy()
    )
    triggered = np.zeros(len(output), dtype=bool)
    planned = np.zeros(len(output), dtype=bool)
    reason = np.full(len(output), "none", dtype=object)
    budget = np.zeros(len(output), dtype=np.int16)
    episode = np.full(len(output), -1, dtype=np.int32)
    ordinal = np.full(len(output), -1, dtype=np.int16)
    arrived = output["packet_arrived"].to_numpy(bool)
    policy_config = config["policies"][policy]
    if policy == "no_recovery":
        output["recovery_triggered"] = triggered.astype(np.uint8)
        output["recovery_attempt_planned"] = planned.astype(np.uint8)
        output["recovery_trigger_reason"] = reason
        output["recovery_attempt_budget"] = budget
        output["recovery_episode_id"] = episode
        output["recovery_attempt_ordinal"] = ordinal
        return output

    next_episode = 0
    for _, indices in output.groupby(["repetition", "window_key"], sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        last_trigger = -(10**9)
        silence_run = 0
        for local_index, position in enumerate(positions):
            if planned[position]:
                continue
            if arrived[position]:
                silence_run = 0
                continue
            silence_run += 1
            should_trigger = False
            trigger_reason = "none"
            if policy == "safe_silence_bounded":
                state = str(output.at[position, "receiver_state"])
                cause = str(output.at[position, "cause_decision"])
                protective = (
                    state in set(policy_config["trigger_states"])
                    or cause in set(policy_config["protective_causes"])
                    or bool(output.at[position, "receiver_contact_due"])
                )
                liveness = (
                    bool(output.at[position, "receiver_contract_valid"])
                    and int(output.at[position, "receiver_contract_age_steps"])
                    >= int(policy_config["liveness_interval_steps"])
                    and int(output.at[position, "receiver_contract_age_steps"])
                    % int(policy_config["liveness_interval_steps"])
                    == 0
                    and float(output.at[position, "environmental_uncertainty"])
                    >= float(policy_config["liveness_environmental_uncertainty_minimum"])
                )
                if protective:
                    should_trigger, trigger_reason = True, "protective_ambiguity"
                elif liveness:
                    should_trigger, trigger_reason = True, "bounded_liveness"
                cooldown = int(policy_config["cooldown_steps"])
                should_trigger &= local_index - last_trigger > cooldown
            else:
                interval = int(policy_config["interval_steps"])
                should_trigger = silence_run % interval == 0
                trigger_reason = policy
            if should_trigger:
                triggered[position] = True
                reason[position] = trigger_reason
                maximum_attempts = int(policy_config["maximum_attempts_per_trigger"])
                for attempt_offset in range(maximum_attempts):
                    scheduled_index = local_index + attempt_offset
                    if scheduled_index >= len(positions):
                        break
                    scheduled_position = positions[scheduled_index]
                    planned[scheduled_position] = True
                    budget[scheduled_position] = 1
                    episode[scheduled_position] = next_episode
                    ordinal[scheduled_position] = attempt_offset
                    if attempt_offset:
                        reason[scheduled_position] = "bounded_followup"
                next_episode += 1
                last_trigger = local_index

    output["recovery_triggered"] = triggered.astype(np.uint8)
    output["recovery_attempt_planned"] = planned.astype(np.uint8)
    output["recovery_trigger_reason"] = reason
    output["recovery_attempt_budget"] = budget
    output["recovery_episode_id"] = episode
    output["recovery_attempt_ordinal"] = ordinal
    return output


def _uniform(base_seed: int, row_key: str, attempt: int, direction: str) -> float:
    material = f"{base_seed}|{row_key}|{attempt}|{direction}".encode()
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return (integer + 0.5) / float(2**64)


def _reference_actions(frame: pd.DataFrame, policy: str, config: dict[str, Any]) -> pd.DataFrame:
    output = (
        make_recovery_observations(frame)
        .sort_values(["repetition", "window_key", "timestamp"])
        .reset_index(drop=True)
    )
    missing = ~output["packet_arrived"].to_numpy(bool)
    truth_frame = frame[
        ["repetition", "window_key", "timestamp", "evaluation_only_true_cause"]
    ].copy()
    truth_frame["timestamp"] = pd.to_datetime(truth_frame["timestamp"], utc=True)
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    truth = (
        output.merge(
            truth_frame,
            on=["repetition", "window_key", "timestamp"],
            how="left",
            validate="one_to_one",
        )["evaluation_only_true_cause"]
        .astype(str)
        .to_numpy()
    )
    trigger_mask = missing & (truth != "valid_suppression")
    if policy == "unbounded_retry_ceiling":
        trigger_mask = missing
    triggered = np.zeros(len(output), dtype=bool)
    planned = np.zeros(len(output), dtype=bool)
    reason = np.full(len(output), "none", dtype=object)
    budget = np.zeros(len(output), dtype=np.int16)
    episode = np.full(len(output), -1, dtype=np.int32)
    ordinal = np.full(len(output), -1, dtype=np.int16)
    next_episode = 0
    maximum_attempts = int(config["policies"][policy]["maximum_attempts_per_trigger"])
    for _, indices in output.groupby(["repetition", "window_key"], sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        for local_index, position in enumerate(positions):
            if planned[position] or not trigger_mask[position]:
                continue
            triggered[position] = True
            reason[position] = policy
            for attempt_offset in range(maximum_attempts):
                scheduled_index = local_index + attempt_offset
                if scheduled_index >= len(positions):
                    break
                scheduled_position = positions[scheduled_index]
                planned[scheduled_position] = True
                budget[scheduled_position] = 1
                episode[scheduled_position] = next_episode
                ordinal[scheduled_position] = attempt_offset
                if attempt_offset:
                    reason[scheduled_position] = "reference_followup"
            next_episode += 1
    output["recovery_triggered"] = triggered.astype(np.uint8)
    output["recovery_attempt_planned"] = planned.astype(np.uint8)
    output["recovery_trigger_reason"] = reason
    output["recovery_attempt_budget"] = budget
    output["recovery_episode_id"] = episode
    output["recovery_attempt_ordinal"] = ordinal
    return output


def replay_recovery(
    frame: pd.DataFrame,
    transport: pd.DataFrame,
    policy: str,
    config: dict[str, Any],
    network_config: dict[str, Any],
    scenario: dict[str, Any],
    grid_minutes: int,
) -> pd.DataFrame:
    """Replay one recovery policy with explicit, byte-accounted packet reconciliation.

    Recovery requests are selected only from receiver-observable evidence.  The
    physical simulator maintains a bounded sender-side cache of causal critical
    packets that were not acknowledged.  A successful response can carry at
    most one cached packet, uses that packet's original encoded wire size, and
    consumes it exactly once.  Generic liveness responses are accounted for
    separately and cannot receive critical-delivery credit.
    """
    if policy in DEPLOYABLE_POLICIES:
        actions = decide_recovery_actions(make_recovery_observations(frame), policy, config)
    else:
        actions = _reference_actions(frame, policy, config)
    keys = ["repetition", "window_key", "timestamp"]
    truth_columns = keys + [
        "realization_seed",
        "packet_delivered_timely",
        "packet_acknowledged",
        "network_bytes",
        "incremental_communication_energy_j",
        "_oracle_link_state",
        "causal_critical_evidence",
        "critical_packet_id",
        "critical_packet_wire_bytes",
        "validity_horizon_steps",
    ]
    physical_source = transport.copy()
    if "packet_acknowledged" not in physical_source.columns:
        if "acknowledged" not in physical_source.columns:
            raise RecoveryError("Task K trace lacks packet acknowledgment evidence")
        physical_source["packet_acknowledged"] = physical_source["acknowledged"]
    if "critical_packet_id" not in physical_source.columns:
        if "silence_contract_id" not in physical_source.columns:
            raise RecoveryError("Task K trace lacks a causal critical-packet identifier")
        physical_source["critical_packet_id"] = physical_source["silence_contract_id"].map(
            str
        )
    if "critical_packet_wire_bytes" not in physical_source.columns:
        if "semantic_payload_bytes" not in physical_source.columns:
            raise RecoveryError("Task K trace lacks exact recovery-response payload bytes")
        physical_source["critical_packet_wire_bytes"] = (
            int(network_config["packet_header_bytes"])
            + pd.to_numeric(
                physical_source["semantic_payload_bytes"], errors="raise"
            ).astype(int)
        )
    missing = set(truth_columns).difference(physical_source.columns)
    if missing:
        raise RecoveryError(f"Task K trace lacks recovery replay fields: {sorted(missing)}")
    physical = physical_source[truth_columns].copy()
    evaluation_columns = keys + [
        "evaluation_only_unsafe_silence",
        "evaluation_only_true_cause",
    ]
    evaluation = frame[evaluation_columns].copy()
    physical["timestamp"] = pd.to_datetime(physical["timestamp"], utc=True)
    evaluation["timestamp"] = pd.to_datetime(evaluation["timestamp"], utc=True)
    actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
    replay = actions.merge(physical, on=keys, how="left", validate="one_to_one")
    replay = replay.merge(evaluation, on=keys, how="left", validate="one_to_one")

    attempts_used = np.zeros(len(replay), dtype=np.int16)
    success = np.zeros(len(replay), dtype=bool)
    critical_response_success = np.zeros(len(replay), dtype=bool)
    bytes_used = np.zeros(len(replay), dtype=np.int32)
    energy_used = np.zeros(len(replay), dtype=np.float64)
    delay = np.full(len(replay), np.nan, dtype=np.float32)
    response_wire_used = np.zeros(len(replay), dtype=np.int32)
    cache_depth_before_response = np.zeros(len(replay), dtype=np.int16)
    cache_expired_items = np.zeros(len(replay), dtype=np.int16)
    response_kind = np.full(len(replay), "none", dtype=object)
    response_id = np.full(len(replay), "", dtype=object)
    retransmitted_packet_id = np.full(len(replay), "", dtype=object)
    recovered_by_response_id = np.full(len(replay), "", dtype=object)
    recovered_by_response_timestamp: list[pd.Timestamp | pd.NaT] = [pd.NaT] * len(replay)
    event_recovery_response_wire_bytes = np.zeros(len(replay), dtype=np.int32)
    request_wire = int(network_config["packet_header_bytes"]) + int(
        config["packet_bytes"]["probe_payload"]
    )
    liveness_response_wire = int(network_config["packet_header_bytes"]) + int(
        config["packet_bytes"]["liveness_response_payload"]
    )
    bitrate = float(network_config["bitrate_bps"])
    hops = int(network_config["hops"])
    power = float(network_config["energy"]["tx_watts"]) + float(
        network_config["energy"]["rx_watts"]
    )
    propagation = (
        hops
        * float(network_config["nominal_distance_per_hop_m"])
        / float(network_config["propagation_speed_m_s"])
    )
    deadline = grid_minutes * 60.0
    base_seed = int(config["matched_replay"]["base_seed"])

    initial = replay["packet_delivered_timely"].to_numpy(bool)
    acknowledged = replay["packet_acknowledged"].to_numpy(bool)
    critical = replay["causal_critical_evidence"].to_numpy(bool)
    packet_ids = replay["critical_packet_id"].map(str).to_numpy()
    packet_wire = pd.to_numeric(
        replay["critical_packet_wire_bytes"], errors="raise"
    ).to_numpy(int)
    if np.any(
        packet_wire[critical] <= int(network_config["packet_header_bytes"])
    ):
        raise RecoveryError("Recovery response must contain a positive encoded payload")
    validity = pd.to_numeric(replay["validity_horizon_steps"], errors="raise").to_numpy(int)
    if np.any(validity < 0):
        raise RecoveryError("Recovery validity horizons cannot be negative")
    critical_identity = replay.loc[
        critical, ["repetition", "window_key", "critical_packet_id"]
    ]
    if critical_identity.duplicated().any():
        raise RecoveryError("Critical packet identifiers must be unique within each source window")

    post_delivery_within_retention = initial.copy()
    post_delivery_timely = initial.copy()
    event_recovery_delay = np.full(len(replay), np.nan, dtype=np.float32)
    retention = int(config["reconciliation"]["critical_alert_retention_steps"])
    budget_config = config.get("extension_common_budget")
    spent_bytes: dict[int, int] = {}
    spent_energy: dict[int, float] = {}
    cap_by_rep: dict[int, tuple[float, float]] = {}
    blocked = np.zeros(len(replay), dtype=np.uint8)
    if budget_config is not None:
        for repetition, group in replay.groupby("repetition", sort=False):
            count = len(group)
            cap_by_rep[int(repetition)] = (
                float(budget_config["bytes_per_opportunity"]) * count,
                float(budget_config["energy_j_per_opportunity"]) * count,
            )
            spent_bytes[int(repetition)] = 0
            spent_energy[int(repetition)] = 0.0
        response_bound = int(budget_config["maximum_response_wire_bytes"])
        if np.any(packet_wire[critical] > response_bound) or liveness_response_wire > response_bound:
            raise RecoveryError("A response exceeds the frozen conservative admission bound")
    successful_episodes: set[int] = set()
    for _, indices in replay.groupby(["repetition", "window_key"], sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        cache: deque[dict[str, Any]] = deque()
        for local_index, position in enumerate(positions):
            while cache and local_index - int(cache[0]["local_index"]) > retention:
                cache.popleft()
                cache_expired_items[position] += 1

            if critical[position] and not acknowledged[position]:
                cache.append(
                    {
                        "position": position,
                        "local_index": local_index,
                        "packet_id": str(packet_ids[position]),
                        "wire_bytes": int(packet_wire[position]),
                    }
                )

            budget = int(replay.at[position, "recovery_attempt_budget"])
            if budget <= 0:
                continue
            episode_id = int(replay.at[position, "recovery_episode_id"])
            if episode_id in successful_episodes:
                replay.at[position, "recovery_attempt_planned"] = 0
                continue

            state = str(replay.at[position, "_oracle_link_state"])
            link_probability = float(scenario["packet_success"][state]) ** hops
            row_key = "|".join(
                [
                    str(replay.at[position, "realization_seed"]),
                    str(replay.at[position, "repetition"]),
                    str(replay.at[position, "window_key"]),
                    pd.Timestamp(replay.at[position, "timestamp"]).isoformat(),
                ]
            )
            for _attempt_within_row in range(budget):
                attempt = int(replay.at[position, "recovery_attempt_ordinal"])
                repetition_key = int(replay.at[position, "repetition"])
                if budget_config is not None:
                    worst_bytes = request_wire + response_bound
                    worst_energy = hops * worst_bytes * 8.0 / bitrate * power
                    byte_cap, energy_cap = cap_by_rep[repetition_key]
                    if spent_bytes[repetition_key] + worst_bytes > byte_cap + 1e-9 or spent_energy[repetition_key] + worst_energy > energy_cap + 1e-9:
                        blocked[position] = 1
                        replay.at[position, "recovery_attempt_planned"] = 0
                        break
                attempts_used[position] += 1
                bytes_used[position] += request_wire
                energy_used[position] += hops * request_wire * 8.0 / bitrate * power
                if budget_config is not None:
                    spent_bytes[repetition_key] += request_wire
                    spent_energy[repetition_key] += hops * request_wire * 8.0 / bitrate * power
                request_ok = (
                    _uniform(base_seed, row_key, attempt, "request") < link_probability
                )
                if not request_ok:
                    continue

                cached = cache[0] if cache else None
                current_response_wire = (
                    int(cached["wire_bytes"]) if cached else liveness_response_wire
                )
                bytes_used[position] += current_response_wire
                energy_used[position] += (
                    hops * current_response_wire * 8.0 / bitrate * power
                )
                if budget_config is not None:
                    spent_bytes[repetition_key] += current_response_wire
                    spent_energy[repetition_key] += hops * current_response_wire * 8.0 / bitrate * power
                response_wire_used[position] = current_response_wire
                cache_depth_before_response[position] = len(cache)
                response_material = "|".join(
                    [row_key, str(episode_id), str(attempt), "response"]
                ).encode()
                current_response_id = hashlib.sha256(response_material).hexdigest()[:24]
                response_id[position] = current_response_id
                response_kind[position] = (
                    "critical_retransmission" if cached is not None else "liveness"
                )
                if cached is not None:
                    retransmitted_packet_id[position] = str(cached["packet_id"])
                response_ok = (
                    _uniform(base_seed, row_key, attempt, "response") < link_probability
                )
                if not response_ok:
                    break

                candidate_delay = (
                    2.0 * propagation
                    + (request_wire + current_response_wire) * 8.0 / bitrate
                    + attempt * float(network_config["retry_backoff_seconds"])
                )
                success[position] = candidate_delay <= deadline
                delay[position] = np.float32(candidate_delay)
                if not success[position]:
                    break

                successful_episodes.add(episode_id)
                if cached is None:
                    break

                delivered_item = cache.popleft()
                event_position = int(delivered_item["position"])
                recovery_offset = local_index - int(delivered_item["local_index"])
                critical_response_success[position] = True
                recovered_by_response_id[event_position] = current_response_id
                recovered_by_response_timestamp[event_position] = pd.Timestamp(
                    replay.at[position, "timestamp"]
                )
                event_recovery_response_wire_bytes[event_position] = current_response_wire
                event_recovery_delay[event_position] = np.float32(recovery_offset)
                if not initial[event_position]:
                    post_delivery_within_retention[event_position] = True
                    if recovery_offset <= int(validity[event_position]):
                        post_delivery_timely[event_position] = True
                break

    linked_ids = [value for value in recovered_by_response_id if value]
    if len(linked_ids) != len(set(linked_ids)):
        raise RecoveryError("A recovery response was credited to more than one critical packet")
    delivered_ids = response_id[critical_response_success]
    if len(delivered_ids) != len(set(delivered_ids)):
        raise RecoveryError("Critical retransmission response identifiers are not unique")
    unsafe = replay["evaluation_only_unsafe_silence"].to_numpy(bool)
    trusted = replay["receiver_state"].astype(str).to_numpy() == "trusted_silence"
    triggered = replay["recovery_triggered"].to_numpy(bool)
    planned = replay["recovery_attempt_planned"].to_numpy(bool)
    post_false_trust = unsafe & trusted & ~success
    post_state = replay["receiver_state"].astype(str).to_numpy().copy()
    post_state[triggered & ~success] = "recovery_pending"
    post_state[success] = "recovered_update"

    replay["extension_budget_blocked"] = blocked
    if budget_config is not None:
        replay["extension_recovery_byte_cap"] = replay["repetition"].map(lambda value: cap_by_rep[int(value)][0])
        replay["extension_recovery_energy_cap_j"] = replay["repetition"].map(lambda value: cap_by_rep[int(value)][1])
        for repetition_key, (byte_cap, energy_cap) in cap_by_rep.items():
            if spent_bytes[repetition_key] > byte_cap + 1e-9 or spent_energy[repetition_key] > energy_cap + 1e-9:
                raise RecoveryError("Realized recovery consumption exceeded a common cap")
    replay["policy"] = policy
    replay["policy_deployable"] = np.uint8(policy in DEPLOYABLE_POLICIES)
    replay["recovery_attempts"] = attempts_used
    replay["recovery_succeeded"] = success.astype(np.uint8)
    replay["critical_retransmission_succeeded"] = critical_response_success.astype(np.uint8)
    replay["recovery_delay_seconds"] = delay
    replay["critical_event_recovery_delay_steps"] = event_recovery_delay
    replay["recovery_response_kind"] = response_kind
    replay["recovery_response_id"] = response_id
    replay["recovery_response_wire_bytes"] = response_wire_used
    replay["recovery_retransmitted_critical_packet_id"] = retransmitted_packet_id
    replay["critical_event_recovery_response_id"] = recovered_by_response_id
    replay["critical_event_recovery_response_timestamp"] = pd.to_datetime(
        recovered_by_response_timestamp, utc=True
    )
    replay["critical_event_recovery_response_wire_bytes"] = (
        event_recovery_response_wire_bytes
    )
    replay["recovery_cache_depth_before_response"] = cache_depth_before_response
    replay["recovery_cache_expired_items"] = cache_expired_items
    replay["recovery_network_bytes"] = bytes_used
    replay["recovery_energy_j"] = energy_used
    replay["post_recovery_packet_delivered_within_retention"] = (
        post_delivery_within_retention.astype(np.uint8)
    )
    replay["post_recovery_packet_delivered_timely"] = post_delivery_timely.astype(np.uint8)
    replay["post_recovery_state"] = post_state
    replay["evaluation_only_post_recovery_false_trust"] = post_false_trust.astype(np.uint8)
    replay["evaluation_only_critical_recovered"] = (
        critical & ~initial & post_delivery_within_retention
    ).astype(np.uint8)
    replay["evaluation_only_critical_recovered_timely"] = (
        critical & ~initial & post_delivery_timely
    ).astype(np.uint8)
    return replay


def recovery_metrics(
    frame: pd.DataFrame,
    protocol: str,
    site: str,
    model_role: str,
    scenario: str,
    repetition: int,
    policy: str,
) -> dict[str, Any]:
    critical = frame["causal_critical_evidence"].to_numpy(bool)
    initial = frame["packet_delivered_timely"].to_numpy(bool)
    post = frame["post_recovery_packet_delivered_within_retention"].to_numpy(bool)
    timely_post = frame["post_recovery_packet_delivered_timely"].to_numpy(bool)
    triggered = frame["recovery_triggered"].to_numpy(bool)
    success = frame["recovery_succeeded"].to_numpy(bool)
    false_trust = frame["evaluation_only_post_recovery_false_trust"].to_numpy(bool)
    true_cause = frame["evaluation_only_true_cause"].astype(str).to_numpy()
    delays = frame["recovery_delay_seconds"].to_numpy(float)
    event_delays = frame["critical_event_recovery_delay_steps"].to_numpy(float)
    bytes_before = float(frame["network_bytes"].sum())
    extra_bytes = float(frame["recovery_network_bytes"].sum())
    return {
        "protocol": protocol,
        "site": site,
        "model_role": model_role,
        "scenario": scenario,
        "repetition": repetition,
        "policy": policy,
        "policy_deployable": policy in DEPLOYABLE_POLICIES,
        "rows": int(len(frame)),
        "available_critical_rows": int(critical.sum()),
        "initial_critical_delivery_recall": (
            float(initial[critical].mean()) if critical.any() else np.nan
        ),
        "post_recovery_critical_delivery_recall": (
            float(post[critical].mean()) if critical.any() else np.nan
        ),
        "post_recovery_timely_critical_delivery_recall": (
            float(timely_post[critical].mean()) if critical.any() else np.nan
        ),
        "critical_false_trust_rate": (
            float(false_trust[critical].mean()) if critical.any() else np.nan
        ),
        "recovery_triggers": int(triggered.sum()),
        "recovery_attempts": int(frame["recovery_attempts"].sum()),
        "recovery_success_rate": (
            float(success.sum() / triggered.sum()) if triggered.any() else np.nan
        ),
        "critical_rows_recovered": int((critical & ~initial & post).sum()),
        "critical_rows_recovered_timely": int(
            (critical & ~initial & timely_post).sum()
        ),
        "critical_retransmission_attempts": int(
            (frame["recovery_response_kind"].astype(str) == "critical_retransmission").sum()
        ),
        "critical_retransmission_successes": int(
            frame["critical_retransmission_succeeded"].sum()
        ),
        "liveness_response_attempts": int(
            (frame["recovery_response_kind"].astype(str) == "liveness").sum()
        ),
        "unnecessary_valid_suppression_trigger_rate": (
            float(triggered[true_cause == "valid_suppression"].mean())
            if np.any(true_cause == "valid_suppression")
            else np.nan
        ),
        "recovery_network_bytes": int(extra_bytes),
        "recovery_wire_overhead_fraction": extra_bytes / bytes_before if bytes_before else np.nan,
        "recovery_energy_j": float(frame["recovery_energy_j"].sum()),
        "p95_successful_recovery_delay_seconds": (
            float(np.nanquantile(delays[success], 0.95)) if success.any() else np.nan
        ),
        "p95_critical_event_recovery_delay_steps": (
            float(np.nanquantile(event_delays[np.isfinite(event_delays)], 0.95))
            if np.isfinite(event_delays).any()
            else np.nan
        ),
        "post_recovery_pending_fraction": float(
            (frame["post_recovery_state"].astype(str).to_numpy() == "recovery_pending").mean()
        ),
    }


def run_bounded_recovery(
    task_m_root: Path,
    task_k_root: Path,
    output_root: Path,
    config: dict[str, Any],
    network_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute Task N over all matched Task M/Task K traces and policies."""
    task_m, _task_k, verified_m, verified_k = verify_inputs(task_m_root, task_k_root)
    metrics: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    trace_records = [
        record
        for record in task_m["artifacts"]
        if all(record.get(key) for key in ("protocol", "site", "scenario"))
    ]
    for record in trace_records:
        protocol = str(record["protocol"])
        site = str(record["site"])
        scenario_name = str(record["scenario"])
        model_role = str(record["model_role"])
        posterior_path = _resolve_artifact(record, task_m_root, "posteriors")
        transport_path = task_k_root / "transport" / protocol / site / f"{scenario_name}.parquet"
        if not transport_path.exists():
            raise RecoveryError(f"Missing matched Task K trace: {transport_path}")
        posterior = pd.read_parquet(posterior_path)
        transport = pd.read_parquet(transport_path)
        posterior["timestamp"] = pd.to_datetime(posterior["timestamp"], utc=True)
        transport["timestamp"] = pd.to_datetime(transport["timestamp"], utc=True)
        keys = ["repetition", "window_key", "timestamp"]
        if posterior.duplicated(keys).any() or transport.duplicated(keys).any():
            raise RecoveryError("Matched recovery keys are not unique")
        if len(posterior) != len(transport):
            raise RecoveryError("Task M and Task K trace lengths differ")
        scenario = network_config["scenarios"][scenario_name]
        grid_minutes = 15 if protocol == "primary_15min" else 60
        bounded_trace: pd.DataFrame | None = None
        for policy in ALL_POLICIES:
            replay = replay_recovery(
                posterior,
                transport,
                policy,
                config,
                network_config,
                scenario,
                grid_minutes,
            )
            replay.insert(0, "recovery_version", str(config["recovery_version"]))
            replay.insert(1, "protocol", protocol)
            replay.insert(2, "site", site)
            replay.insert(3, "model_role", model_role)
            replay.insert(4, "scenario", scenario_name)
            for repetition, group in replay.groupby("repetition", sort=False):
                metrics.append(
                    recovery_metrics(
                        group,
                        protocol,
                        site,
                        model_role,
                        scenario_name,
                        int(repetition),
                        policy,
                    )
                )
            if policy == "safe_silence_bounded":
                bounded_trace = replay
        if bounded_trace is None:
            raise RecoveryError("The deployable bounded-policy trace was not produced")
        output_path = output_root / "recovery" / protocol / site / f"{scenario_name}.parquet"
        _atomic_parquet(output_path, bounded_trace)
        artifacts.append(
            {
                "protocol": protocol,
                "site": site,
                "model_role": model_role,
                "scenario": scenario_name,
                "path": str(output_path),
                "rows": int(len(bounded_trace)),
                "sha256": sha256_file(output_path),
            }
        )

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = output_root / "recovery_metrics.csv"
    _atomic_csv(metrics_path, metrics_frame)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "N",
        "recovery_version": str(config["recovery_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_m_artifacts_reverified": verified_m,
        "task_k_artifacts_reverified": verified_k,
        "task_m_cause_version": task_m["cause_version"],
        "deployable_policy_uses_oracle_fields": False,
        "deployable_policy_uses_event_truth": False,
        "deployable_policy_uses_future_measurements": False,
        "parameters_tuned_on_test_or_external_data": False,
        "algorithm_shared_across_sites_and_scenarios": True,
        "named_cause_is_never_sufficient_for_action": True,
        "matched_counterfactual_recovery_randomness": True,
        "recovery_attempts_are_bounded": True,
        "recovery_attempts_distributed_across_grid_steps": True,
        "critical_alert_retention_steps": int(
            config["reconciliation"]["critical_alert_retention_steps"]
        ),
        "source_alert_cache_is_bounded": True,
        "maximum_critical_packets_per_response": 1,
        "critical_response_uses_original_wire_bytes": True,
        "successful_response_consumes_cached_packet": True,
        "exact_event_response_provenance": True,
        "liveness_and_critical_responses_separated": True,
        "oracle_policies_are_diagnostic_references": True,
        "row_level_trace_policy": "safe_silence_bounded",
        "reference_policies_stored_as_metrics_only": True,
        "energy_is_modeled_not_hardware_measured": True,
        "policies": list(ALL_POLICIES),
        "metrics_rows": int(len(metrics_frame)),
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "task_n_summary.json", summary)
    return summary
