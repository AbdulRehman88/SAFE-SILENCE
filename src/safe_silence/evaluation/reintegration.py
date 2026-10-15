"""Task R relay-receiver-recovery reintegration and paired evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from safe_silence.data.preprocess import sha256_file
from safe_silence.evaluation.integration import (
    replay_selected,
    schedule_reference,
)
from safe_silence.evaluation.integration import (
    verify_inputs as verify_q_upstreams,
)
from safe_silence.protocol.cause import infer_causes
from safe_silence.protocol.receiver import PROTECTIVE_STATES, interpret_receiver
from safe_silence.protocol.recovery import recovery_metrics, replay_recovery
from safe_silence.protocol.scheduler import PROPOSED_POLICY, REFERENCE_POLICY


class ReintegrationError(RuntimeError):
    """Raised when Task R cannot preserve its reintegration contract."""


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


def _resolve(record: dict[str, Any], root: Path, task: str) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    if task == "Q" and all(record.get(key) for key in ("protocol", "site", "scenario")):
        return (
            root
            / "delivery"
            / str(record["protocol"])
            / str(record["site"])
            / f"{record['scenario']}.parquet"
        )
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


def verify_inputs(
    integration_root: Path,
    scheduler_root: Path,
    fusion_root: Path,
    network_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, int]]:
    """Reverify Task Q and all direct frozen replay inputs."""
    summary_path = integration_root / "task_q_summary.json"
    if not summary_path.exists():
        raise ReintegrationError(f"Task R requires {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        task_q = json.load(handle)
    if task_q.get("task") != "Q" or task_q.get("integration_version") != "1.0":
        raise ReintegrationError("Task R requires frozen Task Q contract v1.0")
    verified_q = 0
    for record in task_q.get("artifacts", []):
        path = _resolve(record, integration_root, "Q")
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise ReintegrationError(f"Task Q artifact changed or is missing: {path}")
        verified_q += 1
    if not verified_q:
        raise ReintegrationError("Task Q contains no verifiable artifacts")
    try:
        task_p, task_o, task_k, upstream = verify_q_upstreams(
            scheduler_root, fusion_root, network_root
        )
    except RuntimeError as exc:
        raise ReintegrationError(str(exc)) from exc
    return task_q, task_p, task_o, task_k, {"Q": verified_q, **upstream}


def _find_record(
    summary: dict[str, Any], protocol: str, site: str, scenario: str, task: str
) -> dict[str, Any]:
    matches = [
        record
        for record in summary["artifacts"]
        if record.get("protocol") == protocol
        and record.get("site") == site
        and record.get("scenario") == scenario
    ]
    if len(matches) != 1:
        raise ReintegrationError(
            f"Expected one Task {task} trace for {protocol}/{site}/{scenario}; found {len(matches)}"
        )
    return matches[0]


def _membership_metadata(membership: pd.DataFrame) -> pd.DataFrame:
    required = {
        "batch_id",
        "critical_evidence_score",
        "residual_magnitude",
        "retained",
    }
    missing = required.difference(membership.columns)
    if missing:
        raise ReintegrationError(f"Task O membership lacks: {sorted(missing)}")
    frame = membership.copy()
    frame["critical"] = frame["critical_evidence_score"].astype(float) >= 0.5
    frame["retained_flag"] = frame["retained"].astype(bool)
    frame["retained_critical"] = frame["critical"] & frame["retained_flag"]
    rows: list[dict[str, Any]] = []
    for batch_id, group in frame.groupby("batch_id", sort=False):
        retained = group.loc[group["retained_flag"]]
        rows.append(
            {
                "batch_id": str(batch_id),
                "critical_items": int(group["critical"].sum()),
                "retained_critical_items": int(group["retained_critical"].sum()),
                "retained_residual_magnitude_max": (
                    float(retained["residual_magnitude"].max()) if len(retained) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _contract_id(batch_id: str) -> int:
    """Map a batch identifier to an exactly representable positive 48-bit integer."""
    return int(hashlib.sha256(str(batch_id).encode()).hexdigest()[:12], 16)


def build_relay_transport(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    """Translate fused scheduling outcomes into the frozen receiver packet contract."""
    required = {
        "batch_id",
        "window_key",
        "repetition",
        "relay_flush_timestamp",
        "selected",
        "actual_packet_delivered_timely",
        "actual_acknowledged",
        "actual_network_bytes",
        "actual_modeled_energy_j",
        "encoded_wire_bytes",
        "mandatory_critical",
        "critical_evidence_max",
        "environmental_uncertainty_max",
        "deadline_urgency_max",
        "retained_residual_magnitude_max",
        "_oracle_link_state",
        "realization_seed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ReintegrationError(f"Relay integration input lacks: {sorted(missing)}")
    output = frame.copy()
    selected = output["selected"].to_numpy(bool)
    critical = output["mandatory_critical"].to_numpy(bool)
    output["timestamp"] = pd.to_datetime(output["relay_flush_timestamp"], utc=True)
    output["node_action"] = np.where(
        ~selected,
        str(contract["unselected_summary_action"]),
        np.where(
            critical,
            str(contract["critical_summary_action"]),
            str(contract["ordinary_selected_summary_action"]),
        ),
    )
    evidence = np.column_stack(
        [
            output["critical_evidence_max"].to_numpy(float),
            output["environmental_uncertainty_max"].to_numpy(float),
            output["retained_residual_magnitude_max"].to_numpy(float),
        ]
    )
    output["semantic_risk"] = np.clip(np.max(evidence, axis=1), 0.0, 1.0)
    urgency = np.clip(output["deadline_urgency_max"].to_numpy(float), 1.0e-9, 1.0)
    remaining = np.rint(1.0 / urgency - 1.0)
    output["validity_horizon_steps"] = np.clip(
        remaining, 0, int(contract["maximum_validity_opportunities"])
    ).astype(np.int16)
    output["silence_contract_id"] = [
        _contract_id(batch_id) for batch_id in output["batch_id"].astype(str)
    ]
    output["packet_delivered_timely"] = output["actual_packet_delivered_timely"].astype(np.uint8)
    output["packet_acknowledged"] = output["actual_acknowledged"].astype(np.uint8)
    output["network_bytes"] = output["actual_network_bytes"].astype(np.int32)
    output["incremental_communication_energy_j"] = output["actual_modeled_energy_j"].astype(
        np.float32
    )
    output["causal_critical_evidence"] = critical.astype(np.uint8)
    output["critical_packet_id"] = output["batch_id"].map(str)
    output["critical_packet_wire_bytes"] = output["encoded_wire_bytes"].astype(np.int32)
    keep = [
        "timestamp",
        "window_key",
        "repetition",
        "node_action",
        "semantic_risk",
        "validity_horizon_steps",
        "silence_contract_id",
        "packet_delivered_timely",
        "packet_acknowledged",
        "network_bytes",
        "incremental_communication_energy_j",
        "causal_critical_evidence",
        "critical_packet_id",
        "critical_packet_wire_bytes",
        "_oracle_link_state",
        "realization_seed",
    ]
    transport = output[keep].copy()
    keys = ["repetition", "window_key", "timestamp"]
    if transport.duplicated(keys).any():
        raise ReintegrationError("Relay receiver keys are not unique")
    return transport


def reintegrate_receiver_and_cause(
    transport: pd.DataFrame,
    receiver_config: dict[str, Any],
    cause_config: dict[str, Any],
) -> pd.DataFrame:
    """Run frozen receiver/cause logic and enforce critical-silence safety scoring."""
    interpreted = interpret_receiver(transport, receiver_config)
    missing_critical = interpreted["evaluation_only_critical_event"].to_numpy(bool) & ~interpreted[
        "packet_arrived"
    ].to_numpy(bool)
    unsafe = interpreted["evaluation_only_unsafe_silence"].to_numpy(bool) | missing_critical
    trusted = interpreted["receiver_state"].astype(str).to_numpy() == "trusted_silence"
    interpreted["evaluation_only_unsafe_silence"] = unsafe.astype(np.uint8)
    interpreted["evaluation_only_false_trust"] = (unsafe & trusted).astype(np.uint8)
    inferred = infer_causes(interpreted, cause_config)
    critical_suppression = (
        inferred["evaluation_only_critical_event"].to_numpy(bool)
        & ~inferred["packet_arrived"].to_numpy(bool)
        & (inferred["evaluation_only_source_action"].astype(str).to_numpy() == "silent")
    )
    inferred.loc[critical_suppression, "evaluation_only_true_cause"] = "contract_divergence"
    unsafe = inferred["evaluation_only_unsafe_silence"].to_numpy(bool)
    inferred["evaluation_only_unsafe_predicted_valid"] = (
        unsafe & (inferred["cause_decision"].astype(str).to_numpy() == "valid_suppression")
    ).astype(np.uint8)
    return inferred


def _system_metrics(
    replay: pd.DataFrame,
    evidence: pd.DataFrame,
    protocol: str,
    site: str,
    model_role: str,
    scenario: str,
    repetition: int,
    scheduler_policy: str,
    recovery_policy: str,
) -> dict[str, Any]:
    base = recovery_metrics(
        replay,
        protocol,
        site,
        model_role,
        scenario,
        repetition,
        recovery_policy,
    )
    keys = ["repetition", "window_key", "timestamp"]
    scoring_columns = keys + [
        "tail_loss_before",
        "tail_loss_after_success",
        "expected_loss_before",
        "expected_loss_after_success",
        "critical_items",
        "retained_critical_items",
    ]
    scored = replay.merge(evidence[scoring_columns], on=keys, how="left", validate="one_to_one")
    if scored["tail_loss_before"].isna().any():
        raise ReintegrationError("Recovery trace lost Task P loss-proxy alignment")
    post = scored["post_recovery_packet_delivered_within_retention"].to_numpy(bool)
    post_timely = scored["post_recovery_packet_delivered_timely"].to_numpy(bool)
    initial = scored["packet_delivered_timely"].to_numpy(bool)
    realized_tail = np.where(
        post_timely,
        scored["tail_loss_after_success"].to_numpy(float),
        scored["tail_loss_before"].to_numpy(float),
    )
    realized_expected = np.where(
        post_timely,
        scored["expected_loss_after_success"].to_numpy(float),
        scored["expected_loss_before"].to_numpy(float),
    )
    critical_items = int(scored["critical_items"].sum())
    initial_exact = int(scored.loc[initial, "retained_critical_items"].sum())
    post_exact = int(scored.loc[post, "retained_critical_items"].sum())
    base_bytes = float(scored["network_bytes"].sum())
    recovery_bytes = float(scored["recovery_network_bytes"].sum())
    base_energy = float(scored["incremental_communication_energy_j"].sum())
    recovery_energy = float(scored["recovery_energy_j"].sum())
    total_energy = base_energy + recovery_energy
    tail_reduction = float(np.sum(scored["tail_loss_before"].to_numpy(float) - realized_tail))
    unsafe = scored["evaluation_only_unsafe_silence"].to_numpy(bool)
    protective = np.isin(scored["receiver_state"].astype(str), list(PROTECTIVE_STATES))
    return {
        **base,
        "scheduler_policy": scheduler_policy,
        "recovery_policy": recovery_policy,
        "system_policy": f"{scheduler_policy}+{recovery_policy}",
        "initial_exact_retained_critical_item_recall": (
            initial_exact / critical_items if critical_items else np.nan
        ),
        "post_recovery_exact_retained_critical_item_recall": (
            post_exact / critical_items if critical_items else np.nan
        ),
        "receiver_unsafe_silence_detection_recall": (
            float(protective[unsafe].mean()) if unsafe.any() else np.nan
        ),
        "unsafe_predicted_valid_suppression_rate": (
            float((scored.loc[unsafe, "cause_decision"].astype(str) == "valid_suppression").mean())
            if unsafe.any()
            else 0.0
        ),
        "realized_post_recovery_tail_proxy_reduction": tail_reduction,
        "realized_post_recovery_expected_proxy_reduction": float(
            np.sum(scored["expected_loss_before"].to_numpy(float) - realized_expected)
        ),
        "base_network_bytes": int(base_bytes),
        "total_network_bytes": int(base_bytes + recovery_bytes),
        "base_modeled_energy_j": base_energy,
        "total_modeled_energy_j": total_energy,
        "tail_proxy_reduction_per_total_modeled_joule": (
            tail_reduction / total_energy if total_energy > 0 else np.nan
        ),
    }


STATISTICAL_METRICS: dict[str, bool] = {
    "post_recovery_critical_delivery_recall": True,
    "post_recovery_timely_critical_delivery_recall": True,
    "critical_false_trust_rate": False,
    "post_recovery_exact_retained_critical_item_recall": True,
    "realized_post_recovery_tail_proxy_reduction": True,
    "tail_proxy_reduction_per_total_modeled_joule": True,
    "total_network_bytes": False,
    "total_modeled_energy_j": False,
}


def _paired_bootstrap_interval(
    differences: np.ndarray, confidence: float, resamples: int, seed: int
) -> tuple[float, float]:
    if not len(differences):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def paired_comparisons(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Compute predeclared paired CIs, Wilcoxon tests, and Holm correction."""
    proposed = config["proposed_system"]
    references = config["paired_references"]
    keys = ["protocol", "site", "model_role", "scenario", "repetition"]
    rows: list[dict[str, Any]] = []
    confidence = float(config["statistics"]["confidence_level"])
    resamples = int(config["statistics"]["bootstrap_resamples"])
    base_seed = int(config["statistics"]["seed"])
    proposed_rows = metrics.loc[
        (metrics["scheduler_policy"] == str(proposed["scheduler"]))
        & (metrics["recovery_policy"] == str(proposed["recovery"]))
    ]
    for comparison_name, reference in references.items():
        reference_rows = metrics.loc[
            (metrics["scheduler_policy"] == str(reference["scheduler"]))
            & (metrics["recovery_policy"] == str(reference["recovery"]))
        ]
        joined = proposed_rows.merge(
            reference_rows,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_proposed", "_reference"),
        )
        for cell, group in joined.groupby(keys[:-1], sort=True):
            for metric, higher_is_better in STATISTICAL_METRICS.items():
                proposed_values = group[f"{metric}_proposed"].to_numpy(float)
                reference_values = group[f"{metric}_reference"].to_numpy(float)
                finite = np.isfinite(proposed_values) & np.isfinite(reference_values)
                difference = proposed_values[finite] - reference_values[finite]
                material = "|".join([comparison_name, *map(str, cell), metric]).encode()
                seed = (
                    base_seed + int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
                ) % (2**32 - 1)
                lower, upper = _paired_bootstrap_interval(difference, confidence, resamples, seed)
                if len(difference) and np.any(np.abs(difference) > 0):
                    p_value = float(
                        wilcoxon(difference, alternative="two-sided", zero_method="pratt").pvalue
                    )
                elif len(difference):
                    p_value = 1.0
                else:
                    p_value = np.nan
                mean_difference = float(np.mean(difference)) if len(difference) else np.nan
                favorable = mean_difference > 0 if higher_is_better else mean_difference < 0
                rows.append(
                    {
                        "comparison": comparison_name,
                        "protocol": cell[0],
                        "site": cell[1],
                        "model_role": cell[2],
                        "scenario": cell[3],
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "paired_repetitions": int(len(difference)),
                        "proposed_mean": (
                            float(np.mean(proposed_values[finite])) if finite.any() else np.nan
                        ),
                        "reference_mean": (
                            float(np.mean(reference_values[finite])) if finite.any() else np.nan
                        ),
                        "mean_paired_difference": mean_difference,
                        "bootstrap_ci_lower": lower,
                        "bootstrap_ci_upper": upper,
                        "wilcoxon_p_value": p_value,
                        "directionally_favorable": bool(favorable),
                    }
                )
    output = pd.DataFrame(rows)
    output["holm_adjusted_p_value"] = np.nan
    group_keys = ["comparison", "protocol", "site"]
    for _, indices in output.groupby(group_keys, sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        finite_positions = positions[
            np.isfinite(output.loc[positions, "wilcoxon_p_value"].to_numpy(float))
        ]
        order = finite_positions[
            np.argsort(output.loc[finite_positions, "wilcoxon_p_value"].to_numpy(float))
        ]
        adjusted_previous = 0.0
        total = len(order)
        for rank, position in enumerate(order):
            raw = float(output.at[position, "wilcoxon_p_value"])
            adjusted = min(1.0, max(adjusted_previous, (total - rank) * raw))
            output.at[position, "holm_adjusted_p_value"] = adjusted
            adjusted_previous = adjusted
    alpha = 1.0 - confidence
    output["reject_holm_null"] = output["holm_adjusted_p_value"] < alpha
    return output


def run_reintegration(
    integration_root: Path,
    scheduler_root: Path,
    fusion_root: Path,
    network_root: Path,
    output_root: Path,
    reintegration_config: dict[str, Any],
    integration_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    network_config: dict[str, Any],
    receiver_config: dict[str, Any],
    cause_config: dict[str, Any],
    recovery_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute Task R across frozen Task Q relay candidates and paired policies."""
    task_q, task_p, task_o, task_k, verified = verify_inputs(
        integration_root, scheduler_root, fusion_root, network_root
    )
    if task_q.get("proposed_policy") != integration_config["proposed_policy"]:
        raise ReintegrationError("Task Q proposed policy differs from the frozen configuration")
    if (
        receiver_config["receiver_version"] != "1.0"
        or cause_config["cause_version"] != "1.1"
        or recovery_config["recovery_version"] != "1.1"
    ):
        raise ReintegrationError("Task R requires frozen Tasks L-N logic versions")
    metrics: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    membership_cache: dict[tuple[str, str], pd.DataFrame] = {}
    proposed_trace_rows = 0
    q_records = [
        record
        for record in task_q["artifacts"]
        if all(record.get(key) for key in ("protocol", "site", "scenario"))
    ]
    local_network = deepcopy(network_config)
    local_network["hops"] = int(reintegration_config["relay_to_buoy_hops"])

    for q_record in q_records:
        protocol = str(q_record["protocol"])
        site = str(q_record["site"])
        scenario_name = str(q_record["scenario"])
        model_role = str(q_record["model_role"])
        scenario = network_config["scenarios"][scenario_name]
        p_record = _find_record(task_p, protocol, site, scenario_name, "P")
        k_record = _find_record(task_k, protocol, site, scenario_name, "K")
        schedule = pd.read_parquet(_resolve(p_record, scheduler_root, "P"))
        delivery = pd.read_parquet(_resolve(q_record, integration_root, "Q"))
        transport_k = pd.read_parquet(_resolve(k_record, network_root, "K"))
        for frame, column in (
            (schedule, "relay_flush_timestamp"),
            (delivery, "relay_flush_timestamp"),
            (transport_k, "timestamp"),
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        cache_key = (protocol, site)
        if cache_key not in membership_cache:
            membership_record = next(
                (
                    record
                    for record in task_o["artifacts"]
                    if record.get("protocol") == protocol
                    and record.get("site") == site
                    and record.get("artifact_kind") == "membership"
                ),
                None,
            )
            if membership_record is None:
                raise ReintegrationError(f"Missing Task O membership for {protocol}/{site}")
            membership_cache[cache_key] = _membership_metadata(
                pd.read_parquet(_resolve(membership_record, fusion_root, "O"))
            )
        metadata_columns = [
            "critical_items",
            "retained_critical_items",
            "retained_residual_magnitude_max",
        ]
        schedule = schedule.drop(
            columns=[column for column in metadata_columns if column in schedule.columns]
        ).merge(membership_cache[cache_key], on="batch_id", how="left", validate="many_to_one")
        if schedule["critical_items"].isna().any():
            raise ReintegrationError(f"Task P/Task O mismatch for {protocol}/{site}")
        keys = [
            "batch_id",
            "opportunity_id",
            "window_key",
            "relay_flush_timestamp",
            "repetition",
        ]
        delivery_columns = keys + [
            "selected",
            "actual_attempts",
            "actual_packet_delivered",
            "actual_packet_delivered_timely",
            "actual_acknowledged",
            "actual_first_delivery_delay_seconds",
            "actual_network_bytes",
            "actual_modeled_energy_j",
        ]
        proposed = schedule.merge(
            delivery[delivery_columns],
            on=keys,
            how="left",
            validate="one_to_one",
            suffixes=("", "_task_q"),
        )
        if proposed["actual_packet_delivered_timely"].isna().any():
            raise ReintegrationError("Task Q delivery trace does not align with Task P")
        if not np.array_equal(
            proposed["selected"].to_numpy(int),
            proposed["selected_task_q"].to_numpy(int),
        ):
            raise ReintegrationError("Task Q proposed decisions differ from frozen Task P")
        proposed = proposed.drop(columns=["selected_task_q"])

        oracle_columns = [
            "timestamp",
            "window_key",
            "repetition",
            "realization_seed",
            "_oracle_link_state",
            *[
                f"_oracle_{kind}_u{attempt}"
                for attempt in range(int(network_config["maximum_attempts"]))
                for kind in ("data", "ack", "jitter")
            ],
        ]
        oracle = transport_k[oracle_columns].copy()
        if oracle.duplicated(["window_key", "timestamp", "repetition"]).any():
            raise ReintegrationError("Task K oracle replay keys are not unique")

        reference_parts = [
            schedule_reference(group, REFERENCE_POLICY, scheduler_config)
            for _, group in schedule.groupby(["repetition", "opportunity_id"], sort=False)
        ]
        expected = pd.concat(reference_parts, ignore_index=True)
        scheduler_frames = {
            PROPOSED_POLICY: proposed,
            REFERENCE_POLICY: expected,
        }
        grid_minutes = int(preprocessing_config["protocols"][protocol]["grid_minutes"])
        proposed_system_parts: list[pd.DataFrame] = []
        for scheduler_policy, scheduler_frame in scheduler_frames.items():
            decision_columns = [
                column
                for column in scheduler_frame.columns
                if not column.startswith("_oracle_") and column != "timestamp"
            ]
            with_oracle = scheduler_frame[decision_columns].merge(
                oracle,
                left_on=["window_key", "relay_flush_timestamp", "repetition"],
                right_on=["window_key", "timestamp", "repetition"],
                how="left",
                validate="many_to_one",
            )
            if with_oracle["_oracle_link_state"].isna().any():
                raise ReintegrationError("Task P candidates do not align with Task K oracle")
            if scheduler_policy == REFERENCE_POLICY:
                with_oracle = replay_selected(
                    with_oracle,
                    scenario,
                    network_config,
                    int(reintegration_config["relay_to_buoy_hops"]),
                    grid_minutes * 60.0,
                )
            relay_transport = build_relay_transport(
                with_oracle, reintegration_config["relay_receiver_contract"]
            )
            inferred = reintegrate_receiver_and_cause(
                relay_transport, receiver_config, cause_config
            )
            scoring = with_oracle.copy()
            scoring["timestamp"] = pd.to_datetime(scoring["relay_flush_timestamp"], utc=True)
            for recovery_policy in reintegration_config["recovery_policies"]:
                replay = replay_recovery(
                    inferred,
                    relay_transport,
                    str(recovery_policy),
                    recovery_config,
                    local_network,
                    scenario,
                    grid_minutes,
                )
                for repetition, group in replay.groupby("repetition", sort=False):
                    evidence_group = scoring.loc[scoring["repetition"] == int(repetition)]
                    metrics.append(
                        _system_metrics(
                            group,
                            evidence_group,
                            protocol,
                            site,
                            model_role,
                            scenario_name,
                            int(repetition),
                            scheduler_policy,
                            str(recovery_policy),
                        )
                    )
                if scheduler_policy == str(
                    reintegration_config["proposed_system"]["scheduler"]
                ) and str(recovery_policy) == str(
                    reintegration_config["proposed_system"]["recovery"]
                ):
                    keep = [
                        "timestamp",
                        "window_key",
                        "repetition",
                        "packet_arrived",
                        "packet_acknowledged",
                        "critical_packet_id",
                        "critical_packet_wire_bytes",
                        "receiver_state",
                        "connectivity_belief",
                        "environmental_uncertainty",
                        "receiver_contract_valid",
                        "receiver_contact_due",
                        "cause_decision",
                        "recovery_triggered",
                        "recovery_attempt_planned",
                        "recovery_succeeded",
                        "critical_retransmission_succeeded",
                        "recovery_response_kind",
                        "recovery_response_id",
                        "recovery_response_wire_bytes",
                        "recovery_retransmitted_critical_packet_id",
                        "critical_event_recovery_response_id",
                        "critical_event_recovery_response_timestamp",
                        "critical_event_recovery_response_wire_bytes",
                        "critical_event_recovery_delay_steps",
                        "recovery_cache_depth_before_response",
                        "recovery_cache_expired_items",
                        "post_recovery_packet_delivered_within_retention",
                        "post_recovery_packet_delivered_timely",
                        "post_recovery_state",
                        "evaluation_only_unsafe_silence",
                        "evaluation_only_post_recovery_false_trust",
                        "evaluation_only_critical_recovered",
                        "evaluation_only_critical_recovered_timely",
                        "recovery_network_bytes",
                        "recovery_energy_j",
                    ]
                    proposed_system_parts.append(replay[keep].copy())
        proposed_trace = pd.concat(proposed_system_parts, ignore_index=True)
        proposed_trace.insert(0, "reintegration_version", "1.0")
        proposed_trace.insert(1, "scenario", scenario_name)
        output_path = output_root / "system" / protocol / site / f"{scenario_name}.parquet"
        _atomic_parquet(output_path, proposed_trace)
        artifacts.append(
            {
                "protocol": protocol,
                "site": site,
                "model_role": model_role,
                "scenario": scenario_name,
                "path": str(output_path),
                "rows": int(len(proposed_trace)),
                "sha256": sha256_file(output_path),
            }
        )
        proposed_trace_rows += len(proposed_trace)

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = output_root / "system_metrics.csv"
    _atomic_csv(metrics_path, metrics_frame)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})
    statistics = paired_comparisons(metrics_frame, reintegration_config)
    statistics_path = output_root / "paired_comparisons.csv"
    _atomic_csv(statistics_path, statistics)
    artifacts.append({"path": str(statistics_path), "sha256": sha256_file(statistics_path)})

    config_hash = hashlib.sha256(
        json.dumps(reintegration_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "R",
        "reintegration_version": str(reintegration_config["reintegration_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "upstream_artifacts_reverified": verified,
        "task_q_integration_version": task_q["integration_version"],
        "task_p_scheduler_version": task_p["scheduler_version"],
        "task_o_fusion_version": task_o["fusion_version"],
        "task_k_network_version": task_k["network_version"],
        "receiver_logic_version": receiver_config["receiver_version"],
        "cause_logic_version": cause_config["cause_version"],
        "recovery_logic_version": recovery_config["recovery_version"],
        "proposed_system": reintegration_config["proposed_system"],
        "paired_references": reintegration_config["paired_references"],
        "receiver_uses_oracle_fields": False,
        "receiver_uses_event_truth": False,
        "receiver_uses_scheduler_selection_when_packet_missing": False,
        "recovery_policy_uses_oracle_fields": False,
        "recovery_policy_uses_event_truth": False,
        "future_measurements_used": False,
        "parameters_tuned_on_test_or_external_data": False,
        "temporal_protocols_pooled": False,
        "missing_critical_summary_scored_unsafe": True,
        "matched_counterfactual_delivery_and_recovery": True,
        "recovery_attempts_bounded": True,
        "maximum_critical_packets_per_response": 1,
        "critical_response_uses_original_wire_bytes": True,
        "exact_event_response_provenance": True,
        "late_recovery_excluded_from_tail_proxy_reduction": True,
        "paired_unit": "repetition",
        "bootstrap_confidence_level": float(reintegration_config["statistics"]["confidence_level"]),
        "bootstrap_resamples": int(reintegration_config["statistics"]["bootstrap_resamples"]),
        "multiplicity_correction": str(
            reintegration_config["statistics"]["multiplicity_correction"]
        ),
        "energy_is_modeled_not_hardware_measured": True,
        "actual_field_deployment_claimed": False,
        "claim_scope": str(reintegration_config["reporting"]["claim_scope"]),
        "metrics_rows": int(len(metrics_frame)),
        "paired_comparison_rows": int(len(statistics)),
        "proposed_system_trace_rows": int(proposed_trace_rows),
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "task_r_summary.json", summary)
    return summary
