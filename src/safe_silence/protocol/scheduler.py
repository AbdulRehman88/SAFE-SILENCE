"""Task P causal tail-risk-reduction-per-joule relay scheduling."""

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


class SchedulerError(RuntimeError):
    """Raised when Task P cannot preserve its causal scheduling contract."""


PROPOSED_POLICY = "cvar_risk_reduction_per_joule"
REFERENCE_POLICY = "expected_risk_reduction_per_joule"
ALL_POLICIES = (PROPOSED_POLICY, REFERENCE_POLICY)


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


def _resolve_fusion_artifact(record: dict[str, Any], root: Path) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    if all(record.get(key) for key in ("protocol", "site", "artifact_kind")):
        folder = "membership" if record["artifact_kind"] == "membership" else "summaries"
        return (
            root
            / folder
            / str(record["protocol"])
            / f"{record['site']}.parquet"
        )
    return root / declared.name


def _resolve_network_artifact(record: dict[str, Any], root: Path) -> Path:
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


def verify_inputs(
    fusion_root: Path, network_root: Path
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Reverify every frozen Task O and Task K artifact before scheduling."""
    fusion_summary_path = fusion_root / "task_o_summary.json"
    network_summary_path = network_root / "task_k_summary.json"
    if not fusion_summary_path.exists() or not network_summary_path.exists():
        raise SchedulerError("Task P requires Task O and Task K summaries")
    with fusion_summary_path.open("r", encoding="utf-8") as handle:
        fusion_summary = json.load(handle)
    with network_summary_path.open("r", encoding="utf-8") as handle:
        network_summary = json.load(handle)
    if fusion_summary.get("task") != "O" or fusion_summary.get("fusion_version") != "1.0":
        raise SchedulerError("Task P requires frozen Task O fusion contract v1.0")
    if network_summary.get("task") != "K" or network_summary.get("network_version") != "1.0":
        raise SchedulerError("Task P requires frozen Task K network contract v1.0")

    verified_fusion = 0
    for record in fusion_summary.get("artifacts", []):
        path = _resolve_fusion_artifact(record, fusion_root)
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise SchedulerError(f"Task O artifact changed or is missing: {path}")
        verified_fusion += 1
    verified_network = 0
    for record in network_summary.get("artifacts", []):
        path = _resolve_network_artifact(record, network_root)
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise SchedulerError(f"Task K artifact changed or is missing: {path}")
        verified_network += 1
    if not verified_fusion or not verified_network:
        raise SchedulerError("Task P found no verifiable upstream artifacts")
    return fusion_summary, network_summary, verified_fusion, verified_network


def upper_cvar(values: np.ndarray, alpha: float) -> float:
    """Return empirical upper-tail CVaR with fractional mass at the quantile boundary."""
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        raise SchedulerError("CVaR requires at least one finite loss sample")
    if not 0.0 < alpha < 1.0:
        raise SchedulerError("CVaR alpha must lie strictly between zero and one")
    ordered = np.sort(sample)[::-1]
    tail_mass = (1.0 - alpha) * len(ordered)
    whole = int(np.floor(tail_mass))
    fraction = tail_mass - whole
    total = float(ordered[:whole].sum()) if whole else 0.0
    if fraction > 0.0:
        total += fraction * float(ordered[whole])
    return total / tail_mass


def causal_ack_beliefs(transport: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Estimate reachability using only ACK outcomes strictly preceding each row."""
    required = {"timestamp", "window_key", "repetition", "packet_attempted", "acknowledged"}
    missing = required.difference(transport.columns)
    if missing:
        raise SchedulerError(f"Task K trace lacks causal ACK fields: {sorted(missing)}")
    selected = transport[list(required)].copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True)
    selected = selected.sort_values(["repetition", "window_key", "timestamp"]).reset_index(
        drop=True
    )
    belief = np.zeros(len(selected), dtype=np.float32)
    initial = float(config["initial_probability"])
    gain = float(config["update_gain"])
    lower = float(config["minimum_probability"])
    upper = float(config["maximum_probability"])
    for _, indices in selected.groupby(["repetition", "window_key"], sort=False).groups.items():
        current = initial
        for index in indices:
            belief[index] = np.float32(current)
            if bool(selected.at[index, "packet_attempted"]):
                outcome = float(bool(selected.at[index, "acknowledged"]))
                current = float(np.clip((1.0 - gain) * current + gain * outcome, lower, upper))
    return pd.DataFrame(
        {
            "timestamp": selected["timestamp"],
            "window_key": selected["window_key"].astype(str),
            "repetition": selected["repetition"].astype(int),
            "causal_ack_belief": belief,
        }
    )


def _expected_costs(
    encoded_bytes: int,
    success_probability: float,
    scheduler_config: dict[str, Any],
    network_config: dict[str, Any],
) -> dict[str, float]:
    probability = float(np.clip(success_probability, 0.0, 1.0))
    attempts_limit = int(network_config["maximum_attempts"])
    expected_attempts = float(sum((1.0 - probability) ** k for k in range(attempts_limit)))
    ack_wire_bytes = int(network_config["packet_header_bytes"]) + int(network_config["ack_bytes"])
    expected_bytes_per_attempt = float(encoded_bytes) + probability * ack_wire_bytes
    expected_wire_bytes = expected_attempts * expected_bytes_per_attempt
    bitrate = float(network_config["bitrate_bps"])
    hops = int(scheduler_config["energy_model"]["relay_to_buoy_hops"])
    power = float(network_config["energy"]["tx_watts"]) + float(
        network_config["energy"]["rx_watts"]
    )
    expected_energy = hops * expected_wire_bytes * 8.0 / bitrate * power
    propagation = (
        hops
        * float(network_config["nominal_distance_per_hop_m"])
        / float(network_config["propagation_speed_m_s"])
    )
    expected_airtime = expected_attempts * (
        expected_bytes_per_attempt * 8.0 / bitrate + propagation * (1.0 + probability)
    )
    cost = scheduler_config["cost"]
    denominator = (
        float(cost["energy_weight"]) * expected_energy
        + float(cost["byte_weight"]) * expected_wire_bytes
        + float(cost["airtime_weight"]) * expected_airtime
        + float(cost["epsilon"])
    )
    return {
        "expected_attempts": expected_attempts,
        "expected_wire_bytes": expected_wire_bytes,
        "modeled_energy_j": expected_energy,
        "expected_airtime_seconds": expected_airtime,
        "cost_denominator": denominator,
    }


def candidate_values(
    summary: dict[str, Any],
    membership: pd.DataFrame,
    link_belief: float,
    loss_weights: dict[str, Any],
    scheduler_config: dict[str, Any],
    network_config: dict[str, Any],
) -> dict[str, Any]:
    """Compute causal CVaR and mean marginal-risk values for one fused summary."""
    forbidden = [
        key for key in summary if str(key).startswith(("_oracle_", "evaluation_only_"))
    ]
    if forbidden:
        raise SchedulerError(f"Forbidden scheduler evidence supplied: {sorted(forbidden)}")
    columns = {
        "critical_evidence_score",
        "deadline_urgency",
        "residual_magnitude",
        "environmental_uncertainty",
        "retained",
    }
    missing = columns.difference(membership.columns)
    if missing:
        raise SchedulerError(f"Task O membership lacks scheduler evidence: {sorted(missing)}")
    if not len(membership):
        raise SchedulerError("A fused candidate cannot have empty membership")

    critical = membership["critical_evidence_score"].to_numpy(float)
    deadline = membership["deadline_urgency"].to_numpy(float)
    residual = membership["residual_magnitude"].to_numpy(float)
    uncertainty = membership["environmental_uncertainty"].to_numpy(float)
    retained = membership["retained"].to_numpy(bool)
    ambiguity = uncertainty * (1.0 - float(link_belief))
    before = (
        float(loss_weights["missed_critical_event"]) * critical
        + float(loss_weights["notification_delay"]) * deadline
        + float(loss_weights["reconstruction"]) * residual
        + float(loss_weights["connectivity_ambiguity"]) * ambiguity
    )
    after_success = float(loss_weights["reconstruction"]) * residual * (~retained)
    alpha = float(scheduler_config["cvar_alpha"])
    tail_before = upper_cvar(before, alpha)
    tail_after = upper_cvar(after_success, alpha)
    mean_before = float(np.mean(before))
    mean_after = float(np.mean(after_success))
    probability = float(link_belief)
    tail_delta = probability * max(tail_before - tail_after, 0.0)
    mean_delta = probability * max(mean_before - mean_after, 0.0)
    costs = _expected_costs(
        int(summary["encoded_wire_bytes"]), probability, scheduler_config, network_config
    )
    mandatory_config = scheduler_config["mandatory"]
    mandatory_critical = float(summary["critical_evidence_max"]) >= float(
        mandatory_config["critical_evidence_threshold"]
    )
    mandatory_deadline = float(summary["deadline_urgency_max"]) >= float(
        mandatory_config["deadline_urgency_threshold"]
    )
    return {
        **summary,
        "causal_ack_belief": probability,
        "tail_loss_before": tail_before,
        "tail_loss_after_success": tail_after,
        "expected_loss_before": mean_before,
        "expected_loss_after_success": mean_after,
        "predicted_tail_risk_reduction": tail_delta,
        "predicted_expected_risk_reduction": mean_delta,
        **costs,
        "tail_utility": tail_delta / costs["cost_denominator"],
        "expected_utility": mean_delta / costs["cost_denominator"],
        "mandatory_critical": mandatory_critical,
        "mandatory_deadline": mandatory_deadline,
        "mandatory_candidate": mandatory_critical or mandatory_deadline,
    }


def _feasible(totals: dict[str, float], candidate: dict[str, Any], config: dict[str, Any]) -> bool:
    budgets = config["budgets"]
    queue = config["queue_stress"]
    return (
        totals["packets"] + 1 <= int(queue["maximum_selected_packets"])
        and totals["bytes"] + float(candidate["expected_wire_bytes"])
        <= float(budgets["expected_wire_bytes_per_opportunity"]) + 1e-9
        and totals["energy"] + float(candidate["modeled_energy_j"])
        <= float(budgets["modeled_energy_j_per_opportunity"]) + 1e-9
        and totals["airtime"] + float(candidate["expected_airtime_seconds"])
        <= float(budgets["expected_airtime_seconds_per_opportunity"]) + 1e-9
    )


def schedule_opportunity(
    candidates: list[dict[str, Any]], policy: str, config: dict[str, Any]
) -> pd.DataFrame:
    """Greedily select one constrained opportunity with deterministic tie-breaking."""
    if policy not in ALL_POLICIES:
        raise SchedulerError(f"Unknown scheduling policy: {policy}")
    if not candidates:
        raise SchedulerError("A scheduling opportunity cannot be empty")
    utility_key = "tail_utility" if policy == PROPOSED_POLICY else "expected_utility"
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row[utility_key]),
            pd.Timestamp(row["relay_flush_timestamp"]),
            str(row["batch_id"]),
        ),
    )
    selected: set[str] = set()
    reasons: dict[str, str] = {}
    totals = {"packets": 0.0, "bytes": 0.0, "energy": 0.0, "airtime": 0.0}

    def try_select(candidate: dict[str, Any], reason: str) -> None:
        if str(candidate["batch_id"]) in selected or float(candidate[utility_key]) <= 0.0:
            return
        if not _feasible(totals, candidate, config):
            return
        item_id = str(candidate["batch_id"])
        selected.add(item_id)
        reasons[item_id] = reason
        totals["packets"] += 1
        totals["bytes"] += float(candidate["expected_wire_bytes"])
        totals["energy"] += float(candidate["modeled_energy_j"])
        totals["airtime"] += float(candidate["expected_airtime_seconds"])

    mandatory = [candidate for candidate in ordered if bool(candidate["mandatory_candidate"])]
    reserve = config["liveness_reserve"]
    if bool(reserve["enabled"]) and (
        not bool(reserve["applies_when_no_mandatory_candidate"]) or not mandatory
    ):
        eligible = [
            candidate
            for candidate in candidates
            if float(candidate["environmental_uncertainty_max"])
            >= float(reserve["uncertainty_threshold"])
        ]
        eligible.sort(
            key=lambda row: (
                -float(row["environmental_uncertainty_max"]),
                -float(row[utility_key]),
                pd.Timestamp(row["relay_flush_timestamp"]),
                str(row["batch_id"]),
            )
        )
        for candidate in eligible[: int(reserve["reserve_packets"])]:
            try_select(candidate, "liveness_reserve")
    for candidate in mandatory:
        try_select(candidate, "mandatory_safety")
    for candidate in ordered:
        try_select(candidate, "ranked_utility")

    output = pd.DataFrame(candidates).copy()
    output["policy"] = policy
    output["policy_utility"] = output[utility_key].astype(float)
    output["selected"] = output["batch_id"].astype(str).isin(selected).astype(np.uint8)
    output["selection_reason"] = [
        reasons.get(
            str(batch_id),
            "nonpositive_utility" if float(utility) <= 0.0 else "budget_or_rank_excluded",
        )
        for batch_id, utility in zip(
            output["batch_id"], output[utility_key], strict=True
        )
    ]
    output["opportunity_selected_packets"] = int(totals["packets"])
    output["opportunity_expected_wire_bytes"] = totals["bytes"]
    output["opportunity_modeled_energy_j"] = totals["energy"]
    output["opportunity_expected_airtime_seconds"] = totals["airtime"]
    return output


def scheduler_metrics(
    frame: pd.DataFrame,
    protocol: str,
    site: str,
    model_role: str,
    scenario: str,
    repetition: int,
    loss_profile: str,
    policy: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = frame["selected"].to_numpy(bool)
    critical = frame["mandatory_critical"].to_numpy(bool)
    mandatory = frame["mandatory_candidate"].to_numpy(bool)
    opportunities = frame["opportunity_id"].nunique()
    selected_rows = frame.loc[selected]
    opportunity_totals = frame.drop_duplicates("opportunity_id")
    budgets = config["budgets"]
    queue = config["queue_stress"]
    violations = (
        (
            opportunity_totals["opportunity_selected_packets"]
            > int(queue["maximum_selected_packets"])
        )
        | (
            opportunity_totals["opportunity_expected_wire_bytes"]
            > float(budgets["expected_wire_bytes_per_opportunity"]) + 1e-9
        )
        | (
            opportunity_totals["opportunity_modeled_energy_j"]
            > float(budgets["modeled_energy_j_per_opportunity"]) + 1e-9
        )
        | (
            opportunity_totals["opportunity_expected_airtime_seconds"]
            > float(budgets["expected_airtime_seconds_per_opportunity"]) + 1e-9
        )
    )
    return {
        "protocol": protocol,
        "site": site,
        "model_role": model_role,
        "scenario": scenario,
        "repetition": repetition,
        "loss_profile": loss_profile,
        "policy": policy,
        "opportunities": int(opportunities),
        "candidates": int(len(frame)),
        "selected_candidates": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "critical_candidates": int(critical.sum()),
        "critical_candidate_selection_recall": (
            float(selected[critical].mean()) if critical.any() else np.nan
        ),
        "mandatory_candidates": int(mandatory.sum()),
        "mandatory_candidate_selection_recall": (
            float(selected[mandatory].mean()) if mandatory.any() else np.nan
        ),
        "liveness_reserve_selections": int(
            ((frame["selection_reason"] == "liveness_reserve") & selected).sum()
        ),
        "achieved_predicted_tail_risk_reduction": float(
            selected_rows["predicted_tail_risk_reduction"].sum()
        ),
        "achieved_predicted_expected_risk_reduction": float(
            selected_rows["predicted_expected_risk_reduction"].sum()
        ),
        "expected_wire_bytes": float(selected_rows["expected_wire_bytes"].sum()),
        "modeled_energy_j": float(selected_rows["modeled_energy_j"].sum()),
        "expected_airtime_seconds": float(
            selected_rows["expected_airtime_seconds"].sum()
        ),
        "tail_risk_reduction_per_modeled_joule": (
            float(
                selected_rows["predicted_tail_risk_reduction"].sum()
                / selected_rows["modeled_energy_j"].sum()
            )
            if selected_rows["modeled_energy_j"].sum() > 0
            else np.nan
        ),
        "mean_causal_ack_belief": float(frame["causal_ack_belief"].mean()),
        "budget_violations": int(violations.sum()),
    }


def _load_fusion_site(
    fusion_summary: dict[str, Any], fusion_root: Path, protocol: str, site: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records = [
        record
        for record in fusion_summary["artifacts"]
        if record.get("protocol") == protocol and record.get("site") == site
    ]
    by_kind = {str(record["artifact_kind"]): record for record in records}
    if set(by_kind) != {"membership", "summary"}:
        raise SchedulerError(f"Task O lacks membership/summary pair for {protocol}/{site}")
    summaries = pd.read_parquet(_resolve_fusion_artifact(by_kind["summary"], fusion_root))
    membership = pd.read_parquet(_resolve_fusion_artifact(by_kind["membership"], fusion_root))
    summaries["relay_flush_timestamp"] = pd.to_datetime(
        summaries["relay_flush_timestamp"], utc=True
    )
    if summaries["batch_id"].duplicated().any():
        raise SchedulerError(f"Task O has duplicate batch IDs for {protocol}/{site}")
    groups = {
        str(batch_id): group.reset_index(drop=True)
        for batch_id, group in membership.groupby("batch_id", sort=False)
    }
    if set(summaries["batch_id"].astype(str)) != set(groups):
        raise SchedulerError(f"Task O membership/summary batch mismatch for {protocol}/{site}")
    return summaries, groups


def run_scheduler(
    fusion_root: Path,
    network_root: Path,
    output_root: Path,
    scheduler_config: dict[str, Any],
    network_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute Task P over matched Task O candidates and causal Task K ACK histories."""
    fusion_summary, network_summary, verified_o, verified_k = verify_inputs(
        fusion_root, network_root
    )
    metrics: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    site_cache: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}
    trace_records = [
        record
        for record in network_summary["artifacts"]
        if all(record.get(key) for key in ("protocol", "site", "scenario"))
    ]
    trace_rows = 0
    for record in trace_records:
        protocol = str(record["protocol"])
        site = str(record["site"])
        scenario = str(record["scenario"])
        model_role = str(record["model_role"])
        cache_key = (protocol, site)
        if cache_key not in site_cache:
            site_cache[cache_key] = _load_fusion_site(
                fusion_summary, fusion_root, protocol, site
            )
        summaries, membership_groups = site_cache[cache_key]
        candidate_summaries = summaries.loc[
            summaries["model_role"].astype(str) == model_role
        ].copy()
        transport = pd.read_parquet(_resolve_network_artifact(record, network_root))
        beliefs = causal_ack_beliefs(transport, scheduler_config["causal_link_belief"])
        joined = candidate_summaries.merge(
            beliefs,
            left_on=["window_key", "relay_flush_timestamp"],
            right_on=["window_key", "timestamp"],
            how="inner",
            validate="many_to_many",
        )
        expected_rows = len(candidate_summaries) * transport["repetition"].nunique()
        if len(joined) != expected_rows:
            raise SchedulerError(
                f"Task O candidates did not align with causal Task K history for "
                f"{protocol}/{site}/{scenario}: {len(joined)} != {expected_rows}"
            )
        proposed_trace_parts: list[pd.DataFrame] = []
        for profile_name, weights in scheduler_config["loss_profiles"].items():
            for repetition, repetition_frame in joined.groupby("repetition", sort=False):
                policy_parts: dict[str, list[pd.DataFrame]] = {
                    policy: [] for policy in ALL_POLICIES
                }
                opportunity_sequence = 0
                for _, window in repetition_frame.groupby("window_key", sort=False):
                    window = window.sort_values("relay_flush_timestamp")
                    records_for_window = window.to_dict("records")
                    queue_size = int(
                        scheduler_config["queue_stress"]["candidates_per_opportunity"]
                    )
                    for start in range(0, len(records_for_window), queue_size):
                        queue_records = records_for_window[start : start + queue_size]
                        candidates = [
                            candidate_values(
                                summary,
                                membership_groups[str(summary["batch_id"])],
                                float(summary["causal_ack_belief"]),
                                weights,
                                scheduler_config,
                                network_config,
                            )
                            for summary in queue_records
                        ]
                        opportunity_id = hashlib.sha256(
                            (
                                f"{protocol}|{site}|{scenario}|{repetition}|"
                                f"{profile_name}|{opportunity_sequence}|"
                                + "|".join(str(row["batch_id"]) for row in candidates)
                            ).encode()
                        ).hexdigest()[:24]
                        for policy in ALL_POLICIES:
                            scheduled = schedule_opportunity(candidates, policy, scheduler_config)
                            scheduled["opportunity_id"] = opportunity_id
                            policy_parts[policy].append(scheduled)
                        opportunity_sequence += 1
                for policy, parts in policy_parts.items():
                    scheduled_frame = pd.concat(parts, ignore_index=True)
                    metrics.append(
                        scheduler_metrics(
                            scheduled_frame,
                            protocol,
                            site,
                            model_role,
                            scenario,
                            int(repetition),
                            str(profile_name),
                            policy,
                            scheduler_config,
                        )
                    )
                    if profile_name == "primary" and policy == PROPOSED_POLICY:
                        scheduled_frame.insert(0, "scheduler_version", "1.0")
                        scheduled_frame["scenario"] = scenario
                        scheduled_frame["loss_profile"] = str(profile_name)
                        proposed_trace_parts.append(scheduled_frame)
        proposed_trace = pd.concat(proposed_trace_parts, ignore_index=True)
        output_path = output_root / "schedules" / protocol / site / f"{scenario}.parquet"
        _atomic_parquet(output_path, proposed_trace)
        artifacts.append(
            {
                "protocol": protocol,
                "site": site,
                "model_role": model_role,
                "scenario": scenario,
                "path": str(output_path),
                "rows": int(len(proposed_trace)),
                "sha256": sha256_file(output_path),
            }
        )
        trace_rows += len(proposed_trace)

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = output_root / "scheduler_metrics.csv"
    _atomic_csv(metrics_path, metrics_frame)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})
    if int(metrics_frame["budget_violations"].sum()) != 0:
        raise SchedulerError("Task P produced a budget violation")

    config_hash = hashlib.sha256(
        json.dumps(scheduler_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "P",
        "scheduler_version": str(scheduler_config["scheduler_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_o_artifacts_reverified": verified_o,
        "task_k_artifacts_reverified": verified_k,
        "task_o_fusion_version": fusion_summary["fusion_version"],
        "task_k_network_version": network_summary["network_version"],
        "proposed_policy": PROPOSED_POLICY,
        "reference_policy": REFERENCE_POLICY,
        "cvar_alpha": float(scheduler_config["cvar_alpha"]),
        "loss_profiles": list(scheduler_config["loss_profiles"]),
        "causal_link_belief_uses_only_lagged_ack_history": True,
        "uses_current_candidate_outcome": False,
        "uses_oracle_fields": False,
        "uses_future_measurements": False,
        "parameters_tuned_on_test_or_external_data": False,
        "temporal_protocols_pooled": False,
        "budget_violations": 0,
        "modeled_energy_not_hardware_measured": True,
        "calibration_loss_not_scored_without_probability_forecast": True,
        "component_evaluation_only": True,
        "actual_delivery_replay_deferred": True,
        "matched_cost_full_system_comparison_deferred_to_task_q": True,
        "proposed_trace_rows": int(trace_rows),
        "metrics_rows": int(len(metrics_frame)),
        "artifacts": artifacts,
    }
    summary_path = output_root / "task_p_summary.json"
    _atomic_json(summary_path, summary)
    return summary
