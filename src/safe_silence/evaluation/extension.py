"""Frozen, resumable SAFE-SILENCE confirmation experiments.

The driver writes experiment results to the configured output directory. Source-grid
trust ablations are primary mechanism evidence. Relay trust ablations are secondary.
All policies within a cell reuse identical physical realizations. Resource comparisons
use common caps and report actual consumption; they are not labelled exactly matched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml

from safe_silence.data.preprocess import sha256_file
from safe_silence.simulation.network import simulate_network, _derived_seed, verify_task_j_inputs
from safe_silence.protocol.scheduler import run_scheduler, PROPOSED_POLICY, REFERENCE_POLICY
from safe_silence.evaluation.integration import run_integration, schedule_reference, replay_selected
from safe_silence.evaluation.reintegration import build_relay_transport, _membership_metadata, _system_metrics
from safe_silence.protocol.receiver import interpret_receiver as original_receiver, PROTECTIVE_STATES
from safe_silence.protocol.receiver_ablation import interpret_receiver
from safe_silence.protocol.cause_ablation import infer_causes
from safe_silence.protocol.cause import infer_causes as original_causes
from safe_silence.protocol.recovery_budgeted import replay_recovery, recovery_metrics

VERSION = "0.21.0"

def load_configs(root: Path) -> dict:
    result = {}
    for path in sorted((root / "configs").glob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            result.update(value)
    return result


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def hash_object(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()



def protocol_identity(lock: dict) -> str:
    """Return the archived experiment identity for a scientific release projection.

    A release projection omits non-computational sections and records current
    release source hashes. Its checkpoint identity identifies the archived
    experiment snapshot; release source integrity is verified independently.
    """
    projection = lock.get("release_projection")
    if projection is None:
        return hash_object(lock)
    if projection.get("schema_version") != 1:
        raise ValueError("Unsupported release projection")
    identity = projection.get("archived_experiment_identity", "")
    if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Invalid archived experiment identity")
    return identity


def local_record(root: Path, record: dict) -> Path:
    declared = str(record["path"]).replace("\\", "/")
    marker = "/IEEE_CM/"
    if marker in declared:
        return root / declared.split(marker, 1)[1]
    direct = Path(declared)
    if direct.is_file():
        return direct
    raise RuntimeError(f"Cannot resolve upstream artifact: {declared}")


def verify_recorded(root: Path, summary_path: Path) -> int:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    count = 0
    for record in summary.get("artifacts", []):
        path = local_record(root, record)
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Upstream recorded hash mismatch: {path}")
        count += 1
    return count


def frozen_protocol(root: Path, configs: dict) -> dict:
    ext = configs["extension"]
    files = sorted((root / "configs").glob("*.yaml")) + sorted((root / "src/safe_silence").rglob("*.py")) + [root / "scripts/24_scientific_extension.py", root / "tests/test_scientific_extension.py"]
    verify_paths = [
        root / "results/task_j/task_j_summary.json",
        root / "results/task_o/task_o_summary.json",
        root / "results/task_s/task_s_summary.json",
        root / "results/task_s/replication/task_r/task_r_summary.json",
    ]
    verified = {str(path.relative_to(root)): verify_recorded(root, path) for path in verify_paths}
    budgets = {}
    sources = {"relay": "results/task_r/system_metrics.csv", "source": "results/task_n/recovery_metrics.csv"}
    for scope, relative in sources.items():
        path = root / relative
        frame = pd.read_csv(path)
        chosen = frame[(frame.site == ext["development_site"]) & (frame.model_role == ext["development_role"])]
        if scope == "relay":
            chosen = chosen[(chosen.scheduler_policy == PROPOSED_POLICY) & (chosen.recovery_policy == "safe_silence_bounded")]
        else:
            chosen = chosen[chosen.policy == "safe_silence_bounded"]
        if chosen.empty or (chosen.rows <= 0).any():
            raise RuntimeError(f"No valid development calibration rows in {relative}")
        budgets[scope] = {}
        for protocol in ext["protocols"]:
            part = chosen[chosen.protocol == protocol]
            if part.empty:
                raise RuntimeError(f"Development budget missing: {scope}/{protocol}")
            byte_rate = float(np.quantile(part.recovery_network_bytes / part.rows, ext["budget_quantile"], method="linear"))
            energy_rate = float(np.quantile(part.recovery_energy_j / part.rows, ext["budget_quantile"], method="linear"))
            if scope == "source":
                response_bound = configs["network"]["packet_header_bytes"] + max(
                    value for key, value in configs["semantic"]["payload_bytes"].items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                )
            else:
                fusion = configs["fusion"]
                sizes = fusion["encoding_bytes"]
                response_bound = sizes["relay_packet_header"] + sizes["fixed_summary_metadata"] + fusion["top_k"] * sizes["retained_item"] + fusion["relay_batch_items"] * sizes["provenance_per_source"]
            response_bound = max(response_bound, configs["network"]["packet_header_bytes"] + configs["recovery"]["packet_bytes"]["liveness_response_payload"])
            budgets[scope][protocol] = {
                "bytes_per_opportunity": byte_rate,
                "energy_j_per_opportunity": energy_rate,
                "maximum_response_wire_bytes": int(response_bound),
                "calibration_rows": int(len(part)),
                "calibration_file": relative,
                "calibration_file_sha256": sha256_file(path),
            }
        files.append(path)
    previous = set()
    new_seeds = []
    for protocol in ext["protocols"]:
        for scenario in ext["scenarios"]:
            for repetition in range(ext["repetitions"]):
                parts = (protocol, ext["primary_site"], ext["primary_role"], scenario, repetition)
                new_seeds.append(_derived_seed(ext["new_network_base_seed"], *parts))
    # Compare against all previous site/role/protocol/repetition combinations, not just new CE07 cells.
    summary = json.loads((root / "results/task_j/task_j_summary.json").read_text())
    for base in ext["previous_network_base_seeds"]:
        for protocol, data in summary["protocols"].items():
            for site in data["sites"]:
                role = "internal_test" if site == ext["development_site"] else "external_test"
                for scenario in configs["network"]["scenarios"]:
                    for repetition in range(max(30, configs["network"]["repetitions"])):
                        previous.add(_derived_seed(base, protocol, site, role, scenario, repetition))
    if len(set(new_seeds)) != 420 or set(new_seeds) & previous:
        raise RuntimeError("New 420-seed ledger is not unique and disjoint")
    return {
        "extension_version": VERSION,
        "configs": configs,
        "common_budgets": budgets,
        "new_unique_network_seeds": 420,
        "previous_seed_overlap": 0,
        "seed_ledger": new_seeds,
        "upstream_artifacts_verified": verified,
        "files": {str(path.relative_to(root)): sha256_file(path) for path in sorted(set(files))},
        "primary_family_test_count": int(ext["primary_family"]["expected_test_count"]),
        "timing_disclosure": "Source trust uses protocol-grid rows; relay trust uses candidate-opportunity rows. Fixed timeout uses four elapsed protocol-grid durations in both scopes.",
        "development_disclosure": "Existing CE02 internal-test results are reused as extension development calibration; CE02 is not a held-out confirmation site for this extension.",
        "feedback_disclosure": "Recovery follows the frozen v1.1 cache and response model. Responses do not refilter the entire future receiver belief history; no closed-loop fidelity claim is made.",
        "queue_disclosure": "Existing queue stress groups eight ordered relay candidates; queue residence delay is not replayed. These are offline constrained queue comparisons, not demonstrated real-time deployment.",
    }


def freeze(root: Path, configs: dict) -> tuple[Path, dict]:
    target = root / configs["extension"]["output_directory"] / "protocol_lock.json"
    candidate = frozen_protocol(root, configs)
    if target.exists():
        recorded = json.loads(target.read_text())
        if hash_object(recorded) != hash_object(candidate):
            raise RuntimeError("Frozen protocol or inputs changed. Existing confirmation will not be overwritten.")
        return target, recorded
    atomic_json(target, candidate)
    return target, candidate


def trust_variant(transport: pd.DataFrame, configs: dict, variant: str, grid: int, canonical: pd.DataFrame | None = None) -> pd.DataFrame:
    receiver_config = deepcopy(configs["receiver"])
    receiver_config["extension_ablation"] = variant
    receiver_config["extension_grid_minutes"] = grid
    cause_config = deepcopy(configs["cause"])
    cause_config["extension_ablation"] = variant
    if variant == "no_connectivity":
        cause_config["valid_suppression_admissibility"]["minimum_connectivity"] = 0.0
        for weights in cause_config["logits"].values():
            for field in ("connectivity", "inverse_connectivity"):
                if field in weights:
                    weights[field] = 0.0
    if variant == "collapsed_belief":
        cause_config["valid_suppression_admissibility"]["minimum_connectivity"] = 0.0
        cause_config["valid_suppression_admissibility"]["maximum_environmental_uncertainty"] = 1.0
    receiver = interpret_receiver(transport, receiver_config)
    inferred = infer_causes(receiver, cause_config)
    if variant == "fixed_timeout":
        silence = ~inferred.packet_arrived.astype(bool)
        inferred.loc[silence, "cause_decision"] = np.where(inferred.loc[silence, "receiver_state"].eq("trusted_silence"), "valid_suppression", "unknown")
        inferred.loc[silence, "valid_suppression_admissible"] = inferred.loc[silence, "receiver_state"].eq("trusted_silence").astype(np.uint8)
    if canonical is not None:
        keys = ["repetition", "window_key", "timestamp"]
        truth_columns = ["evaluation_only_unsafe_silence", "evaluation_only_true_cause"]
        inferred = inferred.drop(columns=truth_columns).merge(canonical[keys + truth_columns], on=keys, validate="one_to_one")
    else:
        critical_missing = inferred.evaluation_only_critical_event.astype(bool) & ~inferred.packet_arrived.astype(bool)
        inferred["evaluation_only_unsafe_silence"] = (inferred.evaluation_only_unsafe_silence.astype(bool) | critical_missing).astype(np.uint8)
        critical_suppression = critical_missing & inferred.evaluation_only_source_action.eq("silent")
        inferred.loc[critical_suppression, "evaluation_only_true_cause"] = "contract_divergence"
    unsafe = inferred.evaluation_only_unsafe_silence.astype(bool)
    trusted = inferred.receiver_state.eq("trusted_silence")
    inferred["evaluation_only_false_trust"] = (unsafe & trusted).astype(np.uint8)
    inferred["evaluation_only_unsafe_predicted_valid"] = (unsafe & inferred.cause_decision.eq("valid_suppression")).astype(np.uint8)
    return inferred


def trust_metrics(frame: pd.DataFrame, replay: pd.DataFrame, scope: str, variant: str, protocol: str, site: str, scenario: str) -> list[dict]:
    output = []
    keys = ["repetition", "window_key", "timestamp"]
    extra = ["post_recovery_packet_delivered_within_retention", "recovery_succeeded", "recovery_network_bytes", "recovery_energy_j", "extension_budget_blocked", "extension_recovery_byte_cap", "extension_recovery_energy_cap_j"]
    scored = frame.merge(replay[keys + extra], on=keys, validate="one_to_one")
    for repetition, group in scored.groupby("repetition", sort=False):
        critical = group.evaluation_only_critical_event.to_numpy(bool)
        no_arrival = ~group.packet_arrived.to_numpy(bool)
        unsafe = group.evaluation_only_unsafe_silence.to_numpy(bool)
        trust = group.receiver_state.eq("trusted_silence").to_numpy()
        cause_valid = group.cause_decision.eq("valid_suppression").to_numpy()
        protective = group.receiver_state.isin(PROTECTIVE_STATES).to_numpy() | group.cause_decision.isin(["unknown", "isolated_loss", "persistent_outage", "contract_divergence"]).to_numpy()
        critical_missed = critical & no_arrival
        post = group.post_recovery_packet_delivered_within_retention.to_numpy(bool)
        delays = []
        censored = 0
        for _, window in group.groupby("window_key", sort=False):
            bad = window.evaluation_only_unsafe_silence.to_numpy(bool)
            safe_action = window.receiver_state.isin(PROTECTIVE_STATES).to_numpy() | window.cause_decision.isin(["unknown", "isolated_loss", "persistent_outage", "contract_divergence"]).to_numpy()
            times = pd.to_datetime(window.timestamp, utc=True)
            index = 0
            while index < len(window):
                if not bad[index]:
                    index += 1
                    continue
                stop = index + 1
                while stop < len(window) and bad[stop]:
                    stop += 1
                hits = np.flatnonzero(safe_action[index:stop])
                if len(hits):
                    delays.append((times.iloc[index + int(hits[0])] - times.iloc[index]).total_seconds())
                else:
                    censored += 1
                index = stop
        false = critical_missed & trust
        replay_group = replay[replay.repetition.eq(repetition)]
        diagnostic = recovery_metrics(replay_group, protocol, site, "external_test" if site == "CE07SHSM" else "tuning", scenario, int(repetition), "safe_silence_bounded")
        secondary = {key: diagnostic[key] for key in ["unnecessary_valid_suppression_trigger_rate", "p95_successful_recovery_delay_seconds", "p95_critical_event_recovery_delay_steps", "recovery_attempts", "recovery_triggers", "post_recovery_timely_critical_delivery_recall"]}
        output.append({
            **secondary,
            "protocol": protocol, "site": site, "scenario": scenario, "repetition": int(repetition), "scope": scope, "variant": variant,
            "rows": len(group), "critical_rows": int(critical.sum()), "missed_critical_rows": int(critical_missed.sum()),
            "critical_false_trust_pre_action": float(false.sum() / critical.sum()) if critical.any() else np.nan,
            "critical_false_trust_given_missed": float(false[critical_missed].mean()) if critical_missed.any() else np.nan,
            "critical_unsafe_valid_suppression_assertion_rate": float((critical_missed & cause_valid).sum() / critical.sum()) if critical.any() else np.nan,
            "critical_unsafe_interpretation_union_rate": float((critical_missed & (trust | cause_valid)).sum() / critical.sum()) if critical.any() else np.nan,
            "critical_false_trust_post_action": float((critical & ~post & trust).sum() / critical.sum()) if critical.any() else np.nan,
            "unsafe_silence_detection_recall": float(protective[unsafe].mean()) if unsafe.any() else np.nan,
            "post_recovery_critical_delivery_recall": float(post[critical].mean()) if critical.any() else np.nan,
            "mean_time_to_protective_state_seconds_uncensored": float(np.mean(delays)) if delays else np.nan,
            "protective_episode_detections": len(delays), "protective_episode_non_detections": censored,
            "recovery_network_bytes": int(group.recovery_network_bytes.sum()), "recovery_energy_j": float(group.recovery_energy_j.sum()),
            "budget_blocked_actions": int(group.extension_budget_blocked.sum()),
            "recovery_byte_cap": float(group.extension_recovery_byte_cap.iloc[0]), "recovery_energy_cap_j": float(group.extension_recovery_energy_cap_j.iloc[0]),
        })
    return output


def selection_inputs(root: Path, destination: Path, configs: dict, protocol: str, site: str, role: str, smoke: bool) -> tuple[Path, Path]:
    semantic_root = destination / "semantic_selection"
    fusion_root = destination / "fusion_selection"
    semantic_root.mkdir(parents=True, exist_ok=True)
    fusion_root.mkdir(parents=True, exist_ok=True)
    original_j = json.loads((root / "results/task_j/task_j_summary.json").read_text())
    source = pd.read_parquet(root / f"results/task_j/evidence/{protocol}/{site}.parquet")
    source = source[source.model_role.eq(role)].copy()
    if smoke:
        first_window = sorted(source.window_key.unique())[0]
        source = source[source.window_key.eq(first_window)].sort_values("timestamp").head(512)
    path = semantic_root / f"evidence/{protocol}/{site}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    source.to_parquet(path, index=False, compression="zstd")
    selected_j = deepcopy(original_j)
    selected_j["protocols"] = {protocol: {"grid_minutes": configs["preprocessing"]["protocols"][protocol]["grid_minutes"], "rows": len(source), "sites": [site]}}
    selected_j["artifacts"] = [{"protocol": protocol, "site": site, "path": str(path.resolve()), "rows": len(source), "sha256": sha256_file(path)}]
    atomic_json(semantic_root / "task_j_summary.json", selected_j)
    original_o = json.loads((root / "results/task_o/task_o_summary.json").read_text())
    summaries = pd.read_parquet(root / f"results/task_o/summaries/{protocol}/{site}.parquet")
    summaries = summaries[summaries.model_role.eq(role)].copy()
    if smoke:
        allowed = set(pd.to_datetime(source.timestamp, utc=True))
        summaries = summaries[summaries.window_key.isin(source.window_key.unique()) & pd.to_datetime(summaries.relay_flush_timestamp, utc=True).isin(allowed)]
    membership = pd.read_parquet(root / f"results/task_o/membership/{protocol}/{site}.parquet")
    membership = membership[membership.batch_id.isin(summaries.batch_id)].copy()
    selected_o = deepcopy(original_o)
    selected_o["artifacts"] = []
    for kind, frame, folder in [("membership", membership, "membership"), ("summary", summaries, "summaries")]:
        path = fusion_root / f"{folder}/{protocol}/{site}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
        selected_o["artifacts"].append({"protocol": protocol, "site": site, "artifact_kind": kind, "path": str(path.resolve()), "rows": len(frame), "sha256": sha256_file(path)})
    atomic_json(fusion_root / "task_o_summary.json", selected_o)
    if summaries.empty:
        raise RuntimeError("Selected semantic stream has no aligned relay candidates")
    return semantic_root, fusion_root


def budget_config(configs: dict, lock: dict, scope: str, protocol: str) -> dict:
    result = deepcopy(configs["recovery"])
    result["extension_common_budget"] = deepcopy(lock["common_budgets"][scope][protocol])
    return result


def relay_frame(scheduler_root: Path, fusion_root: Path, network_root: Path, protocol: str, site: str, scenario_name: str, scheduler_policy: str, configs: dict) -> pd.DataFrame:
    schedule = pd.read_parquet(scheduler_root / f"schedules/{protocol}/{site}/{scenario_name}.parquet")
    membership = pd.read_parquet(fusion_root / f"membership/{protocol}/{site}.parquet")
    schedule = schedule.merge(_membership_metadata(membership), on="batch_id", validate="many_to_one")
    if scheduler_policy == REFERENCE_POLICY:
        schedule = pd.concat([schedule_reference(group, REFERENCE_POLICY, configs["scheduler"]) for _, group in schedule.groupby(["repetition", "opportunity_id"], sort=False)], ignore_index=True)
    network = pd.read_parquet(network_root / f"transport/{protocol}/{site}/{scenario_name}.parquet")
    oracle_columns = ["timestamp", "window_key", "repetition", "realization_seed", "_oracle_link_state"] + [f"_oracle_{kind}_u{attempt}" for attempt in range(configs["network"]["maximum_attempts"]) for kind in ("data", "ack", "jitter")]
    oracle = network[oracle_columns].copy()
    oracle["timestamp"] = pd.to_datetime(oracle.timestamp, utc=True)
    schedule["relay_flush_timestamp"] = pd.to_datetime(schedule.relay_flush_timestamp, utc=True)
    schedule = schedule.drop(columns=[c for c in schedule if c.startswith("_oracle_") or c in ("timestamp", "realization_seed")])
    merged = schedule.merge(oracle, left_on=["window_key", "relay_flush_timestamp", "repetition"], right_on=["window_key", "timestamp", "repetition"], validate="many_to_one")
    return replay_selected(merged, configs["network"]["scenarios"][scenario_name], configs["network"], 1, configs["preprocessing"]["protocols"][protocol]["grid_minutes"] * 60.0)


def run_cell(root: Path, cell: Path, protocol: str, scenario_name: str, configs: dict, lock: dict, smoke: bool) -> None:
    ext = configs["extension"]
    site = ext["development_site"] if smoke else ext["primary_site"]
    role = "tuning" if smoke else ext["primary_role"]
    grid = int(configs["preprocessing"]["protocols"][protocol]["grid_minutes"])
    cell.mkdir(parents=True, exist_ok=True)
    sem, fusion = selection_inputs(root, cell, configs, protocol, site, role, smoke)
    network_config = deepcopy(configs["network"])
    network_config["base_seed"] = ext["smoke_network_base_seed"] if smoke else ext["new_network_base_seed"]
    network_config["repetitions"] = 2 if smoke else ext["repetitions"]
    network_config["evaluation_roles"] = [role]
    network_config["scenarios"] = {scenario_name: network_config["scenarios"][scenario_name]}
    network_root = cell / "network"
    print(f"NETWORK {protocol}/{scenario_name}: {network_config['repetitions']} paired repetitions", flush=True)
    simulate_network(sem, network_root, network_config)
    source_transport = pd.read_parquet(network_root / f"transport/{protocol}/{site}/{scenario_name}.parquet")
    seeds = source_transport.groupby("repetition").realization_seed.first().astype("int64").to_dict()
    for repetition, seed in seeds.items():
        expected = _derived_seed(network_config["base_seed"], protocol, site, role, scenario_name, int(repetition))
        if seed != expected:
            raise RuntimeError("Stored physical seed differs from frozen ledger")
    trust_rows = []
    canonical = trust_variant(source_transport, configs, "full", grid)
    source_budget = budget_config(configs, lock, "source", protocol)
    for variant in ext["trust_variants"]:
        print(f"TRUST source {protocol}/{scenario_name}/{variant}", flush=True)
        frame = canonical if variant == "full" else trust_variant(source_transport, configs, variant, grid, canonical)
        replay = replay_recovery(frame, source_transport, "safe_silence_bounded", source_budget, network_config, network_config["scenarios"][scenario_name], grid)
        trust_rows.extend(trust_metrics(frame, replay, "source_protocol_grid", variant, protocol, site, scenario_name))
    del source_transport, canonical
    system_rows = []
    for profile in ext["profiles"]:
        print(f"SCHEDULER {protocol}/{scenario_name}/{profile}", flush=True)
        local_scheduler = deepcopy(configs["scheduler"])
        local_scheduler["loss_profiles"] = {"primary": deepcopy(configs["scheduler"]["loss_profiles"][profile])}
        p_root = cell / profile / "scheduler"
        run_scheduler(fusion, network_root, p_root, local_scheduler, network_config)
        for scheduler_policy in (PROPOSED_POLICY, REFERENCE_POLICY):
            integrated = relay_frame(p_root, fusion, network_root, protocol, site, scenario_name, scheduler_policy, configs)
            transport = build_relay_transport(integrated, configs["reintegration"]["relay_receiver_contract"])
            canonical = trust_variant(transport, configs, "full", grid)
            scoring = integrated.copy()
            scoring["timestamp"] = pd.to_datetime(scoring.relay_flush_timestamp, utc=True)
            relay_network = deepcopy(network_config)
            relay_network["hops"] = 1
            relay_budget = budget_config(configs, lock, "relay", protocol)
            policies = ext["recovery_policies"] if profile == "primary" and scheduler_policy == PROPOSED_POLICY else ["safe_silence_bounded"]
            for policy in policies:
                replay = replay_recovery(canonical, transport, policy, relay_budget, relay_network, network_config["scenarios"][scenario_name], grid)
                for repetition, group in replay.groupby("repetition", sort=False):
                    row = _system_metrics(group, scoring[scoring.repetition.eq(repetition)], protocol, site, role, scenario_name, int(repetition), scheduler_policy, policy)
                    row.update(loss_profile=profile, realization_seed=int(seeds[int(repetition)]), recovery_byte_cap=float(group.extension_recovery_byte_cap.iloc[0]), recovery_energy_cap_j=float(group.extension_recovery_energy_cap_j.iloc[0]), budget_blocked_actions=int(group.extension_budget_blocked.sum()))
                    # Score unresolved critical events using event-specific delivery credit.
                    trusted = group.receiver_state.eq("trusted_silence")
                    critical = group.causal_critical_evidence.astype(bool)
                    unresolved = ~group.post_recovery_packet_delivered_within_retention.astype(bool)
                    row["critical_false_trust_rate"] = float((critical & unresolved & trusted).sum() / critical.sum()) if critical.any() else np.nan
                    system_rows.append(row)
                if profile == "primary" and scheduler_policy == PROPOSED_POLICY and policy == "safe_silence_bounded":
                    trust_rows.extend(trust_metrics(canonical, replay, "relay_candidate_opportunities", "full", protocol, site, scenario_name))
                    if ext["retain_primary_relay_trace"]:
                        replay.to_parquet(cell / "primary_relay_trace.parquet", index=False, compression="zstd")
            if profile == "primary" and scheduler_policy == PROPOSED_POLICY:
                for variant in ext["trust_variants"]:
                    if variant == "full":
                        continue
                    frame = trust_variant(transport, configs, variant, grid, canonical)
                    replay = replay_recovery(frame, transport, "safe_silence_bounded", relay_budget, relay_network, network_config["scenarios"][scenario_name], grid)
                    trust_rows.extend(trust_metrics(frame, replay, "relay_candidate_opportunities", variant, protocol, site, scenario_name))
    trust_frame = pd.DataFrame(trust_rows)
    trust_frame["realization_seed"] = trust_frame.repetition.map(seeds).astype("int64")
    atomic_csv(cell / "trust_metrics.csv", trust_frame)
    atomic_csv(cell / "system_metrics.csv", pd.DataFrame(system_rows))
    files = {name: sha256_file(cell / name) for name in ["trust_metrics.csv", "system_metrics.csv"]}
    if (cell / "primary_relay_trace.parquet").exists():
        files["primary_relay_trace.parquet"] = sha256_file(cell / "primary_relay_trace.parquet")
    atomic_json(cell / "DONE.json", {"protocol_sha256": protocol_identity(lock), "smoke_only": smoke, "site": site, "protocol": protocol, "scenario": scenario_name, "seeds": {str(key): int(value) for key, value in seeds.items()}, "files": files})
    if not ext["retain_full_network_traces"]:
        shutil.rmtree(network_root)
    for profile in ext["profiles"]:
        shutil.rmtree(cell / profile)
    shutil.rmtree(sem)
    shutil.rmtree(fusion)
    print(f"DONE {protocol}/{scenario_name}", flush=True)


def comparisons(system: pd.DataFrame, trust: pd.DataFrame, configs: dict) -> pd.DataFrame:
    ext = configs["extension"]
    rows = []
    endpoints = ext["primary_family"]["recovery_endpoints"]
    declarations = []
    for protocol in ext["protocols"]:
        for scenario in ext["scenarios"]:
            cell = system[system.protocol.eq(protocol) & system.scenario.eq(scenario)]
            proposed = cell[cell.loss_profile.eq("primary") & cell.scheduler_policy.eq(PROPOSED_POLICY) & cell.recovery_policy.eq("safe_silence_bounded")]
            for policy in ext["primary_family"]["recovery_comparators"]:
                reference = cell[cell.loss_profile.eq("primary") & cell.scheduler_policy.eq(PROPOSED_POLICY) & cell.recovery_policy.eq(policy)]
                for metric in endpoints:
                    declarations.append((f"recovery:{policy}", protocol, scenario, metric, True, proposed, reference))
            primary_trust = trust[trust.protocol.eq(protocol) & trust.scenario.eq(scenario) & trust.scope.eq(ext["primary_trust_scope"])]
            for variant in ext["primary_family"]["trust_comparators"]:
                declarations.append((f"trust:{variant}", protocol, scenario, ext["primary_family"]["trust_endpoint"], False, primary_trust[primary_trust.variant.eq("full")], primary_trust[primary_trust.variant.eq(variant)]))
            for profile in ext["profiles"]:
                part = cell[cell.loss_profile.eq(profile) & cell.recovery_policy.eq("safe_silence_bounded")]
                for metric in ext["primary_family"]["scheduler_endpoints"]:
                    declarations.append((f"scheduler:{profile}", protocol, scenario, metric, True, part[part.scheduler_policy.eq(PROPOSED_POLICY)], part[part.scheduler_policy.eq(REFERENCE_POLICY)]))
    for name, protocol, scenario, metric, higher, proposed, reference in declarations:
        paired = proposed[["repetition", "realization_seed", metric]].merge(reference[["repetition", "realization_seed", metric]], on=["repetition", "realization_seed"], suffixes=("_p", "_r"), validate="one_to_one")
        if len(paired) != ext["repetitions"]:
            raise RuntimeError(f"Incomplete declared comparison: {name}/{protocol}/{scenario}/{metric}")
        a = paired[metric + "_p"].to_numpy(float)
        b = paired[metric + "_r"].to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"Undefined primary endpoint: {name}/{protocol}/{scenario}/{metric}. Do not silently drop the hypothesis.")
        difference = a - b
        seed = _derived_seed(ext["bootstrap_seed"], name, protocol, scenario, metric)
        rng = np.random.default_rng(seed)
        sampled = difference[rng.integers(len(difference), size=(ext["bootstrap_resamples"], len(difference)))].mean(axis=1)
        lo, hi = np.quantile(sampled, [(1 - ext["confidence_level"]) / 2, (1 + ext["confidence_level"]) / 2])
        p = 1.0 if np.all(difference == 0) else float(wilcoxon(difference, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
        rows.append({"comparison": name, "protocol": protocol, "scenario": scenario, "metric": metric, "higher_is_better": higher, "paired_repetitions": len(paired), "proposed_mean": a.mean(), "reference_mean": b.mean(), "mean_paired_difference": difference.mean(), "bootstrap_ci_lower": lo, "bootstrap_ci_upper": hi, "wilcoxon_p_value": p})
    result = pd.DataFrame(rows)
    if len(result) != ext["primary_family"]["expected_test_count"]:
        raise RuntimeError("Declared global primary family size changed")
    order = np.argsort(result.wilcoxon_p_value.to_numpy())
    adjusted = np.empty(len(result))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(result.wilcoxon_p_value.to_numpy()[order] * (len(result) - np.arange(len(result)))))
    result["holm_adjusted_p_value"] = adjusted
    result["reject_holm_null"] = adjusted <= 0.05
    result["directionally_favorable"] = np.where(result.higher_is_better, result.mean_paired_difference > 0, result.mean_paired_difference < 0)
    result["confirmatory_support"] = result.reject_holm_null & result.directionally_favorable
    return result


def aggregate_estimates(system: pd.DataFrame, trust: pd.DataFrame, primary: pd.DataFrame, configs: dict) -> pd.DataFrame:
    """Descriptive equal-scenario estimates; never average scenario CI endpoints."""
    ext = configs["extension"]
    rows = []
    for (name, protocol, metric), _ in primary.groupby(["comparison", "protocol", "metric"], sort=True):
        kind, label = name.split(":", 1)
        if kind == "trust":
            part = trust[trust.protocol.eq(protocol) & trust.scope.eq(ext["primary_trust_scope"])]
            proposed = part[part.variant.eq("full")]
            reference = part[part.variant.eq(label)]
        else:
            profile = label if kind == "scheduler" else "primary"
            part = system[system.protocol.eq(protocol) & system.loss_profile.eq(profile)]
            proposed = part[part.scheduler_policy.eq(PROPOSED_POLICY) & part.recovery_policy.eq("safe_silence_bounded")]
            reference = part[part.scheduler_policy.eq(REFERENCE_POLICY if kind == "scheduler" else PROPOSED_POLICY) & part.recovery_policy.eq("safe_silence_bounded" if kind == "scheduler" else label)]
        paired = proposed[["scenario", "repetition", "realization_seed", metric]].merge(reference[["scenario", "repetition", "realization_seed", metric]], on=["scenario", "repetition", "realization_seed"], suffixes=("_p", "_r"), validate="one_to_one")
        if len(paired) != 210 or paired.scenario.nunique() != 7:
            raise RuntimeError("Aggregate estimate lost a prespecified scenario or pair")
        rng = np.random.default_rng(_derived_seed(ext["bootstrap_seed"], "aggregate", name, protocol, metric))
        draws = []
        for _, group in paired.groupby("scenario", sort=True):
            a = group[metric + "_p"].to_numpy(float)
            b = group[metric + "_r"].to_numpy(float)
            indices = rng.integers(len(a), size=(ext["bootstrap_resamples"], len(a)))
            draws.append(np.column_stack([a[indices].mean(axis=1), b[indices].mean(axis=1), (a - b)[indices].mean(axis=1)]))
        means = np.mean(draws, axis=0)
        quantiles = np.quantile(means, [(1 - ext["confidence_level"]) / 2, (1 + ext["confidence_level"]) / 2], axis=0)
        rows.append({"comparison": name, "protocol": protocol, "metric": metric, "scenario_weighting": "equal", "paired_repetitions": len(paired), "proposed_mean": float(paired[metric + "_p"].mean()), "reference_mean": float(paired[metric + "_r"].mean()), "mean_paired_difference": float((paired[metric + "_p"] - paired[metric + "_r"]).mean()), "proposed_ci_lower": quantiles[0, 0], "proposed_ci_upper": quantiles[1, 0], "reference_ci_lower": quantiles[0, 1], "reference_ci_upper": quantiles[1, 1], "difference_ci_lower": quantiles[0, 2], "difference_ci_upper": quantiles[1, 2], "analysis_role": "descriptive_stratified_paired_aggregate"})
    return pd.DataFrame(rows)


def analyze(output: Path, configs: dict, lock: dict) -> None:
    ext = configs["extension"]
    system_parts, trust_parts, seeds = [], [], []
    for protocol in ext["protocols"]:
        for scenario in ext["scenarios"]:
            cell = output / "cells" / protocol / scenario
            done = json.loads((cell / "DONE.json").read_text())
            if done["smoke_only"] or done["protocol_sha256"] != protocol_identity(lock):
                raise RuntimeError("Smoke or incompatible outputs cannot enter confirmation")
            for name, checksum in done["files"].items():
                if sha256_file(cell / name) != checksum:
                    raise RuntimeError(f"Completed artifact changed: {cell/name}")
            system_parts.append(pd.read_csv(cell / "system_metrics.csv"))
            trust_parts.append(pd.read_csv(cell / "trust_metrics.csv"))
            seeds.extend(done["seeds"].values())
    if len(seeds) != 420 or len(set(seeds)) != 420 or set(seeds) != set(lock["seed_ledger"]):
        raise RuntimeError("Confirmation seed ledger is incomplete or changed")
    system = pd.concat(system_parts, ignore_index=True)
    trust = pd.concat(trust_parts, ignore_index=True)
    primary = comparisons(system, trust, configs)
    atomic_csv(output / "system_metrics.csv", system)
    atomic_csv(output / "trust_metrics.csv", trust)
    atomic_csv(output / "primary_comparisons.csv", primary)
    atomic_csv(output / "aggregate_effects.csv", aggregate_estimates(system, trust, primary, configs))
    costs = system.groupby(["protocol", "loss_profile", "scheduler_policy", "recovery_policy"], as_index=False)[["post_recovery_critical_delivery_recall", "recovery_network_bytes", "recovery_energy_j", "recovery_byte_cap", "recovery_energy_cap_j"]].mean()
    atomic_csv(output / "resource_consumption.csv", costs)
    atomic_json(output / "extension_summary.json", {"extension_version": VERSION, "protocol_sha256": protocol_identity(lock), "unique_network_seeds": len(set(seeds)), "primary_tests": len(primary), "primary_holm_rejections": int(primary.reject_holm_null.sum()), "directionally_supported_tests": int(primary.confirmatory_support.sum()), "results_are_simulated": True, "energy_is_modeled": True, "resource_matching": "common caps; realized consumption reported; no exact matching claim", "source_trust_is_mechanism_evidence": True, "relay_trust_is_secondary": True})
    print(f"CONFIRMATION COMPLETE: 420 realizations, {len(primary)} globally Holm-adjusted primary hypotheses", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "freeze", "smoke", "run", "analyze"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    configs = load_configs(root)
    target, lock = freeze(root, configs)
    print(f"PROTOCOL VERIFIED: {target}; {lock['new_unique_network_seeds']} unique disjoint seeds", flush=True)
    output = target.parent
    if args.stage in ("verify", "freeze"):
        return
    if args.stage == "smoke":
        smoke_root = output / "smoke"
        for protocol in configs["extension"]["protocols"]:
            run_cell(root, smoke_root / protocol / "F2_burst", protocol, "F2_burst", configs, lock, True)
        print("SMOKE PASSED: both protocols, CE02 tuning only, two repetitions each; not confirmatory evidence", flush=True)
        return
    if args.stage == "run":
        for protocol in configs["extension"]["protocols"]:
            smoke_done_path = output / "smoke" / protocol / "F2_burst/DONE.json"
            if not smoke_done_path.exists():
                raise RuntimeError("Run both-protocol smoke verification before primary confirmation")
            smoke_done = json.loads(smoke_done_path.read_text())
            if smoke_done["protocol_sha256"] != protocol_identity(lock) or not smoke_done["smoke_only"]:
                raise RuntimeError("Smoke outputs do not match the frozen protocol")
            for name, checksum in smoke_done["files"].items():
                if sha256_file(smoke_done_path.parent / name) != checksum:
                    raise RuntimeError("A smoke artifact changed after verification")
        for protocol in configs["extension"]["protocols"]:
            for scenario in configs["extension"]["scenarios"]:
                cell = output / "cells" / protocol / scenario
                done_path = cell / "DONE.json"
                if done_path.exists():
                    done = json.loads(done_path.read_text())
                    if done["protocol_sha256"] != protocol_identity(lock) or done["smoke_only"]:
                        raise RuntimeError(f"Incompatible completed cell: {cell}")
                    for name, checksum in done["files"].items():
                        if sha256_file(cell / name) != checksum:
                            raise RuntimeError(f"Completed cell artifact changed: {cell/name}")
                    print(f"RESUME SKIP {protocol}/{scenario}", flush=True)
                    continue
                run_cell(root, cell, protocol, scenario, configs, lock, False)
    analyze(output, configs, lock)

if __name__ == "__main__":
    main()
