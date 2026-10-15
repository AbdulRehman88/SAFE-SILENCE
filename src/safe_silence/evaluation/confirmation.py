"""Task S independently seeded confirmatory replication and inference."""

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
from safe_silence.evaluation.integration import run_integration
from safe_silence.evaluation.reintegration import (
    STATISTICAL_METRICS,
    _paired_bootstrap_interval,
    run_reintegration,
)
from safe_silence.protocol.scheduler import run_scheduler
from safe_silence.simulation.network import _derived_seed, simulate_network


class ConfirmationError(RuntimeError):
    """Raised when Task S cannot preserve its independent confirmation contract."""


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


def _resolve_task_r_artifact(record: dict[str, Any], root: Path) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    if all(record.get(key) for key in ("protocol", "site", "scenario")):
        return (
            root
            / "system"
            / str(record["protocol"])
            / str(record["site"])
            / f"{record['scenario']}.parquet"
        )
    return root / declared.name


def verify_prior_task_r(root: Path) -> tuple[dict[str, Any], int]:
    """Reverify the frozen Task R record before running independent confirmation."""
    summary_path = root / "task_r_summary.json"
    if not summary_path.exists():
        raise ConfirmationError(f"Task S requires {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != "R" or summary.get("reintegration_version") != "1.0":
        raise ConfirmationError("Task S requires frozen Task R contract v1.0")
    if summary.get("recovery_logic_version") != "1.1":
        raise ConfirmationError("Task S requires recovery contract v1.1")
    if any(
        bool(summary.get(field))
        for field in (
            "receiver_uses_oracle_fields",
            "receiver_uses_event_truth",
            "recovery_policy_uses_oracle_fields",
            "recovery_policy_uses_event_truth",
            "future_measurements_used",
            "parameters_tuned_on_test_or_external_data",
            "temporal_protocols_pooled",
        )
    ):
        raise ConfirmationError("Frozen Task R summary violates the confirmation boundary")
    verified = 0
    for record in summary.get("artifacts", []):
        path = _resolve_task_r_artifact(record, root)
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise ConfirmationError(f"Task R artifact changed or is missing: {path}")
        verified += 1
    if not verified:
        raise ConfirmationError("Task R contains no verifiable artifacts")
    return summary, verified


def audit_independent_seeds(
    network_summary: dict[str, Any],
    original_network_config: dict[str, Any],
    confirmation_network_config: dict[str, Any],
    network_root: Path,
    previous_confirmation_base_seeds: list[int] | None = None,
) -> dict[str, int]:
    """Prove that every confirmatory realization seed is disjoint from Task K."""
    original: set[int] = set()
    confirmation: set[int] = set()
    previous_confirmation: set[int] = set()
    prior_bases = [int(value) for value in (previous_confirmation_base_seeds or [])]
    trace_records = [
        record
        for record in network_summary["artifacts"]
        if all(record.get(key) for key in ("protocol", "site", "scenario"))
    ]
    for record in trace_records:
        protocol = str(record["protocol"])
        site = str(record["site"])
        role = str(record["model_role"])
        scenario = str(record["scenario"])
        for repetition in range(int(original_network_config["repetitions"])):
            original.add(
                _derived_seed(
                    int(original_network_config["base_seed"]),
                    protocol,
                    site,
                    role,
                    scenario,
                    repetition,
                )
            )
        for prior_base in prior_bases:
            for repetition in range(int(confirmation_network_config["repetitions"])):
                previous_confirmation.add(
                    _derived_seed(
                        prior_base,
                        protocol,
                        site,
                        role,
                        scenario,
                        repetition,
                    )
                )
        path = network_root / "transport" / protocol / site / f"{scenario}.parquet"
        seeds = pd.read_parquet(path, columns=["realization_seed"])["realization_seed"].unique()
        if len(seeds) != int(confirmation_network_config["repetitions"]):
            raise ConfirmationError(
                f"Confirmatory repetition count changed for {protocol}/{site}/{scenario}"
            )
        confirmation.update(int(seed) for seed in seeds)
    overlap = original.intersection(confirmation)
    if overlap:
        raise ConfirmationError("Task S realization seeds overlap frozen Task K")
    previous_overlap = previous_confirmation.intersection(confirmation)
    if previous_overlap:
        raise ConfirmationError("Task S realization seeds overlap the declared reference seed set")
    return {
        "original_task_k_unique_seeds": len(original),
        "previous_confirmation_unique_seeds": len(previous_confirmation),
        "confirmatory_unique_seeds": len(confirmation),
        "seed_overlap_count": 0,
        "previous_confirmation_seed_overlap_count": 0,
    }


def _comparison_rows(
    metrics: pd.DataFrame,
    config: dict[str, Any],
    endpoints: list[str],
    include_tests: bool,
) -> pd.DataFrame:
    proposed = config["proposed_system"]
    references = config["primary_analysis"]["comparisons"]
    keys = ["protocol", "site", "model_role", "scenario", "repetition"]
    confidence = float(config["primary_analysis"]["confidence_level"])
    resamples = int(config["primary_analysis"]["bootstrap_resamples"])
    base_seed = int(config["primary_analysis"]["bootstrap_seed"])
    proposed_rows = metrics.loc[
        (metrics["scheduler_policy"] == str(proposed["scheduler"]))
        & (metrics["recovery_policy"] == str(proposed["recovery"]))
    ]
    rows: list[dict[str, Any]] = []
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
            for metric in endpoints:
                higher_is_better = bool(STATISTICAL_METRICS[metric])
                proposed_values = group[f"{metric}_proposed"].to_numpy(float)
                reference_values = group[f"{metric}_reference"].to_numpy(float)
                finite = np.isfinite(proposed_values) & np.isfinite(reference_values)
                difference = proposed_values[finite] - reference_values[finite]
                material = "|".join([comparison_name, *map(str, cell), metric]).encode()
                seed = (
                    base_seed + int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
                ) % (2**32 - 1)
                lower, upper = _paired_bootstrap_interval(difference, confidence, resamples, seed)
                mean_difference = float(np.mean(difference)) if len(difference) else np.nan
                favorable = mean_difference > 0 if higher_is_better else mean_difference < 0
                if include_tests and len(difference) and np.any(difference != 0):
                    p_value = float(
                        wilcoxon(
                            difference,
                            alternative="two-sided",
                            zero_method="pratt",
                        ).pvalue
                    )
                elif include_tests and len(difference):
                    p_value = 1.0
                else:
                    p_value = np.nan
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
                        "analysis_role": (
                            "primary_confirmation"
                            if cell[1] == str(config["primary_analysis"]["site"])
                            and cell[2] == str(config["primary_analysis"]["model_role"])
                            else "prespecified_sensitivity"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def confirmatory_comparisons(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Run the prospectively narrowed Task S confirmatory test family."""
    endpoints = [str(value) for value in config["primary_analysis"]["endpoints"]]
    output = _comparison_rows(metrics, config, endpoints, include_tests=True)
    output["holm_adjusted_p_value"] = np.nan
    group_keys = ["comparison", "protocol", "site", "model_role"]
    for _, indices in output.groupby(group_keys, sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        finite_positions = positions[
            np.isfinite(output.loc[positions, "wilcoxon_p_value"].to_numpy(float))
        ]
        order = finite_positions[
            np.argsort(output.loc[finite_positions, "wilcoxon_p_value"].to_numpy(float))
        ]
        previous = 0.0
        total = len(order)
        for rank, position in enumerate(order):
            raw = float(output.at[position, "wilcoxon_p_value"])
            adjusted = min(1.0, max(previous, (total - rank) * raw))
            output.at[position, "holm_adjusted_p_value"] = adjusted
            previous = adjusted
    alpha = 1.0 - float(config["primary_analysis"]["confidence_level"])
    output["reject_holm_null"] = output["holm_adjusted_p_value"] < alpha
    output["confirmatory_support"] = output["reject_holm_null"] & output["directionally_favorable"]
    return output


def secondary_descriptive_comparisons(
    metrics: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Report all prespecified safety-cost endpoints without confirmatory claims."""
    endpoints = [str(value) for value in config["secondary_endpoints"]]
    output = _comparison_rows(metrics, config, endpoints, include_tests=False)
    output["exploratory_only"] = True
    return output


def run_confirmation(
    semantic_root: Path,
    fusion_root: Path,
    prior_reintegration_root: Path,
    output_root: Path,
    confirmation_config: dict[str, Any],
    network_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    integration_config: dict[str, Any],
    reintegration_config: dict[str, Any],
    receiver_config: dict[str, Any],
    cause_config: dict[str, Any],
    recovery_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute a disjoint-seed K→P→Q→R replication and confirmatory analysis."""
    prior_r, verified_r = verify_prior_task_r(prior_reintegration_root)
    task_k_root = output_root / "replication" / "task_k"
    task_p_root = output_root / "replication" / "task_p"
    task_q_root = output_root / "replication" / "task_q"
    task_r_root = output_root / "replication" / "task_r"

    local_network = deepcopy(network_config)
    local_network["base_seed"] = int(confirmation_config["independent_network"]["base_seed"])
    local_network["repetitions"] = int(confirmation_config["independent_network"]["repetitions"])
    task_k = simulate_network(semantic_root, task_k_root, local_network)
    seed_audit = audit_independent_seeds(
        task_k,
        network_config,
        local_network,
        task_k_root,
        [
            int(value)
            for value in confirmation_config["independent_network"].get(
                "previous_confirmation_base_seeds", []
            )
        ],
    )
    task_p = run_scheduler(fusion_root, task_k_root, task_p_root, scheduler_config, local_network)
    task_q = run_integration(
        task_p_root,
        fusion_root,
        task_k_root,
        task_q_root,
        integration_config,
        scheduler_config,
        local_network,
        preprocessing_config,
    )
    local_reintegration = deepcopy(reintegration_config)
    local_reintegration["statistics"]["bootstrap_resamples"] = int(
        confirmation_config["primary_analysis"]["bootstrap_resamples"]
    )
    task_r = run_reintegration(
        task_q_root,
        task_p_root,
        fusion_root,
        task_k_root,
        task_r_root,
        local_reintegration,
        integration_config,
        scheduler_config,
        local_network,
        receiver_config,
        cause_config,
        recovery_config,
        preprocessing_config,
    )

    metrics = pd.read_csv(task_r_root / "system_metrics.csv")
    primary = confirmatory_comparisons(metrics, confirmation_config)
    primary_path = output_root / "confirmatory_comparisons.csv"
    _atomic_csv(primary_path, primary)
    secondary = secondary_descriptive_comparisons(metrics, confirmation_config)
    secondary_path = output_root / "secondary_tradeoffs.csv"
    _atomic_csv(secondary_path, secondary)

    primary_rows = primary.loc[primary["analysis_role"] == "primary_confirmation"]
    artifacts: list[dict[str, Any]] = []
    for path in (
        task_k_root / "task_k_summary.json",
        task_p_root / "task_p_summary.json",
        task_q_root / "task_q_summary.json",
        task_r_root / "task_r_summary.json",
        primary_path,
        secondary_path,
    ):
        artifacts.append({"path": str(path), "sha256": sha256_file(path)})

    config_hash = hashlib.sha256(
        json.dumps(confirmation_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "S",
        "confirmation_version": str(confirmation_config["confirmation_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "prior_task_r_artifacts_reverified": verified_r,
        "prior_task_r_reintegration_version": prior_r["reintegration_version"],
        "independent_network_base_seed": int(local_network["base_seed"]),
        "independent_repetitions": int(local_network["repetitions"]),
        **seed_audit,
        "task_k_network_version": task_k["network_version"],
        "task_p_scheduler_version": task_p["scheduler_version"],
        "task_q_integration_version": task_q["integration_version"],
        "task_r_reintegration_version": task_r["reintegration_version"],
        "primary_site": str(confirmation_config["primary_analysis"]["site"]),
        "primary_model_role": str(confirmation_config["primary_analysis"]["model_role"]),
        "primary_endpoints": list(confirmation_config["primary_analysis"]["endpoints"]),
        "primary_comparisons": list(confirmation_config["primary_analysis"]["comparisons"]),
        "primary_family_size_per_protocol_comparison": int(
            primary_rows.groupby(["protocol", "comparison"]).size().max()
        ),
        "primary_comparison_rows": int(len(primary_rows)),
        "primary_directionally_favorable_rows": int(primary_rows["directionally_favorable"].sum()),
        "primary_holm_confirmatory_support_rows": int(primary_rows["confirmatory_support"].sum()),
        "secondary_endpoints_are_exploratory": True,
        "secondary_tradeoff_rows": int(len(secondary)),
        "prior_task_r_results_used_to_tune_algorithms": False,
        "prior_task_s_results_used_to_tune_algorithms": False,
        "recovery_protocol_corrected_before_new_confirmation": True,
        "recovery_correction_was_parameter_tuning": False,
        "network_realizations_independent_of_task_k": True,
        "oracle_fields_used_for_policy_decisions": False,
        "event_truth_used_for_policy_decisions": False,
        "future_measurements_used": False,
        "temporal_protocols_pooled": False,
        "energy_is_modeled_not_hardware_measured": True,
        "actual_field_deployment_claimed": False,
        "claim_scope": str(confirmation_config["claim_scope"]),
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "task_s_summary.json", summary)
    return summary
