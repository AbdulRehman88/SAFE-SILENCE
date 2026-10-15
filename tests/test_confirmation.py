import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.evaluation.confirmation import (
    ConfirmationError,
    audit_independent_seeds,
    confirmatory_comparisons,
    run_confirmation,
    secondary_descriptive_comparisons,
    verify_prior_task_r,
)
from safe_silence.simulation.network import _derived_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")


def test_task_s_configuration_freezes_independent_confirmation() -> None:
    config = _config()
    confirmation = config["confirmation"]
    assert confirmation["independent_network"]["repetitions"] == 30
    assert confirmation["independent_network"]["base_seed"] != config["network"]["base_seed"]
    assert not confirmation["prior_task_r_results_used_to_tune_algorithms"]
    assert confirmation["independent_replication_is_used_for_confirmation"]
    assert not confirmation["primary_analysis"]["temporal_protocols_may_be_pooled"]


def _metrics(repetitions: int = 30) -> pd.DataFrame:
    rows = []
    for scenario_index in range(7):
        scenario = f"F{scenario_index}"
        for repetition in range(repetitions):
            for scheduler, recovery, offset in (
                ("cvar_risk_reduction_per_joule", "safe_silence_bounded", 0.20),
                ("expected_risk_reduction_per_joule", "safe_silence_bounded", 0.10),
                ("cvar_risk_reduction_per_joule", "no_recovery", 0.00),
                ("expected_risk_reduction_per_joule", "no_recovery", -0.10),
            ):
                rows.append(
                    {
                        "protocol": "primary_15min",
                        "site": "CE07SHSM",
                        "model_role": "external_test",
                        "scenario": scenario,
                        "repetition": repetition,
                        "scheduler_policy": scheduler,
                        "recovery_policy": recovery,
                        "post_recovery_critical_delivery_recall": 0.6 + offset,
                        "realized_post_recovery_tail_proxy_reduction": 10.0 + offset,
                        "critical_false_trust_rate": 0.2 - offset / 2,
                        "post_recovery_timely_critical_delivery_recall": 0.55 + offset,
                        "post_recovery_exact_retained_critical_item_recall": 0.5 + offset,
                        "tail_proxy_reduction_per_total_modeled_joule": 0.8 - offset,
                        "total_network_bytes": 1000 + offset * 100,
                        "total_modeled_energy_j": 100 + offset * 10,
                    }
                )
    return pd.DataFrame(rows)


def test_confirmatory_family_has_14_tests_and_can_reject_with_30_pairs() -> None:
    config = deepcopy(_config()["confirmation"])
    config["primary_analysis"]["bootstrap_resamples"] = 1000
    output = confirmatory_comparisons(_metrics(), config)
    assert len(output) == 2 * 7 * 2
    assert output["paired_repetitions"].eq(30).all()
    assert output["confirmatory_support"].all()
    sizes = output.groupby(["comparison", "protocol", "site"]).size()
    assert sizes.eq(14).all()


def test_secondary_tradeoffs_remain_exploratory_without_p_values() -> None:
    config = deepcopy(_config()["confirmation"])
    config["primary_analysis"]["bootstrap_resamples"] = 1000
    output = secondary_descriptive_comparisons(_metrics(), config)
    assert len(output) == 2 * 7 * 6
    assert output["exploratory_only"].all()
    assert output["wilcoxon_p_value"].isna().all()


def test_seed_audit_proves_disjoint_realizations(tmp_path: Path) -> None:
    config = _config()
    local = deepcopy(config["network"])
    local["base_seed"] = config["confirmation"]["independent_network"]["base_seed"]
    local["repetitions"] = 20
    protocol, site, role, scenario = (
        "primary_15min",
        "CE07SHSM",
        "external_test",
        "F0_stable",
    )
    seeds = [
        _derived_seed(local["base_seed"], protocol, site, role, scenario, repetition)
        for repetition in range(20)
    ]
    path = tmp_path / "transport" / protocol / site / f"{scenario}.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"realization_seed": np.repeat(seeds, 2)}).to_parquet(path, index=False)
    summary = {
        "artifacts": [
            {
                "protocol": protocol,
                "site": site,
                "model_role": role,
                "scenario": scenario,
            }
        ]
    }
    result = audit_independent_seeds(summary, config["network"], local, tmp_path)
    assert result["seed_overlap_count"] == 0
    assert result["confirmatory_unique_seeds"] == 20


def test_seed_audit_rejects_task_k_overlap(tmp_path: Path) -> None:
    config = _config()
    local = deepcopy(config["network"])
    local["repetitions"] = config["network"]["repetitions"]
    protocol, site, role, scenario = (
        "primary_15min",
        "CE07SHSM",
        "external_test",
        "F0_stable",
    )
    seeds = [
        _derived_seed(local["base_seed"], protocol, site, role, scenario, repetition)
        for repetition in range(local["repetitions"])
    ]
    path = tmp_path / "transport" / protocol / site / f"{scenario}.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"realization_seed": seeds}).to_parquet(path, index=False)
    summary = {
        "artifacts": [
            {
                "protocol": protocol,
                "site": site,
                "model_role": role,
                "scenario": scenario,
            }
        ]
    }
    with pytest.raises(ConfirmationError, match="overlap"):
        audit_independent_seeds(summary, config["network"], local, tmp_path)


def test_prior_task_r_tampering_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "locked.csv"
    artifact.write_text("locked\n", encoding="utf-8")
    summary = {
        "task": "R",
        "reintegration_version": "1.0",
        "recovery_logic_version": "1.1",
        "receiver_uses_oracle_fields": False,
        "receiver_uses_event_truth": False,
        "recovery_policy_uses_oracle_fields": False,
        "recovery_policy_uses_event_truth": False,
        "future_measurements_used": False,
        "parameters_tuned_on_test_or_external_data": False,
        "temporal_protocols_pooled": False,
        "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}],
    }
    (tmp_path / "task_r_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    verify_prior_task_r(tmp_path)
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ConfirmationError, match="changed or is missing"):
        verify_prior_task_r(tmp_path)


def _write_task_j_and_o(tmp_path: Path) -> tuple[Path, Path]:
    semantic_root = tmp_path / "task_j"
    fusion_root = tmp_path / "task_o"
    protocol, site = "primary_15min", "CE07SHSM"
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    evidence = pd.DataFrame(
        {
            "timestamp": timestamps,
            "window_key": ["w0"] * 8,
            "model_role": ["external_test"] * 8,
            "node_action": ["alert", *("innovation" for _ in range(7))],
            "decision_reason": ["critical_override", *("innovation" for _ in range(7))],
            "semantic_payload_bytes": [114] * 8,
            "causal_critical_evidence": [1, *([0] * 7)],
            "validity_horizon_steps": [0, *([3] * 7)],
            "silence_contract_id": np.arange(8) + 1,
            "contract_age_steps": [0] * 8,
            "semantic_risk": np.linspace(1.0, 0.2, 8),
        }
    )
    evidence_path = semantic_root / "evidence" / protocol / f"{site}.parquet"
    evidence_path.parent.mkdir(parents=True)
    evidence.to_parquet(evidence_path, index=False)
    (semantic_root / "task_j_summary.json").write_text(
        json.dumps(
            {
                "task": "J",
                "semantic_version": "1.0",
                "protocols": {protocol: {"grid_minutes": 15, "sites": [site]}},
                "artifacts": [
                    {
                        "protocol": protocol,
                        "site": site,
                        "path": str(evidence_path),
                        "sha256": sha256_file(evidence_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = []
    memberships = []
    for index, timestamp in enumerate(timestamps):
        batch_id = f"b{index}"
        summaries.append(
            {
                "batch_id": batch_id,
                "protocol": protocol,
                "site": site,
                "model_role": "external_test",
                "window_key": "w0",
                "relay_flush_timestamp": timestamp,
                "earliest_sensing_timestamp": timestamp,
                "critical_evidence_max": float(index == 0),
                "environmental_uncertainty_max": 0.2 + index / 10,
                "deadline_urgency_max": 1.0 if index == 0 else 0.25,
                "encoded_wire_bytes": 134,
            }
        )
        memberships.append(
            {
                "batch_id": batch_id,
                "sensing_timestamp": timestamp,
                "critical_evidence_score": float(index == 0),
                "deadline_urgency": 1.0 if index == 0 else 0.25,
                "residual_magnitude": 0.2 + index / 10,
                "environmental_uncertainty": 0.2 + index / 10,
                "retained": 1,
            }
        )
    summary_path = fusion_root / "summaries" / protocol / f"{site}.parquet"
    membership_path = fusion_root / "membership" / protocol / f"{site}.parquet"
    summary_path.parent.mkdir(parents=True)
    membership_path.parent.mkdir(parents=True)
    pd.DataFrame(summaries).to_parquet(summary_path, index=False)
    pd.DataFrame(memberships).to_parquet(membership_path, index=False)
    records = [
        {
            "protocol": protocol,
            "site": site,
            "artifact_kind": "summary",
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        {
            "protocol": protocol,
            "site": site,
            "artifact_kind": "membership",
            "path": str(membership_path),
            "sha256": sha256_file(membership_path),
        },
    ]
    (fusion_root / "task_o_summary.json").write_text(
        json.dumps({"task": "O", "fusion_version": "1.0", "artifacts": records}),
        encoding="utf-8",
    )
    return semantic_root, fusion_root


def _write_prior_r(tmp_path: Path) -> Path:
    root = tmp_path / "prior_r"
    root.mkdir()
    artifact = root / "system_metrics.csv"
    artifact.write_text("locked\n", encoding="utf-8")
    (root / "task_r_summary.json").write_text(
        json.dumps(
            {
                "task": "R",
                "reintegration_version": "1.0",
                "recovery_logic_version": "1.1",
                "receiver_uses_oracle_fields": False,
                "receiver_uses_event_truth": False,
                "recovery_policy_uses_oracle_fields": False,
                "recovery_policy_uses_event_truth": False,
                "future_measurements_used": False,
                "parameters_tuned_on_test_or_external_data": False,
                "temporal_protocols_pooled": False,
                "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_synthetic_full_confirmation_run(tmp_path: Path) -> None:
    config = _config()
    semantic_root, fusion_root = _write_task_j_and_o(tmp_path)
    prior_r = _write_prior_r(tmp_path)
    output_root = tmp_path / "task_s"
    confirmation = deepcopy(config["confirmation"])
    confirmation["independent_network"]["repetitions"] = 20
    confirmation["primary_analysis"]["bootstrap_resamples"] = 1000
    result = run_confirmation(
        semantic_root,
        fusion_root,
        prior_r,
        output_root,
        confirmation,
        config["network"],
        config["scheduler"],
        config["integration"],
        config["reintegration"],
        config["receiver"],
        config["cause"],
        config["recovery"],
        config["preprocessing"],
    )
    assert result["task"] == "S"
    assert result["independent_repetitions"] == 20
    assert result["seed_overlap_count"] == 0
    assert result["primary_family_size_per_protocol_comparison"] == 14
    assert result["primary_comparison_rows"] == 28
    assert (output_root / "task_s_summary.json").exists()
    assert (output_root / "confirmatory_comparisons.csv").exists()
    assert (output_root / "secondary_tradeoffs.csv").exists()


def test_task_s_cli_is_registered() -> None:
    from safe_silence.cli import build_parser

    args = build_parser().parse_args(["confirm-independent"])
    assert args.command == "confirm-independent"
