import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.evaluation.integration import run_integration
from safe_silence.evaluation.reintegration import (
    ReintegrationError,
    _contract_id,
    build_relay_transport,
    paired_comparisons,
    reintegrate_receiver_and_cause,
    run_reintegration,
    verify_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")


def _candidates() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "batch_id": [f"b{index}" for index in range(8)],
            "opportunity_id": ["o0"] * 8,
            "window_key": ["w0"] * 8,
            "repetition": [0] * 8,
            "relay_flush_timestamp": timestamps,
            "earliest_sensing_timestamp": timestamps - pd.Timedelta(minutes=15),
            "critical_evidence_max": [1.0, *([0.0] * 7)],
            "deadline_urgency_max": [1.0, *([0.25] * 7)],
            "encoded_wire_bytes": [134] * 8,
            "expected_wire_bytes": [150.0] * 8,
            "modeled_energy_j": [2.0] * 8,
            "expected_airtime_seconds": [0.5] * 8,
            "cost_denominator": [3.0] * 8,
            "tail_utility": np.linspace(8.0, 1.0, 8),
            "expected_utility": np.linspace(1.0, 8.0, 8),
            "tail_loss_before": np.linspace(4.0, 1.0, 8),
            "tail_loss_after_success": np.linspace(0.4, 0.1, 8),
            "expected_loss_before": np.linspace(2.0, 0.5, 8),
            "expected_loss_after_success": np.linspace(0.2, 0.05, 8),
            "environmental_uncertainty_max": np.linspace(0.1, 0.8, 8),
            "mandatory_critical": [1, *([0] * 7)],
            "mandatory_deadline": [1, *([0] * 7)],
            "mandatory_candidate": [1, *([0] * 7)],
            "causal_ack_belief": [0.95] * 8,
            "selected": [1, 1, 1, 1, 0, 0, 0, 0],
            "selection_reason": ["ranked_utility"] * 8,
            "opportunity_selected_packets": [4] * 8,
            "opportunity_expected_wire_bytes": [600.0] * 8,
            "opportunity_modeled_energy_j": [8.0] * 8,
            "opportunity_expected_airtime_seconds": [2.0] * 8,
        }
    )


def _relay_rows() -> pd.DataFrame:
    frame = _candidates().iloc[:2].copy()
    frame["actual_packet_delivered_timely"] = [0, 1]
    frame["actual_acknowledged"] = [0, 1]
    frame["actual_network_bytes"] = [0, 134]
    frame["actual_modeled_energy_j"] = [0.0, 2.0]
    frame["retained_residual_magnitude_max"] = [0.3, 0.9]
    frame["_oracle_link_state"] = ["outage", "good"]
    frame["realization_seed"] = [1, 2]
    return frame


def test_task_r_configuration_locks_nonoracle_paired_evaluation() -> None:
    config = _config()["reintegration"]
    assert not config["oracle_fields_are_policy_inputs"]
    assert not config["event_truth_is_policy_input"]
    assert not config["scheduler_selection_is_receiver_input"]
    assert not config["statistics"]["temporal_protocols_may_be_pooled"]
    assert config["statistics"]["paired_unit"] == "repetition"


def test_relay_adapter_is_deterministic_and_uses_only_transmitted_extrema() -> None:
    config = _config()["reintegration"]["relay_receiver_contract"]
    transport = build_relay_transport(_relay_rows(), config)
    assert transport.loc[0, "node_action"] == "alert"
    assert transport.loc[1, "node_action"] == "innovation"
    assert transport.loc[0, "semantic_risk"] == pytest.approx(1.0)
    assert transport.loc[1, "semantic_risk"] == pytest.approx(0.9)
    assert transport.loc[0, "silence_contract_id"] == _contract_id("b0")
    assert transport.loc[0, "critical_packet_id"] == "b0"
    assert transport.loc[0, "critical_packet_wire_bytes"] == 134
    assert _contract_id("b0") == _contract_id("b0")
    assert 0 <= _contract_id("b0") < 2**48


def test_missing_critical_summary_is_scored_unsafe_without_receiver_leakage() -> None:
    config = _config()
    transport = build_relay_transport(
        _relay_rows(), config["reintegration"]["relay_receiver_contract"]
    )
    result = reintegrate_receiver_and_cause(
        transport, config["receiver"], config["cause"]
    )
    assert result.loc[0, "packet_arrived"] == 0
    assert result.loc[0, "received_action"] is None
    assert result.loc[0, "evaluation_only_unsafe_silence"] == 1
    assert result.loc[0, "evaluation_only_true_cause"] == "isolated_loss"


def test_paired_statistics_keep_protocols_separate_and_apply_holm() -> None:
    config = deepcopy(_config()["reintegration"])
    config["statistics"]["bootstrap_resamples"] = 1000
    rows = []
    for protocol in ("primary_15min", "secondary_60min"):
        for repetition in range(4):
            for scheduler, recovery, shift in (
                ("cvar_risk_reduction_per_joule", "safe_silence_bounded", 0.1),
                ("expected_risk_reduction_per_joule", "safe_silence_bounded", 0.0),
                ("cvar_risk_reduction_per_joule", "no_recovery", -0.1),
                ("expected_risk_reduction_per_joule", "no_recovery", -0.2),
            ):
                row = {
                    "protocol": protocol,
                    "site": "S01",
                    "model_role": "external_test",
                    "scenario": "F0_stable",
                    "repetition": repetition,
                    "scheduler_policy": scheduler,
                    "recovery_policy": recovery,
                }
                for metric, higher in {
                    "post_recovery_critical_delivery_recall": True,
                    "post_recovery_timely_critical_delivery_recall": True,
                    "critical_false_trust_rate": False,
                    "post_recovery_exact_retained_critical_item_recall": True,
                    "realized_post_recovery_tail_proxy_reduction": True,
                    "tail_proxy_reduction_per_total_modeled_joule": True,
                    "total_network_bytes": False,
                    "total_modeled_energy_j": False,
                }.items():
                    row[metric] = (1.0 + shift) if higher else (1.0 - shift)
                rows.append(row)
    output = paired_comparisons(pd.DataFrame(rows), config)
    assert len(output) == 2 * 2 * 8
    assert set(output["protocol"]) == {"primary_15min", "secondary_60min"}
    assert output["holm_adjusted_p_value"].notna().all()


def test_task_q_integrity_rejects_tampering(tmp_path: Path) -> None:
    roots = {task: tmp_path / f"task_{task.lower()}" for task in ("Q", "P", "O", "K")}
    versions = {
        "Q": "integration_version",
        "P": "scheduler_version",
        "O": "fusion_version",
        "K": "network_version",
    }
    for task, root in roots.items():
        root.mkdir()
        artifact = root / "locked.csv"
        artifact.write_text("locked\n", encoding="utf-8")
        summary = {
            "task": task,
            versions[task]: "1.0",
            "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}],
        }
        if task == "P":
            summary["component_evaluation_only"] = True
        if task == "K":
            summary["matched_counterfactual_randomness_stored"] = True
        (root / f"task_{task.lower()}_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    verify_inputs(roots["Q"], roots["P"], roots["O"], roots["K"])
    (roots["Q"] / "locked.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ReintegrationError, match="changed or is missing"):
        verify_inputs(roots["Q"], roots["P"], roots["O"], roots["K"])


def _write_upstreams(tmp_path: Path) -> tuple[Path, Path, Path]:
    scheduler_root = tmp_path / "task_p"
    fusion_root = tmp_path / "task_o"
    network_root = tmp_path / "task_k"
    protocol, site, scenario = "primary_15min", "S01", "F0_stable"
    schedule = _candidates()
    schedule_path = scheduler_root / "schedules" / protocol / site / f"{scenario}.parquet"
    schedule_path.parent.mkdir(parents=True)
    schedule.to_parquet(schedule_path, index=False)
    schedule_record = {
        "protocol": protocol,
        "site": site,
        "model_role": "external_test",
        "scenario": scenario,
        "path": str(schedule_path),
        "sha256": sha256_file(schedule_path),
    }
    (scheduler_root / "task_p_summary.json").write_text(
        json.dumps(
            {
                "task": "P",
                "scheduler_version": "1.0",
                "component_evaluation_only": True,
                "artifacts": [schedule_record],
            }
        ),
        encoding="utf-8",
    )

    membership = pd.DataFrame(
        {
            "batch_id": schedule["batch_id"],
            "sensing_timestamp": schedule["earliest_sensing_timestamp"],
            "critical_evidence_score": [1.0, *([0.0] * 7)],
            "residual_magnitude": np.linspace(0.2, 0.9, 8),
            "retained": [1] * 8,
        }
    )
    membership_path = fusion_root / "membership" / protocol / f"{site}.parquet"
    membership_path.parent.mkdir(parents=True)
    membership.to_parquet(membership_path, index=False)
    membership_record = {
        "protocol": protocol,
        "site": site,
        "artifact_kind": "membership",
        "path": str(membership_path),
        "sha256": sha256_file(membership_path),
    }
    (fusion_root / "task_o_summary.json").write_text(
        json.dumps(
            {"task": "O", "fusion_version": "1.0", "artifacts": [membership_record]}
        ),
        encoding="utf-8",
    )

    transport = pd.DataFrame(
        {
            "timestamp": schedule["relay_flush_timestamp"],
            "window_key": schedule["window_key"],
            "repetition": 0,
            "realization_seed": np.arange(8) + 100,
            "_oracle_link_state": "good",
        }
    )
    for attempt in range(3):
        transport[f"_oracle_data_u{attempt}"] = 0.0
        transport[f"_oracle_ack_u{attempt}"] = 0.0
        transport[f"_oracle_jitter_u{attempt}"] = 0.0
    transport_path = network_root / "transport" / protocol / site / f"{scenario}.parquet"
    transport_path.parent.mkdir(parents=True)
    transport.to_parquet(transport_path, index=False)
    transport_record = {
        "protocol": protocol,
        "site": site,
        "model_role": "external_test",
        "scenario": scenario,
        "path": str(transport_path),
        "sha256": sha256_file(transport_path),
    }
    (network_root / "task_k_summary.json").write_text(
        json.dumps(
            {
                "task": "K",
                "network_version": "1.0",
                "matched_counterfactual_randomness_stored": True,
                "artifacts": [transport_record],
            }
        ),
        encoding="utf-8",
    )
    return scheduler_root, fusion_root, network_root


def test_synthetic_full_reintegration_run(tmp_path: Path) -> None:
    config = _config()
    scheduler_root, fusion_root, network_root = _write_upstreams(tmp_path)
    integration_root = tmp_path / "task_q"
    output_root = tmp_path / "task_r"
    run_integration(
        scheduler_root,
        fusion_root,
        network_root,
        integration_root,
        config["integration"],
        config["scheduler"],
        config["network"],
        config["preprocessing"],
    )
    reintegration = deepcopy(config["reintegration"])
    reintegration["statistics"]["bootstrap_resamples"] = 1000
    result = run_reintegration(
        integration_root,
        scheduler_root,
        fusion_root,
        network_root,
        output_root,
        reintegration,
        config["integration"],
        config["scheduler"],
        config["network"],
        config["receiver"],
        config["cause"],
        config["recovery"],
        config["preprocessing"],
    )
    assert result["task"] == "R"
    assert result["metrics_rows"] == 4
    assert result["paired_comparison_rows"] == 16
    assert result["proposed_system_trace_rows"] == 8
    assert (output_root / "task_r_summary.json").exists()
    assert (output_root / "paired_comparisons.csv").exists()


def test_task_r_cli_is_registered() -> None:
    from safe_silence.cli import build_parser

    args = build_parser().parse_args(["reintegrate-system"])
    assert args.command == "reintegrate-system"
