import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.evaluation.integration import (
    IntegrationError,
    integration_metrics,
    replay_selected,
    run_integration,
    schedule_reference,
    verify_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")


def _candidates() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "batch_id": [f"b{index}" for index in range(8)],
            "opportunity_id": ["o0"] * 8,
            "window_key": ["w0"] * 8,
            "repetition": [0] * 8,
            "relay_flush_timestamp": timestamps,
            "earliest_sensing_timestamp": timestamps - pd.Timedelta(minutes=15),
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
            "residual_magnitude_max": np.linspace(0.2, 0.9, 8),
            "mandatory_critical": [1, 0, 0, 0, 0, 0, 0, 0],
            "mandatory_deadline": [1, 0, 0, 0, 0, 0, 0, 0],
            "mandatory_candidate": [1, 0, 0, 0, 0, 0, 0, 0],
            "causal_ack_belief": [0.95] * 8,
            "selected": [1, 1, 1, 1, 0, 0, 0, 0],
            "selection_reason": ["ranked_utility"] * 8,
            "opportunity_selected_packets": [4] * 8,
            "opportunity_expected_wire_bytes": [600.0] * 8,
            "opportunity_modeled_energy_j": [8.0] * 8,
            "opportunity_expected_airtime_seconds": [2.0] * 8,
        }
    )
    return frame


def _oracle(frame: pd.DataFrame, data_u: float = 0.0, ack_u: float = 0.0) -> pd.DataFrame:
    output = frame.copy()
    output["_oracle_link_state"] = "good"
    for attempt in range(3):
        output[f"_oracle_data_u{attempt}"] = data_u
        output[f"_oracle_ack_u{attempt}"] = ack_u
        output[f"_oracle_jitter_u{attempt}"] = 0.0
    return output


def test_task_q_configuration_locks_matched_nonoracle_replay() -> None:
    config = _config()["integration"]
    assert config["matched_realizations"]
    assert config["matched_opportunity_constraints"]
    assert not config["oracle_fields_are_policy_inputs"]
    assert not config["current_outcome_is_policy_input"]
    assert config["relay_to_buoy_hops"] == 1


def test_reference_schedulers_respect_identical_opportunity_limits() -> None:
    config = _config()
    for policy in config["integration"]["reference_policies"]:
        scheduled = schedule_reference(_candidates(), policy, config["scheduler"])
        assert scheduled["selected"].sum() <= 4
        assert scheduled["opportunity_expected_wire_bytes"].iloc[0] <= 2000
        assert scheduled["opportunity_modeled_energy_j"].iloc[0] <= 40


def test_oracle_columns_cannot_change_policy_selection() -> None:
    config = _config()["scheduler"]
    clean = schedule_reference(_candidates(), "uncertainty_only", config)
    contaminated = _candidates().copy()
    contaminated["_oracle_link_state"] = ["outage", *("good" for _ in range(7))]
    replay = schedule_reference(contaminated, "uncertainty_only", config)
    assert clean["selected"].tolist() == replay["selected"].tolist()


def test_matched_replay_uses_stored_draws_and_counts_realized_cost() -> None:
    config = _config()
    frame = _oracle(_candidates())
    replay = replay_selected(
        frame,
        config["network"]["scenarios"]["F0_stable"],
        config["network"],
        1,
        900.0,
    )
    assert replay["actual_packet_delivered_timely"].sum() == 4
    assert replay.loc[replay["selected"] == 0, "actual_network_bytes"].sum() == 0
    assert replay.loc[replay["selected"] == 1, "actual_network_bytes"].sum() > 0


def test_integration_metrics_separate_batch_alert_and_exact_provenance() -> None:
    config = _config()
    frame = _oracle(_candidates())
    frame["critical_items"] = [2, *([0] * 7)]
    frame["retained_critical_items"] = [1, *([0] * 7)]
    frame["earliest_critical_timestamp"] = [
        frame.loc[0, "earliest_sensing_timestamp"],
        *([pd.NaT] * 7),
    ]
    replay = replay_selected(
        frame,
        config["network"]["scenarios"]["F0_stable"],
        config["network"],
        1,
        900.0,
    )
    metrics = integration_metrics(
        replay,
        "primary_15min",
        "S01",
        "external_test",
        "F0_stable",
        "reference",
        0,
        "cvar_risk_reduction_per_joule",
        config["scheduler"],
    )
    assert metrics["critical_batch_timely_delivery_recall"] == 1.0
    assert metrics["exact_retained_critical_item_timely_recall"] == 0.5


def test_upstream_integrity_rejects_task_p_tampering(tmp_path: Path) -> None:
    roots = {task: tmp_path / f"task_{task.lower()}" for task in ("P", "O", "K")}
    versions = {"P": "scheduler_version", "O": "fusion_version", "K": "network_version"}
    for task, root in roots.items():
        root.mkdir()
        artifact = root / "artifact.csv"
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
    verify_inputs(roots["P"], roots["O"], roots["K"])
    (roots["P"] / "artifact.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(IntegrationError, match="changed or is missing"):
        verify_inputs(roots["P"], roots["O"], roots["K"])


def test_synthetic_end_to_end_integration_run(tmp_path: Path) -> None:
    config = _config()
    scheduler_root = tmp_path / "task_p"
    fusion_root = tmp_path / "task_o"
    network_root = tmp_path / "task_k"
    output_root = tmp_path / "task_q"
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

    membership_rows = []
    for index, row in schedule.iterrows():
        membership_rows.append(
            {
                "batch_id": row["batch_id"],
                "sensing_timestamp": row["earliest_sensing_timestamp"],
                "critical_evidence_score": int(index == 0),
                "residual_magnitude": 0.2 + index / 10,
                "retained": 1,
            }
        )
    membership_path = fusion_root / "membership" / protocol / f"{site}.parquet"
    membership_path.parent.mkdir(parents=True)
    pd.DataFrame(membership_rows).to_parquet(membership_path, index=False)
    membership_record = {
        "protocol": protocol,
        "site": site,
        "artifact_kind": "membership",
        "path": str(membership_path),
        "sha256": sha256_file(membership_path),
    }
    (fusion_root / "task_o_summary.json").write_text(
        json.dumps({"task": "O", "fusion_version": "1.0", "artifacts": [membership_record]}),
        encoding="utf-8",
    )

    transport = pd.DataFrame(
        {
            "timestamp": schedule["relay_flush_timestamp"],
            "window_key": schedule["window_key"],
            "repetition": 0,
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

    local_preprocessing = deepcopy(config["preprocessing"])
    result = run_integration(
        scheduler_root,
        fusion_root,
        network_root,
        output_root,
        config["integration"],
        config["scheduler"],
        config["network"],
        local_preprocessing,
    )
    assert result["task"] == "Q"
    assert result["matched_budget_violations"] == 0
    assert result["metrics_rows"] == 9
    metrics = pd.read_csv(output_root / "end_to_end_metrics.csv")
    assert set(metrics["policy"]) == {
        config["integration"]["proposed_policy"],
        *config["integration"]["reference_policies"],
    }
