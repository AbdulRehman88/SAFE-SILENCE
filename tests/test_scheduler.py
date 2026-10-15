import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.protocol.scheduler import (
    PROPOSED_POLICY,
    REFERENCE_POLICY,
    SchedulerError,
    candidate_values,
    causal_ack_beliefs,
    run_scheduler,
    schedule_opportunity,
    upper_cvar,
    verify_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict]:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    return config["scheduler"], config["network"]


def _membership(batch_id: str, critical: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batch_id": [batch_id, batch_id],
            "critical_evidence_score": [int(critical), 0],
            "deadline_urgency": [1.0 if critical else 0.25, 0.25],
            "residual_magnitude": [0.8, 0.3],
            "environmental_uncertainty": [0.7, 0.2],
            "retained": [1, 1],
        }
    )


def _summary(batch_id: str, timestamp: pd.Timestamp, critical: bool = False) -> dict:
    return {
        "batch_id": batch_id,
        "protocol": "primary_15min",
        "site": "S01",
        "model_role": "external_test",
        "window_key": "w0",
        "relay_flush_timestamp": timestamp,
        "critical_evidence_max": float(critical),
        "environmental_uncertainty_max": 0.7,
        "deadline_urgency_max": 1.0 if critical else 0.25,
        "encoded_wire_bytes": 134,
    }


def test_upper_cvar_is_sensitive_to_rare_tail_loss() -> None:
    values = np.asarray([0.0, 0.0, 0.0, 10.0])
    assert upper_cvar(values, 0.75) == 10.0
    assert upper_cvar(values, 0.75) > values.mean()


def test_ack_belief_is_strictly_lagged() -> None:
    config, _ = _configs()
    transport = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC"),
            "window_key": ["w0"] * 3,
            "repetition": [0] * 3,
            "packet_attempted": [1, 1, 1],
            "acknowledged": [0, 1, 1],
            "_oracle_link_state": ["outage", "good", "good"],
        }
    )
    belief = causal_ack_beliefs(transport, config["causal_link_belief"])
    assert belief.loc[0, "causal_ack_belief"] == pytest.approx(0.95)
    assert belief.loc[1, "causal_ack_belief"] == pytest.approx(0.76)
    assert "_oracle_link_state" not in belief


def test_candidate_value_rejects_oracle_input() -> None:
    scheduler, network = _configs()
    summary = _summary("b0", pd.Timestamp("2025-01-01", tz="UTC"), critical=True)
    summary["_oracle_link_state"] = "good"
    with pytest.raises(SchedulerError, match="Forbidden scheduler evidence"):
        candidate_values(
            summary,
            _membership("b0", critical=True),
            0.9,
            scheduler["loss_profiles"]["primary"],
            scheduler,
            network,
        )


def test_cvar_value_credits_delivery_probability() -> None:
    scheduler, network = _configs()
    summary = _summary("b0", pd.Timestamp("2025-01-01", tz="UTC"), critical=True)
    high = candidate_values(
        summary,
        _membership("b0", critical=True),
        0.9,
        scheduler["loss_profiles"]["primary"],
        scheduler,
        network,
    )
    low = candidate_values(
        summary,
        _membership("b0", critical=True),
        0.1,
        scheduler["loss_profiles"]["primary"],
        scheduler,
        network,
    )
    assert high["predicted_tail_risk_reduction"] > low["predicted_tail_risk_reduction"]
    assert high["tail_utility"] > low["tail_utility"]


def test_cvar_and_expected_policies_can_make_distinct_matched_budget_choices() -> None:
    config, _ = _configs()
    candidates = []
    for index in range(5):
        candidate = _summary(
            f"b{index}", pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(index, "h")
        )
        candidate.update(
            {
                "tail_utility": 10.0 if index == 0 else 2.0,
                "expected_utility": 0.1 if index == 0 else 2.0,
                "predicted_tail_risk_reduction": 1.0,
                "predicted_expected_risk_reduction": 1.0,
                "expected_wire_bytes": 100.0,
                "modeled_energy_j": 1.0,
                "expected_airtime_seconds": 0.1,
                "mandatory_candidate": False,
                "mandatory_critical": False,
                "mandatory_deadline": False,
            }
        )
        candidates.append(candidate)
    tail = schedule_opportunity(candidates, PROPOSED_POLICY, config)
    expected = schedule_opportunity(candidates, REFERENCE_POLICY, config)
    assert bool(tail.loc[tail["batch_id"] == "b0", "selected"].iloc[0])
    assert not bool(expected.loc[expected["batch_id"] == "b0", "selected"].iloc[0])
    assert tail["selected"].sum() == expected["selected"].sum() == 4


def test_upstream_integrity_rejects_tampering(tmp_path: Path) -> None:
    fusion_root = tmp_path / "task_o"
    network_root = tmp_path / "task_k"
    fusion_root.mkdir()
    network_root.mkdir()
    fusion_artifact = fusion_root / "fusion_metrics.csv"
    network_artifact = network_root / "network_transport_metrics.csv"
    fusion_artifact.write_text("locked\n", encoding="utf-8")
    network_artifact.write_text("locked\n", encoding="utf-8")
    (fusion_root / "task_o_summary.json").write_text(
        json.dumps(
            {
                "task": "O",
                "fusion_version": "1.0",
                "artifacts": [
                    {"path": str(fusion_artifact), "sha256": sha256_file(fusion_artifact)}
                ],
            }
        ),
        encoding="utf-8",
    )
    (network_root / "task_k_summary.json").write_text(
        json.dumps(
            {
                "task": "K",
                "network_version": "1.0",
                "artifacts": [
                    {"path": str(network_artifact), "sha256": sha256_file(network_artifact)}
                ],
            }
        ),
        encoding="utf-8",
    )
    verify_inputs(fusion_root, network_root)
    fusion_artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(SchedulerError, match="changed or is missing"):
        verify_inputs(fusion_root, network_root)


def test_synthetic_end_to_end_scheduler_run(tmp_path: Path) -> None:
    scheduler, network = _configs()
    fusion_root = tmp_path / "task_o"
    network_root = tmp_path / "task_k"
    output_root = tmp_path / "task_p"
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    summaries = []
    memberships = []
    for index, timestamp in enumerate(timestamps):
        batch_id = f"b{index}"
        value = _summary(batch_id, timestamp, critical=index == 7)
        value.update(
            {
                "fusion_version": "1.0",
                "relay_sequence": index,
                "earliest_sensing_timestamp": timestamp,
                "newest_sensing_timestamp": timestamp,
                "input_items": 2,
                "retained_items": 2,
                "dropped_items": 0,
                "expired_items": 0,
                "quality_rejected_items": 0,
                "source_ids_json": '["S01"]',
                "retained_item_ids_json": "[]",
                "input_set_sha256": "0" * 64,
                "relay_crc32_hex": "00000000",
                "order_audit_passed": 1,
            }
        )
        summaries.append(value)
        memberships.append(_membership(batch_id, critical=index == 7))
    summary_frame = pd.DataFrame(summaries)
    membership_frame = pd.concat(memberships, ignore_index=True)
    summary_path = fusion_root / "summaries" / "primary_15min" / "S01.parquet"
    membership_path = fusion_root / "membership" / "primary_15min" / "S01.parquet"
    summary_path.parent.mkdir(parents=True)
    membership_path.parent.mkdir(parents=True)
    summary_frame.to_parquet(summary_path, index=False)
    membership_frame.to_parquet(membership_path, index=False)
    fusion_records = [
        {
            "protocol": "primary_15min",
            "site": "S01",
            "artifact_kind": "summary",
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        {
            "protocol": "primary_15min",
            "site": "S01",
            "artifact_kind": "membership",
            "path": str(membership_path),
            "sha256": sha256_file(membership_path),
        },
    ]
    fusion_root.mkdir(exist_ok=True)
    (fusion_root / "task_o_summary.json").write_text(
        json.dumps({"task": "O", "fusion_version": "1.0", "artifacts": fusion_records}),
        encoding="utf-8",
    )

    transport = pd.DataFrame(
        {
            "timestamp": timestamps,
            "window_key": ["w0"] * 8,
            "repetition": [0] * 8,
            "packet_attempted": [1] * 8,
            "acknowledged": [1, 1, 1, 0, 1, 1, 1, 1],
        }
    )
    transport_path = (
        network_root / "transport" / "primary_15min" / "S01" / "F0_stable.parquet"
    )
    transport_path.parent.mkdir(parents=True)
    transport.to_parquet(transport_path, index=False)
    network_record = {
        "protocol": "primary_15min",
        "site": "S01",
        "model_role": "external_test",
        "scenario": "F0_stable",
        "path": str(transport_path),
        "sha256": sha256_file(transport_path),
    }
    (network_root / "task_k_summary.json").write_text(
        json.dumps({"task": "K", "network_version": "1.0", "artifacts": [network_record]}),
        encoding="utf-8",
    )
    local_network = deepcopy(network)
    result = run_scheduler(
        fusion_root, network_root, output_root, scheduler, local_network
    )
    assert result["task"] == "P"
    assert result["budget_violations"] == 0
    assert result["metrics_rows"] == 6
    assert result["proposed_trace_rows"] == 8
    metrics = pd.read_csv(output_root / "scheduler_metrics.csv")
    assert set(metrics["policy"]) == {PROPOSED_POLICY, REFERENCE_POLICY}
