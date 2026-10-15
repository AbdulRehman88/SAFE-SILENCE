import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.protocol.receiver import (
    ReceiverInferenceError,
    build_receiver_beliefs,
    interpret_receiver,
    interpret_receiver_group,
    make_receiver_observations,
    receiver_metrics,
    verify_task_k_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")["receiver"]


def _transport(rows: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC"),
            "window_key": ["w0"] * rows,
            "repetition": [0] * rows,
            "node_action": ["heartbeat", *(["silent"] * (rows - 1))],
            "semantic_risk": [0.1] * rows,
            "validity_horizon_steps": [4] * rows,
            "silence_contract_id": [1] * rows,
            "packet_delivered_timely": [1, *([0] * (rows - 1))],
            "causal_critical_evidence": [0] * rows,
            "_oracle_link_state": ["good"] * rows,
        }
    )
    return frame


def test_missing_packet_masks_all_source_packet_fields() -> None:
    frame = _transport(rows=2)
    frame.loc[1, "node_action"] = "alert"
    frame.loc[1, "semantic_risk"] = 1.0
    observations = make_receiver_observations(frame, _config())
    assert observations.loc[1, "received_action"] is None
    assert np.isnan(observations.loc[1, "received_semantic_risk"])
    assert np.isnan(observations.loc[1, "received_validity_horizon_steps"])
    assert np.isnan(observations.loc[1, "received_contract_id"])


def test_oracle_and_truth_fields_are_rejected_as_inference_evidence() -> None:
    observations = make_receiver_observations(_transport(rows=2), _config())
    observations["_oracle_link_state"] = ["good", "outage"]
    with pytest.raises(ReceiverInferenceError, match="Forbidden receiver evidence"):
        interpret_receiver_group(observations, _config())
    observations = make_receiver_observations(_transport(rows=2), _config())
    observations["causal_critical_evidence"] = [0, 1]
    with pytest.raises(ReceiverInferenceError, match="Forbidden receiver evidence"):
        interpret_receiver_group(observations, _config())


def test_valid_contract_supports_bounded_trusted_silence() -> None:
    result = interpret_receiver(_transport(rows=4), _config())
    assert result.loc[0, "receiver_state"] == "updated"
    assert (result.loc[1:3, "receiver_state"] == "trusted_silence").all()
    assert (result.loc[1:3, "receiver_contract_valid"] == 1).all()


def test_expired_contract_becomes_ambiguous_then_stale_unsafe() -> None:
    frame = _transport(rows=4)
    frame.loc[0, "validity_horizon_steps"] = 2
    result = interpret_receiver(frame, _config())
    assert result.loc[1, "receiver_state"] == "trusted_silence"
    assert result.loc[2, "receiver_state"] == "ambiguous"
    assert result.loc[3, "receiver_state"] == "stale_unsafe"
    assert not bool(result.loc[2, "receiver_contract_valid"])


def test_no_received_contract_is_never_trusted() -> None:
    frame = _transport(rows=3)
    frame["packet_delivered_timely"] = 0
    result = interpret_receiver(frame, _config())
    assert result.loc[0, "receiver_state"] == "ambiguous"
    assert result.loc[1, "receiver_state"] == "stale_unsafe"
    assert "trusted_silence" not in set(result["receiver_state"])


def test_lost_source_content_cannot_change_receiver_inference() -> None:
    first = _transport(rows=3)
    second = first.copy()
    second.loc[1, "node_action"] = "alert"
    second.loc[1, "semantic_risk"] = 1.0
    second.loc[1, "validity_horizon_steps"] = 0
    a = interpret_receiver(first, _config())
    b = interpret_receiver(second, _config())
    columns = ["receiver_state", "connectivity_belief", "environmental_uncertainty"]
    pd.testing.assert_frame_equal(a[columns], b[columns])


def test_lost_critical_alert_inside_contract_is_counted_as_false_trust() -> None:
    frame = _transport(rows=2)
    frame.loc[1, "node_action"] = "alert"
    frame.loc[1, "causal_critical_evidence"] = 1
    frame.loc[1, "_oracle_link_state"] = "outage"
    result = interpret_receiver(frame, _config())
    metrics = receiver_metrics(result, "primary_15min", "S01", "external_test", "F13", 0)
    assert result.loc[1, "receiver_state"] == "trusted_silence"
    assert result.loc[1, "evaluation_only_false_trust"] == 1
    assert metrics["critical_false_trust_rate"] == 1.0


def test_task_k_integrity_rejects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "task_k"
    root.mkdir()
    artifact = root / "network_transport_metrics.csv"
    artifact.write_text("locked\n", encoding="utf-8")
    summary = {
        "task": "K",
        "network_version": "1.0",
        "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}],
    }
    (root / "task_k_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _, verified = verify_task_k_inputs(root)
    assert verified == 1
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ReceiverInferenceError, match="changed or is missing"):
        verify_task_k_inputs(root)


def test_receiver_config_locks_scientific_contract() -> None:
    config = _config()
    assert not config["parameters_tuned_on_test_or_external_data"]
    assert config["uses_nominal_model_not_scenario_oracle"]
    assert config["oracle_fields_are_evaluation_only"]
    assert config["event_truth_is_evaluation_only"]
    assert config["state_policy"]["no_normal_fallback_after_contract_expiry"]
    assert config["reporting"]["primary_endpoint"] == "critical_false_trust_rate"


def test_synthetic_end_to_end_receiver_run(tmp_path: Path) -> None:
    input_root = tmp_path / "task_k"
    trace_path = input_root / "transport" / "primary_15min" / "S01" / "F2_burst.parquet"
    trace_path.parent.mkdir(parents=True)
    frame = _transport(rows=4)
    frame.to_parquet(trace_path, index=False)
    record = {
        "protocol": "primary_15min",
        "site": "S01",
        "model_role": "external_test",
        "scenario": "F2_burst",
        "path": str(trace_path),
        "rows": 4,
        "sha256": sha256_file(trace_path),
    }
    summary = {
        "task": "K",
        "network_version": "1.0",
        "artifacts": [record],
    }
    (input_root / "task_k_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    config = deepcopy(_config())
    result = build_receiver_beliefs(input_root, tmp_path / "task_l", config)
    assert result["task"] == "L"
    assert result["metrics_rows"] == 1
    assert not result["receiver_uses_hidden_oracle_state"]
    assert (tmp_path / "task_l" / "receiver_metrics.csv").exists()
