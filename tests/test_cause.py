import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.protocol.cause import (
    CauseInferenceError,
    cause_metrics,
    infer_ambiguity_causes,
    infer_cause_group,
    infer_causes,
    make_cause_observations,
    verify_task_l_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")["cause"]


def _beliefs(rows: int = 4) -> pd.DataFrame:
    timestamp = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "window_key": ["w0"] * rows,
            "repetition": [0] * rows,
            "packet_arrived": [1, *([0] * (rows - 1))],
            "receiver_state": ["updated", "trusted_silence", "ambiguous", "stale_unsafe"][:rows],
            "connectivity_belief": [0.999, 0.99, 0.55, 0.05][:rows],
            "environmental_uncertainty": [0.1, 0.2, 0.8, 1.0][:rows],
            "receiver_contract_age_steps": list(range(rows)),
            "receiver_overdue_steps": [0, 0, 1, 2][:rows],
            "receiver_contract_valid": [1, 1, 0, 0][:rows],
            "receiver_contact_due": [0, 0, 1, 1][:rows],
            "receiver_active_contract_id": [1.0] * rows,
            "received_action": ["heartbeat", *([None] * (rows - 1))],
            "received_semantic_risk": [0.1, *([np.nan] * (rows - 1))],
            "received_validity_horizon_steps": [4.0, *([np.nan] * (rows - 1))],
            "received_contract_id": [1.0, *([np.nan] * (rows - 1))],
            "evaluation_only_source_action": [
                "heartbeat",
                "silent",
                "heartbeat",
                "heartbeat",
            ][:rows],
            "evaluation_only_critical_event": [0] * rows,
            "evaluation_only_unsafe_silence": [0, 0, 1, 1][:rows],
            "evaluation_only_link_state": ["good", "good", "outage", "outage"][:rows],
        }
    )
    return frame


def test_cause_observations_exclude_all_evaluation_truth() -> None:
    observations = make_cause_observations(_beliefs())
    assert not any(column.startswith("evaluation_only_") for column in observations.columns)
    assert not any(column.startswith("_oracle_") for column in observations.columns)


def test_cause_inference_rejects_oracle_or_evaluation_fields() -> None:
    observations = make_cause_observations(_beliefs())
    observations["evaluation_only_true_cause"] = "persistent_outage"
    with pytest.raises(CauseInferenceError, match="Forbidden cause evidence"):
        infer_cause_group(observations, _config())
    observations = make_cause_observations(_beliefs())
    observations["_oracle_link_state"] = "outage"
    with pytest.raises(CauseInferenceError, match="Forbidden cause evidence"):
        infer_cause_group(observations, _config())


def test_valid_bounded_silence_is_identified_as_valid_suppression() -> None:
    result = infer_causes(_beliefs(), _config())
    assert result.loc[1, "cause_decision"] == "valid_suppression"
    assert result.loc[1, "cause_probability_valid_suppression"] > 0.9


def test_valid_suppression_is_hard_gated_by_receiver_trust_contract() -> None:
    frame = _beliefs()
    frame.loc[1, "receiver_state"] = "ambiguous"
    result = infer_causes(frame, _config())
    assert result.loc[1, "cause_decision"] == "unknown"
    assert result.loc[1, "valid_suppression_admissible"] == 0
    assert result.loc[1, "cause_abstained_by_safety_gate"] == 1


def test_long_low_connectivity_silence_is_persistent_outage() -> None:
    result = infer_causes(_beliefs(), _config())
    assert result.loc[3, "cause_decision"] == "persistent_outage"
    assert result.loc[3, "evaluation_only_true_cause"] == "persistent_outage"


def test_received_packet_has_no_silence_cause_posterior() -> None:
    result = infer_causes(_beliefs(), _config())
    assert result.loc[0, "cause_decision"] == "observed_update"
    assert np.isnan(result.loc[0, "cause_probability_unknown"])


def test_changing_evaluation_truth_cannot_change_inference() -> None:
    first = _beliefs()
    second = first.copy()
    second["evaluation_only_source_action"] = "silent"
    second["evaluation_only_link_state"] = "good"
    second["evaluation_only_critical_event"] = 1
    a = infer_causes(first, _config())
    b = infer_causes(second, _config())
    columns = ["cause_decision", *[f"cause_probability_{cause}" for cause in _config()["classes"]]]
    pd.testing.assert_frame_equal(a[columns], b[columns])


def test_unknown_abstention_is_non_benign_for_failure_detection() -> None:
    result = infer_causes(_beliefs(), _config())
    result.loc[2, "cause_decision"] = "unknown"
    metrics = cause_metrics(result, "primary_15min", "S01", "external_test", "F2", 0)
    assert metrics["coarse_failure_detection_recall"] == 1.0


def test_task_l_integrity_rejects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "task_l"
    root.mkdir()
    artifact = root / "receiver_metrics.csv"
    artifact.write_text("locked\n", encoding="utf-8")
    summary = {
        "task": "L",
        "receiver_version": "1.0",
        "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}],
    }
    (root / "task_l_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _, verified = verify_task_l_inputs(root)
    assert verified == 1
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(CauseInferenceError, match="changed or is missing"):
        verify_task_l_inputs(root)


def test_cause_config_locks_taxonomy_and_unknown() -> None:
    config = _config()
    assert not config["parameters_tuned_on_test_or_external_data"]
    assert config["oracle_fields_are_evaluation_only"]
    assert config["exact_physical_cause_is_not_claimed"]
    assert config["abstention"]["unknown_is_retained"]
    assert "unknown" in config["classes"]
    assert config["score_semantics"] == "rule_based_pseudo_posterior"
    assert not config["probabilities_are_empirically_calibrated"]


def test_synthetic_end_to_end_cause_run(tmp_path: Path) -> None:
    input_root = tmp_path / "task_l"
    trace_path = input_root / "beliefs" / "primary_15min" / "S01" / "F2_burst.parquet"
    trace_path.parent.mkdir(parents=True)
    frame = _beliefs()
    frame.to_parquet(trace_path, index=False)
    record = {
        "protocol": "primary_15min",
        "site": "S01",
        "model_role": "external_test",
        "scenario": "F2_burst",
        "path": str(trace_path),
        "rows": len(frame),
        "sha256": sha256_file(trace_path),
    }
    summary = {"task": "L", "receiver_version": "1.0", "artifacts": [record]}
    (input_root / "task_l_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = infer_ambiguity_causes(input_root, tmp_path / "task_m", deepcopy(_config()))
    assert result["task"] == "M"
    assert result["metrics_rows"] == 1
    assert not result["inference_uses_oracle_fields"]
    assert (tmp_path / "task_m" / "cause_metrics.csv").exists()
