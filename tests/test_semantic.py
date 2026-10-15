import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.protocol.semantic import (
    SemanticEvidenceError,
    apply_nominal_policy,
    compute_semantic_evidence,
    validity_horizon,
    verify_task_i_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _semantic_config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")["semantic"]


def _evidence_frame(rows: int = 3) -> pd.DataFrame:
    data: dict[str, object] = {
        "protocol": ["primary_15min"] * rows,
        "site": ["CE02SHSM"] * rows,
        "model_role": ["internal_test"] * rows,
        "window_key": ["w0"] * rows,
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
        "label_available_event": np.zeros(rows, dtype=np.uint8),
        "valid_available_event": np.ones(rows, dtype=np.uint8),
    }
    for channel in ["temperature", "salinity", "oxygen", "chlorophyll"]:
        data[f"actual_{channel}"] = np.ones(rows)
        data[f"prediction_{channel}"] = np.ones(rows)
        data[f"uncertainty_std_{channel}"] = np.full(rows, 0.1)
        data[f"lower_{channel}"] = np.full(rows, 0.8)
        data[f"upper_{channel}"] = np.full(rows, 1.2)
        data[f"target_observed_{channel}"] = np.ones(rows, dtype=np.uint8)
        data[f"age_steps_{channel}"] = np.zeros(rows)
    return pd.DataFrame(data)


def test_semantic_evidence_is_zero_innovation_inside_interval() -> None:
    config = _semantic_config()
    frame = compute_semantic_evidence(
        _evidence_frame(),
        config["channels"],
        {channel: 1.0 for channel in config["channels"]},
        config,
        age_cap_steps=96,
    )
    assert np.allclose(frame["standardized_innovation_temperature"], 0.0)
    assert np.allclose(frame["interval_breach_ratio_temperature"], 0.0)
    assert (frame["semantic_risk"] > 0.0).all()  # predictive uncertainty remains evidence
    assert (frame["semantic_risk"] < config["nominal_policy"]["innovation_threshold"]).all()


def test_unobserved_stale_channel_increases_risk_without_fake_innovation() -> None:
    config = _semantic_config()
    source = _evidence_frame(rows=2)
    source.loc[1, "target_observed_temperature"] = 0
    source.loc[1, "actual_temperature"] = np.nan
    source.loc[1, "age_steps_temperature"] = 96
    frame = compute_semantic_evidence(
        source,
        config["channels"],
        {channel: 1.0 for channel in config["channels"]},
        config,
        age_cap_steps=96,
    )
    assert np.isnan(frame.loc[1, "standardized_innovation_temperature"])
    assert (
        frame.loc[1, "semantic_channel_risk_temperature"]
        > frame.loc[0, "semantic_channel_risk_temperature"]
    )


def test_causal_critical_evidence_forces_alert() -> None:
    config = _semantic_config()
    source = _evidence_frame(rows=2)
    source.loc[1, "label_available_event"] = 1
    evidence = compute_semantic_evidence(
        source,
        config["channels"],
        {channel: 1.0 for channel in config["channels"]},
        config,
        age_cap_steps=96,
    )
    decisions = apply_nominal_policy(evidence, config)
    assert decisions.loc[1, "node_action"] == "alert"
    assert decisions.loc[1, "decision_reason"] == "critical_event"
    assert decisions.loc[1, "semantic_risk"] == 1.0


def test_bounded_silence_never_exceeds_advertised_horizon() -> None:
    config = _semantic_config()
    rows = 12
    frame = pd.DataFrame(
        {
            "window_key": ["w0"] * rows,
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "semantic_risk": np.zeros(rows),
            "causal_critical_evidence": np.zeros(rows, dtype=np.uint8),
            "maximum_interval_breach_ratio": np.zeros(rows),
        }
    )
    decisions = apply_nominal_policy(frame, config)
    transmitted_indices = np.flatnonzero(decisions["node_action"].to_numpy() != "silent")
    assert decisions.loc[0, "node_action"] == "heartbeat"
    assert np.diff(transmitted_indices).max() <= config["bounded_validity"]["maximum_horizon_steps"]
    assert (
        decisions["contract_age_steps"].max() < config["bounded_validity"]["maximum_horizon_steps"]
    )


def test_validity_horizon_contracts_monotonically_with_risk() -> None:
    config = _semantic_config()
    horizons = [validity_horizon(risk, config) for risk in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert horizons == sorted(horizons, reverse=True)
    assert horizons[0] == config["bounded_validity"]["maximum_horizon_steps"]
    assert horizons[-1] == config["bounded_validity"]["minimum_horizon_steps"]


def test_task_i_integrity_rejects_tampering(tmp_path: Path) -> None:
    prediction_root = tmp_path / "results" / "task_i"
    artifact_root = tmp_path / "artifacts" / "task_i"
    prediction_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    artifact = prediction_root / "predictive_metrics.csv"
    artifact.write_text("locked\n", encoding="utf-8")
    summary = {
        "task": "I",
        "predictor_version": "1.1",
        "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}],
    }
    (prediction_root / "task_i_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _, verified = verify_task_i_inputs(prediction_root, artifact_root)
    assert verified == 1
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(SemanticEvidenceError, match="changed or is missing"):
        verify_task_i_inputs(prediction_root, artifact_root)


def test_task_j_config_locks_causal_and_apriori_contract() -> None:
    config = _semantic_config()
    assert config["causal_contract"]["uses_current_observation_at_sensor"]
    assert not config["causal_contract"]["uses_future_measurements"]
    assert not config["causal_contract"]["thresholds_tuned_on_test_or_external_data"]
    assert config["nominal_policy"]["thresholds_are_apriori_not_fitted"]
    assert config["evidence"]["critical_evidence_override"]
