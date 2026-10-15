import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.simulation.network import (
    NetworkSimulationError,
    _derived_seed,
    apply_transport,
    generate_oracle_realization,
    simulate_network,
    verify_task_j_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _network_config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")["network"]


def _evidence(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC"),
            "window_key": ["w0"] * rows,
            "model_role": ["external_test"] * rows,
            "node_action": ["alert"] * rows,
            "decision_reason": ["critical_event"] * rows,
            "semantic_payload_bytes": [40] * rows,
            "validity_horizon_steps": [0] * rows,
            "silence_contract_id": np.arange(rows),
            "contract_age_steps": [0] * rows,
            "causal_critical_evidence": [1] * rows,
            "semantic_risk": [1.0] * rows,
        }
    )


def _perfect_scenario(ack: float = 1.0) -> dict:
    return {
        "start_state": "good",
        "event_coupled": False,
        "packet_success": {state: 1.0 for state in ("good", "degraded", "outage")},
        "ack_success": {state: ack for state in ("good", "degraded", "outage")},
        "dwell_mean_steps": {state: 10.0 for state in ("good", "degraded", "outage")},
        "dwell_shape": {state: 2.0 for state in ("good", "degraded", "outage")},
        "transition": {
            "good": {"degraded": 1.0},
            "degraded": {"outage": 1.0},
            "outage": {"good": 1.0},
        },
    }


def test_primary_realization_is_independent_of_critical_truth() -> None:
    scenario = _network_config()["scenarios"]["F2_burst"]
    first = generate_oracle_realization(
        100, scenario, np.random.default_rng(42), 3, np.zeros(100, dtype=bool)
    )
    second = generate_oracle_realization(
        100, scenario, np.random.default_rng(42), 3, np.ones(100, dtype=bool)
    )
    pd.testing.assert_frame_equal(first, second)


def test_event_coupling_is_confined_to_declared_stress() -> None:
    scenario = dict(_network_config()["scenarios"]["F13_event_coincident_outage"])
    scenario["critical_outage_probability"] = 1.0
    critical = np.array([False, True, False, True])
    oracle = generate_oracle_realization(4, scenario, np.random.default_rng(1), 3, critical)
    assert (oracle.loc[critical, "_oracle_link_state"] == "outage").all()


def test_perfect_transport_delivers_and_acknowledges_once() -> None:
    config = _network_config()
    evidence = _evidence()
    oracle = generate_oracle_realization(
        len(evidence), _perfect_scenario(), np.random.default_rng(2), config["maximum_attempts"]
    )
    result = apply_transport(evidence, oracle, _perfect_scenario(), config, 15)
    assert (result["packet_delivered_timely"] == 1).all()
    assert (result["acknowledged"] == 1).all()
    assert (result["transmission_attempts"] == 1).all()
    assert (result["duplicate_deliveries"] == 0).all()


def test_ack_loss_generates_bounded_idempotent_duplicates() -> None:
    config = _network_config()
    evidence = _evidence(rows=2)
    scenario = _perfect_scenario(ack=0.0)
    oracle = generate_oracle_realization(
        len(evidence), scenario, np.random.default_rng(3), config["maximum_attempts"]
    )
    result = apply_transport(evidence, oracle, scenario, config, 15)
    assert (result["packet_delivered"] == 1).all()
    assert (result["acknowledged"] == 0).all()
    assert (result["transmission_attempts"] == config["maximum_attempts"]).all()
    assert (result["duplicate_deliveries"] == config["maximum_attempts"] - 1).all()


def test_silence_never_consumes_network_resources() -> None:
    config = _network_config()
    evidence = _evidence(rows=1)
    evidence.loc[0, "node_action"] = "silent"
    evidence.loc[0, "semantic_payload_bytes"] = 0
    oracle = generate_oracle_realization(
        1, _perfect_scenario(), np.random.default_rng(4), config["maximum_attempts"]
    )
    result = apply_transport(evidence, oracle, _perfect_scenario(), config, 15)
    assert result.loc[0, "transmission_attempts"] == 0
    assert result.loc[0, "network_bytes"] == 0
    assert result.loc[0, "incremental_communication_energy_j"] == 0


def test_seed_derivation_is_stable_and_context_specific() -> None:
    seed = _derived_seed(3407, "primary", "site", "F2", 0)
    assert seed == _derived_seed(3407, "primary", "site", "F2", 0)
    assert seed != _derived_seed(3407, "primary", "site", "F2", 1)
    assert 0 <= seed < 2**63 - 1


def test_task_j_integrity_rejects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "task_j"
    root.mkdir()
    artifact = root / "semantic_decision_metrics.csv"
    artifact.write_text("locked\n", encoding="utf-8")
    summary = {
        "task": "J",
        "semantic_version": "1.0",
        "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}],
    }
    (root / "task_j_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _, verified = verify_task_j_inputs(root)
    assert verified == 1
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(NetworkSimulationError, match="changed or is missing"):
        verify_task_j_inputs(root)


def test_network_config_locks_oracle_and_primary_independence() -> None:
    config = _network_config()
    assert not config["outage"]["primary_generation_uses_event_labels"]
    assert config["hidden_oracle_state_is_never_receiver_input"]
    assert config["store_counterfactual_randomness"]
    assert config["scenarios"]["F13_event_coincident_outage"]["event_coupled"]
    for scenario in config["scenarios"].values():
        if scenario["role"] in {"primary", "reference"}:
            assert not scenario["event_coupled"]


def test_synthetic_end_to_end_network_run(tmp_path: Path) -> None:
    input_root = tmp_path / "task_j"
    evidence_path = input_root / "evidence" / "primary_15min" / "S01.parquet"
    evidence_path.parent.mkdir(parents=True)
    evidence = _evidence(rows=4)
    evidence.to_parquet(evidence_path, index=False)
    artifact = {
        "protocol": "primary_15min",
        "site": "S01",
        "path": str(evidence_path),
        "sha256": sha256_file(evidence_path),
    }
    summary = {
        "task": "J",
        "semantic_version": "1.0",
        "protocols": {"primary_15min": {"grid_minutes": 15, "sites": ["S01"], "rows": 4}},
        "artifacts": [artifact],
    }
    (input_root / "task_j_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    config = deepcopy(_network_config())
    config["repetitions"] = 1
    config["scenarios"] = {"F0_stable": config["scenarios"]["F0_stable"]}
    result = simulate_network(input_root, tmp_path / "task_k", config)
    assert result["task"] == "K"
    assert result["metrics_rows"] == 1
    assert not result["primary_outages_use_event_labels"]
    assert (tmp_path / "task_k" / "network_transport_metrics.csv").exists()
