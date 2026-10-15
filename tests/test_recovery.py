import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.protocol.recovery import (
    RecoveryError,
    decide_recovery_actions,
    make_recovery_observations,
    replay_recovery,
    run_bounded_recovery,
    verify_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict]:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    return config["recovery"], config["network"]


def _posterior(rows: int = 6) -> pd.DataFrame:
    timestamp = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "window_key": ["w0"] * rows,
            "repetition": [0] * rows,
            "packet_arrived": [1, 0, 0, 0, 0, 1][:rows],
            "receiver_state": [
                "updated",
                "trusted_silence",
                "ambiguous",
                "failure_suspected",
                "stale_unsafe",
                "updated",
            ][:rows],
            "connectivity_belief": [0.99, 0.98, 0.6, 0.2, 0.05, 0.99][:rows],
            "environmental_uncertainty": [0.1, 0.2, 0.5, 0.8, 0.95, 0.1][:rows],
            "receiver_contract_age_steps": [0, 1, 2, 3, 4, 0][:rows],
            "receiver_overdue_steps": [0, 0, 1, 2, 3, 0][:rows],
            "receiver_contract_valid": [1, 1, 0, 0, 0, 1][:rows],
            "receiver_contact_due": [0, 0, 1, 1, 1, 0][:rows],
            "cause_decision": [
                "observed_update",
                "valid_suppression",
                "unknown",
                "persistent_outage",
                "persistent_outage",
                "observed_update",
            ][:rows],
            "valid_suppression_admissible": [0, 1, 0, 0, 0, 0][:rows],
            "evaluation_only_true_cause": [
                "observed_update",
                "valid_suppression",
                "isolated_loss",
                "persistent_outage",
                "persistent_outage",
                "observed_update",
            ][:rows],
            "evaluation_only_unsafe_silence": [0, 0, 1, 1, 1, 0][:rows],
            "evaluation_only_critical_event": [0, 0, 1, 0, 1, 0][:rows],
        }
    )


def _transport(rows: int = 6) -> pd.DataFrame:
    posterior = _posterior(rows)
    return pd.DataFrame(
        {
            "timestamp": posterior["timestamp"],
            "window_key": posterior["window_key"],
            "repetition": posterior["repetition"],
            "realization_seed": [123] * rows,
            "packet_delivered_timely": posterior["packet_arrived"],
            "acknowledged": posterior["packet_arrived"],
            "semantic_payload_bytes": [24, 0, 40, 0, 40, 24][:rows],
            "silence_contract_id": list(range(1, rows + 1)),
            "validity_horizon_steps": [4, 4, 0, 4, 0, 4][:rows],
            "network_bytes": [50] * rows,
            "incremental_communication_energy_j": [1.0] * rows,
            "_oracle_link_state": ["good"] * rows,
            "causal_critical_evidence": posterior["evaluation_only_critical_event"],
        }
    )


def test_recovery_observations_exclude_truth_and_oracle() -> None:
    observations = make_recovery_observations(_posterior())
    assert not any(column.startswith("evaluation_only_") for column in observations)
    assert not any(column.startswith("_oracle_") for column in observations)


def test_deployable_policy_rejects_oracle_evidence() -> None:
    observations = make_recovery_observations(_posterior())
    observations["_oracle_link_state"] = "good"
    with pytest.raises(RecoveryError, match="Forbidden recovery evidence"):
        decide_recovery_actions(observations, "safe_silence_bounded", _configs()[0])


def test_changing_truth_cannot_change_deployable_actions() -> None:
    first = _posterior()
    second = first.copy()
    second["evaluation_only_true_cause"] = "valid_suppression"
    second["evaluation_only_critical_event"] = 1
    config, _ = _configs()
    a = decide_recovery_actions(make_recovery_observations(first), "safe_silence_bounded", config)
    b = decide_recovery_actions(make_recovery_observations(second), "safe_silence_bounded", config)
    pd.testing.assert_frame_equal(
        a[["recovery_triggered", "recovery_trigger_reason", "recovery_attempt_budget"]],
        b[["recovery_triggered", "recovery_trigger_reason", "recovery_attempt_budget"]],
    )


def test_valid_suppression_continues_without_unnecessary_probe() -> None:
    actions = decide_recovery_actions(
        make_recovery_observations(_posterior()), "safe_silence_bounded", _configs()[0]
    )
    assert actions.loc[1, "recovery_triggered"] == 0


def test_ambiguity_triggers_bounded_recovery() -> None:
    config, _ = _configs()
    actions = decide_recovery_actions(
        make_recovery_observations(_posterior()), "safe_silence_bounded", config
    )
    assert actions.loc[2, "recovery_triggered"] == 1
    assert actions.loc[3, "recovery_triggered"] == 0
    assert actions.loc[2, "recovery_attempt_planned"] == 1
    assert actions.loc[3, "recovery_attempt_planned"] == 1
    assert actions.loc[2, "recovery_episode_id"] == actions.loc[3, "recovery_episode_id"]
    assert actions["recovery_attempt_budget"].max() == 1


def test_recovery_succeeds_on_certain_round_trip_link() -> None:
    recovery_config, network_config = _configs()
    scenario = deepcopy(network_config["scenarios"]["F0_stable"])
    scenario["packet_success"] = {"good": 1.0, "degraded": 1.0, "outage": 1.0}
    replay = replay_recovery(
        _posterior(),
        _transport(),
        "safe_silence_bounded",
        recovery_config,
        network_config,
        scenario,
        15,
    )
    assert replay.loc[2, "recovery_succeeded"] == 1
    assert replay.loc[2, "critical_retransmission_succeeded"] == 1
    assert replay.loc[2, "recovery_response_kind"] == "critical_retransmission"
    assert replay.loc[2, "recovery_response_wire_bytes"] == 60
    assert replay.loc[2, "recovery_network_bytes"] == 92
    assert replay["recovery_attempts"].max() <= 1
    assert replay.loc[3, "recovery_attempts"] == 0


def test_cross_step_reconciliation_recovers_buffered_critical_alert() -> None:
    recovery_config, network_config = _configs()
    scenario = deepcopy(network_config["scenarios"]["F8_hard_partition"])
    scenario["packet_success"] = {"good": 1.0, "degraded": 1.0, "outage": 0.0}
    transport = _transport()
    transport.loc[[2, 4, 5], "_oracle_link_state"] = "outage"
    transport.loc[3, "_oracle_link_state"] = "good"
    replay = replay_recovery(
        _posterior(),
        transport,
        "safe_silence_bounded",
        recovery_config,
        network_config,
        scenario,
        15,
    )
    assert replay.loc[2, "recovery_succeeded"] == 0
    assert replay.loc[3, "recovery_succeeded"] == 1
    assert replay.loc[2, "evaluation_only_critical_recovered"] == 1
    assert replay.loc[2, "evaluation_only_critical_recovered_timely"] == 0
    assert replay.loc[2, "critical_event_recovery_delay_steps"] == 1
    assert replay.loc[3, "recovery_response_wire_bytes"] == 60
    assert replay.loc[2, "critical_event_recovery_response_wire_bytes"] == 60
    assert replay.loc[2, "critical_event_recovery_response_id"] == replay.loc[
        3, "recovery_response_id"
    ]


def test_one_successful_response_cannot_recover_two_cached_packets() -> None:
    recovery_config, network_config = _configs()
    scenario = deepcopy(network_config["scenarios"]["F8_hard_partition"])
    scenario["packet_success"] = {"good": 1.0, "degraded": 1.0, "outage": 0.0}
    transport = _transport()
    transport["causal_critical_evidence"] = [0, 0, 1, 1, 0, 0]
    transport.loc[[2, 3], "semantic_payload_bytes"] = 40
    transport.loc[[2, 3], "validity_horizon_steps"] = 0
    transport.loc[[2, 3], "acknowledged"] = 0
    transport.loc[[2, 4, 5], "_oracle_link_state"] = "outage"
    transport.loc[3, "_oracle_link_state"] = "good"

    replay = replay_recovery(
        _posterior(),
        transport,
        "safe_silence_bounded",
        recovery_config,
        network_config,
        scenario,
        15,
    )

    assert replay.loc[2, "evaluation_only_critical_recovered"] == 1
    assert replay.loc[3, "evaluation_only_critical_recovered"] == 0
    assert replay["critical_retransmission_succeeded"].sum() == 1
    linked = replay.loc[
        replay["critical_event_recovery_response_id"].astype(str) != "",
        "critical_event_recovery_response_id",
    ]
    assert linked.is_unique


def test_recovery_uses_exact_original_packet_size() -> None:
    recovery_config, network_config = _configs()
    scenario = deepcopy(network_config["scenarios"]["F0_stable"])
    scenario["packet_success"] = {"good": 1.0, "degraded": 1.0, "outage": 1.0}
    transport = _transport()
    transport.loc[2, "semantic_payload_bytes"] = 73

    replay = replay_recovery(
        _posterior(),
        transport,
        "safe_silence_bounded",
        recovery_config,
        network_config,
        scenario,
        15,
    )

    expected_response_wire = network_config["packet_header_bytes"] + 73
    expected_request_wire = (
        network_config["packet_header_bytes"]
        + recovery_config["packet_bytes"]["probe_payload"]
    )
    assert replay.loc[2, "recovery_response_wire_bytes"] == expected_response_wire
    assert replay.loc[2, "recovery_network_bytes"] == (
        expected_request_wire + expected_response_wire
    )


def test_expired_cached_packet_cannot_receive_recovery_credit() -> None:
    recovery_config, network_config = _configs()
    scenario = deepcopy(network_config["scenarios"]["F8_hard_partition"])
    scenario["packet_success"] = {"good": 1.0, "degraded": 1.0, "outage": 0.0}
    posterior = _posterior()
    posterior.loc[0, "packet_arrived"] = 0
    transport = _transport()
    transport["causal_critical_evidence"] = [1, 0, 0, 0, 0, 0]
    transport.loc[0, "packet_delivered_timely"] = 0
    transport.loc[0, "acknowledged"] = 0
    transport.loc[0, "semantic_payload_bytes"] = 40
    transport.loc[0, "validity_horizon_steps"] = 0
    transport["_oracle_link_state"] = ["outage"] * 5 + ["good"]

    replay = replay_recovery(
        posterior,
        transport,
        "safe_silence_bounded",
        recovery_config,
        network_config,
        scenario,
        15,
    )

    assert replay.loc[0, "evaluation_only_critical_recovered"] == 0
    assert replay["recovery_cache_expired_items"].sum() == 1
    assert replay.loc[5, "recovery_response_kind"] == "liveness"


def test_no_recovery_adds_no_cost() -> None:
    recovery_config, network_config = _configs()
    replay = replay_recovery(
        _posterior(),
        _transport(),
        "no_recovery",
        recovery_config,
        network_config,
        network_config["scenarios"]["F0_stable"],
        15,
    )
    assert replay["recovery_network_bytes"].sum() == 0
    assert replay["recovery_energy_j"].sum() == 0


def test_task_n_config_locks_oracle_and_budget_boundaries() -> None:
    config, _ = _configs()
    assert not config["oracle_fields_are_policy_inputs"]
    assert not config["event_truth_is_policy_input"]
    assert config["named_cause_is_never_sufficient_for_action"]
    assert config["reconciliation"]["one_attempt_per_grid_step"]
    assert config["reconciliation"]["critical_alert_retention_steps"] == 4
    assert config["reconciliation"]["maximum_critical_packets_per_response"] == 1
    assert config["reconciliation"]["critical_response_uses_original_wire_bytes"]
    assert config["reconciliation"]["successful_response_consumes_cached_packet"]
    assert config["reconciliation"]["exact_event_response_provenance"]
    assert (
        config["policies"]["safe_silence_bounded"]["maximum_attempts_per_trigger"]
        < config["policies"]["unbounded_retry_ceiling"]["maximum_attempts_per_trigger"]
    )


def test_upstream_integrity_rejects_tampering(tmp_path: Path) -> None:
    task_m = tmp_path / "task_m"
    task_k = tmp_path / "task_k"
    task_m.mkdir()
    task_k.mkdir()
    m_artifact = task_m / "cause_metrics.csv"
    k_artifact = task_k / "network_transport_metrics.csv"
    m_artifact.write_text("locked\n", encoding="utf-8")
    k_artifact.write_text("locked\n", encoding="utf-8")
    (task_m / "task_m_summary.json").write_text(
        json.dumps(
            {
                "task": "M",
                "cause_version": "1.1",
                "artifacts": [{"path": str(m_artifact), "sha256": sha256_file(m_artifact)}],
            }
        ),
        encoding="utf-8",
    )
    (task_k / "task_k_summary.json").write_text(
        json.dumps(
            {
                "task": "K",
                "network_version": "1.0",
                "artifacts": [{"path": str(k_artifact), "sha256": sha256_file(k_artifact)}],
            }
        ),
        encoding="utf-8",
    )
    verify_inputs(task_m, task_k)
    m_artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RecoveryError, match="changed or is missing"):
        verify_inputs(task_m, task_k)


def test_synthetic_end_to_end_recovery_run(tmp_path: Path) -> None:
    task_m = tmp_path / "task_m"
    task_k = tmp_path / "task_k"
    m_trace = task_m / "posteriors" / "primary_15min" / "S01" / "F0_stable.parquet"
    k_trace = task_k / "transport" / "primary_15min" / "S01" / "F0_stable.parquet"
    m_trace.parent.mkdir(parents=True)
    k_trace.parent.mkdir(parents=True)
    _posterior().to_parquet(m_trace, index=False)
    _transport().to_parquet(k_trace, index=False)
    record = {
        "protocol": "primary_15min",
        "site": "S01",
        "model_role": "external_test",
        "scenario": "F0_stable",
    }
    m_record = {**record, "path": str(m_trace), "sha256": sha256_file(m_trace)}
    k_record = {**record, "path": str(k_trace), "sha256": sha256_file(k_trace)}
    (task_m / "task_m_summary.json").write_text(
        json.dumps({"task": "M", "cause_version": "1.1", "artifacts": [m_record]}),
        encoding="utf-8",
    )
    (task_k / "task_k_summary.json").write_text(
        json.dumps({"task": "K", "network_version": "1.0", "artifacts": [k_record]}),
        encoding="utf-8",
    )
    recovery_config, network_config = _configs()
    summary = run_bounded_recovery(
        task_m, task_k, tmp_path / "task_n", recovery_config, network_config
    )
    assert summary["task"] == "N"
    assert summary["metrics_rows"] == 6
    assert not summary["deployable_policy_uses_oracle_fields"]
    assert summary["row_level_trace_policy"] == "safe_silence_bounded"
    assert summary["artifacts"][0]["rows"] == len(_posterior())
    assert (tmp_path / "task_n" / "recovery_metrics.csv").exists()
