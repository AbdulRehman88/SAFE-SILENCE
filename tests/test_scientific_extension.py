"""Meaningful invariants for the new scientific extension."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.evaluation.extension import load_configs, trust_variant, comparisons
from safe_silence.protocol.receiver import interpret_receiver as original_receiver
from safe_silence.protocol.receiver_ablation import interpret_receiver
from safe_silence.protocol.recovery import replay_recovery as original_replay
from safe_silence.protocol.recovery_budgeted import replay_recovery, RecoveryError
from safe_silence.protocol.scheduler import PROPOSED_POLICY, REFERENCE_POLICY
from safe_silence.simulation.network import _derived_seed

ROOT = Path(__file__).resolve().parents[1]


def configs():
    return load_configs(ROOT)


def source(rows=8):
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC"),
        "window_key": ["w0"] * rows, "repetition": [0] * rows,
        "realization_seed": [123] * rows,
        "node_action": ["heartbeat", *(["silent"] * (rows - 1))],
        "semantic_risk": [0.1] * rows, "validity_horizon_steps": [4] * rows,
        "silence_contract_id": list(range(rows)),
        "packet_delivered_timely": [1, *([0] * (rows - 1))],
        "acknowledged": [1, *([0] * (rows - 1))],
        "causal_critical_evidence": [0, 0, 1, *([0] * (rows - 3))],
        "semantic_payload_bytes": [8, 0, 40, *([0] * (rows - 3))],
        "network_bytes": [28, *([0] * (rows - 1))],
        "incremental_communication_energy_j": [1.0, *([0.0] * (rows - 1))],
        "_oracle_link_state": ["good"] * rows,
    })


def budget_config(byte_rate, energy_rate=1e9):
    c = deepcopy(configs()["recovery"])
    c["extension_common_budget"] = {"bytes_per_opportunity": byte_rate, "energy_j_per_opportunity": energy_rate, "maximum_response_wire_bytes": 60}
    return c


def test_full_receiver_reproduces_original():
    c = configs()["receiver"]
    pd.testing.assert_frame_equal(original_receiver(source(), c), interpret_receiver(source(), c))


@pytest.mark.parametrize("variant", configs()["extension"]["trust_variants"])
def test_truth_cannot_change_receiver_or_cause_decisions(variant):
    a = source()
    b = a.copy()
    b["causal_critical_evidence"] = 1 - b.causal_critical_evidence
    b["_oracle_link_state"] = "outage"
    fields = ["receiver_state", "connectivity_belief", "environmental_uncertainty", "cause_decision"]
    pd.testing.assert_frame_equal(trust_variant(a, configs(), variant, 15)[fields], trust_variant(b, configs(), variant, 15)[fields])


@pytest.mark.parametrize("variant", configs()["extension"]["trust_variants"])
def test_future_arrivals_cannot_change_past_decisions(variant):
    a = source()
    b = a.copy()
    b.loc[5:, "packet_delivered_timely"] = 1
    fields = ["receiver_state", "connectivity_belief", "environmental_uncertainty", "cause_decision"]
    pd.testing.assert_frame_equal(trust_variant(a, configs(), variant, 15)[fields].iloc[:5], trust_variant(b, configs(), variant, 15)[fields].iloc[:5])


def test_ablation_truth_is_fixed_to_full_receiver():
    a = source()
    full = trust_variant(a, configs(), "full", 15)
    no_expiry = trust_variant(a, configs(), "no_expiration", 15, full)
    assert no_expiry.receiver_contract_valid.iloc[-1] == 1
    assert full.receiver_contract_valid.iloc[-1] == 0
    assert full.loc[2, "evaluation_only_true_cause"] == "contract_divergence"
    pd.testing.assert_series_equal(full.evaluation_only_unsafe_silence, no_expiry.evaluation_only_unsafe_silence)
    pd.testing.assert_series_equal(full.evaluation_only_true_cause, no_expiry.evaluation_only_true_cause)


def test_timeout_uses_elapsed_grid_time_not_candidate_count():
    a = source(4)
    a.loc[1:, "timestamp"] = pd.date_range("2025-01-01T01:15:00Z", periods=3, freq="15min")
    result = trust_variant(a, configs(), "fixed_timeout", 15)
    assert result.receiver_state.iloc[1] == "stale_unsafe"


def test_no_unknown_forces_known_interpretation():
    result = trust_variant(source(), configs(), "no_unknown", 15)
    assert not result.loc[result.packet_arrived.eq(0), "cause_decision"].eq("unknown").any()


def test_gate_ablations_remove_connectivity_restriction():
    c = configs()
    c["receiver"]["connectivity_filter"]["reachable_to_reachable"] = 0.5
    c["receiver"]["connectivity_filter"]["unreachable_to_reachable"] = 0.5
    assert trust_variant(source(), c, "full", 15).receiver_state.iloc[1] == "ambiguous"
    assert trust_variant(source(), c, "no_connectivity", 15).receiver_state.iloc[1] == "trusted_silence"
    assert trust_variant(source(), c, "collapsed_belief", 15).receiver_state.iloc[1] == "trusted_silence"


@pytest.mark.parametrize("policy", ["safe_silence_bounded", "fixed_probe", "periodic_retry", "no_recovery"])
def test_zero_common_budget_permits_no_transmissions(policy):
    c = configs()
    a = source()
    full = trust_variant(a, c, "full", 15)
    replay = replay_recovery(full, a, policy, budget_config(0), c["network"], c["network"]["scenarios"]["F2_burst"], 15)
    assert replay.recovery_attempts.sum() == 0
    assert replay.recovery_network_bytes.sum() == 0
    assert replay.recovery_energy_j.sum() == 0


def test_energy_cap_is_enforced_independently():
    c = configs()
    a = source()
    replay = replay_recovery(trust_variant(a, c, "full", 15), a, "safe_silence_bounded", budget_config(1e9, 0), c["network"], c["network"]["scenarios"]["F2_burst"], 15)
    assert replay.recovery_attempts.sum() == 0


def test_affordability_does_not_peek_at_request_outcome():
    c = configs()
    a = source()
    scenario = deepcopy(c["network"]["scenarios"]["F2_burst"])
    scenario["packet_success"] = {"good": 0.0, "degraded": 0.0, "outage": 0.0}
    # Total budget 32 bytes can cover a request but not its worst-case response.
    replay = replay_recovery(trust_variant(a, c, "full", 15), a, "safe_silence_bounded", budget_config(32 / len(a)), c["network"], scenario, 15)
    assert replay.recovery_attempts.sum() == 0


def test_cancelled_request_does_not_erase_false_trust():
    c = configs()
    a = source()
    full = trust_variant(a, c, "full", 15)
    full.loc[2, "receiver_state"] = "trusted_silence"
    full.loc[2, "cause_decision"] = "valid_suppression"
    replay = replay_recovery(full, a, "safe_silence_bounded", budget_config(0), c["network"], c["network"]["scenarios"]["F2_burst"], 15)
    assert replay.loc[2, "evaluation_only_post_recovery_false_trust"] == 1
    assert replay.loc[2, "post_recovery_packet_delivered_within_retention"] == 0


def test_large_budget_preserves_original_delivery_randomness():
    c = configs()
    a = source()
    full = trust_variant(a, c, "full", 15)
    scenario = c["network"]["scenarios"]["F2_burst"]
    old = original_replay(full, a, "safe_silence_bounded", c["recovery"], c["network"], scenario, 15)
    new = replay_recovery(full, a, "safe_silence_bounded", budget_config(1e9), c["network"], scenario, 15)
    for field in ["recovery_attempts", "recovery_network_bytes", "recovery_response_id", "post_recovery_packet_delivered_within_retention"]:
        pd.testing.assert_series_equal(old[field], new[field])


def test_oversized_response_fails_instead_of_breaking_cap():
    c = configs()
    a = source()
    bounded = budget_config(1e9)
    bounded["extension_common_budget"]["maximum_response_wire_bytes"] = 40
    with pytest.raises(RecoveryError, match="response exceeds"):
        replay_recovery(trust_variant(a, c, "full", 15), a, "safe_silence_bounded", bounded, c["network"], c["network"]["scenarios"]["F2_burst"], 15)


def synthetic_metrics():
    c = configs()
    systems, trusts = [], []
    for protocol in c["extension"]["protocols"]:
        for scenario in c["extension"]["scenarios"]:
            for repetition in range(30):
                base = {"protocol": protocol, "scenario": scenario, "repetition": repetition, "realization_seed": _derived_seed(293211, protocol, "CE07SHSM", "external_test", scenario, repetition)}
                for profile in c["extension"]["profiles"]:
                    for scheduler in [PROPOSED_POLICY, REFERENCE_POLICY]:
                        policies = c["extension"]["recovery_policies"] if profile == "primary" and scheduler == PROPOSED_POLICY else ["safe_silence_bounded"]
                        for recovery in policies:
                            systems.append({**base, "loss_profile": profile, "scheduler_policy": scheduler, "recovery_policy": recovery, "post_recovery_critical_delivery_recall": .9, "realized_post_recovery_tail_proxy_reduction": 100.0})
                for variant in c["extension"]["trust_variants"]:
                    trusts.append({**base, "scope": "source_protocol_grid", "variant": variant, "critical_false_trust_pre_action": 0.0})
    return pd.DataFrame(systems), pd.DataFrame(trusts)


def test_global_family_keeps_all_null_results():
    system, trust = synthetic_metrics()
    result = comparisons(system, trust, configs())
    assert len(result) == 210
    assert result.wilcoxon_p_value.eq(1).all()
    assert not result.confirmatory_support.any()


def test_incomplete_pairing_cannot_silently_reduce_family():
    system, trust = synthetic_metrics()
    with pytest.raises(RuntimeError, match="Incomplete declared comparison"):
        comparisons(system.iloc[1:], trust, configs())


def test_common_caps_reset_per_repetition_and_cover_every_window():
    c = configs()
    a = source()
    b = source()
    b["repetition"] = 1
    b["realization_seed"] = 456
    combined = pd.concat([a, b], ignore_index=True)
    full = trust_variant(combined, c, "full", 15)
    bounded = budget_config(92 / len(a))
    scenario = deepcopy(c["network"]["scenarios"]["F2_burst"])
    scenario["packet_success"] = {"good": 1.0, "degraded": 1.0, "outage": 1.0}
    replay = replay_recovery(full, combined, "safe_silence_bounded", bounded, c["network"], scenario, 15)
    sums = replay.groupby("repetition").recovery_network_bytes.sum()
    assert (sums <= 92).all()
    assert (sums > 0).all()


def test_stratified_aggregate_preserves_pairing():
    from safe_silence.evaluation.extension import aggregate_estimates
    system, trust = synthetic_metrics()
    system["post_recovery_critical_delivery_recall"] += system.repetition / 10000.0
    c = configs()
    primary = comparisons(system, trust, c)
    aggregate = aggregate_estimates(system, trust, primary, c)
    assert len(aggregate) == 30
    assert np.allclose(aggregate.mean_paired_difference, 0)
    assert np.allclose(aggregate.difference_ci_lower, 0)
    assert np.allclose(aggregate.difference_ci_upper, 0)
    recall = aggregate[aggregate.metric.eq("post_recovery_critical_delivery_recall")]
    assert (recall.proposed_ci_upper > recall.proposed_ci_lower).all()


def test_cause_assertion_is_not_misreported_as_receiver_false_trust():
    from safe_silence.evaluation.extension import trust_metrics
    c = configs()
    a = source()
    frame = trust_variant(a, c, "full", 15)
    frame.loc[2, "receiver_state"] = "ambiguous"
    frame.loc[2, "cause_decision"] = "valid_suppression"
    replay = replay_recovery(frame, a, "safe_silence_bounded", budget_config(0), c["network"], c["network"]["scenarios"]["F2_burst"], 15)
    metric = trust_metrics(frame, replay, "source_protocol_grid", "no_unknown", "primary_15min", "CE07SHSM", "F2_burst")[0]
    assert metric["critical_false_trust_pre_action"] == 0.0
    assert metric["critical_unsafe_valid_suppression_assertion_rate"] == 1.0
