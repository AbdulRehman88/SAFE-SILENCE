"""YAML composition and invariant validation.

The YAML files are authoritative. This module only composes them and enforces
cross-component invariants that would otherwise create scientifically invalid runs.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration violates the experiment contract."""


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {path}")
    return value


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the pipeline YAML and merge includes in declaration order."""
    pipeline_path = Path(path).expanduser().resolve()
    root = _read_yaml(pipeline_path)
    includes = root.pop("includes", [])
    if not isinstance(includes, list) or not all(isinstance(x, str) for x in includes):
        raise ConfigurationError("pipeline.includes must be a list of YAML filenames")

    merged: dict[str, Any] = {}
    for relative in includes:
        include_path = (pipeline_path.parent / relative).resolve()
        if pipeline_path.parent not in include_path.parents:
            raise ConfigurationError(f"Include escapes config directory: {relative}")
        merged = _deep_merge(merged, _read_yaml(include_path))
    merged = _deep_merge(merged, root)
    validate_config(merged)
    return merged


def _require(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise ConfigurationError(f"Missing required configuration: {dotted_key}")
        cursor = cursor[part]
    return cursor


def validate_config(config: dict[str, Any]) -> None:
    """Enforce the invariants that protect the paper's central claims."""
    required = [
        "schema_version",
        "project.seed",
        "data.primary_site",
        "data.audit.availability_windows.selection_uses_measurement_values",
        "data.audit.availability_windows.network_outages_injected_after_selection",
        "preprocessing.measurement_values_used_for_availability_selection",
        "preprocessing.post_qc_gate_uses_qartod_finiteness_and_physical_validity",
        "preprocessing.fit_scalers_during_preprocessing",
        "preprocessing.split.uses_measurement_values",
        "events.split.external_sites_are_test_only",
        "events.split.reuse_task_g_splits",
        "events.split.calibration_split",
        "events.calibration.site",
        "events.calibration.observed_measurements_only",
        "events.reporting.pool_temporal_protocols",
        "events.reporting.score_zero_support_event_classes",
        "network.outage.event_correlated_outages_primary",
        "network.outage.primary_generation_uses_event_labels",
        "network.hidden_oracle_state_is_never_receiver_input",
        "protocol.bounded_silence.enabled",
        "protocol.dual_belief.environmental_uncertainty",
        "protocol.dual_belief.connectivity_uncertainty",
        "protocol.dual_belief.unknown_cause_class",
        "model.scaling.fit_site",
        "model.scaling.fit_split",
        "model.scaling.observed_only",
        "model.validation_roles.disjoint_tuning_and_calibration",
        "model.uncertainty.calibration_is_never_used_for_early_stopping",
        "model.evaluation.pool_temporal_protocols",
        "semantic.causal_contract.uses_current_observation_at_sensor",
        "semantic.causal_contract.uses_future_measurements",
        "semantic.causal_contract.thresholds_tuned_on_test_or_external_data",
        "semantic.causal_contract.temporal_protocols_may_be_pooled",
        "receiver.temporal_protocols_may_be_pooled",
        "receiver.algorithm_shared_across_sites_and_scenarios",
        "receiver.parameters_tuned_on_test_or_external_data",
        "receiver.uses_nominal_model_not_scenario_oracle",
        "receiver.oracle_fields_are_evaluation_only",
        "receiver.event_truth_is_evaluation_only",
        "receiver.future_measurements_used",
        "cause.temporal_protocols_may_be_pooled",
        "cause.algorithm_shared_across_sites_and_scenarios",
        "cause.parameters_tuned_on_test_or_external_data",
        "cause.uses_future_measurements",
        "cause.oracle_fields_are_evaluation_only",
        "cause.event_truth_is_evaluation_only",
        "cause.exact_physical_cause_is_not_claimed",
        "recovery.temporal_protocols_may_be_pooled",
        "recovery.algorithm_shared_across_sites_and_scenarios",
        "recovery.parameters_tuned_on_test_or_external_data",
        "recovery.uses_future_measurements",
        "recovery.oracle_fields_are_policy_inputs",
        "recovery.event_truth_is_policy_input",
        "recovery.named_cause_is_never_sufficient_for_action",
        "recovery.reconciliation.one_attempt_per_grid_step",
        "recovery.reconciliation.source_retains_critical_alerts",
        "recovery.reconciliation.critical_alert_retention_steps",
        "recovery.reconciliation.maximum_critical_packets_per_response",
        "recovery.reconciliation.critical_response_uses_original_wire_bytes",
        "recovery.reconciliation.successful_response_consumes_cached_packet",
        "recovery.reconciliation.exact_event_response_provenance",
        "recovery.packet_bytes.liveness_response_payload",
        "fusion.temporal_protocols_may_be_pooled",
        "fusion.algorithm_shared_across_sites_and_splits",
        "fusion.parameters_tuned_on_test_or_external_data",
        "fusion.uses_future_measurements",
        "fusion.oracle_fields_are_fusion_inputs",
        "fusion.critical_evidence_override",
        "fusion.deterministic_total_order",
        "fusion.exact_provenance",
        "scheduler.temporal_protocols_may_be_pooled",
        "scheduler.algorithm_shared_across_sites_scenarios_and_splits",
        "scheduler.parameters_tuned_on_test_or_external_data",
        "scheduler.uses_future_measurements",
        "scheduler.oracle_fields_are_scheduler_inputs",
        "scheduler.current_outcome_is_scheduler_input",
        "scheduler.component_evaluation_only",
        "integration.temporal_protocols_may_be_pooled",
        "integration.parameters_tuned_on_test_or_external_data",
        "integration.uses_future_measurements",
        "integration.oracle_fields_are_policy_inputs",
        "integration.event_truth_is_policy_input",
        "integration.current_outcome_is_policy_input",
        "integration.matched_realizations",
        "integration.matched_opportunity_constraints",
        "reintegration.parameters_tuned_on_test_or_external_data",
        "reintegration.uses_future_measurements",
        "reintegration.oracle_fields_are_policy_inputs",
        "reintegration.event_truth_is_policy_input",
        "reintegration.scheduler_selection_is_receiver_input",
        "reintegration.statistics.temporal_protocols_may_be_pooled",
        "confirmation.prior_task_r_results_used_to_tune_algorithms",
        "confirmation.independent_replication_is_used_for_confirmation",
        "confirmation.uses_future_measurements",
        "confirmation.oracle_fields_are_policy_inputs",
        "confirmation.event_truth_is_policy_input",
        "confirmation.frozen_algorithms.recovery_version",
        "confirmation.independent_network.previous_confirmation_base_seeds",
        "confirmation.independent_network.must_be_disjoint_from_prior_confirmation",
        "confirmation.primary_analysis.temporal_protocols_may_be_pooled",
        "experiments.primary_endpoint",
        "experiments.baselines",
        "experiments.ablations",
    ]
    for key in required:
        _require(config, key)

    if config["schema_version"] != 1:
        raise ConfigurationError("Unsupported schema_version; expected 1")
    if not _require(config, "events.split.external_sites_are_test_only"):
        raise ConfigurationError("External sites must remain test-only")
    if _require(config, "network.outage.event_correlated_outages_primary"):
        raise ConfigurationError("Event-correlated outages are stress tests, not primary scenarios")
    if _require(config, "network.outage.primary_generation_uses_event_labels"):
        raise ConfigurationError("Primary network realizations cannot inspect event labels")
    if not _require(config, "network.hidden_oracle_state_is_never_receiver_input"):
        raise ConfigurationError("Hidden simulator state must never become receiver evidence")
    scenarios = _require(config, "network.scenarios")
    for name, scenario in scenarios.items():
        if scenario.get("role") in {"primary", "reference"} and scenario.get("event_coupled"):
            raise ConfigurationError(f"Primary/reference scenario {name} cannot be event-coupled")
    coupled = [name for name, scenario in scenarios.items() if scenario.get("event_coupled")]
    if coupled != ["F13_event_coincident_outage"]:
        raise ConfigurationError("Only declared F13 stress may couple outages to events")
    if _require(config, "data.audit.availability_windows.selection_uses_measurement_values"):
        raise ConfigurationError("Availability selection must never inspect measurement values")
    if not _require(
        config, "data.audit.availability_windows.network_outages_injected_after_selection"
    ):
        raise ConfigurationError(
            "Simulated network outages must be injected after environmental-data selection"
        )
    if _require(config, "preprocessing.measurement_values_used_for_availability_selection"):
        raise ConfigurationError("Availability selection cannot inspect measurement values")
    if not _require(
        config, "preprocessing.post_qc_gate_uses_qartod_finiteness_and_physical_validity"
    ):
        raise ConfigurationError("Post-QC retention must enforce the declared quality gate")
    if _require(config, "preprocessing.split.uses_measurement_values"):
        raise ConfigurationError("Chronological split assignment cannot inspect measurements")
    if _require(config, "preprocessing.fit_scalers_during_preprocessing"):
        raise ConfigurationError("Scalers must be fit later on training data only")
    if set(_require(config, "preprocessing.accepted_qartod_flags")) != {1}:
        raise ConfigurationError("Primary canonical channels must use QARTOD PASS (1) only")
    if int(_require(config, "preprocessing.maximum_causal_fill_steps")) > 1:
        raise ConfigurationError("Causal filling cannot exceed one grid step")
    primary = _require(config, "preprocessing.protocols.primary_15min")
    if (
        int(primary.get("grid_minutes", -1)) != 15
        or primary.get("development_site") != "CE02SHSM"
        or primary.get("external_sites") != ["CE07SHSM"]
    ):
        raise ConfigurationError("The locked primary protocol is CE02SHSM to CE07SHSM at 15 min")
    secondary = _require(config, "preprocessing.protocols.secondary_60min")
    if int(secondary.get("grid_minutes", -1)) != 60:
        raise ConfigurationError("The locked secondary protocol uses a 60-minute grid")
    history_hours = float(_require(config, "model.history_hours"))
    input_steps = _require(config, "model.input_steps_by_protocol")
    for name, protocol_config in (("primary_15min", primary), ("secondary_60min", secondary)):
        represented_hours = (
            int(input_steps.get(name, 0)) * int(protocol_config["grid_minutes"]) / 60.0
        )
        if represented_hours != history_hours:
            raise ConfigurationError(
                f"{name} predictor history must represent exactly {history_hours:g} hours"
            )
    sparse_sites = secondary.get("site_overrides", {})
    for site in ("CE01ISSM", "CE06ISSM"):
        override = sparse_sites.get(site, {})
        if override.get("sparse_channels") != ["oxygen"]:
            raise ConfigurationError(f"{site} must remain an oxygen-sparse external stress test")
        if "oxygen" in override.get("quality_gate_channels", []):
            raise ConfigurationError(f"{site} oxygen cannot gate window eligibility")
    prep_train = float(_require(config, "preprocessing.split.train_fraction"))
    prep_valid = float(_require(config, "preprocessing.split.validation_fraction"))
    prep_test = float(_require(config, "preprocessing.split.internal_test_fraction"))
    if abs(prep_train + prep_valid + prep_test - 1.0) > 1e-9:
        raise ConfigurationError("Preprocessing split fractions must sum to 1")
    if not _require(config, "protocol.bounded_silence.enabled"):
        raise ConfigurationError("Bounded silence cannot be disabled in the full method")
    if not _require(config, "protocol.dual_belief.unknown_cause_class"):
        raise ConfigurationError("Failure attribution must retain an UNKNOWN class")
    if _require(config, "experiments.primary_endpoint") != "critical_false_trust_rate":
        raise ConfigurationError("The locked primary endpoint is critical_false_trust_rate")

    if not _require(config, "events.split.reuse_task_g_splits"):
        raise ConfigurationError("Task H must reuse Task G splits without reassignment")
    if _require(config, "events.split.calibration_split") != "train":
        raise ConfigurationError("Event thresholds must be calibrated on train only")
    if _require(config, "events.calibration.site") != "CE02SHSM":
        raise ConfigurationError("Event thresholds must be calibrated on CE02SHSM")
    if not _require(config, "events.calibration.observed_measurements_only"):
        raise ConfigurationError("Event calibration and ground truth must use observations only")
    if not _require(config, "events.causal_labels_only"):
        raise ConfigurationError("Task H labels must be causal")
    if _require(config, "events.reporting.pool_temporal_protocols"):
        raise ConfigurationError("Overlapping temporal protocols must not be pooled")
    if _require(config, "events.reporting.score_zero_support_event_classes"):
        raise ConfigurationError("Zero-support event classes cannot receive event metrics")
    if _require(config, "model.scaling.fit_site") != "CE02SHSM":
        raise ConfigurationError("Predictor scaling must be fit on CE02SHSM")
    if _require(config, "model.scaling.fit_split") != "train":
        raise ConfigurationError("Predictor scaling must use train only")
    if not _require(config, "model.scaling.observed_only"):
        raise ConfigurationError("Predictor scaling must use genuinely observed values only")
    if not _require(config, "model.validation_roles.disjoint_tuning_and_calibration"):
        raise ConfigurationError("Tuning and uncertainty calibration windows must be disjoint")
    if not _require(config, "model.uncertainty.calibration_is_never_used_for_early_stopping"):
        raise ConfigurationError("Calibration windows cannot be used for early stopping")
    if _require(config, "model.evaluation.pool_temporal_protocols"):
        raise ConfigurationError("Predictor results from temporal protocols cannot be pooled")
    if _require(config, "model.predictor") != "persistence_residual_tcn":
        raise ConfigurationError("The locked predictor is the persistence-residual TCN")
    if _require(config, "model.uncertainty.calibration") != (
        "worst_window_normalized_split_conformal"
    ):
        raise ConfigurationError("Task I requires worst-window conformal calibration")

    if not _require(config, "semantic.causal_contract.uses_current_observation_at_sensor"):
        raise ConfigurationError(
            "Semantic innovations require the current causal sensor observation"
        )
    if _require(config, "semantic.causal_contract.uses_future_measurements"):
        raise ConfigurationError("Task J decisions cannot use future measurements")
    if _require(config, "semantic.causal_contract.thresholds_tuned_on_test_or_external_data"):
        raise ConfigurationError("Task J policy thresholds cannot be tuned on test/external data")
    if _require(config, "semantic.causal_contract.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task J cannot pool overlapping temporal protocols")
    if not _require(config, "semantic.nominal_policy.thresholds_are_apriori_not_fitted"):
        raise ConfigurationError("The nominal Task J policy must remain a priori")
    evidence_weights = _require(config, "semantic.evidence.channel_weights")
    if abs(sum(float(value) for value in evidence_weights.values()) - 1.0) > 1e-9:
        raise ConfigurationError("Task J channel evidence weights must sum to one")
    if _require(config, "semantic.evidence.cross_channel_aggregation") != "maximum":
        raise ConfigurationError("Task J must preserve the maximum safety-critical channel risk")
    if not _require(config, "semantic.evidence.critical_evidence_override"):
        raise ConfigurationError(
            "Causal critical-event evidence must override semantic compression"
        )
    if _require(config, "semantic.nominal_policy.decision_priority") != [
        "critical_event",
        "extreme_risk",
        "innovation",
        "validity_expiry",
        "heartbeat_expiry",
        "silent",
    ]:
        raise ConfigurationError("Task J decision priority changed from the locked safety order")
    semantic_payloads = _require(config, "semantic.payload_bytes")
    if int(semantic_payloads.get("silent", -1)) != 0:
        raise ConfigurationError("Intentional silence must transmit zero payload bytes")
    if not semantic_payloads.get("excludes_network_headers"):
        raise ConfigurationError("Task J payload accounting must exclude later network headers")

    if _require(config, "receiver.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task L cannot pool overlapping temporal protocols")
    if not _require(config, "receiver.algorithm_shared_across_sites_and_scenarios"):
        raise ConfigurationError("Task L must use one receiver algorithm across test conditions")
    if _require(config, "receiver.parameters_tuned_on_test_or_external_data"):
        raise ConfigurationError("Task L cannot tune receiver parameters on evaluation data")
    if not _require(config, "receiver.uses_nominal_model_not_scenario_oracle"):
        raise ConfigurationError("Task L must not use scenario-specific oracle parameters")
    if not _require(config, "receiver.oracle_fields_are_evaluation_only"):
        raise ConfigurationError("Task L oracle fields must remain evaluation-only")
    if not _require(config, "receiver.event_truth_is_evaluation_only"):
        raise ConfigurationError("Task L event truth must remain evaluation-only")
    if _require(config, "receiver.future_measurements_used"):
        raise ConfigurationError("Task L cannot use future measurements")
    if not _require(config, "receiver.state_policy.no_normal_fallback_after_contract_expiry"):
        raise ConfigurationError("Expired contracts cannot silently fall back to normal")
    if not _require(config, "receiver.state_policy.unknown_is_retained"):
        raise ConfigurationError("Task L must preserve unresolved ambiguity/UNKNOWN")
    if _require(config, "receiver.reporting.primary_endpoint") != "critical_false_trust_rate":
        raise ConfigurationError("Task L primary endpoint must remain critical false trust")
    receiver_states = set(_require(config, "receiver.state_policy.states"))
    if receiver_states != set(_require(config, "protocol.receiver_states")):
        raise ConfigurationError("Task L receiver states must match the protocol contract")

    if _require(config, "cause.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task M cannot pool overlapping temporal protocols")
    if not _require(config, "cause.algorithm_shared_across_sites_and_scenarios"):
        raise ConfigurationError("Task M must use one cause model across test conditions")
    if _require(config, "cause.parameters_tuned_on_test_or_external_data"):
        raise ConfigurationError("Task M cannot tune cause parameters on evaluation data")
    if _require(config, "cause.uses_future_measurements"):
        raise ConfigurationError("Task M cannot use future measurements")
    if not _require(config, "cause.oracle_fields_are_evaluation_only"):
        raise ConfigurationError("Task M oracle fields must remain evaluation-only")
    if not _require(config, "cause.event_truth_is_evaluation_only"):
        raise ConfigurationError("Task M event truth must remain evaluation-only")
    if not _require(config, "cause.exact_physical_cause_is_not_claimed"):
        raise ConfigurationError("Task M cannot claim exact cause identification")
    if _require(config, "cause.score_semantics") != "rule_based_pseudo_posterior":
        raise ConfigurationError("Task M scores must be identified as rule-based pseudo-posteriors")
    if _require(config, "cause.probabilities_are_empirically_calibrated"):
        raise ConfigurationError("Task M cannot claim empirical probability calibration")
    cause_classes = set(_require(config, "cause.classes"))
    expected_causes = {
        "valid_suppression",
        "isolated_loss",
        "persistent_outage",
        "contract_divergence",
        "unknown",
    }
    if cause_classes != expected_causes:
        raise ConfigurationError("Task M coarse-cause taxonomy changed")
    if not _require(config, "cause.abstention.unknown_is_retained"):
        raise ConfigurationError("Task M must retain UNKNOWN abstention")
    if _require(config, "cause.valid_suppression_admissibility.receiver_state") != (
        "trusted_silence"
    ):
        raise ConfigurationError("Valid suppression requires the trusted-silence state")
    if not _require(config, "cause.valid_suppression_admissibility.require_valid_contract"):
        raise ConfigurationError("Valid suppression requires a valid contract")
    if not _require(config, "cause.valid_suppression_admissibility.require_contact_not_due"):
        raise ConfigurationError("Valid suppression requires contact not to be due")
    if _require(config, "cause.valid_suppression_admissibility.inadmissible_decision") != (
        "unknown"
    ):
        raise ConfigurationError("Inadmissible valid suppression must abstain to UNKNOWN")
    if _require(config, "cause.reporting.zero_support_classes_are_scored"):
        raise ConfigurationError("Task M cannot score zero-support cause classes")

    if _require(config, "recovery.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task N cannot pool overlapping temporal protocols")
    if not _require(config, "recovery.algorithm_shared_across_sites_and_scenarios"):
        raise ConfigurationError("Task N must use one recovery policy across test conditions")
    if _require(config, "recovery.parameters_tuned_on_test_or_external_data"):
        raise ConfigurationError("Task N cannot tune recovery parameters on evaluation data")
    if _require(config, "recovery.uses_future_measurements"):
        raise ConfigurationError("Task N cannot use future measurements")
    if _require(config, "recovery.oracle_fields_are_policy_inputs"):
        raise ConfigurationError("Task N deployable policy cannot use oracle fields")
    if _require(config, "recovery.event_truth_is_policy_input"):
        raise ConfigurationError("Task N deployable policy cannot use event truth")
    if not _require(config, "recovery.named_cause_is_never_sufficient_for_action"):
        raise ConfigurationError("Task N cannot trust a named cause as sufficient evidence")
    if not _require(config, "recovery.reconciliation.one_attempt_per_grid_step"):
        raise ConfigurationError("Task N attempts must be distributed across grid steps")
    if not _require(config, "recovery.reconciliation.source_retains_critical_alerts"):
        raise ConfigurationError("Task N requires a bounded critical-alert cache")
    if int(_require(config, "recovery.reconciliation.critical_alert_retention_steps")) != 4:
        raise ConfigurationError("Task N alert retention must match the four-step silence horizon")
    if (
        int(
            _require(
                config,
                "recovery.reconciliation.maximum_critical_packets_per_response",
            )
        )
        != 1
    ):
        raise ConfigurationError("Each recovery response must carry at most one cached packet")
    if not _require(
        config, "recovery.reconciliation.critical_response_uses_original_wire_bytes"
    ):
        raise ConfigurationError("Critical recovery must charge the original encoded packet size")
    if not _require(
        config, "recovery.reconciliation.successful_response_consumes_cached_packet"
    ):
        raise ConfigurationError("Successful recovery must consume exactly one cached packet")
    if not _require(config, "recovery.reconciliation.exact_event_response_provenance"):
        raise ConfigurationError("Recovery requires exact event-to-response provenance")
    if int(_require(config, "recovery.packet_bytes.liveness_response_payload")) <= 0:
        raise ConfigurationError("Liveness response payload must be positive")
    if _require(config, "recovery.reporting.primary_endpoint") != "critical_false_trust_rate":
        raise ConfigurationError("Task N primary endpoint must remain critical false trust")
    bounded = _require(config, "recovery.policies.safe_silence_bounded")
    if int(bounded["maximum_attempts_per_trigger"]) < 1:
        raise ConfigurationError("Task N bounded policy must permit at least one attempt")
    if int(bounded["maximum_attempts_per_trigger"]) >= int(
        _require(config, "recovery.policies.unbounded_retry_ceiling.maximum_attempts_per_trigger")
    ):
        raise ConfigurationError("The bounded recovery budget must remain below the ceiling")

    if _require(config, "fusion.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task O cannot pool overlapping temporal protocols")
    if not _require(config, "fusion.algorithm_shared_across_sites_and_splits"):
        raise ConfigurationError("Task O must use one fusion algorithm everywhere")
    if _require(config, "fusion.parameters_tuned_on_test_or_external_data"):
        raise ConfigurationError("Task O cannot tune fusion parameters on evaluation data")
    if _require(config, "fusion.uses_future_measurements"):
        raise ConfigurationError("Task O cannot use future sensor measurements")
    if _require(config, "fusion.oracle_fields_are_fusion_inputs"):
        raise ConfigurationError("Task O fusion cannot use oracle fields")
    if not _require(config, "fusion.critical_evidence_override"):
        raise ConfigurationError("Task O must preserve critical evidence")
    if not _require(config, "fusion.deterministic_total_order"):
        raise ConfigurationError("Task O requires a deterministic total order")
    if not _require(config, "fusion.exact_provenance"):
        raise ConfigurationError("Task O must retain exact provenance in this evaluation")
    if _require(config, "fusion.method") != "associative_top_k":
        raise ConfigurationError("Task O proposed method must be associative_top_k")
    if int(_require(config, "fusion.top_k")) != int(_require(config, "protocol.relay_fusion.k")):
        raise ConfigurationError("Task O top-K must match the protocol contract")
    if int(_require(config, "fusion.relay_batch_items")) <= int(_require(config, "fusion.top_k")):
        raise ConfigurationError("Task O relay batch must exceed K to test bounded fusion")
    if set(_require(config, "fusion.references")) != {
        "no_relay_fusion",
        "mean_relay_fusion",
    }:
        raise ConfigurationError("Task O requires no-fusion and mean-fusion references")

    if _require(config, "scheduler.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task P cannot pool overlapping temporal protocols")
    if not _require(config, "scheduler.algorithm_shared_across_sites_scenarios_and_splits"):
        raise ConfigurationError("Task P must use one scheduling algorithm everywhere")
    if _require(config, "scheduler.parameters_tuned_on_test_or_external_data"):
        raise ConfigurationError("Task P cannot tune scheduler parameters on evaluation data")
    if _require(config, "scheduler.uses_future_measurements"):
        raise ConfigurationError("Task P cannot use future measurements")
    if _require(config, "scheduler.oracle_fields_are_scheduler_inputs"):
        raise ConfigurationError("Task P scheduler cannot use oracle fields")
    if _require(config, "scheduler.current_outcome_is_scheduler_input"):
        raise ConfigurationError("Task P cannot inspect a candidate's current outcome")
    if not _require(config, "scheduler.component_evaluation_only"):
        raise ConfigurationError("Task P must retain its component-evaluation claim boundary")
    if _require(config, "scheduler.proposed_policy") != ("cvar_risk_reduction_per_joule"):
        raise ConfigurationError("Task P proposed policy must use CVaR reduction per joule")
    if _require(config, "scheduler.reference_policy") != ("expected_risk_reduction_per_joule"):
        raise ConfigurationError("Task P requires the expected-risk ablation")
    if float(_require(config, "scheduler.cvar_alpha")) != float(
        _require(config, "protocol.scheduler.cvar_alpha")
    ):
        raise ConfigurationError("Task P CVaR alpha must match the protocol contract")
    for task_key, protocol_key in (
        ("energy_weight", "energy_cost_weight"),
        ("byte_weight", "byte_cost_weight"),
        ("airtime_weight", "airtime_cost_weight"),
    ):
        if float(_require(config, f"scheduler.cost.{task_key}")) != float(
            _require(config, f"protocol.scheduler.{protocol_key}")
        ):
            raise ConfigurationError(f"Task P {task_key} must match the protocol contract")
    if not _require(config, "scheduler.causal_link_belief.update_only_after_attempted_packet"):
        raise ConfigurationError("Task P link belief must use observed attempts only")
    if int(_require(config, "scheduler.queue_stress.candidates_per_opportunity")) <= int(
        _require(config, "scheduler.queue_stress.maximum_selected_packets")
    ):
        raise ConfigurationError("Task P queue stress must create a constrained choice")
    for key, value in _require(config, "scheduler.budgets").items():
        if float(value) <= 0:
            raise ConfigurationError(f"Task P budget must be positive: {key}")
    profiles = _require(config, "scheduler.loss_profiles")
    if "primary" not in profiles or len(profiles) < 2:
        raise ConfigurationError("Task P requires a primary loss profile and sensitivities")
    for name, profile in profiles.items():
        if float(profile.get("calibration", -1)) != 0.0:
            raise ConfigurationError(
                f"Task P cannot score unavailable probability calibration: {name}"
            )

    if _require(config, "integration.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task Q cannot pool overlapping temporal protocols")
    if _require(config, "integration.parameters_tuned_on_test_or_external_data"):
        raise ConfigurationError("Task Q cannot tune policies on test/external data")
    if _require(config, "integration.uses_future_measurements"):
        raise ConfigurationError("Task Q cannot use future measurements")
    if _require(config, "integration.oracle_fields_are_policy_inputs"):
        raise ConfigurationError("Task Q policies cannot use stored oracle fields")
    if _require(config, "integration.event_truth_is_policy_input"):
        raise ConfigurationError("Task Q policies cannot use event truth")
    if _require(config, "integration.current_outcome_is_policy_input"):
        raise ConfigurationError("Task Q policies cannot inspect current outcomes")
    if not _require(config, "integration.matched_realizations"):
        raise ConfigurationError("Task Q requires matched counterfactual realizations")
    if not _require(config, "integration.matched_opportunity_constraints"):
        raise ConfigurationError("Task Q requires matched opportunity constraints")
    if int(_require(config, "integration.relay_to_buoy_hops")) != 1:
        raise ConfigurationError("Task Q integrates the one-hop relay-to-buoy path")
    if _require(config, "integration.proposed_policy") != _require(
        config, "scheduler.proposed_policy"
    ):
        raise ConfigurationError("Task Q proposed policy must be the frozen Task P policy")
    if _require(config, "integration.loss_profile") != "primary":
        raise ConfigurationError("Task Q primary replay must use the frozen primary loss profile")
    references = set(_require(config, "integration.reference_policies"))
    if _require(config, "scheduler.reference_policy") not in references:
        raise ConfigurationError("Task Q must retain the expected-risk scheduler reference")

    if _require(config, "reintegration.parameters_tuned_on_test_or_external_data"):
        raise ConfigurationError("Task R cannot tune policies on test/external data")
    if _require(config, "reintegration.uses_future_measurements"):
        raise ConfigurationError("Task R cannot use future measurements")
    if _require(config, "reintegration.oracle_fields_are_policy_inputs"):
        raise ConfigurationError("Task R deployable policies cannot use oracle fields")
    if _require(config, "reintegration.event_truth_is_policy_input"):
        raise ConfigurationError("Task R deployable policies cannot use event truth")
    if _require(config, "reintegration.scheduler_selection_is_receiver_input"):
        raise ConfigurationError("Task R receiver cannot observe missing scheduler decisions")
    if _require(config, "reintegration.statistics.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task R cannot pool overlapping temporal protocols")
    if int(_require(config, "reintegration.relay_to_buoy_hops")) != 1:
        raise ConfigurationError("Task R integrates the one-hop relay-to-buoy path")
    if int(
        _require(
            config,
            "reintegration.relay_receiver_contract.maximum_validity_opportunities",
        )
    ) != int(_require(config, "protocol.bounded_silence.validity_horizon_steps")):
        raise ConfigurationError("Task R validity bound must match the silence contract")
    if not _require(
        config, "reintegration.relay_receiver_contract.missing_critical_summary_is_unsafe"
    ):
        raise ConfigurationError("Task R must score missing critical summaries as unsafe")
    if int(_require(config, "reintegration.statistics.bootstrap_resamples")) < 1000:
        raise ConfigurationError("Task R requires at least 1,000 paired bootstrap resamples")
    if _require(config, "reintegration.statistics.test") != ("two_sided_wilcoxon_signed_rank"):
        raise ConfigurationError("Task R requires the predeclared paired Wilcoxon test")
    if _require(config, "reintegration.statistics.multiplicity_correction") != (
        "holm_within_site_protocol_comparison"
    ):
        raise ConfigurationError("Task R requires the predeclared Holm correction")
    if set(_require(config, "reintegration.scheduler_policies")) != {
        "cvar_risk_reduction_per_joule",
        "expected_risk_reduction_per_joule",
    }:
        raise ConfigurationError("Task R requires the two frozen Task P schedulers")
    if set(_require(config, "reintegration.recovery_policies")) != {
        "safe_silence_bounded",
        "no_recovery",
    }:
        raise ConfigurationError("Task R requires bounded and no-recovery policies")
    if _require(config, "reintegration.proposed_system.scheduler") != _require(
        config, "scheduler.proposed_policy"
    ) or _require(config, "reintegration.proposed_system.recovery") != ("safe_silence_bounded"):
        raise ConfigurationError("Task R proposed system must retain the frozen policies")

    if _require(config, "confirmation.prior_task_r_results_used_to_tune_algorithms"):
        raise ConfigurationError("Task S cannot tune algorithms using Task R outcomes")
    if not _require(config, "confirmation.independent_replication_is_used_for_confirmation"):
        raise ConfigurationError("Task S requires an independent confirmatory replication")
    if _require(config, "confirmation.uses_future_measurements"):
        raise ConfigurationError("Task S cannot use future measurements")
    if _require(config, "confirmation.oracle_fields_are_policy_inputs"):
        raise ConfigurationError("Task S deployable policies cannot use oracle fields")
    if _require(config, "confirmation.event_truth_is_policy_input"):
        raise ConfigurationError("Task S deployable policies cannot use event truth")
    if _require(config, "confirmation.primary_analysis.temporal_protocols_may_be_pooled"):
        raise ConfigurationError("Task S cannot pool overlapping temporal protocols")
    if _require(config, "confirmation.frozen_algorithms.recovery_version") != _require(
        config, "recovery.recovery_version"
    ):
        raise ConfigurationError("Task S recovery version must match the frozen recovery logic")
    if int(_require(config, "confirmation.independent_network.repetitions")) < 20:
        raise ConfigurationError("Task S requires at least 20 independent repetitions")
    if int(_require(config, "confirmation.independent_network.base_seed")) == int(
        _require(config, "network.base_seed")
    ):
        raise ConfigurationError("Task S seed must differ from the frozen Task K seed")
    if not _require(config, "confirmation.independent_network.must_be_disjoint_from_task_k"):
        raise ConfigurationError("Task S must audit seed disjointness")
    previous_confirmation_base_seeds = [
        int(value)
        for value in _require(
            config, "confirmation.independent_network.previous_confirmation_base_seeds"
        )
    ]
    if not previous_confirmation_base_seeds:
        raise ConfigurationError("Task S must declare the previously inspected confirmation seeds")
    if int(_require(config, "confirmation.independent_network.base_seed")) in set(
        previous_confirmation_base_seeds
    ):
        raise ConfigurationError("Task S confirmation seed must be disjoint from reference seeds")
    if not _require(
        config, "confirmation.independent_network.must_be_disjoint_from_prior_confirmation"
    ):
        raise ConfigurationError("Task S must audit disjointness from prior confirmation seeds")
    if set(_require(config, "confirmation.primary_analysis.endpoints")) != {
        "post_recovery_critical_delivery_recall",
        "realized_post_recovery_tail_proxy_reduction",
    }:
        raise ConfigurationError("Task S confirmatory endpoints are frozen")
    if _require(config, "confirmation.primary_analysis.test") != ("two_sided_wilcoxon_signed_rank"):
        raise ConfigurationError("Task S requires the paired two-sided Wilcoxon test")
    if _require(config, "confirmation.primary_analysis.multiplicity_correction") != (
        "holm_across_primary_endpoints_and_scenarios"
    ):
        raise ConfigurationError("Task S requires the predeclared primary-family correction")
    if int(_require(config, "confirmation.primary_analysis.bootstrap_resamples")) < 1000:
        raise ConfigurationError("Task S requires at least 1,000 bootstrap resamples")

    actions = set(_require(config, "protocol.node_actions"))
    if actions != {"silent", "heartbeat", "innovation", "alert"}:
        raise ConfigurationError("Protocol actions must be silent/heartbeat/innovation/alert")
