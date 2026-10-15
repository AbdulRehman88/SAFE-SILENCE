"""Isolated UNKNOWN ablation, with declared known-cause-order tie-breaking."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from safe_silence.data.preprocess import sha256_file


class CauseInferenceError(RuntimeError):
    """Raised when Task M cannot preserve its cause-inference contract."""


KNOWN_CAUSES = (
    "valid_suppression",
    "isolated_loss",
    "persistent_outage",
    "contract_divergence",
)
ALL_CAUSES = (*KNOWN_CAUSES, "unknown")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _resolve_task_l_artifact(record: dict[str, Any], root: Path) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    if all(record.get(key) for key in ("protocol", "site", "scenario")):
        return (
            root
            / "beliefs"
            / str(record["protocol"])
            / str(record["site"])
            / f"{record['scenario']}.parquet"
        )
    return root / declared.name


def verify_task_l_inputs(root: Path) -> tuple[dict[str, Any], int]:
    """Reverify every frozen Task L artifact after possible project relocation."""
    summary_path = root / "task_l_summary.json"
    if not summary_path.exists():
        raise CauseInferenceError(f"Missing Task L summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != "L" or summary.get("receiver_version") != "1.0":
        raise CauseInferenceError("Task M requires the frozen Task L receiver contract v1.0")
    verified = 0
    for record in summary.get("artifacts", []):
        path = _resolve_task_l_artifact(record, root)
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise CauseInferenceError(f"Task L artifact changed or is missing: {path}")
        verified += 1
    if verified == 0:
        raise CauseInferenceError("Task L summary contains no verifiable artifacts")
    return summary, verified


def make_cause_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only receiver-observable current and historical evidence."""
    columns = [
        "timestamp",
        "window_key",
        "repetition",
        "packet_arrived",
        "receiver_state",
        "connectivity_belief",
        "environmental_uncertainty",
        "receiver_contract_age_steps",
        "receiver_overdue_steps",
        "receiver_contract_valid",
        "receiver_contact_due",
        "receiver_active_contract_id",
        "received_action",
        "received_semantic_risk",
        "received_validity_horizon_steps",
        "received_contract_id",
    ]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise CauseInferenceError(f"Task L beliefs lack receiver evidence: {sorted(missing)}")
    return frame[columns].copy()


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1.0e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=1) / np.log(probabilities.shape[1])


def infer_cause_group(observations: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Infer coarse cause probabilities for one chronological receiver window."""
    forbidden_prefixes = tuple(str(value) for value in config["forbidden_inference_prefixes"])
    forbidden = [column for column in observations.columns if column.startswith(forbidden_prefixes)]
    if forbidden:
        raise CauseInferenceError(f"Forbidden cause evidence supplied: {sorted(forbidden)}")
    required = {
        "timestamp",
        "packet_arrived",
        "receiver_state",
        "connectivity_belief",
        "environmental_uncertainty",
        "receiver_overdue_steps",
        "receiver_contract_valid",
        "receiver_contact_due",
    }
    missing = required.difference(observations.columns)
    if missing:
        raise CauseInferenceError(f"Cause observation lacks: {sorted(missing)}")

    frame = observations.sort_values("timestamp").reset_index(drop=True).copy()
    arrived = frame["packet_arrived"].to_numpy(bool)
    connectivity = np.clip(frame["connectivity_belief"].to_numpy(float), 0.0, 1.0)
    environmental = np.clip(frame["environmental_uncertainty"].to_numpy(float), 0.0, 1.0)
    contract_valid = frame["receiver_contract_valid"].to_numpy(bool).astype(float)
    contact_due = frame["receiver_contact_due"].to_numpy(bool).astype(float)
    contract_invalid = 1.0 - contract_valid
    overdue_cap = max(int(config["evidence"]["overdue_cap_steps"]), 1)
    silence_cap = max(int(config["evidence"]["silence_run_cap_steps"]), 1)
    overdue = np.clip(frame["receiver_overdue_steps"].to_numpy(float) / overdue_cap, 0, 1)
    states = frame["receiver_state"].astype(str).to_numpy()
    trusted = (states == "trusted_silence").astype(float)
    ambiguous = (states == "ambiguous").astype(float)
    stale = np.isin(states, ["failure_suspected", "stale_unsafe"]).astype(float)

    silence_run = np.zeros(len(frame), dtype=float)
    run = 0
    for index, packet_arrived in enumerate(arrived):
        run = 0 if packet_arrived else run + 1
        silence_run[index] = min(run, silence_cap) / silence_cap
    first_due = ((contact_due == 1.0) & (frame["receiver_overdue_steps"].to_numpy() <= 1)).astype(
        float
    )

    feature_map = {
        "contract_valid": contract_valid,
        "contract_invalid": contract_invalid,
        "connectivity": connectivity,
        "inverse_connectivity": 1.0 - connectivity,
        "trusted_state": trusted,
        "ambiguous_state": ambiguous,
        "stale_state": stale,
        "contact_due": contact_due,
        "first_due_absence": first_due,
        "silence_run": silence_run,
        "overdue": overdue,
        "environmental_uncertainty": environmental,
    }
    logits = np.zeros((len(frame), len(KNOWN_CAUSES)), dtype=float)
    for cause_index, cause in enumerate(KNOWN_CAUSES):
        weights = config["logits"][cause]
        logits[:, cause_index] = float(weights["intercept"])
        for feature, weight in weights.items():
            if feature == "intercept":
                continue
            if feature not in feature_map:
                raise CauseInferenceError(f"Unknown configured cause feature: {feature}")
            logits[:, cause_index] += float(weight) * feature_map[feature]

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    known_probability = exponent / exponent.sum(axis=1, keepdims=True)
    entropy = _normalized_entropy(known_probability)
    entropy_floor = float(config["abstention"]["entropy_floor"])
    unknown_probability = np.clip(
        (entropy - entropy_floor) / max(1.0 - entropy_floor, 1.0e-9), 0.0, 1.0
    ) * float(config["abstention"]["maximum_unknown_probability"])
    final_known = known_probability * (1.0 - unknown_probability[:, None])
    full_probability = np.column_stack([final_known, unknown_probability])

    admissibility = config["valid_suppression_admissibility"]
    valid_admissible = (
        (states == str(admissibility["receiver_state"]))
        & (contract_valid == 1.0)
        & (contact_due == 0.0)
        & (connectivity >= float(admissibility["minimum_connectivity"]))
        & (environmental <= float(admissibility["maximum_environmental_uncertainty"]))
    )
    preliminary_best = np.argmax(final_known, axis=1)
    gate_transfer = ~valid_admissible & ~arrived
    if config.get("extension_ablation") == "no_unknown":
        unknown_probability[:] = 0.0
        final_known = known_probability.copy()
        full_probability = np.column_stack([final_known, unknown_probability])
        gate_transfer[:] = False
    transferred_mass = np.where(gate_transfer, full_probability[:, 0], 0.0)
    full_probability[gate_transfer, 0] = 0.0
    full_probability[:, -1] += transferred_mass
    final_known = full_probability[:, : len(KNOWN_CAUSES)]
    unknown_probability = full_probability[:, -1]

    known_order = np.argsort(final_known, axis=1)
    best_known_index = np.argmax(final_known, axis=1) if config.get("extension_ablation") == "no_unknown" else known_order[:, -1]
    best_known = final_known[np.arange(len(frame)), best_known_index]
    second_known = final_known[np.arange(len(frame)), known_order[:, -2]]
    margin = best_known - second_known
    decisions = np.asarray([KNOWN_CAUSES[index] for index in best_known_index], dtype=object)
    abstain = (
        (unknown_probability >= best_known)
        | (best_known < float(config["abstention"]["decision_minimum_known_probability"]))
        | (margin < float(config["abstention"]["decision_minimum_margin"]))
    )
    if config.get("extension_ablation") == "no_unknown":
        abstain[:] = False
    decisions[abstain] = "unknown"
    inadmissible_valid = gate_transfer & (preliminary_best == 0)
    decisions[inadmissible_valid] = str(admissibility["inadmissible_decision"])
    decisions[arrived] = "observed_update"

    for index, cause in enumerate(ALL_CAUSES):
        probability = full_probability[:, index].astype(np.float32)
        probability[arrived] = np.nan
        frame[f"cause_probability_{cause}"] = probability
    confidence = np.where(decisions == "unknown", unknown_probability, best_known).astype(
        np.float32
    )
    confidence[arrived] = np.nan
    entropy_full = _normalized_entropy(full_probability).astype(np.float32)
    entropy_full[arrived] = np.nan
    frame["cause_decision"] = decisions
    frame["valid_suppression_admissible"] = valid_admissible.astype(np.uint8)
    frame["cause_abstained_by_safety_gate"] = inadmissible_valid.astype(np.uint8)
    frame["cause_confidence"] = confidence
    frame["cause_posterior_entropy"] = entropy_full
    frame["receiver_silence_run_steps"] = np.rint(silence_run * silence_cap).astype(np.int32)
    return frame


def infer_causes(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Infer causes without truth, then attach evaluation-only oracle labels."""
    observations = make_cause_observations(frame)
    pieces: list[pd.DataFrame] = []
    for _, group in observations.groupby(["repetition", "window_key"], sort=False):
        pieces.append(infer_cause_group(group, config))
    inferred = pd.concat(pieces, ignore_index=True)
    keys = ["repetition", "window_key", "timestamp"]
    truth_columns = keys + [
        "evaluation_only_source_action",
        "evaluation_only_critical_event",
        "evaluation_only_unsafe_silence",
        "evaluation_only_link_state",
    ]
    missing = set(truth_columns).difference(frame.columns)
    if missing:
        raise CauseInferenceError(f"Task L lacks cause evaluation truth: {sorted(missing)}")
    truth = frame[truth_columns].copy()
    truth["timestamp"] = pd.to_datetime(truth["timestamp"], utc=True)
    inferred["timestamp"] = pd.to_datetime(inferred["timestamp"], utc=True)
    result = inferred.merge(truth, on=keys, how="left", validate="one_to_one")

    labels = np.full(len(result), "observed_update", dtype=object)
    for _, indices in result.groupby(["repetition", "window_key"], sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        link_state = result.loc[positions, "evaluation_only_link_state"].astype(str).to_numpy()
        outage_run = 0
        for local_index, position in enumerate(positions):
            outage_run = outage_run + 1 if link_state[local_index] == "outage" else 0
            if bool(result.at[position, "packet_arrived"]):
                continue
            source_action = str(result.at[position, "evaluation_only_source_action"])
            contract_valid = bool(result.at[position, "receiver_contract_valid"])
            if source_action == "silent" and contract_valid:
                labels[position] = "valid_suppression"
            elif source_action == "silent":
                labels[position] = "contract_divergence"
            elif outage_run >= 2:
                labels[position] = "persistent_outage"
            else:
                labels[position] = "isolated_loss"
    result["evaluation_only_true_cause"] = labels
    unsafe = result["evaluation_only_unsafe_silence"].to_numpy(bool)
    false_valid = unsafe & (result["cause_decision"].to_numpy() == "valid_suppression")
    result["evaluation_only_unsafe_predicted_valid"] = false_valid.astype(np.uint8)
    return result


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence >= lower) & (
            confidence <= upper if upper == 1.0 else confidence < upper
        )
        if selected.any():
            error += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return float(error)


def cause_metrics(
    frame: pd.DataFrame,
    protocol: str,
    site: str,
    model_role: str,
    scenario: str,
    repetition: int,
) -> dict[str, Any]:
    silence = ~frame["packet_arrived"].to_numpy(bool)
    subset = frame.loc[silence]
    truth = subset["evaluation_only_true_cause"].astype(str).to_numpy()
    decision = subset["cause_decision"].astype(str).to_numpy()
    supported = [cause for cause in KNOWN_CAUSES if np.any(truth == cause)]
    recalls = [float((decision[truth == cause] == cause).mean()) for cause in supported]
    probabilities = subset[[f"cause_probability_{cause}" for cause in ALL_CAUSES]].to_numpy(float)
    targets = np.zeros_like(probabilities)
    cause_index = {cause: index for index, cause in enumerate(ALL_CAUSES)}
    for row, cause in enumerate(truth):
        targets[row, cause_index[cause]] = 1.0
    true_probability = np.asarray(
        [probabilities[row, cause_index[cause]] for row, cause in enumerate(truth)]
    )
    non_unknown = decision != "unknown"
    confidence = subset["cause_confidence"].to_numpy(float)
    correct = decision == truth
    unsafe = subset["evaluation_only_unsafe_silence"].to_numpy(bool)
    failure_truth = truth != "valid_suppression"
    failure_flag = decision != "valid_suppression"
    return {
        "protocol": protocol,
        "site": site,
        "model_role": model_role,
        "scenario": scenario,
        "repetition": repetition,
        "silence_rows": int(len(subset)),
        "supported_true_classes": int(len(supported)),
        "macro_f1_supported_classes": (
            float(f1_score(truth, decision, labels=supported, average="macro", zero_division=0))
            if supported
            else np.nan
        ),
        "balanced_accuracy_supported_classes": float(np.mean(recalls)) if recalls else np.nan,
        "selective_coverage": float(non_unknown.mean()) if len(subset) else np.nan,
        "selective_accuracy": (float(correct[non_unknown].mean()) if non_unknown.any() else np.nan),
        "unknown_fraction": float((decision == "unknown").mean()) if len(subset) else np.nan,
        "multiclass_brier_score": (
            float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))
            if len(subset)
            else np.nan
        ),
        "multiclass_log_loss": (
            float(-np.mean(np.log(np.clip(true_probability, 1.0e-9, 1.0))))
            if len(subset)
            else np.nan
        ),
        "decision_ece": _ece(confidence, correct.astype(float)) if len(subset) else np.nan,
        "mean_posterior_entropy": (
            float(subset["cause_posterior_entropy"].mean()) if len(subset) else np.nan
        ),
        "unsafe_silence_rows": int(unsafe.sum()),
        "unsafe_predicted_valid_suppression_rate": (
            float((decision[unsafe] == "valid_suppression").mean()) if unsafe.any() else 0.0
        ),
        "coarse_failure_detection_recall": (
            float(failure_flag[failure_truth].mean()) if failure_truth.any() else np.nan
        ),
        **{f"support_{cause}": int((truth == cause).sum()) for cause in KNOWN_CAUSES},
        **{
            f"recall_{cause}": (
                float((decision[truth == cause] == cause).mean())
                if np.any(truth == cause)
                else np.nan
            )
            for cause in KNOWN_CAUSES
        },
    }


def infer_ambiguity_causes(
    input_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute Task M across every frozen Task L receiver trace."""
    task_l, verified = verify_task_l_inputs(input_root)
    metrics: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    trace_records = [
        record
        for record in task_l["artifacts"]
        if all(record.get(key) for key in ("protocol", "site", "scenario"))
    ]
    for record in trace_records:
        input_path = _resolve_task_l_artifact(record, input_root)
        beliefs = pd.read_parquet(input_path)
        protocol = str(record["protocol"])
        site = str(record["site"])
        scenario = str(record["scenario"])
        model_role = str(record["model_role"])
        inferred = infer_causes(beliefs, config)
        inferred.insert(0, "cause_version", str(config["cause_version"]))
        inferred.insert(1, "protocol", protocol)
        inferred.insert(2, "site", site)
        inferred.insert(3, "model_role", model_role)
        inferred.insert(4, "scenario", scenario)
        for repetition, group in inferred.groupby("repetition", sort=False):
            metrics.append(
                cause_metrics(group, protocol, site, model_role, scenario, int(repetition))
            )
            silence = group[group["packet_arrived"] == 0]
            counts = silence.groupby(
                ["evaluation_only_true_cause", "cause_decision"], dropna=False
            ).size()
            for (true_cause, decision), count in counts.items():
                confusion.append(
                    {
                        "protocol": protocol,
                        "site": site,
                        "model_role": model_role,
                        "scenario": scenario,
                        "repetition": int(repetition),
                        "true_cause": str(true_cause),
                        "predicted_cause": str(decision),
                        "rows": int(count),
                    }
                )
        output_path = output_root / "posteriors" / protocol / site / f"{scenario}.parquet"
        _atomic_parquet(output_path, inferred)
        artifacts.append(
            {
                "protocol": protocol,
                "site": site,
                "model_role": model_role,
                "scenario": scenario,
                "path": str(output_path),
                "rows": int(len(inferred)),
                "sha256": sha256_file(output_path),
            }
        )

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = output_root / "cause_metrics.csv"
    _atomic_csv(metrics_path, metrics_frame)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})
    confusion_frame = pd.DataFrame(confusion)
    confusion_path = output_root / "cause_confusion.csv"
    _atomic_csv(confusion_path, confusion_frame)
    artifacts.append({"path": str(confusion_path), "sha256": sha256_file(confusion_path)})
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "M",
        "cause_version": str(config["cause_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_l_artifacts_reverified": verified,
        "task_l_receiver_version": task_l["receiver_version"],
        "inference_uses_oracle_fields": False,
        "inference_uses_event_truth": False,
        "inference_uses_future_measurements": False,
        "parameters_tuned_on_test_or_external_data": False,
        "algorithm_shared_across_sites_and_scenarios": True,
        "unknown_abstention_retained": True,
        "score_semantics": str(config["score_semantics"]),
        "probabilities_are_empirically_calibrated": bool(
            config["probabilities_are_empirically_calibrated"]
        ),
        "valid_suppression_has_hard_admissibility_gate": True,
        "exact_physical_cause_claimed": False,
        "exact_cause_accuracy_is_diagnostic_only": True,
        "metrics_rows": int(len(metrics_frame)),
        "confusion_rows": int(len(confusion_frame)),
        "artifacts": artifacts,
    }
    _atomic_json(output_root / "task_m_summary.json", summary)
    return summary
