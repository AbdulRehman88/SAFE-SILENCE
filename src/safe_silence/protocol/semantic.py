"""Task J causal semantic innovations and bounded-silence validity contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_silence.data.preprocess import sha256_file
from safe_silence.models.predictor import verify_task_h_inputs


class SemanticEvidenceError(RuntimeError):
    """Raised when Task J cannot preserve its causal or integrity contract."""


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


def _resolve_task_i_artifact(
    record: dict[str, Any], prediction_root: Path, artifact_root: Path
) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    name = declared.name
    protocol = record.get("protocol")
    if protocol and record.get("site"):
        return prediction_root / "predictions" / str(protocol) / name
    if protocol:
        return artifact_root / str(protocol) / name
    return prediction_root / name


def verify_task_i_inputs(prediction_root: Path, artifact_root: Path) -> tuple[dict[str, Any], int]:
    """Reverify every Task I artifact, supporting a relocated project root."""
    summary_path = prediction_root / "task_i_summary.json"
    if not summary_path.exists():
        raise SemanticEvidenceError(f"Missing Task I summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != "I" or summary.get("predictor_version") != "1.1":
        raise SemanticEvidenceError("Task J requires the frozen Task I predictor contract v1.1")
    verified = 0
    for record in summary.get("artifacts", []):
        if not record.get("path") or not record.get("sha256"):
            raise SemanticEvidenceError("Task I artifact lacks path or SHA-256")
        path = _resolve_task_i_artifact(record, prediction_root, artifact_root)
        if not path.exists() or sha256_file(path) != str(record["sha256"]):
            raise SemanticEvidenceError(f"Task I artifact changed or is missing: {path}")
        verified += 1
    if verified == 0:
        raise SemanticEvidenceError("Task I summary contains no verifiable artifacts")
    return summary, verified


def _soft_saturation(value: np.ndarray, scale: float, clip: float) -> np.ndarray:
    finite = np.nan_to_num(value, nan=0.0, posinf=clip, neginf=0.0)
    finite = np.clip(finite, 0.0, clip)
    return 1.0 - np.exp(-finite / scale)


def compute_semantic_evidence(
    frame: pd.DataFrame,
    channels: list[str],
    training_scales: dict[str, float],
    config: dict[str, Any],
    age_cap_steps: int,
) -> pd.DataFrame:
    """Construct dimensionless, causal evidence available locally at sensing time."""
    result = frame.copy()
    evidence = config["evidence"]
    weights = evidence["channel_weights"]
    epsilon = float(evidence["numerical_epsilon"])
    clip = float(evidence["clip_dimensionless_evidence_at"])
    channel_risk_columns: list[str] = []

    for channel in channels:
        required = {
            f"actual_{channel}",
            f"prediction_{channel}",
            f"uncertainty_std_{channel}",
            f"lower_{channel}",
            f"upper_{channel}",
            f"target_observed_{channel}",
            f"age_steps_{channel}",
        }
        missing = required.difference(result.columns)
        if missing:
            raise SemanticEvidenceError(
                f"Semantic evidence for {channel} lacks columns: {sorted(missing)}"
            )
        actual = pd.to_numeric(result[f"actual_{channel}"], errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(result[f"prediction_{channel}"], errors="coerce").to_numpy(float)
        standard_deviation = pd.to_numeric(
            result[f"uncertainty_std_{channel}"], errors="coerce"
        ).to_numpy(float)
        lower = pd.to_numeric(result[f"lower_{channel}"], errors="coerce").to_numpy(float)
        upper = pd.to_numeric(result[f"upper_{channel}"], errors="coerce").to_numpy(float)
        observed = result[f"target_observed_{channel}"].fillna(0).to_numpy(bool)
        observed &= np.isfinite(actual) & np.isfinite(prediction)
        scale = float(training_scales[channel])
        uncertainty_floor = max(
            epsilon,
            float(evidence["uncertainty_floor_fraction_of_training_scale"]) * scale,
        )
        denominator = np.maximum(np.abs(standard_deviation), uncertainty_floor)
        innovation = np.full(len(result), np.nan, dtype=float)
        innovation[observed] = (
            np.abs(actual[observed] - prediction[observed]) / denominator[observed]
        )
        half_width = np.maximum((upper - lower) / 2.0, uncertainty_floor)
        breach_amount = np.maximum(np.maximum(lower - actual, actual - upper), 0.0)
        breach = np.full(len(result), np.nan, dtype=float)
        breach[observed] = breach_amount[observed] / half_width[observed]
        normalized_uncertainty = np.maximum(standard_deviation, 0.0) / max(scale, epsilon)
        age = pd.to_numeric(result[f"age_steps_{channel}"], errors="coerce").to_numpy(float)
        age_risk = np.clip(age / max(int(age_cap_steps), 1), 0.0, 1.0)
        age_risk[~observed] = np.maximum(age_risk[~observed], 1.0)

        innovation_risk = _soft_saturation(
            innovation, float(evidence["innovation_soft_scale"]), clip
        )
        breach_risk = _soft_saturation(breach, float(evidence["breach_soft_scale"]), clip)
        uncertainty_risk = _soft_saturation(
            normalized_uncertainty,
            float(evidence["uncertainty_soft_scale"]),
            clip,
        )
        channel_risk = (
            float(weights["innovation"]) * innovation_risk
            + float(weights["interval_breach"]) * breach_risk
            + float(weights["predictive_uncertainty"]) * uncertainty_risk
            + float(weights["observation_age"]) * age_risk
        )
        channel_risk = np.clip(channel_risk, 0.0, 1.0)
        result[f"standardized_innovation_{channel}"] = innovation.astype(np.float32)
        result[f"interval_breach_ratio_{channel}"] = breach.astype(np.float32)
        result[f"normalized_uncertainty_{channel}"] = normalized_uncertainty.astype(np.float32)
        result[f"semantic_channel_risk_{channel}"] = channel_risk.astype(np.float32)
        channel_risk_columns.append(f"semantic_channel_risk_{channel}")

    risk_matrix = result[channel_risk_columns].to_numpy(dtype=float)
    result["environmental_semantic_risk"] = np.nanmax(risk_matrix, axis=1).astype(np.float32)
    if "label_available_event" not in result or "valid_available_event" not in result:
        raise SemanticEvidenceError("Task H available-event evidence is missing")
    valid_event = result["valid_available_event"].fillna(0).to_numpy(bool)
    event = result["label_available_event"].fillna(0).to_numpy(bool) & valid_event
    result["causal_critical_evidence"] = event.astype(np.uint8)
    semantic_risk = result["environmental_semantic_risk"].to_numpy(dtype=float)
    if bool(evidence["critical_evidence_override"]):
        semantic_risk = np.maximum(semantic_risk, event.astype(float))
    result["semantic_risk"] = np.clip(semantic_risk, 0.0, 1.0).astype(np.float32)
    innovation_columns = [f"standardized_innovation_{channel}" for channel in channels]
    breach_columns = [f"interval_breach_ratio_{channel}" for channel in channels]
    result["maximum_standardized_innovation"] = np.nanmax(
        result[innovation_columns].to_numpy(float), axis=1
    ).astype(np.float32)
    result["maximum_interval_breach_ratio"] = np.nanmax(
        result[breach_columns].to_numpy(float), axis=1
    ).astype(np.float32)
    return result


def validity_horizon(risk: float, config: dict[str, Any]) -> int:
    bounded = config["bounded_validity"]
    minimum = int(bounded["minimum_horizon_steps"])
    maximum = int(bounded["maximum_horizon_steps"])
    if minimum < 1 or maximum < minimum:
        raise SemanticEvidenceError("Invalid bounded-validity horizon limits")
    horizon = int(math.ceil(maximum * (1.0 - float(np.clip(risk, 0.0, 1.0)))))
    return min(max(horizon, minimum), maximum)


def apply_nominal_policy(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply the a-priori action policy independently within each immutable window."""
    required = {
        "window_key",
        "timestamp",
        "semantic_risk",
        "causal_critical_evidence",
        "maximum_interval_breach_ratio",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise SemanticEvidenceError(f"Nominal policy lacks columns: {sorted(missing)}")
    policy = config["nominal_policy"]
    bounded = config["bounded_validity"]
    payloads = config["payload_bytes"]
    output_groups: list[pd.DataFrame] = []
    contract_id = 0
    for _, group in frame.groupby("window_key", sort=False):
        group = group.sort_values("timestamp").copy()
        actions: list[str] = []
        reasons: list[str] = []
        horizons: list[int] = []
        contract_ids: list[int] = []
        contract_ages: list[int] = []
        last_transmission_age: int | None = None
        active_horizon: int | None = None
        for row in group.itertuples(index=False):
            risk = float(row.semantic_risk)
            critical = bool(row.causal_critical_evidence)
            breach = float(row.maximum_interval_breach_ratio)
            dynamic_horizon = validity_horizon(risk, config)
            elapsed = None if last_transmission_age is None else last_transmission_age + 1
            if critical:
                action, reason = "alert", "critical_event"
            elif risk >= float(policy["alert_threshold"]) or breach >= float(
                policy["extreme_interval_breach_ratio"]
            ):
                action, reason = "alert", "extreme_risk"
            elif risk >= float(policy["innovation_threshold"]):
                action, reason = "innovation", "semantic_innovation"
            elif last_transmission_age is None:
                action, reason = "heartbeat", "contract_bootstrap"
            elif active_horizon is not None and elapsed is not None and elapsed >= active_horizon:
                action, reason = "heartbeat", "validity_expiry"
            elif elapsed is not None and elapsed >= int(bounded["heartbeat_interval_steps"]):
                action, reason = "heartbeat", "heartbeat_expiry"
            else:
                action, reason = "silent", "bounded_silence"

            if action == "silent":
                last_transmission_age = int(elapsed or 0)
                advertised_horizon = int(active_horizon or dynamic_horizon)
            else:
                contract_id += 1
                last_transmission_age = 0
                active_horizon = dynamic_horizon
                advertised_horizon = (
                    int(bounded["alert_advertised_horizon_steps"])
                    if action == "alert"
                    else dynamic_horizon
                )
            actions.append(action)
            reasons.append(reason)
            horizons.append(advertised_horizon)
            contract_ids.append(contract_id)
            contract_ages.append(int(last_transmission_age))
        group["node_action"] = actions
        group["decision_reason"] = reasons
        group["validity_horizon_steps"] = np.asarray(horizons, dtype=np.int16)
        group["silence_contract_id"] = np.asarray(contract_ids, dtype=np.int64)
        group["contract_age_steps"] = np.asarray(contract_ages, dtype=np.int16)
        group["intentional_silence"] = (group["node_action"] == "silent").astype(np.uint8)
        group["semantic_payload_bytes"] = group["node_action"].map(payloads).astype(np.int16)
        output_groups.append(group)
    return pd.concat(output_groups, ignore_index=True)


def _decision_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_columns = ["protocol", "site", "model_role"]
    raw_bytes = int(config["payload_bytes"]["periodic_raw_reference"])
    for keys, group in frame.groupby(group_columns, sort=False):
        transmitted = group["node_action"] != "silent"
        valid_truth = group["valid_available_event"].fillna(0).to_numpy(bool)
        critical = group["label_available_event"].fillna(0).to_numpy(bool) & valid_truth
        alert = group["node_action"].to_numpy() == "alert"
        semantic_bytes = int(group["semantic_payload_bytes"].sum())
        periodic_bytes = int(len(group) * raw_bytes)
        row: dict[str, Any] = {
            "protocol": keys[0],
            "site": keys[1],
            "model_role": keys[2],
            "rows": int(len(group)),
            "silent": int((group["node_action"] == "silent").sum()),
            "heartbeat": int((group["node_action"] == "heartbeat").sum()),
            "innovation": int((group["node_action"] == "innovation").sum()),
            "alert": int((group["node_action"] == "alert").sum()),
            "transmission_fraction": float(transmitted.mean()),
            "intentional_silence_fraction": float((~transmitted).mean()),
            "semantic_payload_bytes": semantic_bytes,
            "periodic_raw_reference_bytes": periodic_bytes,
            "payload_reduction_fraction": float(1.0 - semantic_bytes / periodic_bytes),
            "available_critical_rows": int(critical.sum()),
            "pre_network_critical_transmission_recall": (
                float(transmitted.to_numpy()[critical].mean()) if critical.any() else np.nan
            ),
            "pre_network_critical_alert_recall": (
                float(alert[critical].mean()) if critical.any() else np.nan
            ),
            "noncritical_alert_fraction": (
                float(alert[valid_truth & ~critical].mean())
                if (valid_truth & ~critical).any()
                else np.nan
            ),
            "mean_semantic_risk": float(group["semantic_risk"].mean()),
            "p95_semantic_risk": float(group["semantic_risk"].quantile(0.95)),
        }
        rows.append(row)
    return rows


def _load_training_scales(
    artifact_root: Path, protocol: str, channels: list[str]
) -> dict[str, float]:
    path = artifact_root / protocol / "scaler.json"
    if not path.exists():
        raise SemanticEvidenceError(f"Missing Task I scaler: {path}")
    with path.open("r", encoding="utf-8") as handle:
        scaler = json.load(handle)
    try:
        return {channel: float(scaler["channels"][channel]["scale"]) for channel in channels}
    except (KeyError, TypeError, ValueError) as exc:
        raise SemanticEvidenceError(f"Invalid Task I scaler schema: {path}") from exc


def _join_causal_task_h(
    predictions: pd.DataFrame,
    labeled: pd.DataFrame,
    channels: list[str],
) -> pd.DataFrame:
    keys = ["window_key", "timestamp"]
    for frame in (predictions, labeled):
        missing = set(keys).difference(frame.columns)
        if missing:
            raise SemanticEvidenceError(f"Task H/Task I join lacks keys: {sorted(missing)}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if labeled.duplicated(keys).any():
        raise SemanticEvidenceError("Task H has duplicate causal join keys")
    event_columns = [
        column
        for column in labeled.columns
        if column.startswith("label_") or column.startswith("valid_")
    ]
    source_columns = keys + [f"age_steps_{channel}" for channel in channels] + event_columns
    joined = predictions.drop(
        columns=[column for column in event_columns if column in predictions.columns]
    ).merge(labeled[source_columns], on=keys, how="left", validate="many_to_one")
    if joined["valid_available_event"].isna().any():
        raise SemanticEvidenceError(
            "Task I predictions could not be fully joined to causal Task H rows"
        )
    return joined


def build_semantic_evidence(
    prediction_root: Path,
    prediction_artifact_root: Path,
    event_root: Path,
    output_root: Path,
    semantic_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute Task J without fitting or consulting future observations."""
    _, task_i_verified = verify_task_i_inputs(prediction_root, prediction_artifact_root)
    task_h_summary, task_h_verified = verify_task_h_inputs(event_root)
    channels = [str(value) for value in semantic_config["channels"]]
    all_metrics: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    rows_total = 0
    protocol_summaries: dict[str, Any] = {}

    for protocol, protocol_config in preprocessing_config["protocols"].items():
        scales = _load_training_scales(prediction_artifact_root, protocol, channels)
        sites = [protocol_config["development_site"], *protocol_config["external_sites"]]
        protocol_rows = 0
        for site in sites:
            prediction_path = prediction_root / "predictions" / protocol / f"{site}.parquet"
            event_path = event_root / protocol / f"{site}.parquet"
            if not prediction_path.exists() or not event_path.exists():
                raise SemanticEvidenceError(
                    f"Missing Task I/Task H site input for {protocol}/{site}"
                )
            predictions = pd.read_parquet(prediction_path)
            labeled = pd.read_parquet(event_path)
            joined = _join_causal_task_h(predictions, labeled, channels)
            evidence = compute_semantic_evidence(
                joined,
                channels,
                scales,
                semantic_config,
                int(preprocessing_config["age_cap_steps"]),
            )
            decisions = apply_nominal_policy(evidence, semantic_config)
            decisions.insert(0, "semantic_version", str(semantic_config["semantic_version"]))
            output_path = output_root / "evidence" / protocol / f"{site}.parquet"
            _atomic_parquet(output_path, decisions)
            all_metrics.extend(_decision_metrics(decisions, semantic_config))
            artifacts.append(
                {
                    "protocol": protocol,
                    "site": site,
                    "path": str(output_path),
                    "rows": int(len(decisions)),
                    "sha256": sha256_file(output_path),
                }
            )
            rows_total += len(decisions)
            protocol_rows += len(decisions)
        protocol_summaries[protocol] = {
            "grid_minutes": int(protocol_config["grid_minutes"]),
            "sites": sites,
            "rows": int(protocol_rows),
        }

    metrics = pd.DataFrame(all_metrics)
    metrics_path = output_root / "semantic_decision_metrics.csv"
    _atomic_csv(metrics_path, metrics)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})
    config_hash = hashlib.sha256(
        json.dumps(semantic_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "J",
        "semantic_version": str(semantic_config["semantic_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_i_artifacts_reverified": task_i_verified,
        "task_h_artifacts_reverified": task_h_verified,
        "task_h_labeling_version": task_h_summary["labeling_version"],
        "uses_current_causal_observation_at_sensor": True,
        "uses_future_measurements": False,
        "event_evidence_uses_frozen_training_calibrated_rules": True,
        "policy_thresholds_tuned_on_test_or_external_data": False,
        "temporal_protocols_may_be_pooled": False,
        "critical_evidence_overrides_compression": True,
        "payload_bytes_exclude_network_headers": True,
        "event_recall_is_pre_network_upper_bound": True,
        "rows": int(rows_total),
        "protocols": protocol_summaries,
        "artifacts": artifacts,
    }
    summary_path = output_root / "task_j_summary.json"
    _atomic_json(summary_path, summary)
    return summary
