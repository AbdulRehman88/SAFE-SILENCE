"""Task O criticality-preserving, order-robust relay fusion."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_silence.data.preprocess import sha256_file


class FusionError(RuntimeError):
    """Raised when Task O cannot preserve the relay-fusion contract."""


PROPOSED_POLICY = "associative_top_k"
REFERENCE_POLICIES = ("no_relay_fusion", "mean_relay_fusion")
ALL_POLICIES = (PROPOSED_POLICY, *REFERENCE_POLICIES)


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


def _resolve_artifact(record: dict[str, Any], root: Path) -> Path:
    declared = Path(str(record["path"]))
    if declared.exists():
        return declared
    if record.get("protocol") and record.get("site"):
        return root / "evidence" / str(record["protocol"]) / f"{record['site']}.parquet"
    return root / declared.name


def verify_task_j_inputs(task_j_root: Path) -> tuple[dict[str, Any], int]:
    """Reverify every frozen Task J artifact before forming relay batches."""
    summary_path = task_j_root / "task_j_summary.json"
    if not summary_path.exists():
        raise FusionError(f"Task O requires the Task J summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != "J" or summary.get("semantic_version") != "1.0":
        raise FusionError("Task O requires frozen Task J semantic contract v1.0")
    verified = 0
    for record in summary.get("artifacts", []):
        path = _resolve_artifact(record, task_j_root)
        if not path.exists() or sha256_file(path) != str(record.get("sha256", "")):
            raise FusionError(f"Task J artifact changed or is missing: {path}")
        verified += 1
    if not verified:
        raise FusionError("Task O found no verifiable Task J artifacts")
    return summary, verified


def _finite_max(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros(len(frame), dtype=float)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values[~np.isfinite(values)] = 0.0
    return np.max(values, axis=1)


def make_fusion_items(frame: pd.DataFrame, protocol: str, site: str) -> pd.DataFrame:
    """Map generated Task J packets to the relay's causal evidence contract."""
    required = {
        "timestamp",
        "window_key",
        "model_role",
        "node_action",
        "causal_critical_evidence",
        "semantic_payload_bytes",
        "validity_horizon_steps",
        "contract_age_steps",
        "maximum_standardized_innovation",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise FusionError(f"Task J trace lacks relay evidence: {sorted(missing)}")
    forbidden = [
        column
        for column in frame.columns
        if column.startswith(("_oracle_", "evaluation_only_"))
    ]
    if forbidden:
        raise FusionError(f"Forbidden fusion evidence supplied: {sorted(forbidden)}")

    generated = frame.loc[frame["node_action"].astype(str) != "silent"].copy()
    generated["timestamp"] = pd.to_datetime(generated["timestamp"], utc=True)
    generated = generated.sort_values(["model_role", "window_key", "timestamp"]).reset_index(
        drop=True
    )
    uncertainty_columns = [
        column for column in generated.columns if column.startswith("normalized_uncertainty_")
    ]
    uncertainty = np.clip(_finite_max(generated, uncertainty_columns), 0.0, 1.0)
    residual = pd.to_numeric(
        generated["maximum_standardized_innovation"], errors="coerce"
    ).to_numpy(float)
    residual[~np.isfinite(residual)] = 0.0
    residual_score = np.clip(residual / (1.0 + np.maximum(residual, 0.0)), 0.0, 1.0)
    horizon = pd.to_numeric(generated["validity_horizon_steps"], errors="coerce").fillna(0)
    age = pd.to_numeric(generated["contract_age_steps"], errors="coerce").fillna(0)
    remaining = np.maximum(horizon.to_numpy(float) - age.to_numpy(float), 0.0)
    deadline_urgency = 1.0 / (remaining + 1.0)

    items = pd.DataFrame(
        {
            "protocol": protocol,
            "site": site,
            "model_role": generated["model_role"].astype(str),
            "window_key": generated["window_key"].astype(str),
            "sensing_timestamp": generated["timestamp"],
            "source_id": site,
            "node_action": generated["node_action"].astype(str),
            "critical_evidence_score": generated["causal_critical_evidence"]
            .fillna(0)
            .astype(np.uint8),
            "environmental_uncertainty": uncertainty.astype(np.float32),
            "deadline_urgency": deadline_urgency.astype(np.float32),
            "residual_magnitude": residual_score.astype(np.float32),
            "payload_bytes": pd.to_numeric(
                generated["semantic_payload_bytes"], errors="raise"
            ).astype(np.int16),
        }
    )
    sequence = items.groupby(["model_role", "window_key"], sort=False).cumcount()
    items["source_sequence"] = sequence.astype(np.int64)
    items["item_id"] = [
        hashlib.sha256(
            f"{protocol}|{site}|{role}|{window}|{timestamp.isoformat()}|{seq}".encode()
        ).hexdigest()[:24]
        for role, window, timestamp, seq in zip(
            items["model_role"],
            items["window_key"],
            items["sensing_timestamp"],
            items["source_sequence"],
            strict=True,
        )
    ]
    return items


def priority_score(item: dict[str, Any], config: dict[str, Any]) -> float:
    """Compute the locked deterministic evidence-per-byte priority score."""
    weights = config["priority"]
    numerator = (
        float(weights["critical_evidence_weight"])
        * float(item["critical_evidence_score"])
        + float(weights["environmental_uncertainty_weight"])
        * float(item["environmental_uncertainty"])
        + float(weights["deadline_urgency_weight"]) * float(item["deadline_urgency"])
        + float(weights["residual_magnitude_weight"]) * float(item["residual_magnitude"])
    )
    if bool(weights["divide_by_payload_bytes"]):
        return numerator / max(int(item["payload_bytes"]), 1)
    return numerator


def _canonical_item(item: dict[str, Any]) -> str:
    values = {
        key: (value.isoformat() if isinstance(value, pd.Timestamp) else value)
        for key, value in item.items()
        if key != "priority_score"
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)


def _deduplicate(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    canonical: dict[str, str] = {}
    for source in items:
        item = dict(source)
        item_id = str(item["item_id"])
        encoded = _canonical_item(item)
        if item_id in canonical and canonical[item_id] != encoded:
            raise FusionError(f"Conflicting duplicate fusion item: {item_id}")
        canonical[item_id] = encoded
        unique[item_id] = item
    return list(unique.values())


def associative_top_k(
    items: Iterable[dict[str, Any]], k: int, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return a deterministic bounded summary; duplicates are idempotently removed."""
    if k < 1:
        raise FusionError("Top-K must be positive")
    unique = _deduplicate(items)
    for item in unique:
        item["sensing_timestamp"] = pd.Timestamp(item["sensing_timestamp"])
        item["priority_score"] = priority_score(item, config)
    return sorted(
        unique,
        key=lambda item: (
            -float(item["priority_score"]),
            item["sensing_timestamp"],
            str(item["source_id"]),
            str(item["item_id"]),
        ),
    )[:k]


def merge_top_k(
    left: Iterable[dict[str, Any]],
    right: Iterable[dict[str, Any]],
    k: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Associative binary merge operator F_K(A, B)."""
    return associative_top_k([*left, *right], k, config)


def _ordered_variants(
    items: list[dict[str, Any]], names: list[str]
) -> list[list[dict[str, Any]]]:
    variants: list[list[dict[str, Any]]] = []
    for name in names:
        if name == "original":
            variant = list(items)
        elif name == "reversed":
            variant = list(reversed(items))
        elif name == "rotated":
            variant = items[1:] + items[:1]
        elif name == "item_id_descending":
            variant = sorted(items, key=lambda item: str(item["item_id"]), reverse=True)
        else:
            raise FusionError(f"Unknown order-audit permutation: {name}")
        variants.append(variant)
    return variants


def order_audit(items: list[dict[str, Any]], k: int, config: dict[str, Any]) -> bool:
    """Verify order and grouping invariance for one relay batch."""
    expected = [item["item_id"] for item in associative_top_k(items, k, config)]
    names = [str(value) for value in config["order_audit"]["permutations"]]
    for variant in _ordered_variants(items, names):
        direct = [item["item_id"] for item in associative_top_k(variant, k, config)]
        midpoint = len(variant) // 2
        left = associative_top_k(variant[:midpoint], k, config)
        right = associative_top_k(variant[midpoint:], k, config)
        grouped = [item["item_id"] for item in merge_top_k(left, right, k, config)]
        if direct != expected or grouped != expected:
            return False
    return True


def _wire_bytes(
    policy: str, items: list[dict[str, Any]], retained: int, config: dict[str, Any]
) -> int:
    encoding = config["encoding_bytes"]
    header = int(encoding["relay_packet_header"])
    if policy == "no_relay_fusion":
        return int(sum(header + int(item["payload_bytes"]) for item in items))
    sources = len({str(item["source_id"]) for item in items})
    if policy == "mean_relay_fusion":
        return header + int(encoding["mean_summary_payload"])
    return (
        header
        + int(encoding["fixed_summary_metadata"])
        + retained * int(encoding["retained_item"])
        + sources * int(encoding["provenance_per_source"])
    )


def evaluate_batch(
    items: list[dict[str, Any]], batch_id: str, relay_sequence: int, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Fuse one causal relay batch and evaluate proposed/reference representations."""
    if not items:
        raise FusionError("Cannot fuse an empty relay batch")
    k = int(config["top_k"])
    retained = associative_top_k(items, k, config)
    retained_ids = {str(item["item_id"]) for item in retained}
    ranked_ids = {str(item["item_id"]): rank + 1 for rank, item in enumerate(retained)}
    critical = [item for item in items if bool(item["critical_evidence_score"])]
    highest_critical = associative_top_k(critical, 1, config)[0] if critical else None
    q_max = max(float(item["critical_evidence_score"]) for item in items)
    q_mean = float(np.mean([float(item["critical_evidence_score"]) for item in items]))
    u_max = max(float(item["environmental_uncertainty"]) for item in items)
    d_max = max(float(item["deadline_urgency"]) for item in items)
    earliest = min(pd.Timestamp(item["sensing_timestamp"]) for item in items)
    newest = max(pd.Timestamp(item["sensing_timestamp"]) for item in items)
    all_ids = {str(item["item_id"]) for item in items}
    all_sources = sorted({str(item["source_id"]) for item in items})
    canonical = "|".join(sorted(all_ids))
    crc = hashlib.sha256(canonical.encode()).hexdigest()[:8]
    proposed_bytes = _wire_bytes(PROPOSED_POLICY, items, len(retained), config)

    membership: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item["item_id"])
        membership.append(
            {
                "fusion_version": str(config["fusion_version"]),
                "batch_id": batch_id,
                "relay_sequence": relay_sequence,
                **item,
                "priority_score": priority_score(item, config),
                "retained": int(item_id in retained_ids),
                "retained_rank": ranked_ids.get(item_id, -1),
                "membership_reason": (
                    "top_k_critical" if item_id in retained_ids and item["critical_evidence_score"]
                    else "top_k_priority" if item_id in retained_ids
                    else "bounded_capacity_drop"
                ),
            }
        )

    summary = {
        "fusion_version": str(config["fusion_version"]),
        "batch_id": batch_id,
        "relay_sequence": relay_sequence,
        "protocol": str(items[0]["protocol"]),
        "site": str(items[0]["site"]),
        "model_role": str(items[0]["model_role"]),
        "window_key": str(items[0]["window_key"]),
        "relay_flush_timestamp": newest,
        "earliest_sensing_timestamp": earliest,
        "newest_sensing_timestamp": newest,
        "critical_evidence_max": q_max,
        "environmental_uncertainty_max": u_max,
        "deadline_urgency_max": d_max,
        "input_items": len(items),
        "retained_items": len(retained),
        "dropped_items": len(items) - len(retained),
        "expired_items": 0,
        "quality_rejected_items": 0,
        "source_ids_json": json.dumps(all_sources, separators=(",", ":")),
        "retained_item_ids_json": json.dumps(
            [str(item["item_id"]) for item in retained], separators=(",", ":")
        ),
        "input_set_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "relay_crc32_hex": crc,
        "encoded_wire_bytes": proposed_bytes,
        "order_audit_passed": int(order_audit(items, k, config)),
    }

    evaluations: list[dict[str, Any]] = []
    threshold = float(config["mean_reference"]["critical_evidence_threshold"])
    for policy in ALL_POLICIES:
        if policy == PROPOSED_POLICY:
            policy_retained = retained_ids
            detects_critical = q_max >= threshold
            maxima_preserved = True
            provenance_preserved = True
            audit_passed = bool(summary["order_audit_passed"])
        elif policy == "no_relay_fusion":
            policy_retained = all_ids
            detects_critical = q_max >= threshold
            maxima_preserved = True
            provenance_preserved = True
            audit_passed = True
        else:
            policy_retained = set()
            detects_critical = q_mean >= threshold
            maxima_preserved = False
            provenance_preserved = False
            audit_passed = True
        highest_preserved = (
            highest_critical is None or str(highest_critical["item_id"]) in policy_retained
        )
        evaluations.append(
            {
                "protocol": str(items[0]["protocol"]),
                "site": str(items[0]["site"]),
                "model_role": str(items[0]["model_role"]),
                "batch_id": batch_id,
                "policy": policy,
                "input_items": len(items),
                "critical_items": len(critical),
                "retained_critical_items": sum(
                    str(item["item_id"]) in policy_retained for item in critical
                ),
                "critical_batch": bool(critical),
                "highest_ranked_critical_preserved": bool(highest_preserved),
                "critical_evidence_detected": bool(detects_critical),
                "critical_maximum_preserved": maxima_preserved,
                "uncertainty_maximum_preserved": maxima_preserved,
                "deadline_extreme_preserved": maxima_preserved,
                "individual_provenance_preserved": provenance_preserved,
                "encoded_wire_bytes": _wire_bytes(
                    policy,
                    items,
                    len(retained) if policy == PROPOSED_POLICY else len(items),
                    config,
                ),
                "order_audit_passed": audit_passed,
            }
        )
    return membership, summary, evaluations


def _aggregate_metrics(evaluations: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["protocol", "site", "model_role", "policy"]
    for values, group in evaluations.groupby(keys, sort=True, dropna=False):
        critical_batches = group["critical_batch"].to_numpy(bool)
        critical_items = int(group["critical_items"].sum())
        wire = int(group["encoded_wire_bytes"].sum())
        no_fusion = evaluations.loc[
            (evaluations["protocol"] == values[0])
            & (evaluations["site"] == values[1])
            & (evaluations["model_role"] == values[2])
            & (evaluations["policy"] == "no_relay_fusion"),
            "encoded_wire_bytes",
        ].sum()
        records.append(
            {
                "protocol": values[0],
                "site": values[1],
                "model_role": values[2],
                "policy": values[3],
                "batches": int(len(group)),
                "input_items": int(group["input_items"].sum()),
                "critical_items": critical_items,
                "critical_item_retention_recall": (
                    float(group["retained_critical_items"].sum() / critical_items)
                    if critical_items
                    else np.nan
                ),
                "highest_ranked_critical_preservation_rate": (
                    float(group.loc[critical_batches, "highest_ranked_critical_preserved"].mean())
                    if critical_batches.any()
                    else np.nan
                ),
                "critical_batch_detection_recall": (
                    float(group.loc[critical_batches, "critical_evidence_detected"].mean())
                    if critical_batches.any()
                    else np.nan
                ),
                "critical_maximum_preservation_rate": float(
                    group["critical_maximum_preserved"].mean()
                ),
                "uncertainty_maximum_preservation_rate": float(
                    group["uncertainty_maximum_preserved"].mean()
                ),
                "deadline_extreme_preservation_rate": float(
                    group["deadline_extreme_preserved"].mean()
                ),
                "individual_provenance_preservation_rate": float(
                    group["individual_provenance_preserved"].mean()
                ),
                "order_audit_failures": int((~group["order_audit_passed"].to_numpy(bool)).sum()),
                "encoded_wire_bytes": wire,
                "wire_byte_reduction_vs_no_fusion": (
                    float(1.0 - wire / float(no_fusion)) if no_fusion else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def run_relay_fusion(
    task_j_root: Path, output_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Execute Task O as a component-level relay-fusion evaluation."""
    task_j, verified = verify_task_j_inputs(task_j_root)
    artifacts: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    rows_total = 0
    batches_total = 0
    trace_records = [
        record
        for record in task_j["artifacts"]
        if record.get("protocol") and record.get("site")
    ]
    for record in trace_records:
        protocol = str(record["protocol"])
        site = str(record["site"])
        frame = pd.read_parquet(_resolve_artifact(record, task_j_root))
        items = make_fusion_items(frame, protocol, site)
        memberships: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        relay_sequence = 0
        batch_size = int(config["relay_batch_items"])
        for _, group in items.groupby(["model_role", "window_key"], sort=False):
            records = group.to_dict("records")
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                batch_id = hashlib.sha256(
                    "|".join(str(item["item_id"]) for item in batch).encode()
                ).hexdigest()[:24]
                membership, summary, batch_evaluations = evaluate_batch(
                    batch, batch_id, relay_sequence, config
                )
                memberships.extend(membership)
                summaries.append(summary)
                evaluations.extend(batch_evaluations)
                relay_sequence += 1
        membership_frame = pd.DataFrame(memberships)
        summary_frame = pd.DataFrame(summaries)
        membership_path = output_root / "membership" / protocol / f"{site}.parquet"
        summary_path = output_root / "summaries" / protocol / f"{site}.parquet"
        _atomic_parquet(membership_path, membership_frame)
        _atomic_parquet(summary_path, summary_frame)
        for kind, path, rows in (
            ("membership", membership_path, len(membership_frame)),
            ("summary", summary_path, len(summary_frame)),
        ):
            artifacts.append(
                {
                    "protocol": protocol,
                    "site": site,
                    "artifact_kind": kind,
                    "path": str(path),
                    "rows": int(rows),
                    "sha256": sha256_file(path),
                }
            )
        rows_total += len(membership_frame)
        batches_total += len(summary_frame)

    evaluation_frame = pd.DataFrame(evaluations)
    metrics = _aggregate_metrics(evaluation_frame)
    metrics_path = output_root / "fusion_metrics.csv"
    _atomic_csv(metrics_path, metrics)
    artifacts.append({"path": str(metrics_path), "sha256": sha256_file(metrics_path)})
    if int(metrics["order_audit_failures"].sum()) != 0:
        raise FusionError("Associative top-K failed the order/grouping audit")

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "O",
        "fusion_version": str(config["fusion_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_j_artifacts_reverified": verified,
        "task_j_semantic_version": task_j["semantic_version"],
        "proposed_method": PROPOSED_POLICY,
        "reference_methods": list(REFERENCE_POLICIES),
        "top_k": int(config["top_k"]),
        "relay_batch_items": int(config["relay_batch_items"]),
        "generated_packet_items": int(rows_total),
        "relay_batches": int(batches_total),
        "temporal_protocols_pooled": False,
        "uses_future_measurements": False,
        "uses_oracle_fields": False,
        "parameters_tuned_on_test_or_external_data": False,
        "deterministic_total_order": True,
        "commutative_associative_idempotent_merge": True,
        "order_audit_failures": 0,
        "critical_maxima_carried_separately": True,
        "uncertainty_maxima_carried_separately": True,
        "deadline_extreme_carried_separately": True,
        "exact_provenance_used": True,
        "component_evaluation_only": True,
        "acoustic_delivery_replayed": False,
        "energy_evaluation_deferred_to_task_p": True,
        "artifacts": artifacts,
    }
    task_summary_path = output_root / "task_o_summary.json"
    _atomic_json(task_summary_path, summary)
    return summary
