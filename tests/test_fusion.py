import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safe_silence.config import load_config
from safe_silence.data.preprocess import sha256_file
from safe_silence.protocol.fusion import (
    FusionError,
    associative_top_k,
    evaluate_batch,
    make_fusion_items,
    merge_top_k,
    order_audit,
    priority_score,
    run_relay_fusion,
    verify_task_j_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")["fusion"]


def _item(index: int, critical: int = 0, source: str = "S01") -> dict:
    return {
        "protocol": "primary_15min",
        "site": "S01",
        "model_role": "external_test",
        "window_key": "w0",
        "sensing_timestamp": pd.Timestamp("2025-01-01", tz="UTC")
        + pd.Timedelta(minutes=15 * index),
        "source_id": source,
        "source_sequence": index,
        "item_id": f"item-{index:02d}",
        "node_action": "alert" if critical else "innovation",
        "critical_evidence_score": critical,
        "environmental_uncertainty": 0.2 + index / 100,
        "deadline_urgency": 0.25,
        "residual_magnitude": 0.3,
        "payload_bytes": 40 if critical else 32,
    }


def _semantic_frame(rows: int = 10) -> pd.DataFrame:
    critical = np.zeros(rows, dtype=np.uint8)
    critical[5] = 1
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC"),
            "window_key": ["w0"] * rows,
            "model_role": ["external_test"] * rows,
            "node_action": ["heartbeat", *(["innovation"] * (rows - 2)), "alert"],
            "causal_critical_evidence": critical,
            "semantic_payload_bytes": [8, *([32] * (rows - 2)), 40],
            "validity_horizon_steps": [4] * rows,
            "contract_age_steps": [0] * rows,
            "maximum_standardized_innovation": np.linspace(0.0, 5.0, rows),
            "normalized_uncertainty_temperature": np.linspace(0.1, 0.8, rows),
            "normalized_uncertainty_salinity": np.linspace(0.2, 0.7, rows),
        }
    )


def test_task_o_config_locks_safety_contract() -> None:
    config = _config()
    assert config["method"] == "associative_top_k"
    assert config["top_k"] == 4
    assert config["relay_batch_items"] > config["top_k"]
    assert not config["uses_future_measurements"]
    assert not config["oracle_fields_are_fusion_inputs"]
    assert config["critical_evidence_override"]
    assert config["exact_provenance"]


def test_critical_alert_outranks_lower_ranked_noncritical_items() -> None:
    config = _config()
    critical = _item(0, critical=1)
    ordinary = _item(1, critical=0)
    ordinary["environmental_uncertainty"] = 1.0
    ordinary["deadline_urgency"] = 1.0
    ordinary["residual_magnitude"] = 1.0
    ordinary["payload_bytes"] = 8
    assert priority_score(critical, config) > priority_score(ordinary, config)


def test_top_k_is_commutative_associative_and_idempotent() -> None:
    config = _config()
    items = [_item(index, critical=int(index == 6), source=f"S{index % 3}") for index in range(8)]
    direct = associative_top_k(items, 4, config)
    reversed_result = associative_top_k(reversed(items), 4, config)
    grouped_left = merge_top_k(items[:3], items[3:6], 4, config)
    grouped = merge_top_k(grouped_left, items[6:], 4, config)
    duplicate = associative_top_k([*items, items[0], items[6]], 4, config)
    expected = [item["item_id"] for item in direct]
    assert [item["item_id"] for item in reversed_result] == expected
    assert [item["item_id"] for item in grouped] == expected
    assert [item["item_id"] for item in duplicate] == expected
    assert order_audit(items, 4, config)


def test_top_k_preserves_rare_critical_item_that_mean_dilutes() -> None:
    config = _config()
    items = [_item(index, critical=int(index == 7)) for index in range(8)]
    membership, summary, evaluations = evaluate_batch(items, "batch", 0, config)
    by_policy = {row["policy"]: row for row in evaluations}
    assert any(row["retained"] and row["critical_evidence_score"] for row in membership)
    assert summary["critical_evidence_max"] == 1.0
    assert by_policy["associative_top_k"]["critical_evidence_detected"]
    assert not by_policy["mean_relay_fusion"]["critical_evidence_detected"]
    assert by_policy["associative_top_k"]["highest_ranked_critical_preserved"]


def test_fusion_rejects_oracle_columns() -> None:
    frame = _semantic_frame()
    frame["_oracle_link_state"] = "good"
    with pytest.raises(FusionError, match="Forbidden fusion evidence"):
        make_fusion_items(frame, "primary_15min", "S01")


def test_task_j_integrity_rejects_tampering(tmp_path: Path) -> None:
    task_j = tmp_path / "task_j"
    artifact = task_j / "evidence" / "primary_15min" / "S01.parquet"
    artifact.parent.mkdir(parents=True)
    _semantic_frame().to_parquet(artifact, index=False)
    summary = {
        "task": "J",
        "semantic_version": "1.0",
        "artifacts": [
            {
                "protocol": "primary_15min",
                "site": "S01",
                "path": str(artifact),
                "sha256": sha256_file(artifact),
            }
        ],
    }
    (task_j / "task_j_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    verify_task_j_inputs(task_j)
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(FusionError, match="changed or is missing"):
        verify_task_j_inputs(task_j)


def test_synthetic_end_to_end_fusion_run(tmp_path: Path) -> None:
    task_j = tmp_path / "task_j"
    output = tmp_path / "task_o"
    artifact = task_j / "evidence" / "primary_15min" / "S01.parquet"
    artifact.parent.mkdir(parents=True)
    _semantic_frame().to_parquet(artifact, index=False)
    record = {
        "protocol": "primary_15min",
        "site": "S01",
        "path": str(artifact),
        "sha256": sha256_file(artifact),
    }
    (task_j / "task_j_summary.json").write_text(
        json.dumps({"task": "J", "semantic_version": "1.0", "artifacts": [record]}),
        encoding="utf-8",
    )
    summary = run_relay_fusion(task_j, output, _config())
    assert summary["task"] == "O"
    assert summary["order_audit_failures"] == 0
    metrics = pd.read_csv(output / "fusion_metrics.csv")
    assert set(metrics["policy"]) == {
        "associative_top_k",
        "mean_relay_fusion",
        "no_relay_fusion",
    }
    assert (output / "membership" / "primary_15min" / "S01.parquet").exists()
    assert (output / "summaries" / "primary_15min" / "S01.parquet").exists()
