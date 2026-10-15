from copy import deepcopy
from pathlib import Path

import pytest

from safe_silence.config import ConfigurationError, load_config, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_config_is_valid() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    assert config["project"]["name"] == "SAFE-SILENCE"
    assert config["experiments"]["primary_endpoint"] == "critical_false_trust_rate"


def test_external_sites_cannot_leak_into_training() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    invalid = deepcopy(config)
    invalid["events"]["split"]["external_sites_are_test_only"] = False
    with pytest.raises(ConfigurationError, match="External sites"):
        validate_config(invalid)


def test_primary_outages_cannot_be_event_correlated() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    invalid = deepcopy(config)
    invalid["network"]["outage"]["event_correlated_outages_primary"] = True
    with pytest.raises(ConfigurationError, match="stress tests"):
        validate_config(invalid)


def test_unknown_failure_class_is_mandatory() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    invalid = deepcopy(config)
    invalid["protocol"]["dual_belief"]["unknown_cause_class"] = False
    with pytest.raises(ConfigurationError, match="UNKNOWN"):
        validate_config(invalid)


def test_availability_selection_cannot_use_measurement_values() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    invalid = deepcopy(config)
    invalid["data"]["audit"]["availability_windows"]["selection_uses_measurement_values"] = True
    with pytest.raises(ConfigurationError, match="must never inspect"):
        validate_config(invalid)


def test_event_thresholds_must_be_training_only() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    invalid = deepcopy(config)
    invalid["events"]["split"]["calibration_split"] = "validation"
    with pytest.raises(ConfigurationError, match="train only"):
        validate_config(invalid)


def test_overlapping_temporal_protocols_cannot_be_pooled() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    invalid = deepcopy(config)
    invalid["events"]["reporting"]["pool_temporal_protocols"] = True
    with pytest.raises(ConfigurationError, match="must not be pooled"):
        validate_config(invalid)


def test_predictor_calibration_cannot_control_early_stopping() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    invalid = deepcopy(config)
    invalid["model"]["uncertainty"]["calibration_is_never_used_for_early_stopping"] = False
    with pytest.raises(ConfigurationError, match="cannot be used for early stopping"):
        validate_config(invalid)
