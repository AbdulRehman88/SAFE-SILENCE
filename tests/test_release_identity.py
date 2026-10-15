from copy import deepcopy
import pytest
from safe_silence.evaluation.extension import hash_object, protocol_identity


def test_release_projection_preserves_archive_identity_without_redefining_source():
    original = {"seed_ledger": [1, 2], "configs": {"network": {"rate": 5}}}
    projected = deepcopy(original)
    identity = hash_object(original)
    projected["release_projection"] = {"schema_version": 1, "archived_experiment_identity": identity}
    assert protocol_identity(projected) == identity
    assert hash_object(projected) != identity
    assert protocol_identity(original) == identity


@pytest.mark.parametrize("metadata", [{"schema_version": 2}, {"schema_version": 1, "archived_experiment_identity": "bad"}])
def test_invalid_projection_identity_is_rejected(metadata):
    with pytest.raises(ValueError):
        protocol_identity({"release_projection": metadata})
