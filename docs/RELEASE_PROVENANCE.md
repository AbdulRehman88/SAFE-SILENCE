# Experiment and software provenance

SAFE-SILENCE provides computational research code and scientific result visualization.

## Source and experiment records

The release protocol records the distributed source files, scientific configurations,
calibration inputs, random seeds, and resource budgets. The archived experiment identity
identifies the experiment snapshot associated with the result checkpoints. The experiment
snapshot and software distribution are distinct provenance records; their source trees
are not asserted to be byte-identical.

`results/task_x/protocol_lock.json` contains the release input checksums. Its
`release_projection.archived_experiment_identity` links to the identity stored in the
14 archived confirmation checkpoints. The full experiment snapshot is preserved
separately by the author. The public protocol contains the scientific execution settings.

## Verification

Verification checks 59 release inputs, 42 cell artifacts, and 420 unique seeds. Statistical
reproduction evaluates all 210 primary hypotheses, compares the reference tables, and
generates the result figures in a separate output directory. The development simulation
check executes both protocols on CE02 tuning data with two paired repetitions each.
The three reference statistical tables reproduce exactly in the verified Windows environment.

The project test suite contains 147 passing tests in that environment. A complete
420-realization simulation rerun and raw-observation model retraining are separate
execution scopes and are not claimed by statistical reproduction or smoke testing.

`provenance/release_manifest.json` records distributed file hashes, excluding itself,
the virtual environment, Git metadata, caches, and reproduction output directories.
