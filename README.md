# SAFE-SILENCE

Receiver-verifiable semantic telemetry for underwater digital twins.

Companion software for **When Silence Is Ambiguous: Receiver-Verifiable Semantic Telemetry for Underwater Digital Twins**.

The pipeline includes ocean-data preparation, predictive semantic evidence, simulated acoustic transport, receiver trust, relay fusion, scheduling, bounded recovery, and paired statistical evaluation.

Results describe simulated communication outcomes and modeled communication energy. Field performance and digital-twin fidelity were not evaluated.

## Contents

- src/safe_silence/: scientific implementations and reporting.
- configs/: experiment and figure configurations.
- scripts/: acquisition, processing, modeling, simulation, analysis, and verification.
- tests/: project tests.
- docs/: scientific protocol documentation.
- provenance/: environment records, file selection, and integrity manifests.
- data/, artifacts/, results/: companion scientific evidence, excluded from ordinary Git commits.
- reproduction/: generated verification outputs, excluded from Git.

## Verified environment

The verified Windows environment uses Python 3.11.9 and PyTorch 2.13.0+cu126. Editable installation, dependency checks, configuration validation, and all 147 project tests pass without skips.

requirements-observed-windows.txt records the package versions used for the verified installation. PyTorch is installed separately.

The supplied requirements snapshot is the installation reference.

Other platforms and dependency versions have not been certified.

## Installation

Open PowerShell in the repository root with Python 3.11 installed.

Run each command separately. Stop if any command fails.

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install pip==26.2.1 setuptools==84.0.0
    .\.venv\Scripts\python.exe -m pip install -r requirements-observed-windows.txt
    .\.venv\Scripts\python.exe -m pip install --no-deps torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    .\.venv\Scripts\python.exe -m pip install --no-deps -e .
    .\.venv\Scripts\python.exe -m pip check
    .\.venv\Scripts\python.exe -m pytest -q

These commands describe the verified Windows installation. GPU availability must be checked separately before model training.

## Restore scientific evidence

Statistical reproduction and simulation checks require the companion evidence archive.

Restore its data/, artifacts/, and results/ directories under the repository root, preserving relative paths. A source-only checkout does not contain these ignored directories.

Do not modify or regenerate the frozen protocol lock to bypass integrity failures.

## Reproduce statistics and manuscript figures

From the repository root:

    .\.venv\Scripts\python.exe scripts\27_verify_release.py

The command verifies the scientific release inputs, 42 cell artifacts, and 420 unique seeds; reanalyzes archived confirmation cells; compares reference tables; and regenerates manuscript result figures.

Expected table sizes:

- primary_comparisons.csv: 210 rows.
- aggregate_effects.csv: 30 rows.
- resource_consumption.csv: 18 rows.

The verified Windows execution reproduced all three tables with a maximum absolute numeric difference of zero.

Outputs, including the verification report and figures, are written under reproduction/release_check/. Frozen evidence is preserved.

This command reanalyzes archived outputs. It does not rerun complete simulations or retrain models.

## Execute the development simulation check

    .\.venv\Scripts\python.exe scripts\28_check_simulation.py

This executes the F2 burst scenario for both protocols using CE02 tuning data and two paired repetitions per protocol. It exercises the network, six trust variants, three scheduler profiles, and recovery.

Outputs are written under reproduction/simulation_smoke/. Frozen evidence is checked before and after execution.

This is a development smoke test, not confirmatory evidence.

## Repeat verification

Commands refuse to overwrite an existing output directory. Supply a new output path for another run:

    .\.venv\Scripts\python.exe scripts\27_verify_release.py --output reproduction\release_check_02
    .\.venv\Scripts\python.exe scripts\28_check_simulation.py --output reproduction\simulation_smoke_02

## Scientific provenance and verification scope

Archived confirmation comprises seven scenarios, two protocols, and 30 paired repetitions per cell: 420 network realizations and 210 globally Holm-adjusted primary hypotheses.

Release verification covers installation and dependency checks, configuration validation, all 147 project tests, exact reproduction of all three reference statistical tables, figure generation, and development simulation checks for both protocols. Verification reports are supplied under provenance/.

Full reconstruction from raw observations and model retraining have not been repeated as part of release verification.

Scientific protocol identifier 0.21.0 identifies the experiment protocol; package version 0.20.1 identifies the installed software distribution.

The protocol records scientific configurations, release-source checksums, resource budgets, and random seeds. Checkpoint identities link the archived experiment to its result artifacts; see docs/RELEASE_PROVENANCE.md.

scripts/26_generate_results_figures.py generates the current manuscript result figures.

Use scripts/27_verify_release.py to verify archived evidence. scripts/24_scientific_extension.py freezes new experiment trees; use it only in a separate project copy with a new output directory. Do not overwrite an archived lock to bypass integrity checks.

## Data attribution

Ocean observations originate from the NSF Ocean Observatories Initiative (OOI). Dataset identifiers and acquisition provenance are recorded in the configurations and evidence.

Retain applicable OOI attribution and citation requirements, including acknowledgment of OOI and US National Science Foundation support.

OOI attribution guidance:
https://oceanobservatories.org/how-to-use-acknowledge-and-cite-data/

## Citation

When publishing work using SAFE-SILENCE, please cite both the associated manuscript and the specific archived software version used.

Software v0.20.1: https://doi.org/10.5281/zenodo.23386629

Companion data, models, and experimental evidence: https://doi.org/10.5281/zenodo.23386811

Download SAFE-SILENCE_Evidence_v0.20.1.zip from the evidence record and extract it into the software project root. The data/, artifacts/, and results/ directories must be alongside configs/ and scripts/.

Expected evidence ZIP SHA-256:
565eaa720a65f03c9a655bf4587a29e7cbe805eaf572c13ee5417890021a4b1c

The software uses the MIT License. Author-generated evidence uses CC BY 4.0; original OOI observations retain their provider terms and acknowledgment requirements.

## License

SAFE-SILENCE software is released under the MIT License; see LICENSE.

Third-party software and OOI data retain their respective licenses and terms. The MIT License does not impose an academic citation requirement.
