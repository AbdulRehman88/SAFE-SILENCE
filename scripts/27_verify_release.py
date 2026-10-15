"""Verify frozen inputs, reproduce statistical analysis, and regenerate figures.

Scientific algorithms and result artifacts are preserved. A scientific release
projection records the release source files separately from archived checkpoint identity.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_join(root, relative):
    relative = str(relative).replace('\\', '/')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or ':' in relative:
        raise ValueError(f'Invalid recorded relative path: {relative}')
    result = root.joinpath(*parts.parts)
    if root.resolve() not in result.resolve().parents:
        raise ValueError(f'Recorded path escapes its root: {relative}')
    return result


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify(root, evidence):
    from safe_silence.simulation.network import _derived_seed
    from safe_silence.evaluation.extension import protocol_identity
    lock_path = evidence / 'protocol_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if lock['new_unique_network_seeds'] != 420 or lock['primary_family_test_count'] != 210:
        raise ValueError('Wrong confirmation protocol')
    verified = {}
    for relative, expected in lock['files'].items():
        path = safe_join(root, relative)
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'Frozen input hash mismatch: {relative}')
        verified[relative.replace('\\', '/')] = actual
    cfg = lock['configs']['extension']
    seeds, artifacts, checkpoints = [], {}, {}
    for protocol in cfg['protocols']:
        for scenario in cfg['scenarios']:
            cell = evidence / 'cells' / protocol / scenario
            done_path = cell / 'DONE.json'
            done = json.loads(done_path.read_text(encoding='utf-8'))
            if (done['smoke_only'] or done['protocol_sha256'] != protocol_identity(lock)
                    or done['protocol'] != protocol or done['scenario'] != scenario
                    or done['site'] != cfg['primary_site']):
                raise ValueError(f'Incompatible checkpoint: {cell}')
            if set(done['seeds']) != {str(i) for i in range(30)}:
                raise ValueError(f'Incomplete paired repetitions: {cell}')
            for repetition, seed in done['seeds'].items():
                expected = _derived_seed(cfg['new_network_base_seed'], protocol,
                                         cfg['primary_site'], cfg['primary_role'],
                                         scenario, int(repetition))
                if seed != expected:
                    raise ValueError(f'Unexpected seed: {cell}/{repetition}')
                seeds.append(seed)
            for name, expected in done['files'].items():
                path = safe_join(cell, name)
                actual = digest(path)
                if actual != expected:
                    raise ValueError(f'Cell result hash mismatch: {path}')
                artifacts[path.relative_to(evidence).as_posix()] = actual
            checkpoints[done_path.relative_to(evidence).as_posix()] = digest(done_path)
    if len(seeds) != 420 or len(set(seeds)) != 420 or set(seeds) != set(lock['seed_ledger']):
        raise ValueError('Independent seed ledger mismatch')
    print(f'SCIENTIFIC RELEASE INPUTS VERIFIED: {len(verified)} locked inputs; '
          f'{len(artifacts)} cell artifacts; 420 unique seeds', flush=True)
    return lock, {'protocol_lock_sha256': digest(lock_path),
                  'locked_inputs': verified, 'cell_artifacts': artifacts,
                  'archived_experiment_identity': protocol_identity(lock),
                  'release_projection': lock.get('release_projection'),
                  'checkpoints': checkpoints, 'unique_network_realizations': 420}


def compare_table(expected_path, actual_path, keys):
    import numpy as np
    import pandas as pd
    expected, actual = pd.read_csv(expected_path), pd.read_csv(actual_path)
    if set(expected.columns) != set(actual.columns):
        raise ValueError(f'Column mismatch: {expected_path.name}')
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        raise ValueError(f'Duplicate comparison keys: {expected_path.name}')
    expected = expected.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    if len(expected) != len(actual):
        raise ValueError(f'Row count mismatch: {expected_path.name}')
    maximum_difference = 0.0
    for column in expected.columns:
        a, b = expected[column], actual[column]
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            aa, bb = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            if not np.allclose(aa, bb, rtol=1e-10, atol=1e-12, equal_nan=True):
                raise ValueError(f'Numerical mismatch: {expected_path.name}/{column}')
            finite = np.isfinite(aa) & np.isfinite(bb)
            if finite.any():
                maximum_difference = max(maximum_difference, float(np.max(np.abs(aa[finite]-bb[finite]))))
        elif not a.fillna('<NA>').equals(b.fillna('<NA>')):
            raise ValueError(f'Value/decision mismatch: {expected_path.name}/{column}')
    print(f'REPRODUCED: {expected_path.name}; {len(expected)} rows; '
          f'maximum absolute numeric difference {maximum_difference:.3g}', flush=True)
    return {'rows': len(expected), 'maximum_absolute_numeric_difference': maximum_difference,
            'expected_sha256': digest(expected_path), 'reproduced_sha256': digest(actual_path)}


def reproduce(root, evidence, output):
    import importlib.metadata as metadata
    import yaml
    from safe_silence.evaluation.extension import analyze
    from safe_silence.reporting.confirmation_figures import generate
    lock, report = verify(root, evidence)
    if output.exists():
        raise ValueError('Reproduction output already exists; choose a new --output directory')
    if output == evidence or evidence in output.parents or output in evidence.parents:
        raise ValueError('Reproduction output must be separate from frozen evidence')
    output.mkdir(parents=True)
    marker = output / 'REPRODUCTION_INCOMPLETE'
    marker.write_text('Verification not complete.\n')
    working = output / 'confirmation'
    working.mkdir()
    shutil.copyfile(evidence / 'protocol_lock.json', working / 'protocol_lock.json')
    for relative in [*report['cell_artifacts'], *report['checkpoints']]:
        target = working / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evidence / relative, target)
    analyze(working, lock['configs'], lock)
    specifications = {
        'primary_comparisons.csv': ['comparison', 'protocol', 'scenario', 'metric'],
        'aggregate_effects.csv': ['comparison', 'protocol', 'metric'],
        'resource_consumption.csv': ['protocol', 'loss_profile', 'scheduler_policy', 'recovery_policy'],
    }
    report['statistical_reproduction'] = {
        name: compare_table(evidence / name, working / name, keys)
        for name, keys in specifications.items()}
    expected = json.loads((evidence / 'extension_summary.json').read_text())
    actual = json.loads((working / 'extension_summary.json').read_text())
    if expected != actual:
        raise ValueError('Confirmation summary or global decision counts differ')
    plot_config = root / 'configs/manuscript_figures.yaml'
    cfg = yaml.safe_load(plot_config.read_text())
    generate(working, output / 'figures', cfg)
    report['status'] = 'PASSED'
    report['scope'] = 'Frozen-input verification; statistical and figure reproduction from archived cells'
    report['full_experiments_rerun'] = False
    report['raw_data_retraining_verified'] = False
    report['global_holm_rejections'] = actual['primary_holm_rejections']
    report['python'] = sys.version
    report['packages'] = {}
    for name in ['numpy', 'pandas', 'scipy', 'matplotlib', 'pyarrow', 'PyYAML', 'torch']:
        try:
            report['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report['packages'][name] = 'distribution metadata unavailable in this analysis runtime'
    report['release_verifier_sha256'] = digest(Path(__file__))
    report['figure_generator_sha256'] = digest(root / 'src/safe_silence/reporting/confirmation_figures.py')
    report['plot_configuration_sha256'] = digest(plot_config)
    # Verify input and archived artifact integrity after reproduction.
    for relative, before in report['locked_inputs'].items():
        if digest(safe_join(root, relative)) != before:
            raise ValueError('A frozen input changed during reproduction')
    for relative, before in {**report['cell_artifacts'], **report['checkpoints']}.items():
        if digest(evidence / relative) != before:
            raise ValueError('An archived cell file changed during reproduction')
    for name, values in report['statistical_reproduction'].items():
        if digest(evidence / name) != values['expected_sha256']:
            raise ValueError('An archived statistical table changed during reproduction')
    (output / 'release_verification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    marker.unlink()
    print(f'RELEASE STATISTICS AND FIGURES REPRODUCED: {output}', flush=True)
    print('Frozen experiment and evidence files preserved.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--evidence-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'src'))
    evidence = (args.evidence_dir or root / 'results/task_x').resolve()
    output = (args.output or root / 'reproduction/release_check').resolve()
    reproduce(root, evidence, output)


if __name__ == '__main__':
    main()
