"""Run an isolated development-data simulation check using frozen algorithms."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def upstream_path(root, declared):
    # Remap archived Windows paths into this release, never the old working tree.
    parts = [p for p in str(declared).replace('\\', '/').split('/') if p]
    if 'IEEE_CM' in parts:
        parts = parts[parts.index('IEEE_CM') + 1:]
    elif parts and (':' in parts[0] or str(declared).startswith('/')):
        raise ValueError(f'Cannot remap archived path: {declared}')
    if '..' in parts or not parts:
        raise ValueError(f'Unsafe archived path: {declared}')
    result = root.joinpath(*parts)
    if root.resolve() not in result.resolve().parents:
        raise ValueError('Artifact path escapes release')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evidence-dir', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'src'))
    spec = importlib.util.spec_from_file_location('release_verifier', root / 'scripts/27_verify_release.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    from safe_silence.evaluation.extension import run_cell, protocol_identity
    from safe_silence.simulation.network import _derived_seed
    evidence = (args.evidence_dir or root / 'results/task_x').resolve()
    lock, _ = verifier.verify(root, evidence)
    inputs = {}
    for relative in ['results/task_j/task_j_summary.json', 'results/task_o/task_o_summary.json']:
        summary_path = root / relative
        inputs[relative] = verifier.digest(summary_path)
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        for record in summary['artifacts']:
            path = upstream_path(root, record['path'])
            actual = verifier.digest(path)
            if actual != record['sha256']:
                raise ValueError(f'Archived semantic/fusion input hash mismatch: {path}')
            inputs[path.relative_to(root).as_posix()] = actual
    print(f'SEMANTIC AND FUSION INPUTS VERIFIED: {len(inputs)} records', flush=True)
    output = (args.output or root / 'reproduction/simulation_smoke').resolve()
    if output == root or root not in output.parents or (root / 'results') in output.parents:
        raise ValueError('Use a new output within the release, outside frozen results')
    if output.exists():
        raise ValueError('Output already exists; choose a new --output directory')
    output.mkdir(parents=True)
    marker = output / 'SIMULATION_CHECK_INCOMPLETE'
    marker.write_text('Do not treat this development check as confirmatory evidence.\n')
    cfg = lock['configs']
    ext = cfg['extension']
    cells = []
    for protocol in ext['protocols']:
        cell = output / protocol / 'F2_burst'
        run_cell(root, cell, protocol, 'F2_burst', cfg, lock, True)
        done = json.loads((cell / 'DONE.json').read_text())
        if not done['smoke_only'] or done['protocol_sha256'] != protocol_identity(lock):
            raise ValueError('Smoke checkpoint does not match frozen protocol')
        if set(done['seeds']) != {'0', '1'}:
            raise ValueError('Expected two paired development repetitions')
        for repetition, seed in done['seeds'].items():
            expected = _derived_seed(ext['smoke_network_base_seed'], protocol,
                                     ext['development_site'], 'tuning', 'F2_burst', int(repetition))
            if seed != expected:
                raise ValueError('Unexpected development-check seed')
        for name, checksum in done['files'].items():
            if verifier.digest(cell / name) != checksum:
                raise ValueError('Simulation checkpoint hash mismatch')
        cells.append({'protocol': protocol, 'scenario': 'F2_burst',
                      'paired_repetitions': 2, 'checkpoint_sha256': verifier.digest(cell / 'DONE.json')})
    for relative, expected in inputs.items():
        if verifier.digest(root / relative) != expected:
            raise ValueError('Archived input changed during simulation')
    verifier.verify(root, evidence)
    report = {'status': 'PASSED', 'scope': 'Development-data simulation smoke check',
              'site': ext['development_site'], 'confirmation_evidence': False,
              'full_confirmation_rerun': False, 'model_retraining_performed': False,
              'cells': cells, 'input_sha256': inputs,
              'simulation_check_script_sha256': verifier.digest(Path(__file__))}
    (output / 'simulation_check_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    marker.unlink()
    print('SIMULATION CHECK PASSED: both protocols; CE02 tuning data; two paired repetitions each.', flush=True)
    print('Network, six trust variants, three scheduler profiles, and recovery executed.', flush=True)
    print('Development smoke only; archived confirmation evidence preserved.', flush=True)


if __name__ == '__main__':
    main()
