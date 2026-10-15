# Task S — Independently seeded confirmatory replication

## Evaluation scope

Task R supplies the development evaluation with ten paired repetitions. Task S
uses 30 independently seeded paired repetitions and a predefined primary analysis.
The development analysis uses a 49-test Holm family within each comparison,
protocol, and site; the Task S primary analysis uses 14 tests per comparison/protocol.
These families describe separate studies and are not pooled or selected after inference.

The seven-scenario, two-protocol experiment under results/task_x has its own global
210-hypothesis family, described in SCIENTIFIC_EXTENSION_v0.21.0.md.

## Independent replication

The complete frozen K→P→Q→R algorithmic path is rerun with:

- 30 matched acoustic repetitions;
- base seed 93211, disjoint from the Task K base seed 3407;
- an explicit audit proving zero realization-seed overlap;
- unchanged network, scheduler, integration, receiver, cause, and recovery algorithms; and
- outputs under `results/task_s/replication/`, with separately stored Tasks K–R evidence.

Task S first reverifies every declared Task R artifact. The nested replication independently
reverifies Task J and Task O inputs through the frozen upstream contracts.

## Primary family

The primary external evaluation site remains CE07SHSM. The 15-minute and 60-minute protocols are
analyzed separately because they overlap in physical observations. Two endpoints are fixed from
the scientific objectives, not selected by Task R significance:

1. post-recovery critical-delivery recall; and
2. realized post-recovery tail-loss-proxy reduction.

The proposed CVaR-plus-bounded-recovery system is compared separately with:

1. expected-risk scheduling plus the same bounded recovery; and
2. CVaR scheduling without recovery.

Each comparison/protocol family contains 14 tests: two primary endpoints across seven acoustic
scenarios. Repetition is the paired unit. Task S uses two-sided Wilcoxon signed-rank tests,
10,000-resample paired-bootstrap 95% intervals, and Holm correction within each declared family.

## Secondary trade-offs

False trust, exact retained-critical provenance, risk reduction per modeled joule, total bytes,
and total modeled energy remain fully reported. They are explicitly exploratory in Task S and are
not silently promoted into or removed from the confirmatory family. The primary and secondary endpoints have distinct declared analysis roles.

## Outputs

- `results/task_s/replication/task_k/`: independent acoustic traces;
- `results/task_s/replication/task_p/`: frozen scheduler replay;
- `results/task_s/replication/task_q/`: matched relay-delivery replay;
- `results/task_s/replication/task_r/`: frozen receiver/recovery reintegration;
- `results/task_s/confirmatory_comparisons.csv`: primary paired inference;
- `results/task_s/secondary_tradeoffs.csv`: exploratory safety-cost endpoints; and
- `results/task_s/task_s_summary.json`: integrity, version, seed-disjointness, family-size,
  leakage, and claim-boundary record.

## Claim boundary

Task S can confirm reproducibility across independently seeded simulations under the declared
acoustic model. It cannot convert modeled energy into hardware measurements, establish battery
lifetime, validate physical deployment, or demonstrate empirically calibrated cause probabilities.

## Command

```powershell
python -m safe_silence confirm-independent --config configs/pipeline.yaml
```
