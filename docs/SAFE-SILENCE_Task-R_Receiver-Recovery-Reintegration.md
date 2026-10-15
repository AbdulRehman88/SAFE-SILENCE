# Task R — Receiver and bounded-recovery reintegration

## Purpose

Task R closes the simulated SAFE-SILENCE path evaluated in this repository by connecting
Task Q relay-to-buoy delivery to the frozen Task L receiver, Task M coarse-cause, and Task N
bounded-recovery logic. It does not alter or refit Tasks A–Q.

The predeclared proposed system is:

- Task P CVaR risk-reduction-per-joule scheduling; and
- Task N `safe_silence_bounded` receiver recovery.

Two paired references isolate the contribution of each component:

1. expected-risk scheduling with the same bounded recovery; and
2. CVaR scheduling with no recovery.

## Frozen inputs and integrity

Task R requires the frozen v1.0 Task Q, P, O, and K contracts and reverifies the SHA-256
digest of every declared upstream artifact before execution. The receiver, cause, and
recovery versions must remain Task L v1.0, Task M v1.1, and Task N v1.0. Relocated project
directories are supported through deterministic artifact-path reconstruction.

## Relay-to-receiver adapter

Task Q evaluates fused relay candidates rather than the original Task J source packets.
Each fused candidate therefore becomes one receiver opportunity. A delivered critical
summary exposes `alert`; a delivered ordinary selected summary exposes `innovation`; and an
unselected summary is source-side `silent`. If no packet arrives, all source-side fields are
masked by the frozen Task L observation function.

The semantic risk carried by a summary is the maximum of its preserved critical-evidence,
environmental-uncertainty, and retained-residual extrema. The validity horizon is derived
from the separately preserved deadline-urgency extreme and clipped to the frozen four-step
silence bound. Contract identifiers are deterministic 48-bit hashes of relay batch IDs.

Scheduler selection is not supplied as a receiver observation. A missing critical summary
is labeled unsafe only after receiver inference for evaluation. Likewise, a suppressed
critical summary is relabeled `contract_divergence` only in evaluation truth; neither label
can change cause inference or recovery decisions.

## Matched evaluation

Both schedulers use identical Task P opportunity constraints and the stored Task K acoustic
realizations. Recovery replays share the frozen Task N deterministic matched-randomness
contract. Results are reported independently for every protocol, site, model role, acoustic
scenario, and repetition; the overlapping 15-minute and 60-minute protocols are never pooled.

The repetition is the paired statistical unit. For each protocol/site/scenario cell and each
predeclared endpoint, Task R reports the proposed and reference means, mean paired difference,
10,000-resample paired-bootstrap 95% percentile interval, two-sided Wilcoxon signed-rank
test, and Holm-adjusted p-value. Holm correction is applied within each comparison,
protocol, and site across the declared endpoints and scenarios.

Primary endpoints include post-recovery critical delivery recall, critical false trust,
exact retained-critical provenance recall, realized tail-loss-proxy reduction, total wire
bytes, total modeled communication energy, and tail-proxy reduction per modeled joule.

## Outputs

- `results/task_r/system/<protocol>/<site>/<scenario>.parquet`: proposed-system trace;
- `results/task_r/system_metrics.csv`: repetition-level 2×2 system metrics;
- `results/task_r/paired_comparisons.csv`: paired inference results; and
- `results/task_r/task_r_summary.json`: versions, integrity counts, leakage declarations,
  claim boundary, output hashes, and row counts.

## Claim boundary

Task R supports claims about a simulated semantic-packet → relay fusion/scheduling → acoustic
delivery → receiver interpretation → bounded recovery path under declared models and
real-ocean-derived evidence. It does not establish field deployment, hardware-measured
energy or battery lifetime, exact physical-cause diagnosis, or empirically calibrated cause
probabilities. Those limitations must remain explicit in the paper and public release.

## Command

```powershell
python -m safe_silence reintegrate-system --config configs/pipeline.yaml
```
