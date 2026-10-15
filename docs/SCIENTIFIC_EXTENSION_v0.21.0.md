# SAFE-SILENCE experiment protocol v0.21.0

This document describes the confirmation experiment protocol. The archived confirmation comprises 420 network realizations; release verification scope is recorded in README.md.

## Scientific components

The pipeline combines causal semantic evidence, source packet decisions, acoustic-network simulation, associative top-k fusion (k=4), constrained scheduling, and a sender-side critical-packet cache. Receiver and cause variants implement the ablations; budget admission bounds recovery traffic. The experiment driver executes scheduling and physical delivery under paired network realizations.

## Frozen evaluation

- Primary site: CE07SHSM, external_test only.
- Two protocols: primary_15min and secondary_60min.
- Seven scenarios: F0, F2, F3, F5, F8, F13, F14.
- Thirty paired repetitions per cell: 420 unique physical network realizations.
- Confirmation network base seed: 293211. Disjoint reference bases: 3407, 93211, 193211. Derived seeds are audited across sites and roles.
- All compared policies, ablations, and loss profiles share the realization within each pair.
- Smoke testing uses CE02 tuning data, a separate seed base (393211), two repetitions, and a short window under both protocols. Smoke data cannot enter confirmation.
- Algorithms, input hashes, budgets, profiles, and hypothesis definitions are frozen before primary execution.
- Primary outcomes are excluded from parameter tuning. Unfavorable or null outcomes must remain in the report.

## Common-budget recovery

Policies: SAFE-SILENCE bounded recovery, periodic_retry, fixed_probe, and the no_recovery zero-budget anchor.

Both common caps are calibrated exclusively from CE02 data: the linear 95th percentile of recovery bytes/receiver opportunity and modeled energy/receiver opportunity for the full policy. CE02 internal_test is explicitly reused as development calibration for this evaluation; it must not be described as an independent held-out confirmation site for the evaluation. Source calibration uses Task N; relay calibration uses Task R metrics. All calibration files are hashed.

Each repetition receives the same cap for every policy, proportional to its fixed available receiver opportunities. Source and relay scopes have separate cap rates because their opportunity units and hop counts differ. A request is admitted only if a request plus a conservatively bounded response fits BOTH remaining caps. Admission cannot inspect the future request outcome or hidden link state. Only transmissions actually made are charged. The source response bound is 60 bytes, the relay bound is 148 bytes under the frozen encoding configuration.

Actual bytes, modeled joules, cap values, blocked actions, and recall are reported. These are common-budget comparisons, not exact realized-resource matching. If consumption differs materially, use effect-versus-cost/Pareto reporting and retain that wording. The energy model is proportional to transmitted bytes for a fixed hop count and bitrate; its two caps are not two independent empirical hardware measures.

## Direct trust ablations

Full receiver versus:

1. No expiration: a received contract remains active until replaced by a newly received contract; environmental uncertainty continues to age.
2. No connectivity gate: environmental/contract evidence determines trust; connectivity-only cause weights and the connectivity admissibility threshold are removed.
3. Collapsed gate: connectivity times environmental confidence is compared with the product of the declared boundary values, 0.95*(1-0.85)=0.1425. This tests gate composition, not replacement of the two latent estimators by a newly trained model.
4. No UNKNOWN: ambiguous cases select the highest-scoring known cause without UNKNOWN mass or the UNKNOWN safety transfer. Ties follow the declared known-cause order. Physical truth is never an input.
5. Fixed timeout: trust lasts less than four elapsed protocol-grid durations after contact. This baseline does not use adaptive belief gating.

Unsafe-silence and true-cause labels are fixed to the full receiver's common reference for every variant. A weakened receiver cannot redefine its own unsafe observations as safe. Critical missing summaries are unsafe even if scheduling suppressed their transmission.

Primary trust evidence is evaluated on source protocol-grid telemetry. Secondary trust evidence is also evaluated on the integrated relay path. Their timing units are explicit: the relay receiver counts relay-candidate opportunities, while the fixed-timeout baseline uses elapsed protocol-grid time. Do not relabel relay retention/expiration opportunities as minutes or protocol steps.

The primary trust outcome is critical false trust BEFORE recovery: the number of critical rows with no arrival whose receiver state is trusted_silence, divided by all critical rows. Unsafe valid-suppression cause assertions and the union of unsafe receiver/cause interpretations are separate secondary outcomes: a forced known-cause assertion does not necessarily change a protective receiver action. No UNKNOWN-specific receiver-safety benefit is claimed if only the cause assertion changes. Also report conditional-on-missing false trust, unsafe detection recall, recovery delivery, unnecessary valid-suppression triggers, delays, consumption, and protective-episode detections/non-detections. Protective delay means exclude non-detections and are labelled uncensored; non-detection counts must accompany them.

A planned or budget-cancelled request does not remove false trust. Post-action critical false trust requires event-specific delivery credit; generic liveness cannot count as delivery of a missing critical packet. Post-action scoring uses the observed event-specific delivery outcome.

## End-to-end scheduler sensitivity

Three prespecified profiles:

| Profile | Criticality | Urgency | Residual | Uncertainty/link ambiguity |
|---|---:|---:|---:|---:|
| Primary | 4 | 2 | 1 | 2 |
| Miss-dominant | 8 | 1 | 1 | 1 |
| Uncertainty-dominant | 4 | 1 | 2 | 3 |

For each profile, the complete frozen candidate-selection, physical-delivery, receiver, and common-budget recovery path is replayed for CVaR and expected-risk scheduling. CVaR alpha is 0.95. Loss before delivery is weighted criticality + urgency + residual + uncertainty*(1-causal ACK belief). Successful-delivery residual loss is reconstruction weight times residual for unretained items. Realized reduction is summed over candidates using their event-specific timely post-recovery delivery state. This is a policy-aligned proxy, not a measured digital-twin-fidelity score.

The implementation computes each profile in an isolated scheduler directory using the runner's primary-profile trace slot; exported system rows retain the actual profile name. All profiles share network realizations and opportunity constraints.

## Statistical family and outputs

One global two-sided Wilcoxon/Holm family of 210 primary hypotheses:

- Recovery: 2 comparators * 2 endpoints * 14 cells = 56.
- Source trust: 5 ablations * 1 endpoint * 14 cells = 70.
- Scheduler sensitivity: 3 profiles * 2 endpoints * 14 cells = 84.

Recovery and scheduler endpoints are critical-delivery recall and realized post-recovery tail-proxy reduction. Source trust uses pre-action critical false trust. All-zero paired differences receive p=1 and remain in the family. An undefined primary endpoint stops analysis instead of silently deleting a hypothesis.

Scenario intervals use paired repetition resampling. Aggregate intervals resample pairs independently inside each scenario and average the seven scenario estimates with equal weights. Protocols are never pooled; scenario confidence limits are never averaged. Aggregate intervals are descriptive and introduce no extra significance claims. A zero observed rate and a degenerate bootstrap interval do not establish zero population risk.

Final outputs under results/task_x:

- protocol_lock.json: algorithm/input hashes, configuration snapshot, budgets, and seed ledger.
- cells/<protocol>/<scenario>/DONE.json: completed-cell checkpoint and output hashes.
- system_metrics.csv and trust_metrics.csv: all repetition-level outcomes.
- primary_comparisons.csv: all 210 hypotheses and global adjusted p-values.
- aggregate_effects.csv: absolute means and stratified paired confidence intervals.
- resource_consumption.csv: actual costs and common caps.
- extension_summary.json: confirmation accounting and claim boundaries.

Completion is resumable at the protocol-scenario cell level. Re-running `run` skips completed, hash-verified cells and recomputes an interrupted cell. Do not edit frozen code/configurations between runs. Large temporary physical traces are removed after a completed cell by default; the primary relay trace and all final metrics/hashes are retained.

## Verification

The verified Windows release passes 147 project tests. Its statistical reproduction
checks 59 release inputs, 42 cell artifacts, and 420 unique seeds, and reproduces all
three reference tables exactly. Figure generation and the CE02 tuning-data simulation
check pass for both protocols. The simulation check uses two paired repetitions per
protocol and is separate from the archived confirmation evidence.

## Interpretation boundaries

- Recovery responses do not rerun the full future receiver-belief history. Do not claim validated closed-loop twin fidelity.
- The scheduler stress queue groups eight ordered relay candidates and does not replay queue residence delay. Treat its findings as offline constrained queue comparisons, not demonstrated real-time deployment.
- Top-k fusion preserves critical batch evidence/maxima but does not preserve every individual critical item. Its component comparison is not an end-to-end fusion ablation.
- Communications energy is modeled, outages are simulated, and the digital-twin application is a telemetry-substrate context.
- Interpret component and policy benefits within the corresponding measured endpoints and confidence intervals.

## Windows execution

Run from the repository root using the project environment:

1. `.venv/Scripts/python.exe scripts/27_verify_release.py`
2. `.venv/Scripts/python.exe scripts/28_check_simulation.py`

Outputs are written under `reproduction/`; an existing output directory is not overwritten.
Use a distinct `--output` path for another verification run.

`scripts/24_scientific_extension.py` creates a protocol for a separately configured
experiment tree and executes resumable simulation cells. Use a separate project copy
and output directory for that workflow; archived confirmation locks remain immutable.
`scripts/25_existing_evidence_reanalysis.py` evaluates the independently seeded
component-confirmation evidence under `results/task_s/`.
