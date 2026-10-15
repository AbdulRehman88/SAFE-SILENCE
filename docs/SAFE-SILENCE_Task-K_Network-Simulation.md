# SAFE-SILENCE Task K — Matched Acoustic-Network Realizations

## Purpose

Task K moves the frozen Task J actions through explicitly lossy underwater acoustic links.
It measures the gap between pre-network event preservation and actual packet delivery while
creating replayable network traces for every later baseline and ablation.

## Scientific locks

- Reference and primary outages are generated without reading measurements or event labels.
- Event-coupled outage is isolated to declared F13 stress testing.
- Every method receives the same hidden states and per-attempt random draws.
- Hidden link state and oracle failure cause are evaluation fields, never receiver inputs.
- Data loss, ACK loss, retry exhaustion, duplicate delivery, delay, bytes, and modeled
  communication energy are recorded separately.
- Retry count is bounded; there is no unlimited recovery during a partition.
- Energy is a parameterized simulation estimate, not hardware measurement.

## Scenarios

- F0 stable high-reliability reference;
- F2 independent burst outage (primary);
- F3 independent long heavy-tailed outage (primary);
- F5 ACK-asymmetry stress;
- F8 hard-partition stress;
- F13 event-coincident outage stress;
- F14 unseen outage-duration generalization.

Each scenario is repeated ten times with deterministic context-specific seeds. Transport
records retain the fixed oracle realization so periodic, threshold, AoI, AoII, uncertainty,
and SAFE-SILENCE policies can later be compared under exactly matched conditions.

## Interpretation boundary

Task K evaluates transport only. It does not infer whether missing reception means
intentional silence or network failure. Receiver-side connectivity belief, false trust,
stale-unsafe transitions, and bounded recovery are subsequent tasks.
