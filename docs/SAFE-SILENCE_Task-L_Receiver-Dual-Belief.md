# SAFE-SILENCE Task L — Receiver-Side Dual Belief

## Purpose

Task L determines whether a missing reception can still be interpreted as valid bounded
silence or must be marked ambiguous, failure-suspected, or stale-unsafe. It maintains a
reachability belief separately from environmental uncertainty and makes no recovery action.

## Receiver evidence boundary

The receiver may use only:

- packet arrival at the current epoch;
- action, semantic risk, contract identifier, and validity horizon carried by a packet that
  actually arrived;
- elapsed sensing epochs and the receiver's own previously accepted contract state.

It cannot use source action or semantic risk from a missing packet, event truth, future
measurements, network scenario identity, hidden link state, random draws, failure cause,
attempt count, or ACK success. These remain evaluation-only.

## Fixed dual belief

The connectivity belief uses one declared nominal two-state Bayesian filter across every
site, protocol, scenario, and repetition. No test or external result changes its transition,
emission, or decision parameters. The environmental belief begins with the risk in the last
accepted packet and grows monotonically with contract age.

## State logic

- `updated`: a timely packet was accepted;
- `trusted_silence`: no packet arrived, but the accepted contract remains valid and both
  beliefs satisfy the declared trust limits;
- `ambiguous`: evidence is insufficient to distinguish valid silence from failure;
- `failure_suspected`: reachability belief crosses the failure threshold;
- `recovery_pending`: reserved for the subsequent bounded-recovery task;
- `stale_unsafe`: an expired contract remains unresolved beyond the declared bound.

There is no normal fallback after contract expiry. Unresolved evidence remains UNKNOWN.

## Evaluation

Oracle link state, source action, and critical-event truth are joined only after receiver
inference. The primary endpoint is critical false-trust rate. Supporting outcomes are unsafe
silence detection, critical safe handling, time to ambiguity, time to stale-unsafe, Brier
score, expected calibration error, and state occupancy.

## Interpretation boundary

Task L diagnoses unsafe ambiguity. Task M evaluates coarse cause inference; Task N
evaluates bounded recovery, and Task R evaluates integrated receiver/recovery outcomes.
