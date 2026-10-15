# SAFE-SILENCE Task M — Coarse Ambiguity-Cause Posterior

## Purpose

Task M converts Task L receiver evidence into a probabilistic, deliberately coarse account
of why no packet arrived. It supports later recovery selection without claiming that silence
reveals an exact modem, link, relay, node, or physical failure.

## Taxonomy

- `valid_suppression`: the last accepted contract still supports intentional silence;
- `isolated_loss`: a due or expected contact is absent without sustained outage evidence;
- `persistent_outage`: prolonged silence, low reachability, and stale evidence support a
  continuing communication failure;
- `contract_divergence`: the receiver's accepted contract has expired while source and
  receiver state may have diverged;
- `unknown`: evidence is insufficient for a defensible named cause.

An arrived packet is recorded as `observed_update` and is not assigned a silence cause.

## Evidence and abstention

The fixed evidence model uses only Task L receiver state, reachability belief,
environmental uncertainty, contract validity/age, overdue evidence, and causal receiver
history. One coefficient set is used across every site, protocol, scenario, and repetition.
The known-cause softmax is converted into explicit UNKNOWN mass from normalized score
entropy. Low confidence or a small decision margin triggers abstention. In addition,
`valid_suppression` is admissible only when the frozen Task D/L trust contract is satisfied:
the receiver is in `trusted_silence`, the contract is valid, contact is not due,
reachability is at least 0.95, and environmental uncertainty is at most 0.85. A proposed
`valid_suppression` decision that fails this gate becomes `unknown`.

The resulting distribution is a **rule-based pseudo-posterior**, not an empirically
calibrated probability estimate. Brier score, log loss, and ECE are therefore diagnostic
score-quality measures; they must not be described as proof of probabilistic calibration.

## Leakage boundary

Scenario identity, hidden link state, source action during a missing packet, critical-event
truth, network random draws, and future measurements cannot enter the posterior. Oracle
fields are joined only after inference to create diagnostic coarse labels and scores.

## Evaluation

Task M reports supported-class macro-F1 and balanced accuracy, selective coverage and
accuracy, UNKNOWN rate, multiclass Brier score, log loss, calibration error, posterior
entropy, per-class support/recall, and the primary safety error: unsafe silence predicted as
valid suppression. Zero-support classes are not scored.

## Interpretation boundary

Exact cause accuracy is diagnostic only. Task N consumes the pseudo-posterior and
abstention state to select bounded probes, retransmissions, reconciliation, or stale-unsafe
escalation. The cause-inference contract is v1.1.
