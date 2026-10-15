# Task N — Cause-Aware Bounded Recovery

Task N converts Task L protective receiver states and Task M coarse evidence into finite
recovery actions. The deployable controller never reads hidden acoustic state, missing
packet content, event truth, scenario identity as a decision feature, or future data.

The shared policy continues silence only under Task M's frozen valid-suppression gate.
Ambiguity, failure suspicion, stale-unsafe state, UNKNOWN, or a due contact can trigger a
bounded two-attempt round trip. A low-rate liveness check inside a still-valid contract is
the general safety mechanism for events whose alerts are lost during an apparently valid
silence interval; it is not activated by knowing that a run is F13.

The two-attempt budget is distributed over consecutive grid steps rather than compressed
inside one unchanged acoustic state. The source retains critical alerts for at most four
steps—the already-frozen maximum silence horizon—and returns still-valid cached alerts
during reconciliation. This permits recovery after an outage clears while keeping memory,
attempts, and staleness explicitly bounded.

Every recovery request and response is replayed through the Task K physical state using
deterministic matched draws. All policies therefore face the same realization. The
oracle-cause and unbounded-retry policies are diagnostic references only. Reported joules
are communication-model estimates, not measurements from an acoustic modem.

Primary outcomes are critical false trust and post-recovery critical delivery. Secondary
outcomes include recovered critical rows, success probability, recovery delay, wire and
energy overhead, unnecessary probing during truly valid suppression, and unresolved
recovery-pending exposure.

Only the deployable `safe_silence_bounded` policy is retained as a row-level trace.
Matched reference policies are stored in the metric registry only, preventing a sixfold
duplication of upstream observations while preserving every comparison endpoint.
