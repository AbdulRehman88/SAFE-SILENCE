# Task Q — Matched-Realization End-to-End Relay Evaluation

Task Q converts the frozen Task P scheduling decisions into realized relay-to-buoy delivery
outcomes. Every policy sees the same Task O candidates, Task P opportunity constraints, Task K
hidden link states, retry limit, and per-attempt counterfactual random draws. Policy decisions are
completed before oracle columns are attached; hidden state, current delivery outcome, event truth,
and future measurements therefore cannot influence selection.

## Policies and matching

The proposed CVaR-risk-reduction-per-joule scheduler is compared with the frozen expected-risk
ablation and fixed-threshold, FIFO, AoI, AoII, residual-only, uncertainty-only, and semantic-only
rankers. All operate on identical eight-candidate opportunities with the same packet-count,
expected-byte, modeled-energy, and expected-airtime constraints. Realized bytes and modeled energy
are reported rather than forced to equality after the fact; their ratios to the proposed method
make any residual cost difference explicit.

The stored Task K realization is replayed at the Task O relay-flush timestamp. The integrated path
is one acoustic hop from relay to buoy. Task O's encoded packet size already includes its relay
header, so Task Q does not add a second data header. ACK bytes, bounded retries, propagation,
duplicate delivery, and modeled transmit/receive energy follow the frozen Task K contract.

## Outcomes

Task Q reports critical-batch selection and timely-delivery recall, critical miss probability,
exact retained-critical-item recall, notification delay, realized tail/mean operational-loss proxy
reduction, bytes, modeled energy, duplicate deliveries, and tail reduction per modeled joule.
Critical-batch detection credits the separately preserved Task O critical maximum; exact critical
provenance credits only individually retained critical items. These endpoints must not be merged.

## Claim boundary

This is an actual simulated acoustic-delivery replay over real-ocean-derived semantic evidence, not
a field deployment. Energy is modeled, not measured. The completed path is semantic packet through
relay fusion and scheduling to buoy delivery. Receiver belief and bounded recovery are evaluated in Task R.
Task Q alone does not establish whole-pipeline superiority,
physical modem lifetime, or calibrated critical-event probabilities.
