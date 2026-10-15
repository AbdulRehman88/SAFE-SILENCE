# Task P — Tail-Risk Reduction per Joule

Task P implements the transparent online scheduler specified in Task C. Frozen Task O relay
summaries are the candidate packets. Task K supplies only packet-attempt and ACK observations that
strictly precede the candidate decision. Hidden link regimes, current outcomes, missing packet
content, event truth unavailable to the relay, and future measurements are prohibited inputs.

## Causal link belief

Each repetition and immutable window begins from the declared nominal reachability probability.
Before a candidate is scored, the scheduler records the current belief. Only afterward may an
attempted packet's ACK outcome update the EWMA. Therefore the ACK generated at the candidate's own
timestamp cannot influence its decision. Unattempted rows do not update reachability.

## Operational safety-loss proxy

For every item represented by a fused candidate, the pre-delivery loss contains weighted missed
critical evidence, notification urgency, residual reconstruction evidence, and connectivity
ambiguity. Successful delivery removes the critical maximum and deadline terms carried by the
summary, resolves contact ambiguity, and resolves reconstruction loss only for individually
retained top-K items. Calibration loss is assigned zero because Task J does not output a calibrated
critical-event probability; Task P must not invent one.

The primary physical-priority weights are accompanied by miss-dominant and
uncertainty-dominant sensitivity profiles. None is selected or changed using test/external results.
Empirical upper-tail CVaR uses alpha 0.95 with fractional mass at the quantile boundary.

For candidate p, the predicted reduction is

\[
\Delta_p = \widehat p_{\mathrm{ACK}}
\left[\operatorname{CVaR}_{0.95}(L_{\mathrm{before}})
-\operatorname{CVaR}_{0.95}(L_{\mathrm{after\ success}})\right]_+.
\]

The expected-risk reference replaces both CVaR terms with their means. Both policies use the same
causal link belief and communication-cost estimate.

## Communication cost and constraints

Expected attempts follow the Task K bounded-retry limit. Expected wire bytes include the fused
packet and ACK, while modeled energy uses the Task K bitrate and transmit/receive powers for the
relay-to-buoy hop. Propagation and transmission time enter expected airtime. The utility denominator
uses the frozen energy, byte-equivalent, and airtime coefficients plus a numerical epsilon.

The controlled congestion opportunity contains eight candidates and permits at most four subject
to 2,000 expected bytes, 40 modeled joules, and 12 expected seconds. This creates a reproducible
component-level scheduling choice. It is not a claim that the sequential candidates were physically
co-resident for their full timestamp interval, and Task P reports no queue-residence delay.

Mandatory critical/deadline candidates are considered before ordinary utility-ranked traffic.
When no mandatory candidate exists, one sufficiently uncertain candidate may reserve liveness
capacity. If mandatory demand exceeds feasibility, the same deterministic policy utility and
timestamp/batch tie-break select the feasible subset; no budget is silently exceeded.

## Outputs and claim boundary

Metrics include critical/mandatory selection recall, achieved predicted tail and expected-risk
reduction, modeled energy, expected bytes/airtime, causal ACK belief, tail reduction per modeled
joule, and budget violations. Row-level traces are stored only for the primary CVaR policy and
primary loss profile; the expected-risk ablation and sensitivity profiles are metric-only.

Task P demonstrates behavior under its declared causal proxy and budgets. It does not claim global
optimality, measured modem energy, actual selected-packet delivery, end-to-end critical-event
improvement, or final matched-cost superiority. Task Q must replay delivery and compare complete
systems at matched bytes and modeled energy.
