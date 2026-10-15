# SAFE-SILENCE — Task D Failure Ontology, Outage Model, and Recovery Policy

**Version:** 1.0  
**Date:** 16 September 2026  
**Scope:** Conceptual model; executable settings and evaluated mechanisms are defined by the source code and experiment configuration.

## 1. Purpose

This document defines what “failure” means in SAFE-SILENCE, which failure causes are hidden, what evidence is available to the digital twin, which distinctions are identifiable, how burst and correlated outages are generated, and how the system responds without oracle knowledge.

The governing safety principle is:

> When the cause of silence cannot be reliably identified, the system must recognize unsafe ambiguity and limit confidence rather than force an unjustified exact diagnosis.

The primary inference target is therefore **trusted versus ambiguous silence**. Exact root-cause classification is secondary.

## 2. Nonfailure states that must not be mislabeled

| Code | State | Meaning |
|---|---|---|
| NF0 | Normal update | A packet was generated and successfully received |
| NF1 | Intentional valid silence | No packet was generated because the active contract and node policy allowed suppression |
| NF2 | Predictable environmental evolution | The environment remained within the declared predictive validity region |
| NF3 | Genuine critical event | A real environmental event occurred; this is a target condition, not a network or sensor failure |

A critical event and a communication failure may occur simultaneously. The evaluation representation must support multiple concurrent labels.

## 3. Failure ontology

### 3.1 Sensor and measurement failures

| Code | Failure | Operational definition | Receiver identifiability |
|---|---|---|---|
| S1 | Missing sample | Expected measurement absent at the node | Only if reported later or inferred indirectly |
| S2 | Transient spike | Isolated physically implausible deviation | Partial through quality rules and neighboring sensors |
| S3 | Stuck-at/flatline | Value remains nearly constant beyond physically plausible duration | Partial |
| S4 | Bias/drift | Gradual systematic deviation from reference or peers | Difficult; probabilistic |
| S5 | Excess noise | Variance increases beyond normal operating range | Partial |
| S6 | Saturation/range failure | Sensor output clipped at instrument boundary | Often observable if value/quality flag arrives |
| S7 | Calibration/quality rejection | Observation fails declared quality-control rules | Observable only when quality metadata arrives |

### 3.2 Node and computation failures

| Code | Failure | Operational definition | Receiver identifiability |
|---|---|---|---|
| N1 | Energy depletion | Node cannot sense or transmit because usable energy is exhausted | Probabilistic from last energy report and silence |
| N2 | Reboot/reset | State, sequence, predictor, or contract continuity is broken | Often identifiable after restart packet |
| N3 | Clock desynchronization | Timestamps exceed allowed offset/drift | Identifiable when packets arrive |
| N4 | Predictor divergence | Local and receiver predictors become materially inconsistent | Identifiable after anchor/update comparison |
| N5 | Buffer overflow | Generated packet dropped before acoustic transmission | Partial if local counters later arrive |
| N6 | Compute overrun | Inference or encoding misses its decision deadline | Partial |
| N7 | Software/protocol fault | Malformed state or illegal protocol transition | Observable only if evidence arrives |

### 3.3 Link and packet failures

| Code | Failure | Operational definition | Receiver identifiability |
|---|---|---|---|
| C1 | Isolated packet loss | Single transmission attempt fails outside a sustained outage | Usually not separable from no generation without evidence |
| C2 | Packet corruption | Packet arrives but fails CRC/decoding | Observable |
| C3 | Collision/interference | Simultaneous transmissions prevent decoding | Partial through network evidence |
| C4 | Degraded link | Success probability/delay worsens but connectivity remains possible | Probabilistic |
| C5 | Burst outage | Consecutive period of near-zero or zero link availability | Detectable with delay |
| C6 | Excessive delay | Packet arrives after its semantic usefulness deadline | Observable after arrival; uncertain before |
| C7 | Reordering | Packets arrive out of sensing-time order | Observable through timestamps/sequence numbers |
| C8 | Duplicate/replay | A previously processed packet reappears | Observable |
| C9 | ACK loss | Data delivery and sender/receiver contract beliefs may diverge | Ambiguous without cumulative acknowledgment |

### 3.4 Route, relay, and topology failures

| Code | Failure | Operational definition | Receiver identifiability |
|---|---|---|---|
| R1 | Route disappearance | No viable current acoustic path from source to buoy | Probabilistic unless topology is measured |
| R2 | Relay failure | Relay stops receiving, processing, or forwarding | Partial |
| R3 | Relay energy exhaustion | Relay becomes unavailable due to energy state | Probabilistic from last report |
| R4 | Queue congestion | Packets expire or drop in a relay queue | Partial through queue summaries |
| R5 | Partition | A source subset becomes disconnected from the buoy | Detectable with delay and spatial correlation |
| R6 | Mobility-induced handover gap | AUV/relay movement temporarily removes the path | Partial if planned trajectory is known |
| R7 | Fusion/summary loss | Relay summary is lost after source packets were received | Indistinguishable from upstream loss without relay evidence |

### 3.5 Common-cause failures

| Code | Failure | Operational definition |
|---|---|---|
| G1 | Regional acoustic disturbance | Several spatially related links degrade simultaneously |
| G2 | Buoy/gateway interruption | Multiple source paths disappear because the common sink is unavailable |
| G3 | Shared relay failure | All sources depending on a relay become unreachable |
| G4 | Environmental interference | Physical conditions jointly change communication quality across links |
| G5 | Coordinated energy exhaustion | Several nodes fail after similar duty cycles |

Common-cause events must be modeled because independent Bernoulli packet loss unrealistically underestimates simultaneous information loss.

## 4. Hidden ground-truth state

For source $i$, the simulator maintains an oracle state:

\[
\Omega_{i,t}=
\{z_{i,t},s_{i,t}^{H},n_{i,t}^{H},
\mathbf{c}_{i,t}^{H},r_{i,t}^{H},g_t^{H}\},
\]

where:

- $z_{i,t}$: genuine environmental event state;
- $s_{i,t}^{H}$: sensor-health state;
- $n_{i,t}^{H}$: node-health state;
- $\mathbf{c}_{i,t}^{H}$: relevant link states;
- $r_{i,t}^{H}$: route/relay state;
- $g_t^{H}$: common-cause disturbance state.

Superscript $H$ denotes hidden ground truth. This state is written to an evaluation-only log and is never supplied to SAFE-SILENCE, its predictor, trigger, fusion module, or scheduler.

## 5. Primary acoustic outage process

### 5.1 Link-level hidden semi-Markov model

Each link $(u,v)$ has state

\[
c_{uv,t}\in\{G,D,O\},
\]

for good, degraded, and outage. State transitions follow:

\[
P_{uv}=
\begin{bmatrix}
p_{GG}&p_{GD}&p_{GO}\\
p_{DG}&p_{DD}&p_{DO}\\
p_{OG}&p_{OD}&p_{OO}
\end{bmatrix}.
\]

The primary model is hidden semi-Markov rather than memoryless Markov: after entering state $s$, dwell duration is sampled from

\[
D_s\sim F_s(\vartheta_s).
\]

Candidate duration families are geometric, negative-binomial, Weibull, and log-normal. The selected family and parameters must be fitted from available channel traces or transparently defined as simulation factors. The two-state Gilbert–Elliott process is retained as a required baseline/sensitivity model.

Conditional packet-success probabilities satisfy:

\[
p_G>p_D>p_O,
\]

with $p_O$ near zero but not automatically forced to zero unless the scenario represents a hard partition.

### 5.2 Packet delay and corruption

For a generated packet $p$:

\[
\delta_p=\delta_p^{\mathrm{prop}}+
\delta_p^{\mathrm{queue}}+
\delta_p^{\mathrm{access}}+
\delta_p^{\mathrm{retx}}.
\]

The packet is semantically expired if its arrival time exceeds its packet-specific usefulness deadline, even if decoding succeeds.

CRC failure, collision, queue drop, expiry, and acoustic channel loss remain distinct oracle causes. They may share the same receiver observation: no usable packet.

### 5.3 End-to-end reachability

For source $i$, define:

\[
A_{i,t}^{\mathrm{path}}=
\mathbb{1}\!\left[
\exists\ \text{viable path }i\rightsquigarrow b
\right].
\]

This oracle path state is not directly revealed. SAFE-SILENCE estimates:

\[
\widehat{\gamma}_{i,t}
=P(A_{i,t}^{\mathrm{path}}=1\mid\mathcal{I}^{D}_t).
\]

### 5.4 Spatially correlated outages

Let nodes and links be grouped into spatial/acoustic regions $k$. Introduce a latent regional disturbance:

\[
g_{k,t}\in\{0,1\}.
\]

When $g_{k,t}=1$, transition probabilities and packet-success parameters of affected links are jointly modified. This produces correlated loss without forcing identical outcomes.

At minimum, experiments must separate:

- independent link outages;
- shared-relay outages;
- regional correlated degradation;
- hard network partition.

### 5.5 Relationship to environmental events

The primary experiment generates communication outages independently of event labels:

\[
\Omega^{\mathrm{comm}}\perp z
\quad\text{for the primary matched comparison}.
\]

Event-correlated outages are a prespecified stress test:

\[
P(\Omega^{\mathrm{comm}}\mid z=1)
\ne
P(\Omega^{\mathrm{comm}}\mid z=0).
\]

This separation prevents the main result from depending on an artificially favorable or adversarial coupling while still testing the dangerous case in which a critical environmental change coincides with reduced acoustic reliability.

## 6. Sensor and node failure processes

### 6.1 Sensor health

Sensor health follows:

\[
s_{i,t}^{H}\in
\{\mathrm{HEALTHY},\mathrm{NOISY},\mathrm{DRIFT},
\mathrm{STUCK},\mathrm{MISSING},\mathrm{SATURATED}\}.
\]

The primary analysis uses authentic quality flags and natural missingness where available. Controlled faults are sensitivity tests and must be injected only into held-out copies, never into the source data.

### 6.2 Node availability

\[
n_{i,t}^{H}\in
\{\mathrm{ACTIVE},\mathrm{DEGRADED},\mathrm{REBOOTING},
\mathrm{ENERGY\_EMPTY},\mathrm{FAILED}\}.
\]

Node availability governs whether sensing, computation, packet generation, and transmission can occur. A node may sense successfully yet fail to transmit because of buffer or compute failure.

### 6.3 Battery evolution

\[
E_{i,t+1}=
\min\{E_i^{\max},
E_{i,t}-E_{i,t}^{\mathrm{sense}}
-E_{i,t}^{\mathrm{compute}}
-E_{i,t}^{\mathrm{tx}}
-E_{i,t}^{\mathrm{rx}}
-E_{i,t}^{\mathrm{idle}}
+E_{i,t}^{\mathrm{harvest}}\}.
\]

Energy harvesting is optional and disabled unless explicitly studied. A node cannot transmit when the required energy exceeds its available budget.

## 7. Receiver-observable evidence

The digital twin may observe only:

- successful packet arrivals;
- sensing/transmit/arrival timestamps;
- source and relay sequence numbers;
- CRC/decoding outcome for detected packets;
- cumulative ACK/NACK and probe outcomes;
- round-trip-time and delay history;
- declared contract horizons and liveness deadlines;
- last reported energy class;
- delivered sensor quality/provenance flags;
- relay provenance, queue summaries, and retained/dropped counts;
- known routing priors or planned AUV trajectories;
- cross-source spatial/temporal consistency.

It does not observe:

- whether an unreceived packet was generated;
- exact channel state;
- exact queue-drop location;
- current battery after the last report;
- current sensor health without delivered evidence;
- event truth;
- exact root cause of silence.

## 8. Receiver hypothesis space

For each source and epoch, define:

\[
K_{i,t}\in
\{\mathrm{VALID\_SUPPRESSION},
\mathrm{ISOLATED\_LOSS},
\mathrm{BURST\_OUTAGE},
\mathrm{NODE\_UNAVAILABLE},
\mathrm{ROUTE\_RELAY\_FAILURE},
\mathrm{DELAYED\_IN\_FLIGHT},
\mathrm{SENSOR\_INVALID},
\mathrm{UNKNOWN}\}.
\]

SAFE-SILENCE estimates:

\[
\mathbf{p}^{K}_{i,t}
=P(K_{i,t}\mid\mathcal{I}^{D}_t).
\]

UNKNOWN is mandatory. The model must not redistribute irreducible uncertainty into a falsely confident named cause.

## 9. Hierarchical inference targets

### Level 1: safety trust

\[
Y^{(1)}_{i,t}\in
\{\mathrm{TRUSTED},\mathrm{AMBIGUOUS}\}.
\]

This is the primary detection target and determines whether the twin may continue presenting a nonstale environmental estimate without a warning.

### Level 2: reachability

\[
Y^{(2)}_{i,t}\in
\{\mathrm{REACHABLE},\mathrm{DEGRADED},\mathrm{UNREACHABLE}\}.
\]

### Level 3: coarse cause

The posterior over $K_{i,t}$ supports recovery selection. Exact cause accuracy is secondary because several causes are observationally equivalent.

### Level 4: detailed oracle cause

Fine-grained simulator labels C1–C9, R1–R7, S1–S7, and N1–N7 are used only for diagnostic evaluation, not as a required deployable prediction target.

## 10. Connectivity and cause inference

The connectivity filter updates:

\[
b^{C}_{i,t}
\propto
p(o_{i,t}\mid c^{\mathrm{path}}_{i,t})
\sum_{c'}
p(c^{\mathrm{path}}_{i,t}\mid c')
b^{C}_{i,t-1}(c'),
\]

where $o_{i,t}$ contains arrivals, deadline misses, ACK/probe results, sequence gaps, and correlated-source evidence.

The cause posterior combines:

\[
\mathbf{p}^{K}_{i,t}
=\mathcal{H}
\left(
b^{C}_{i,t},
V^{E}_{i,t},
\text{sequence evidence},
\text{relay evidence},
\text{quality evidence},
\text{cross-source evidence}
\right).
\]

The primary implementation should be transparent: Bayesian/HMM filtering plus deterministic evidence rules or calibrated lightweight classification. A large black-box failure classifier is not required.

## 11. Contract-safe silence decision

Silence may be classified as trusted only when all conditions hold:

1. environmental contract remains valid;
2. liveness deadline has not expired or fresh evidence supports reachability;
3. estimated critical-event risk remains below threshold;
4. environmental uncertainty remains below threshold;
5. no sequence, quality, relay, or cross-source evidence indicates failure;
6. posterior trusted-suppression probability exceeds $\tau_T$.

\[
P^{T}_{i,t}
=P(K_{i,t}=\mathrm{VALID\_SUPPRESSION}
\mid\mathcal{I}^{D}_t).
\]

\[
\mathrm{TRUSTED}_{i,t}
=\mathbb{1}
\left[
P^{T}_{i,t}\ge\tau_T
\land V^{E}_{i,t}=1
\land \widehat{q}_{i,t}\le\tau_q
\land u^E_{i,t}\le\tau_u
\right].
\]

Failure of any condition yields ambiguous silence; it does not automatically identify the root cause.

## 12. Failure-aware silence state machine

| Current state | Trigger | Next state | Required response |
|---|---|---|---|
| UPDATED | Valid contract, no new update | TRUSTED_SILENCE | Continue forecast with growing uncertainty |
| TRUSTED_SILENCE | Horizon/deadline expiry or falling reachability belief | AMBIGUOUS_SILENCE | Mark uncertainty; evaluate low-cost probe |
| TRUSTED_SILENCE | High predicted event risk | STALE_UNSAFE or RECOVERY_PENDING | Prioritize evidence acquisition |
| AMBIGUOUS_SILENCE | Fresh valid update/heartbeat plus environmental evidence | UPDATED or TRUSTED_SILENCE | Reconcile state |
| AMBIGUOUS_SILENCE | Repeated missed checks or low connectivity posterior | FAILURE_SUSPECTED | Select recovery |
| FAILURE_SUSPECTED | Recovery action issued | RECOVERY_PENDING | Await bounded response |
| RECOVERY_PENDING | Valid evidence arrives | UPDATED | Reconcile contracts and sequence |
| RECOVERY_PENDING | Retry/recovery budget exhausted | STALE_UNSAFE | Stop presenting unqualified normality |
| STALE_UNSAFE | Valid update and resynchronization | UPDATED | Clear warning only after verification |

There is no direct transition from prolonged ambiguous silence to “normal.”

## 13. Recovery action set

\[
\mathcal{A}^{\mathrm{rec}}=
\{\mathrm{WAIT},
\mathrm{PROBE},
\mathrm{HEARTBEAT\_REQUEST},
\mathrm{TARGETED\_RETX},
\mathrm{REROUTE},
\mathrm{RELAY\_CHANGE},
\mathrm{STORE\_CARRY\_FORWARD},
\mathrm{SLOT\_RESERVE},
\mathrm{CROSS\_CHECK},
\mathrm{RESYNC},
\mathrm{FLAG\_STALE\_UNSAFE}\}.
\]

| Dominant posterior condition | Preferred response |
|---|---|
| Valid suppression likely | WAIT |
| Weak ambiguity, contract near expiry | PROBE or HEARTBEAT_REQUEST |
| Isolated packet loss likely | TARGETED_RETX |
| Burst outage likely | Backoff plus REROUTE or STORE_CARRY_FORWARD |
| Congestion/collision likely | SLOT_RESERVE and reprioritization |
| Shared relay/route failure likely | RELAY_CHANGE or REROUTE |
| Node unavailable likely | CROSS_CHECK neighboring sources; flag source unavailable |
| Sensor invalid likely | CROSS_CHECK; suppress unsupported event interpretation |
| Clock/predictor mismatch | RESYNC |
| Evidence remains insufficient | FLAG_STALE_UNSAFE |

Recovery actions are bounded. Unlimited probing or retransmission is prohibited because it can waste energy during long outages.

## 14. Recovery decision rule

For candidate recovery action $a$:

\[
J^{\mathrm{rec}}_a(t)=
\mathbb{E}
\left[
\Delta\operatorname{CVaR}_{\alpha}(L)
\mid a,\mathbf{p}^{K}_{i,t}
\right]
-\lambda_E E_a
-\lambda_B B_a
-\lambda_D D_a.
\]

Choose:

\[
a_t^*=
\begin{cases}
\arg\max_a J^{\mathrm{rec}}_a(t),
&\max_a J^{\mathrm{rec}}_a(t)>0,\\
\mathrm{FLAG\_STALE\_UNSAFE},
&\text{otherwise}.
\end{cases}
\]

Mandatory safety overrides may select a probe or alert slot when declared critical-risk thresholds are crossed. This is a transparent expected-value policy, not a claim of globally optimal recovery.

## 15. Contract reconciliation under ACK loss

SAFE-SILENCE requires:

- monotonic source sequence numbers;
- contract identifiers;
- cumulative acknowledgment of latest accepted sequence/contract;
- idempotent processing of retransmissions;
- explicit resynchronization after reboot or state mismatch;
- sender-side contract renewal only under the locked acknowledgment policy.

An ACK packet has nonzero byte, delay, and energy cost and is included in accounting.

## 16. Scenario matrix

Every method is evaluated on the same stored network realizations.

| Scenario | Core condition | Purpose |
|---|---|---|
| F0 | Stable high-reliability links | Verify low unnecessary recovery overhead |
| F1 | Independent random packet loss | Conventional reference condition |
| F2 | Short burst outages | Test rapid ambiguity recognition |
| F3 | Long heavy-tailed outages | Test bounded retries and stale-unsafe behavior |
| F4 | Degraded high-delay links | Separate delay from hard loss |
| F5 | ACK loss/asymmetry | Test contract divergence and reconciliation |
| F6 | Relay failure | Test correlated downstream silence and rerouting |
| F7 | Regional correlated outage | Test multi-source evidence and common-cause inference |
| F8 | Hard partition | Test safe degradation when recovery is impossible |
| F9 | Congestion and queue drops | Test packet prioritization and expiry |
| F10 | Node energy exhaustion/reboot | Test source availability inference |
| F11 | Sensor drift/stuck/noisy | Test sensor versus communication uncertainty |
| F12 | Clock drift/reordering/duplicates | Test protocol integrity |
| F13 | Event coincident with outage | Primary safety stress test |
| F14 | Unseen outage-duration distribution | Generalization test |
| F15 | Unseen topology/node density | Structural generalization test |

## 17. Primary failure-awareness metrics

### 17.1 Safety metrics

- **False-trust rate:** unsafe/failed silence epochs classified as trusted.
- **Critical false-trust rate:** false trust while a critical event is active.
- **Ambiguity recall:** unsafe silence epochs correctly flagged ambiguous.
- **Time to ambiguity:** delay from failure onset to ambiguous classification.
- **Time to stale-unsafe:** delay until unqualified normal presentation stops.
- **Critical-event miss probability under outage.**
- **Event notification delay.**

False trust is the primary failure-awareness error; false ambiguity is secondary because it costs communication but is less safety-critical.

### 17.2 Probabilistic metrics

- Brier score and log loss for reachability belief;
- calibration curve and expected calibration error;
- environmental uncertainty coverage;
- cause-posterior entropy;
- selective accuracy versus abstention/UNKNOWN rate.

### 17.3 Cause metrics

- macro-F1 and balanced accuracy for coarse causes;
- per-cause recall and confusion matrix;
- hierarchical loss distinguishing a wrong trusted decision from a wrong named cause.

Exact detailed-cause accuracy is diagnostic, not the primary success criterion.

### 17.4 Recovery and resource metrics

- successful recovery fraction and delay;
- probe/heartbeat/retransmission bytes and energy;
- unnecessary recovery under valid silence;
- retry-budget exhaustion;
- network lifetime;
- duplicate and stale-packet processing rate.

## 18. Required baselines and ablations

1. periodic transmission;
2. event-triggered prediction residual;
3. AoI;
4. AoII;
5. uncertainty-only semantic scheduling;
6. loss detector without environmental semantics;
7. naive silence treated as normal;
8. every silence treated as failure;
9. SAFE-SILENCE without connectivity belief;
10. SAFE-SILENCE without contract horizon;
11. SAFE-SILENCE without regional/cross-source evidence;
12. SAFE-SILENCE without UNKNOWN/abstention;
13. SAFE-SILENCE without recovery-cost accounting;
14. oracle silence cause;
15. full SAFE-SILENCE.

The oracle is an upper bound, not a deployable competitor.

## 19. Split and parameter discipline

- Outage parameters are fitted on training/validation evidence only.
- Thresholds $\tau_T,\tau_q,\tau_u$ are selected on validation data.
- Test outage realizations use independent seeds.
- Generalization tests change duration distribution, transition intensity, topology, or correlation structure.
- Network realization files are generated once and replayed identically across methods.
- Event labels do not influence primary outage generation.
- Failure injection records onset, magnitude, duration, and recovery parameters.
- Hyperparameters cannot change after final test inspection.

## 20. Required implementation tests

1. **Oracle isolation:** hidden causes never enter model inputs.
2. **Concurrent labels:** event plus outage can coexist.
3. **Cause conservation:** every unusable packet has one immediate drop cause and may have parent/common causes.
4. **Duration fidelity:** sampled dwell times match configured distributions.
5. **Correlation fidelity:** regional scenarios produce target inter-link correlation.
6. **ACK asymmetry:** data-success/ACK-failure states are represented.
7. **Expiry correctness:** late success is not counted as timely delivery.
8. **Reordering safety:** assimilation uses sensing time and sequence checks.
9. **Contract reconciliation:** duplicates and lost ACKs do not create silent divergence.
10. **Bounded recovery:** probes/retries stop at configured budget.
11. **No normal fallback:** exhausted recovery leads to STALE_UNSAFE.
12. **Matched realizations:** methods see identical environment and network traces.
13. **Primary independence:** event labels and primary outages are independent.
14. **Stress coupling:** event-correlated outage appears only in declared stress scenarios.
15. **Calibration separation:** connectivity and environmental calibration are reported separately.

## 21. Parameters deliberately left open

- state transition and dwell-time distributions;
- packet-success and delay distributions per channel state;
- regional correlation grouping;
- ACK loss and asymmetry rates;
- route discovery and relay-change delay;
- controlled sensor-fault onset, duration, and magnitude;
- recovery cost weights and retry budgets;
- trust, event-risk, and uncertainty thresholds;
- prior probability of each coarse cause;
- cross-source evidence model.

These must be version-controlled and included in sensitivity analysis.

## 22. Claims permitted and prohibited

Permitted if supported:

- SAFE-SILENCE reduces dangerous false trust under tested intermittent conditions;
- separate connectivity belief improves ambiguity detection;
- bounded-contract and recovery outcomes are evaluated under common caps with actual costs reported;
- recovery remains bounded under long outages;
- the method generalizes to specified unseen outage distributions/topologies.

Prohibited:

- exact identification of every silence cause;
- guaranteed node-failure detection;
- guaranteed critical-event delivery;
- physically measured energy without hardware measurement;
- robustness to arbitrary failures not tested;
- field validation without a real deployment.

