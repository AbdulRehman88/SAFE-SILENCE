# SAFE-SILENCE — Task C System Model and End-to-End Method Specification

**Version:** 1.0  
**Date:** 16 September 2026  
**Scope:** Conceptual model; executable settings and evaluated mechanisms are defined by the source code and experiment configuration.

## 1. Purpose and design boundary

This document defines the causal, mathematical, and operational system model for SAFE-SILENCE. It specifies:

- physical and computational entities;
- environmental, communication, and knowledge states;
- information available to each entity at decision time;
- semantic packet and bounded-silence-contract fields;
- node, relay, and digital-twin operations;
- dual environmental/connectivity beliefs;
- criticality-preserving in-network fusion;
- tail-risk-per-joule scheduling;
- constraints, outputs, and required implementation invariants.

This is a design specification, not a statement of experimental results. It does not assume perfect failure detection, perfect event labels, error-free acknowledgments, globally available channel state, or field-validated energy measurements.

## 2. System objective

Consider an intermittently connected underwater sensing network that updates a surface-side environmental digital twin. The system must reduce acoustic communication while preventing suppressed, delayed, or lost updates from creating false confidence that the environment remains normal.

The operational objective is:

> Minimize safety-weighted digital-twin tail loss and critical-event misses subject to communication, energy, packet-size, latency, and causal-information constraints.

The primary comparative condition is equal transmitted bytes; equal estimated communication energy is a co-primary resource-matched analysis.

## 3. Network entities and topology

Let the directed time-varying communication graph be

\[
\mathcal{G}_t=(\mathcal{V},\mathcal{E}_t),
\]

with:

- sensor nodes $\mathcal{N}=\{1,\ldots,N\}$;
- relay/AUV nodes $\mathcal{R}=\{1,\ldots,R\}$;
- one surface gateway/buoy $b$;
- one surface or shore digital twin $D$ logically connected to $b$.

Thus,

\[
\mathcal{V}=\mathcal{N}\cup\mathcal{R}\cup\{b\}.
\]

The primary architecture is sensor $\rightarrow$ relay/AUV $\rightarrow$ buoy $\rightarrow$ digital twin. Direct sensor-to-buoy transmission and multiple relay hops are supported as topology variants. The digital twin is treated as co-located with or reliably backhauled from the buoy; unreliable underwater acoustic communication is the modeled bottleneck.

## 4. Time model

Environmental sensing occurs at discrete epochs $t\in\{0,1,\ldots,T\}$ separated by $\Delta_s$. Acoustic transmission uses slots or packet events at a finer network time scale. Every packet carries source sensing time and transmit time; receiver-side freshness is computed from sensing time, not arrival order.

The system supports nonzero propagation delay, queueing delay, retransmission delay, and out-of-order arrival. No method may use observations generated after its decision time.

## 5. Environmental state and observations

### 5.1 Latent environmental state

For node $i$, let

\[
\mathbf{x}_{i,t}\in\mathbb{R}^{M}
\]

denote the latent local environmental state. Candidate dimensions include temperature, salinity, dissolved oxygen, pressure, turbidity, and chlorophyll. The final variable set is determined by Task E–G data availability and event validity.

### 5.2 Sensor observation

Node $i$ observes

\[
\mathbf{y}_{i,t}=\mathbf{x}_{i,t}+\boldsymbol{\epsilon}_{i,t},
\]

where $\boldsymbol{\epsilon}_{i,t}$ includes measurement noise, calibration error, drift, and quality-control effects. Missing or quality-rejected observations are not silently imputed as valid measurements.

### 5.3 Environmental predictor

Each node uses a compact causal predictor $f_{\theta}$:

\[
(\widehat{\mathbf{y}}_{i,t|t-1},\boldsymbol{\sigma}^{E}_{i,t})
=f_{\theta}(\mathbf{y}_{i,t-W:t-1},\mathbf{m}_{i,t-W:t-1}),
\]

where $W$ is the history length, $\mathbf{m}$ is a validity/missingness mask, $\widehat{\mathbf{y}}$ is the forecast, and $\boldsymbol{\sigma}^{E}$ is environmental predictive uncertainty. The primary candidate is a small causal TCN; compact GRU and persistence/linear state-space predictors are required alternatives.

The semantic innovation is

\[
\mathbf{r}_{i,t}=\mathbf{y}_{i,t}-\widehat{\mathbf{y}}_{i,t|t-1}.
\]

Residual magnitude alone is insufficient because small residuals can still be safety-critical near an event boundary.

### 5.4 Critical-event probability

Let $z_{i,t}\in\{0,1\}$ denote whether the observation belongs to a declared critical-event state. Node $i$ estimates

\[
q_{i,t}=P(z_{i,t}=1\mid\mathcal{I}^{N}_{i,t}),
\]

where $\mathcal{I}^{N}_{i,t}$ is the node's causally available information. Event definitions and label construction are locked later in Task G. The architecture supports rule-derived, calibrated probabilistic, or hybrid event scores.

## 6. Underwater acoustic communication state

For directed link $(u,v)$, define latent connectivity state

\[
c_{uv,t}\in\{G,B,O\},
\]

where $G$ is usable/good, $B$ is degraded, and $O$ is unavailable/outage. The primary burst model is a finite-state Markov process; a two-state Gilbert–Elliott model is the minimum baseline.

Conditional packet success is

\[
s_{uv,t}\sim\operatorname{Bernoulli}\!\left(p_{\mathrm{succ}}
(c_{uv,t},\mathrm{SNR}_{uv,t},B_p,M_p)\right),
\]

where $B_p$ is packet size and $M_p$ is the modulation/coding configuration. The network model must also represent:

- propagation delay;
- half-duplex operation;
- packet collision/interference;
- queue capacity;
- retransmission limit;
- transmit/receive/idle energy;
- burst loss and route disappearance.

Independent packet loss is a sensitivity condition, not the primary outage model.

For source-level receiver inference, let $c^{\mathrm{path}}_{i,t}$ denote the latent end-to-end reachability state from source $i$ to the buoy after accounting for the currently available route and all constituent links.

## 7. Causal information sets

### 7.1 Sensor node

At time $t$, node $i$ may use only:

\[
\mathcal{I}^{N}_{i,t}=
\{\mathbf{y}_{i,0:t},\mathbf{m}_{i,0:t},
\text{battery}_{i,t},\text{local ACK/NACK history},
\text{local queue},\text{local contract state}\}.
\]

It may not use future data, receiver-side truth, global topology truth, or oracle channel state.

### 7.2 Relay

Relay $r$ may use:

\[
\mathcal{I}^{R}_{r,t}=
\{\text{received packets}_{0:t},\text{local link evidence},
\text{local ACK/NACK history},\text{queue},\text{energy}\}.
\]

It does not have raw measurements from sources whose packets were not received.

### 7.3 Digital twin

The twin may use:

\[
\mathcal{I}^{D}_{t}=
\{\text{successfully received packets}_{0:t},
\text{arrival times},\text{ACK/probe outcomes},
\text{known protocol and topology priors}\}.
\]

It may not know whether a missing packet was intentionally suppressed, lost, never generated, or blocked by node failure except through probabilistic inference. Oracle cause labels are reserved for an upper-bound baseline.

## 8. Node action space

At each sensing epoch, node $i$ chooses

\[
a_{i,t}\in\{\mathrm{SILENT},\mathrm{HEARTBEAT},
\mathrm{INNOVATION},\mathrm{ALERT}\}.
\]

- **SILENT:** no packet generated; allowed only under an active local contract and locked risk rules.
- **HEARTBEAT:** minimal liveness/contract-renewal packet.
- **INNOVATION:** quantized residual and semantic state update.
- **ALERT:** highest-priority packet for a likely critical event or expired unsafe state.

The action is chosen before transmission success is known. A generated packet and a successfully received packet are distinct events.

## 9. Semantic innovation packet

A source packet is

\[
\mathcal{P}_{i,t}=
\{v,\mathrm{id}_i,n_i,t_s,t_x,\tau_i,\widehat{\mathbf{y}}^{\,a}_{i,t},
Q(\mathbf{r}_{i,t}),q_{i,t},u^{E}_{i,t},
h_{i,t},d_{i,t},\ell_{i,t},e_{i,t},
\pi_{i,t},\mathrm{crc}\}.
\]

Fields:

| Field | Meaning |
|---|---|
| $v$ | Protocol version |
| $\mathrm{id}_i$ | Source identifier |
| $n_i$ | Monotonic sequence number |
| $t_s,t_x$ | Sensing and transmit timestamps |
| $\tau_i$ | Timestamp of previous successfully acknowledged contract |
| $\widehat{\mathbf{y}}^{\,a}$ | Optional compact anchor/predictor synchronization state |
| $Q(\mathbf{r})$ | Quantized innovation or sufficient statistic |
| $q$ | Calibrated critical-event probability |
| $u^E$ | Environmental predictive uncertainty summary |
| $h$ | Environmental validity horizon |
| $d$ | Next liveness/check deadline |
| $\ell$ | Compact link/liveness evidence |
| $e$ | Remaining node energy or energy class |
| $\pi$ | Provenance and quality-control bits |
| crc | Error-detection field |

Every field must have an explicit encoded bit cost. Optional fields require flags. Packet-size accounting includes headers, timestamps, CRC, retransmissions, and relay packets.

## 10. Bounded silence contract

### 10.1 Contract creation

After a successful acknowledged update at time $\tau_i$, node $i$ and the twin share:

\[
\mathcal{C}_{i,\tau_i}=
\{\tau_i,h_{i,\tau_i},d_{i,\tau_i},
q_{i,\tau_i},u^E_{i,\tau_i},\ell_{i,\tau_i}\}.
\]

The environmental validity horizon is

\[
h_{i,\tau_i}=\max\{k\ge 0:
\ \forall j\in\{1,\ldots,k\},\
\widehat{R}^{E}_{i,\tau_i+j}\le\rho_E
\ \land\
\widehat{P}(z_{i,\tau_i+j}=1)\le\rho_z\},
\]

capped by $h_{\max}$. Here $\rho_E$ and $\rho_z$ are validation-selected thresholds. The deadline $d_{i,\tau_i}$ is the latest time at which fresh liveness evidence is expected.

### 10.2 Receiver-side trust

Let $\tau_i^D(t)$ be the latest successfully received contract for source $i$. Define environmental contract validity:

\[
V^E_{i,t}=\mathbb{1}[t-\tau_i^D(t)\le h_{i,\tau_i^D(t)}].
\]

Define the connectivity belief:

\[
\gamma_{i,t}=P(c^{\mathrm{path}}_{i,t}\neq O\mid\mathcal{I}^{D}_{t}).
\]

The receiver-side trusted-silence score is

\[
\chi_{i,t}=V^E_{i,t}\,\gamma_{i,t}\,
\left(1-\widehat{P}(z_{i,t}=1\mid\mathcal{I}^{D}_{t})\right).
\]

Silence is **trusted** only when $\chi_{i,t}\ge\tau_{\mathrm{trust}}$. It becomes **ambiguous** when the environmental horizon or liveness deadline expires, the connectivity posterior falls, or predicted event risk rises.

The term “contract” denotes a shared validity rule, not guaranteed delivery or guaranteed safety.

### 10.3 Contract renewal

A heartbeat can renew liveness evidence but cannot automatically renew the environmental horizon unless it contains enough environmental evidence to justify renewal. This prevents empty heartbeats from indefinitely extending a stale environmental prediction.

## 11. Dual digital-twin beliefs

### 11.1 Environmental belief

\[
b^E_{i,t}=p(\mathbf{x}_{i,t}\mid\mathcal{R}_{0:t},\mathcal{Q}_{0:t}),
\]

where $\mathcal{R}$ contains received innovations and $\mathcal{Q}$ contains sensor quality/provenance information.

### 11.2 Connectivity belief

\[
b^C_{i,t}=p(c_{i,t}\mid\mathcal{A}_{0:t},
\mathcal{K}_{0:t},\mathcal{L}_{0:t}),
\]

where $\mathcal{A}$ contains packet arrivals/absences, $\mathcal{K}$ contains ACK/NACK/probe evidence, and $\mathcal{L}$ contains relay/link observations.

The two beliefs interact in the decision loss but remain separately represented, calibrated, and evaluated. Environmental uncertainty may increase during a perfectly healthy but intentionally silent link; connectivity uncertainty may increase while the environmental prediction remains temporarily precise.

## 12. Digital-twin reconstruction

For source $i$, the twin performs:

\[
\widetilde{\mathbf{x}}_{i,t}=
\begin{cases}
\mathcal{U}(\widetilde{\mathbf{x}}_{i,t^-},\mathcal{P}_{i,t}),
& \text{if an update is received},\\[2mm]
\mathcal{F}(\widetilde{\mathbf{x}}_{i,t-1}),
& \text{otherwise},
\end{cases}
\]

where $\mathcal{U}$ assimilates the packet and $\mathcal{F}$ propagates the synchronized predictor. Receiver uncertainty must expand with forecast horizon, contract expiry, and connectivity ambiguity. Missing updates cannot freeze uncertainty.

## 13. Criticality-preserving in-network fusion

### 13.1 Fusion objective

Relays suppress redundant information while guaranteeing that the summary retains the highest-ranked critical evidence it has actually received.

For packet $p$, define a deterministic priority key:

\[
\kappa(p)=
\left[
\omega_q q_p+
\omega_u u^E_p+
\omega_d\phi(d_p-t)+
\omega_r\|Q(\mathbf{r}_p)\|
\right]\big/\max(B_p,1),
\]

with a fixed tie-break by timestamp and source ID.

### 13.2 Associative bounded summary

Let $\operatorname{TopK}_K(S)$ return the $K$ highest packets under the total ordering induced by $\kappa$. Define:

\[
F_K(A,B)=\operatorname{TopK}_K(A\cup B).
\]

With deterministic total ordering, this operator is commutative, associative, and idempotent:

\[
F_K(F_K(A,B),C)=F_K(A,F_K(B,C)).
\]

Therefore changing relay grouping or aggregation order does not remove an item that belongs to the global top-$K$ evidence set.

### 13.3 Fused packet metadata

A relay summary carries:

- top-$K$ evidence items or compressed sufficient statistics;
- $q_{\max}=\max_p q_p$;
- $u_{\max}=\max_p u^E_p$;
- $d_{\min}=\min_p d_p$;
- earliest sensing time and newest sensing time;
- source/provenance bitmap or compact sketch;
- count of merged, dropped, expired, and quality-rejected items;
- relay sequence number and CRC.

Mean-only fusion is prohibited in the proposed method. If residuals from correlated sensors are averaged, the top-$K$ critical items and maxima remain separately preserved.

### 13.4 Limits

Fusion preserves critical evidence among received packets; it cannot recover evidence lost before the relay. Bloom/sketch collisions and quantization error must be measured.

## 14. Safety-weighted twin loss

For source $i$:

\[
L_{i,t}=
\lambda_m L^{\mathrm{miss}}_{i,t}
+\lambda_d L^{\mathrm{delay}}_{i,t}
+\lambda_r L^{\mathrm{recon}}_{i,t}
+\lambda_a L^{\mathrm{ambiguity}}_{i,t}
+\lambda_c L^{\mathrm{cal}}_{i,t}.
\]

Components:

- $L^{\mathrm{miss}}$: false negative for a declared critical event;
- $L^{\mathrm{delay}}$: event-notification delay penalty;
- $L^{\mathrm{recon}}$: normalized multivariate twin reconstruction error;
- $L^{\mathrm{ambiguity}}$: penalty for high-confidence normal interpretation under weak connectivity evidence;
- $L^{\mathrm{cal}}$: probabilistic miscalibration penalty.

Network loss is a declared weighted aggregate over sources:

\[
L_t=\sum_{i=1}^{N}\alpha_i L_{i,t}.
\]

Weights $\lambda$ and $\alpha$ are fixed using validation data and physical/event priorities before final testing. Results must include sensitivity analysis rather than relying on one convenient weighting.

## 15. Tail-risk-per-joule scheduling

For queued candidate packet $p$, define conditional marginal tail-risk reduction:

\[
\Delta_p(t)=
\operatorname{CVaR}_{\alpha}(L_{t:t+H}\mid\mathcal{I}_t)
-
\mathbb{E}_{s_p}
\left[
\operatorname{CVaR}_{\alpha}
(L_{t:t+H}\mid\mathcal{I}_t,\operatorname{outcome}(p))
\right].
\]

The expectation includes delivery success/failure. Thus a packet on a highly unreliable link does not receive full credit merely because its content is important.

Define:

\[
U_p(t)=
\frac{\max(\Delta_p(t),0)}
{E_p+\eta_B B_p+\eta_D D_p+\varepsilon},
\]

where $E_p$ is estimated communication energy, $B_p$ is total bytes, $D_p$ is expected delay/airtime cost, and $\varepsilon>0$ prevents division by zero.

At each scheduling opportunity:

1. mark mandatory alert/deadline packets;
2. discard expired packets whose utility is no longer positive;
3. reserve minimum liveness capacity if required;
4. rank feasible packets by $U_p(t)$;
5. select greedily subject to airtime, energy, queue, interference, and half-duplex constraints;
6. apply deterministic tie-breaking.

This transparent scheduler is the primary method. Learned MARL is not required.

## 16. Optimization problem

Let $\pi$ denote all node triggering, relay fusion, and network scheduling decisions. The conceptual problem is:

\[
\begin{aligned}
\min_{\pi}\quad&
\operatorname{CVaR}_{\alpha}
\left(\sum_{t=1}^{T}L_t(\pi)\right)\\
\text{s.t.}\quad&
\sum_t E_{i,t}^{\mathrm{comm}}(\pi)
\le E_i^{\max},\ \forall i,\\
&
\sum_{p\in\mathcal{S}_t}B_p\le C_t,\\
&
B_p\le B_{\max},\\
&
\text{half-duplex, interference, route, queue, and causality constraints},\\
&
a_{i,t}\text{ is measurable with respect to }\mathcal{I}^{N}_{i,t},\\
&
a_{r,t}\text{ is measurable with respect to }\mathcal{I}^{R}_{r,t},\\
&
a_{D,t}\text{ is measurable with respect to }\mathcal{I}^{D}_{t}.
\end{aligned}
\]

The implemented algorithm is a tractable online approximation, not a claim of globally optimal control.

## 17. Receiver silence-state machine

For every source, the twin maintains:

1. **UPDATED:** fresh semantic update received;
2. **TRUSTED_SILENCE:** contract valid and connectivity belief sufficient;
3. **AMBIGUOUS_SILENCE:** contract/link evidence insufficient;
4. **FAILURE_SUSPECTED:** connectivity posterior below threshold or repeated deadlines missed;
5. **RECOVERY_PENDING:** probe, heartbeat request, relay change, or retransmission requested;
6. **STALE_UNSAFE:** environmental confidence no longer acceptable; twin output flagged rather than treated as normal.

Transitions are driven by received updates, contract expiry, liveness deadlines, predicted event risk, ACK/probe evidence, and recovery outcomes. The exact failure-state transition probabilities are locked in Task D.

## 18. End-to-end operational algorithm

### At sensor node $i$

1. Acquire observation and quality mask.
2. Generate causal forecast and environmental uncertainty.
3. Compute residual, event probability, and remaining contract validity.
4. Construct candidate SILENT, HEARTBEAT, INNOVATION, and ALERT actions.
5. Estimate local safety value and communication cost.
6. Choose an action under causal node policy.
7. If transmitting, encode packet and contract fields; record generation separately from delivery.
8. Update local contract only according to the locked ACK policy.

### At relay $r$

1. Verify CRC, timestamps, sequence numbers, and provenance.
2. Reject or flag duplicates, expired packets, and invalid evidence.
3. Merge received evidence using associative top-$K$ fusion.
4. Preserve critical maxima and minimum deadline.
5. Compute forwarding utility with link-success probability.
6. Forward subject to energy, queue, and airtime constraints.

### At digital twin $D$

1. Assimilate received innovations in sensing-time order.
2. Update environmental belief.
3. Update connectivity belief from arrivals, absences, ACKs, probes, and relay evidence.
4. Recompute contract validity and silence state.
5. Propagate twin state and expand uncertainty.
6. Issue event alert, liveness probe, retransmission request, route-change request, or stale-unsafe flag as required.
7. Log enough state to reconstruct every transmission and recovery decision.

## 19. Required logs for reproducibility

Every run must store:

- random seed and configuration hash;
- sensing, generation, transmit, arrival, ACK, and assimilation timestamps;
- node action and reason components;
- packet fields and encoded bytes;
- link-state realization and delivery outcome;
- queue and energy state;
- environmental and connectivity beliefs;
- contract state and state-machine transition;
- relay fusion membership, retained/dropped evidence, and reason;
- scheduler candidate utilities and final selection;
- twin estimate, uncertainty, event score, and alert;
- oracle event and outage cause for evaluation only.

Evaluation code must never feed oracle causes back into SAFE-SILENCE.

## 20. Primary invariants and implementation tests

The implementation must verify:

1. **Causality:** no future observation influences a decision.
2. **Generation/delivery separation:** missing reception is not treated as no generation.
3. **Contract ACK integrity:** sender and receiver do not silently diverge on active contract state.
4. **No indefinite heartbeat renewal:** liveness alone cannot perpetually extend environmental validity.
5. **Uncertainty growth:** unobserved twin states do not become more certain without evidence.
6. **Belief separation:** environmental and connectivity uncertainties remain separately inspectable.
7. **Fusion associativity:** relay ordering produces identical top-$K$ evidence for identical inputs.
8. **Criticality preservation:** a highest-ranked alert cannot be removed by lower-ranked averaging.
9. **Byte accounting:** all protocol overhead and retries are counted.
10. **Budget feasibility:** selected transmissions respect energy, airtime, queue, and half-duplex limits.
11. **Determinism:** identical configuration and seed reproduce identical event traces.
12. **Oracle isolation:** oracle labels and outage causes are accessible only to evaluation.

## 21. Configuration parameters

The executable configuration specifies the following quantities using the declared data and validation roles:

- sensor variables and sampling interval;
- predictor history and architecture;
- event definitions and prevalence;
- uncertainty estimator and calibration method;
- $h_{\max}$, $\rho_E$, $\rho_z$, and $\tau_{\mathrm{trust}}$;
- packet quantization and field bit allocation;
- connectivity transition probabilities;
- acoustic energy parameters;
- top-$K$ and relay buffer size;
- loss weights $\lambda$ and source weights $\alpha$;
- CVaR level $\alpha$ and horizon $H$;
- recovery thresholds and retry limits.

The configuration files record the values used in each experiment.

## 22. Claim boundary

This system can estimate whether silence remains trustworthy and can reduce risk under tested conditions. It does not prove that a silent node is healthy, guarantee delivery, guarantee event detection, or constitute a physical sea trial. The bounded silence contract is a probabilistic operational mechanism.

