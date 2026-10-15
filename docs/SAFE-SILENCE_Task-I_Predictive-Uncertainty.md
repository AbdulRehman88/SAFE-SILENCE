# SAFE-SILENCE Task I — Predictive Environmental Uncertainty

## Purpose

Task I supplies the environmental belief used by the bounded-silence contract. It predicts
the next multivariate ocean state and provides calibrated uncertainty without mixing that
uncertainty with future connectivity uncertainty.

## Leakage contract

1. Each temporal protocol is modeled independently.
2. Channel scaling is fitted only from genuinely observed CE02SHSM training values.
3. Task G validation windows are assigned chronologically to two disjoint roles: tuning
   controls early stopping; calibration fits split-conformal multipliers.
4. Calibration never controls training or checkpoint selection.
5. Internal-test and external-site data affect no learned or calibrated quantity.
6. Sequences remain inside one immutable 30-day `window_key`.

## Predictor

The node model is a compact persistence-residual causal TCN with three 32-channel blocks
and fewer than 250,000 parameters. Persistence is the explicit causal backbone and the TCN
learns only its correction, preventing a flexible model from needlessly relearning the
strong short-horizon identity dynamic. At every step it receives standardized values plus
observed, imputed, and age evidence. Its diagonal Gaussian head estimates a mean and
aleatoric variance for temperature, salinity, oxygen, and chlorophyll. Five independently
initialized members add epistemic dispersion through total-variance decomposition.
Both protocols use the same eight-hour physical history: 32 steps at 15 minutes and eight
steps at 60 minutes.

## Calibration and evaluation

Worst-window normalized split conformal calibration uses

`|observed - ensemble_mean| / predictive_standard_deviation`

on each calibration window separately, then freezes the largest window-specific multiplier.
This is more conservative under temporal regime variation than a pooled calibration
quantile while remaining test- and external-site blind. Results report RMSE, MAE, coverage,
and interval width overall and
on valid critical-event rows, alongside a persistence baseline. Protocol results must not
be pooled.

## Outputs

- `artifacts/task_i/<protocol>/`: training-only scaler, five model checkpoints, and frozen
  conformal multipliers.
- `results/task_i/predictions/`: one-step means, uncertainties, intervals, targets, masks,
  and persistence predictions.
- `results/task_i/predictive_metrics.csv`: protocol/site/role/channel metrics.
- `results/task_i/training_history.csv`: member-level learning histories.
- `results/task_i/task_i_summary.json`: device, leakage declarations, role counts, hashes,
  and artifact inventory.
