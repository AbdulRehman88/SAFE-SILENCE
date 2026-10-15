# SAFE-SILENCE Task H — Leakage-Safe Critical-Event Labels

## Scientific contract

Task H converts the immutable Task G partitions into observation-grounded event labels.
It does not change window eligibility, site roles, or split assignments.

1. Statistical thresholds are fitted separately for each temporal protocol using only
   `CE02SHSM` rows whose immutable Task G split is `train`.
2. Only QARTOD-PASS measurements with `observed_<channel> == 1` can calibrate a threshold
   or define event truth. A causally filled value is context, never ground truth.
3. Thresholds are serialized before validation, internal-test, or external-site labels
   are produced. External data cannot alter them.
4. Causal differences reset at each `window_key`; no transition crosses a 30-day window.
5. Persistence uses trailing confirmation and never retroactively labels an onset.

## Locked labels

| Label | Definition | Validity requirement |
|---|---|---|
| Absolute hypoxia | Oxygen ≤ 60.979 µmol/kg for two observed steps | Current oxygen observed |
| Low-oxygen anomaly | Oxygen ≤ CE02-train 5th percentile for two observed steps | Current oxygen observed |
| Rapid temperature change | Absolute causal rate ≥ CE02-train 99.5th percentile | Current and previous temperature observed and contiguous |
| High-chlorophyll anomaly | Chlorophyll ≥ CE02-train 99.5th percentile for two observed steps | Current chlorophyll observed |
| Multivariate innovation anomaly | Robust L2 causal-innovation score ≥ CE02-train 99.5th percentile | Current and previous values observed for all four channels |

The absolute hypoxia value is the NOAA ≤2 mg/L operational reference converted with the
declared 31.998 g/mol oxygen molar mass and 1.025 kg/L seawater-density assumption. It is
not pooled with the empirical low-oxygen threshold.

## Output contract

`data/processed/task_h/` contains labeled Parquet datasets, a frozen threshold document,
an episode registry, event counts by protocol/site/split, and a SHA-256 artifact manifest
inside `labeling_summary.json`.

`valid_critical_event` is true only when every critical constituent is observable. This
prevents oxygen-absent sites from being counted as verified no-event intervals.
`label_available_event` remains available for explicitly named partial-observability stress
analyses, but it is not interchangeable with the primary critical-event ground truth.

The 15-minute and 60-minute protocols resample overlapping physical observations. Their
episode totals are therefore reported separately and must never be pooled or described as
a count of unique ecological events. Event classes with zero positive support are retained
as transparent negative findings but are not assigned recall, delay, or miss-rate scores.
