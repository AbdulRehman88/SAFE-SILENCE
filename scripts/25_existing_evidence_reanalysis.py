"""Reproduce frozen CE07 recall aggregates and global-56 Holm sensitivity."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
metrics_path = root / "results/task_s/replication/task_r/system_metrics.csv"
comparisons_path = root / "results/task_s/confirmatory_comparisons.csv"
frame = pd.read_csv(metrics_path)
frame = frame[frame.site.eq("CE07SHSM") & frame.model_role.eq("external_test")]
rows = []
for protocol, part in frame.groupby("protocol", sort=True):
    proposed = part[part.scheduler_policy.eq("cvar_risk_reduction_per_joule") & part.recovery_policy.eq("safe_silence_bounded")]
    for name, scheduler, recovery in [
        ("no_recovery", "cvar_risk_reduction_per_joule", "no_recovery"),
        ("expected_scheduler", "expected_risk_reduction_per_joule", "safe_silence_bounded"),
    ]:
        reference = part[part.scheduler_policy.eq(scheduler) & part.recovery_policy.eq(recovery)]
        paired = proposed.merge(reference, on=["scenario", "repetition"], suffixes=("_p", "_r"), validate="one_to_one")
        assert len(paired) == 210 and paired.scenario.nunique() == 7
        rng = np.random.default_rng(59321)
        draws = []
        for _, group in paired.groupby("scenario", sort=True):
            assert len(group) == 30
            a = group.post_recovery_critical_delivery_recall_p.to_numpy(float)
            b = group.post_recovery_critical_delivery_recall_r.to_numpy(float)
            indices = rng.integers(len(a), size=(10000, len(a)))
            draws.append(np.column_stack([a[indices].mean(1), b[indices].mean(1), (a - b)[indices].mean(1)]))
        ci = np.quantile(np.mean(draws, axis=0), [.025, .975], axis=0)
        rows.append({
            "protocol": protocol, "comparison": name, "paired_repetitions": 210,
            "proposed_mean": paired.post_recovery_critical_delivery_recall_p.mean(),
            "reference_mean": paired.post_recovery_critical_delivery_recall_r.mean(),
            "mean_paired_difference": (paired.post_recovery_critical_delivery_recall_p - paired.post_recovery_critical_delivery_recall_r).mean(),
            "proposed_ci_lower": ci[0, 0], "proposed_ci_upper": ci[1, 0],
            "reference_ci_lower": ci[0, 1], "reference_ci_upper": ci[1, 1],
            "difference_ci_lower": ci[0, 2], "difference_ci_upper": ci[1, 2],
        })
comparisons = pd.read_csv(comparisons_path)
comparisons = comparisons[comparisons.site.eq("CE07SHSM") & comparisons.model_role.eq("external_test")].copy()
assert len(comparisons) == 56
p = comparisons.wilcoxon_p_value.to_numpy(float)
order = np.argsort(p)
adjusted = np.empty(len(p))
adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
comparisons["global_56_holm_adjusted_p_value"] = adjusted
comparisons["global_56_reject"] = adjusted <= .05
output = root / "results/existing_evidence_reanalysis"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(output / "recall_aggregate_intervals.csv", index=False)
comparisons.to_csv(output / "global_56_holm.csv", index=False)
summary = {
    "analysis_scope": "existing frozen CE07 evidence only; no new experiments",
    "scenario_weighting": "equal",
    "bootstrap_method": "paired repetition resampling independently within each scenario",
    "bootstrap_resamples": 10000, "bootstrap_seed": 59321,
    "global_primary_hypotheses": 56, "global_holm_rejections": int((adjusted <= .05).sum()),
    "same_rejection_decisions": bool(np.array_equal(adjusted <= .05, comparisons.reject_holm_null.to_numpy(bool))),
    "input_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [metrics_path, comparisons_path]},
}
(output / "reanalysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
