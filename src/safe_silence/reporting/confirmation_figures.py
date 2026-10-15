"""Read-only plotting of frozen paired confirmation estimates.

No statistics are refitted. Exported intervals are the declared pointwise
paired-bootstrap intervals, not simultaneous or Holm-adjusted intervals.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

RECALL = "post_recovery_critical_delivery_recall"
TRUST = "critical_false_trust_pre_action"
COLORS = ["#0072B2", "#D55E00"]
MARKERS = ["o", "s"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root, cfg):
    lock = json.loads((root / "protocol_lock.json").read_text())
    if lock["new_unique_network_seeds"] != 420 or lock["primary_family_test_count"] != 210:
        raise ValueError("Expected the complete 420-realization, 210-hypothesis protocol")
    seeds, inputs = set(), {}
    for protocol in cfg["protocols"]:
        for scenario in cfg["scenarios"]:
            cell = root / "cells" / protocol / scenario
            done = json.loads((cell / "DONE.json").read_text())
            if done["smoke_only"] or len(done["seeds"]) != 30:
                raise ValueError(f"Incomplete or smoke-only cell: {cell}")
            for seed in done["seeds"].values():
                if seed in seeds:
                    raise ValueError("Network seeds are not unique across cells")
                seeds.add(seed)
            for name, expected in done["files"].items():
                path = cell / name
                actual = digest(path)
                if actual != expected:
                    raise ValueError(f"Checkpoint hash mismatch: {path}")
                inputs[str(path.relative_to(root))] = actual
    if len(seeds) != 420:
        raise ValueError("Expected exactly 420 unique seeds")
    comparisons = pd.read_csv(root / "primary_comparisons.csv")
    aggregate = pd.read_csv(root / "aggregate_effects.csv")
    if len(comparisons) != 210 or comparisons["paired_repetitions"].ne(30).any():
        raise ValueError("Incomplete primary comparisons")
    if comparisons.duplicated(["comparison", "protocol", "scenario", "metric"]).any():
        raise ValueError("Duplicate comparison rows")
    for _, row in aggregate.iterrows():
        part = comparisons.loc[(comparisons.comparison == row.comparison) &
                               (comparisons.protocol == row.protocol) &
                               (comparisons.metric == row.metric)]
        if set(part.scenario) != set(cfg["scenarios"]):
            raise ValueError("Aggregate does not cover the declared seven scenarios")
        for key in ["proposed_mean", "reference_mean", "mean_paired_difference"]:
            if not np.isclose(part[key].mean(), row[key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Aggregate inconsistency: {key}")
        if not row.difference_ci_lower <= row.mean_paired_difference <= row.difference_ci_upper:
            raise ValueError("Invalid aggregate interval")
    for name in ["protocol_lock.json", "primary_comparisons.csv", "aggregate_effects.csv"]:
        inputs[name] = digest(root / name)
    return comparisons, aggregate, inputs


def pick(frame, comparison, protocol, metric):
    return frame.loc[(frame.comparison == comparison) &
                     (frame.protocol == protocol) & (frame.metric == metric)].copy()


def errors(values, low, high):
    return np.vstack([values - low, high - values])


def finish(fig, out, stem, data, cfg):
    for suffix in ["pdf", "png"]:
        fig.savefig(out / f"{stem}.{suffix}", dpi=cfg["dpi"], facecolor="white")
    data.to_csv(out / f"{stem}_data.csv", index=False)
    plt.close(fig)


def generate(root, out, cfg):
    comparisons, aggregate, inputs = verify(root, cfg)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": cfg["font_family"], "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    width = cfg["width_in"]
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.25), sharey=True)
    plotted = []
    for j, (ax, protocol) in enumerate(zip(axes, cfg["protocols"])):
        for k, comparator in enumerate(["periodic_retry", "fixed_probe"]):
            rows = pick(comparisons, "recovery:" + comparator, protocol, RECALL)
            rows = rows.set_index("scenario").loc[cfg["scenarios"]].reset_index()
            y = rows.mean_paired_difference.to_numpy() * 100
            ax.errorbar(np.arange(7) + (k - .5) * .16, y,
                        yerr=errors(y, rows.bootstrap_ci_lower.to_numpy()*100,
                                    rows.bootstrap_ci_upper.to_numpy()*100),
                        fmt=MARKERS[k], color=COLORS[k], ms=2.2,
                        markerfacecolor=COLORS[k], markeredgewidth=.7,
                        elinewidth=1.1, capsize=2.2, capthick=.9, barsabove=True,
                        label=["Periodic retry", "Fixed probing"][k])
            plotted.append(rows)
        ax.set_title(["15 min", "60 min"][j], fontsize=8)
        ax.set_xticks(range(7), cfg["scenario_labels"])
        ax.axhline(0, color=".45", lw=.7, ls="--")
        ax.grid(axis="y", color=".9", lw=.5)
        ax.set_xlabel("Acoustic scenario")
    axes[0].set_ylabel("Recall gain (percentage points)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(.5, 1))
    fig.subplots_adjust(left=.16, right=.98, bottom=.21, top=.76, wspace=.15)
    finish(fig, out, "figure2_recovery", pd.concat(plotted), cfg)

    fig, ax = plt.subplots(figsize=(width, 2.45))
    plotted = []
    for j, protocol in enumerate(cfg["protocols"]):
        rows = pd.concat([pick(aggregate, "trust:" + v, protocol, TRUST)
                          for v in cfg["trust_variants"]])
        x = rows.mean_paired_difference.to_numpy()*100
        y = np.arange(len(rows)) + (j - .5)*.23
        ax.errorbar(x, y, xerr=errors(x, rows.difference_ci_lower.to_numpy()*100,
                                   rows.difference_ci_upper.to_numpy()*100),
                    fmt=MARKERS[j], color=COLORS[j], ms=2.2,
                    markerfacecolor=COLORS[j], markeredgewidth=.7,
                    elinewidth=1.1, capsize=2.2, capthick=.9, barsabove=True,
                    label=["15 min", "60 min"][j])
        for xx, yy in zip(x, y):
            ax.text(.48, yy, f"{xx:+.3f}", ha="left", va="center",
                    fontsize=6.5, color=COLORS[j])
        plotted.append(rows)
    ax.set_yticks(range(5), cfg["trust_labels"])
    ax.invert_yaxis()
    ax.set_xlim(-8.2, 1.85)
    ax.axvline(0, color=".45", lw=.7, ls="--")
    ax.grid(axis="x", color=".9", lw=.5)
    ax.set_xlabel("False-trust difference (pp)\nFull receiver minus comparator")
    ax.legend(loc="upper center", bbox_to_anchor=(.55, 1.25), ncol=2, frameon=False)
    fig.subplots_adjust(left=.32, right=.98, bottom=.26, top=.83)
    finish(fig, out, "figure3_trust", pd.concat(plotted), cfg)

    fig, ax = plt.subplots(figsize=(width, 2.1))
    plotted = []
    for j, protocol in enumerate(cfg["protocols"]):
        rows = pd.concat([pick(aggregate, "scheduler:" + p, protocol, RECALL)
                          for p in cfg["profiles"]])
        x = rows.mean_paired_difference.to_numpy()*100
        ax.errorbar(x, np.arange(3)+(j-.5)*.24,
                    xerr=errors(x, rows.difference_ci_lower.to_numpy()*100,
                                rows.difference_ci_upper.to_numpy()*100),
                    fmt=MARKERS[j], color=COLORS[j], ms=2.2,
                    markerfacecolor=COLORS[j], markeredgewidth=.7,
                    elinewidth=1.1, capsize=2.2, capthick=.9, barsabove=True,
                    label=["15 min", "60 min"][j])
        plotted.append(rows)
    ax.set_yticks(range(3), [s.replace("-dominant", "-\ndominant") for s in cfg["profile_labels"]])
    ax.invert_yaxis()
    ax.set_xlim(0, .48)
    ax.axvline(0, color=".45", lw=.7, ls="--")
    ax.grid(axis="x", color=".9", lw=.5)
    ax.set_xlabel("Recall gain (pp)\nCVaR minus expected risk")
    ax.legend(loc="upper center", bbox_to_anchor=(.55, 1.34), ncol=2, frameon=False)
    fig.subplots_adjust(left=.34, right=.98, bottom=.29, top=.80)
    finish(fig, out, "figure4_scheduler", pd.concat(plotted), cfg)
    inputs["plot_config"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    inputs["plot_module"] = digest(Path(__file__))
    artifacts = {p.name: digest(p) for p in sorted(out.iterdir())
                 if p.suffix in [".pdf", ".png", ".csv"]}
    manifest = {"input_sha256": inputs, "output_sha256": artifacts,
                "verified_unique_realizations": 420, "primary_hypotheses": 210,
                "intervals": "Pointwise 95% paired bootstrap; aggregate intervals descriptive",
                "python": sys.version, "matplotlib": matplotlib.__version__,
                "numpy": np.__version__, "pandas": pd.__version__}
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"FIGURES VERIFIED AND GENERATED: {out}; 420 realizations; 210 hypotheses")


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config or project_root / "configs/manuscript_figures.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    root = (args.evidence_dir or project_root / cfg["evidence_directory"]).resolve()
    out = (args.output_dir or project_root / cfg["output_directory"]).resolve()
    if out == root or root in out.parents:
        raise ValueError("Figure outputs must be separate from frozen evidence")
    generate(root, out, cfg)
