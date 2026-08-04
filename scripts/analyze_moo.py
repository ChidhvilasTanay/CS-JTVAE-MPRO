"""Merge the v3 MOO catalogue shards and reproduce the optimiser comparison.

Ranks the optimisers, finds per-anchor winners, runs the Friedman and
pairwise-Wilcoxon significance tests, and draws the HV_final/HV_accum figure.

    python scripts/analyze_moo.py
Inputs : results/moo_catalogue_v3_a*.csv  (per-anchor shards)
Outputs: results/moo_catalogue_v3.csv     (merged 270-row table)
         results/moo_catalogue_v3.png     (figure)
         results/moo_catalogue_v3_stats.txt(ranking + significance, for the paper)
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ALGOS = ["NSGA2", "NSGA3", "CTAEA", "SMSEMOA", "SPEA2", "RVEA"]
EXPECTED_ROWS = len(ALGOS) * 3  # 18 per anchor


def merge():
    shards = sorted(glob.glob(str(ROOT / "results" / "moo_catalogue_v3_a*.csv")))
    frames, short = [], []
    for f in shards:
        df = pd.read_csv(f)
        frames.append(df)
        if len(df) != EXPECTED_ROWS:
            short.append((Path(f).name, len(df)))
    merged = pd.concat(frames, ignore_index=True)
    out = ROOT / "results" / "moo_catalogue_v3.csv"
    merged.to_csv(out, index=False)
    print(f"merged {len(shards)} shards -> {out}  ({len(merged)} rows)")
    if short:
        print("WARNING incomplete shards:", short)
    return merged


def main():
    d = merge()
    n_anchor = d.anchor_idx.nunique()
    print(f"{len(d)} runs | {n_anchor} anchors {sorted(d.anchor_idx.unique())} | "
          f"seeds/anchor {dict(d.groupby('anchor_idx').seed.nunique())}")

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ranking
    g = (d.groupby("algo")
           .agg(HV_final=("hv_final", "mean"), std=("hv_final", "std"),
                HV_accum=("hv_accum", "mean"), front=("front_final", "mean"),
                feasible=("n_feasible", "mean"))
           .sort_values("HV_final", ascending=False))
    g["gap(acc-fin)"] = g.HV_accum - g.HV_final
    out("\n=== ranking by HV_final (standard res.F metric) ===")
    out(g.round(4).to_string())

    # per-anchor winners
    out("\n=== per-anchor winner / loser (HV_final) ===")
    for a in sorted(d.anchor_idx.unique()):
        s = d[d.anchor_idx == a].groupby("algo").hv_final.mean().sort_values(ascending=False)
        out(f"  anchor{a}: 1st={s.index[0]} ({s.iloc[0]:.3f})   "
            f"last={s.index[-1]} ({s.iloc[-1]:.3f})")

    # significance
    d["block"] = d.anchor_idx.astype(str) + "_" + d.seed.astype(str)
    wide = d.pivot_table(index="block", columns="algo", values="hv_final").dropna()
    fr = stats.friedmanchisquare(*[wide[a] for a in ALGOS])
    out("\n=== significance ===")
    verdict = ("algorithms DIFFER" if fr.pvalue < 0.05
               else "no significant difference (tied)")
    out(f"Friedman (blocks={len(wide)}): chi2={fr.statistic:.3f}  "
        f"p={fr.pvalue:.4f}  -> {verdict}")
    top = g.index[0]
    alpha = 0.05 / (len(ALGOS) - 1)
    out(f"\nPairwise Wilcoxon vs {top} (Bonferroni alpha={alpha:.3f}):")
    for a in ALGOS:
        if a == top:
            continue
        w = stats.wilcoxon(wide[top], wide[a])
        flag = "  * significant" if w.pvalue < alpha else ""
        out(f"  {top} vs {a:8s}: p={w.pvalue:.4f}{flag}")

    # figure
    order = g.index.tolist()
    m = [g.loc[a, "HV_final"] for a in order]
    e = [g.loc[a, "std"] for a in order]
    acc = [g.loc[a, "HV_accum"] for a in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w/2, m, w, yerr=e, capsize=4,
           label="HV_final (res.F, standard)", color="#2c7fb8")
    ax.bar(x + w/2, acc, w, label="HV_accum (analog-yield)", color="#a6bddb")
    for xi, val, err in zip(x - w/2, m, e):
        ax.text(xi, val + err + 0.004, f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#1f5f8b")
    for xi, val in zip(x + w/2, acc):
        ax.text(xi, val + 0.004, f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, color="#4a7fa5")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("hypervolume")
    ax.set_ylim(0, max(max(np.array(m) + np.array(e)), max(acc)) + 0.035)
    ax.set_title(f"MOO algorithm catalogue ({n_anchor} productive anchors x 3 seeds)")
    ax.legend(fontsize=8, framealpha=0.9, handlelength=1.3,
              borderpad=0.4, labelspacing=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    figout = ROOT / "results" / "moo_catalogue_v3.png"
    fig.savefig(figout, dpi=150, bbox_inches="tight")
    out(f"\nsaved figure -> {figout}")

    statout = ROOT / "results" / "moo_catalogue_v3_stats.txt"
    statout.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved stats -> {statout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
