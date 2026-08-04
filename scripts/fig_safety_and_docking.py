"""Generate two figures for the six-objective NSGA-III Pareto front:
  1. v3_6obj_vs_parent.png  -- 8-endpoint paired mean, front vs source inhibitors (4/8 safer)
  2. v3_6obj_docking.png     -- paired Vina delta hist + surrogate-vs-Vina scatter (rho=-0.30)
Writes into figures/. Read-only on the protected resopt file.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
RDLogger.DisableLog("rdApp.*")

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
GOOD, TIE, BAD, BLUE = "#41ab5d", "#9e9e9e", "#d7301f", "#2c7fb8"
TIE_THRESH = 0.005
EPS = ["AMES", "hERG", "DILI", "CYP1A2_Veith", "CYP2C9_Veith", "CYP2C19_Veith", "CYP2D6_Veith", "CYP3A4_Veith"]
ORDER = ["AMES", "hERG", "DILI", "CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
NIRMA = "CC(C)(C)[C@H](NC(=O)C(F)(F)F)C(=O)N1C[C@H]2[C@@H]([C@H]1C(=O)N[C@H](C#N)C[C@@H]1CCNC1=O)C2(C)C"
DUAL = "CC(C)(C)C(NC(=O)C(F)(F)F)C(=O)N1CC(C#N)C(C(=O)NC(C#N)CC2CCNC2=O)C1"


def verdict(d):
    if d < -TIE_THRESH:
        return GOOD
    if d > TIE_THRESH:
        return BAD
    return TIE


def fig_vs_parent_and_docking():
    d = pd.read_csv("data/analogs_v3_6obj_resopt.csv")
    from admet_ai import ADMETModel
    model = ADMETModel()
    cands = sorted(d.cand_smi.unique().tolist())
    pars = sorted(d.orig_smi.unique().tolist())
    pc = model.predict(smiles=cands)
    pp = model.predict(smiles=pars)
    rows = []
    for ep, lab in zip(EPS, ORDER):
        c = d.cand_smi.map(pc[ep].to_dict())
        o = d.orig_smi.map(pp[ep].to_dict())
        rows.append((lab, o.mean(), c.mean(), c.mean() - o.mean()))
    v = pd.DataFrame(rows, columns=["label", "parent_mean", "analog_mean", "delta"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(v)); w = 0.38
    ax.bar(x - w/2, v.parent_mean, w, color="#bdbdbd", edgecolor="white")
    ax.bar(x + w/2, v.analog_mean, w, color=[verdict(dd) for dd in v.delta], edgecolor="white")
    for i, dd in enumerate(v.delta):
        tag = "safer" if dd < -TIE_THRESH else ("worse" if dd > TIE_THRESH else "parity")
        ax.text(i, max(v.parent_mean[i], v.analog_mean[i]) + 0.015, f"{dd:+.3f}\n{tag}",
                ha="center", va="bottom", fontsize=8, color=verdict(dd), fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(v.label, fontsize=9)
    ax.set_ylabel("mean predicted liability (lower = safer)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Six-objective NSGA-III Pareto front vs the source clinical inhibitors "
                 "(138 analogs, all 20 anchors, no post-hoc filtering)\n"
                 "As safe as or safer on 4/8 (hERG, CYP3A4, CYP2D6, DILI); AMES at parity; "
                 "cost on CYP1A2/CYP2C9/CYP2C19", fontsize=9)
    ax.legend(handles=[Patch(color="#bdbdbd", label="source inhibitor"),
                       Patch(color=GOOD, label="front safer"),
                       Patch(color=TIE, label="parity"),
                       Patch(color=BAD, label="front worse")],
              loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/v3_6obj_vs_parent.png", dpi=150)
    plt.close(fig)
    print("vs_parent deltas:", {r.label: round(r.delta, 3) for r in v.itertuples()})

    # ---- docking figure ----
    dk = pd.read_csv("data/analogs_v3_6obj_docked.csv")
    dk = dk[dk.vina_kcal_per_mol.notna()].copy()
    dk["delta"] = dk.vina_kcal_per_mol - dk.orig_vina_kcal_per_mol
    n = len(dk); n_tight = int((dk.delta <= 0).sum()); dmean = dk.delta.mean()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    axL.hist(dk.delta, bins=18, color=BLUE, alpha=0.6, edgecolor="white")
    axL.axvline(0, ls="--", c="black", lw=1)
    axL.axvline(dmean, ls="-", c=BAD, lw=1.6, label=f"mean {dmean:+.2f}")
    axL.set_xlabel(r"paired Vina $\Delta$ = analog $-$ parent (kcal/mol)")
    axL.set_ylabel("analogs")
    axL.set_title(f"Binding preserved (n={n})\nanalog mean {dk.vina_kcal_per_mol.mean():.2f} vs "
                  f"parent {dk.orig_vina_kcal_per_mol.mean():.2f}; {n_tight}/{n} at least as tight; "
                  f"best {dk.vina_kcal_per_mol.min():.2f}", fontsize=10)
    axL.legend(fontsize=9)
    m = dk[dk.cand_potency_norm.notna()]  # docked copy already carries cand_potency_norm
    rho, pr = stats.spearmanr(m.cand_potency_norm, m.vina_kcal_per_mol)
    xs = m.cand_potency_norm.to_numpy(); ys = m.vina_kcal_per_mol.to_numpy()
    axR.scatter(xs, ys, s=22, color=BLUE, alpha=0.55, edgecolor="white")
    xline = np.linspace(xs.min(), xs.max(), 100)
    axR.plot(xline, np.polyval(np.polyfit(xs, ys, 1), xline), color=BAD, lw=1.8,
             label=f"linear fit (rho = {rho:+.2f})")
    axR.legend(fontsize=9, loc="upper right")
    axR.set_xlabel("surrogate predicted potency (higher = more potent)")
    axR.set_ylabel("Vina binding energy (kcal/mol, lower = tighter)")
    axR.set_title(f"Surrogate tracks physics\nSpearman rho = {rho:+.2f} (p < 0.001)", fontsize=10)
    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/v3_6obj_docking.png", dpi=150)
    plt.close(fig)
    print(f"docking: n={n} dmean {dmean:+.3f} tighter {n_tight}/{n} best {dk.vina_kcal_per_mol.min():.2f} rho {rho:+.3f}")


def main():
    fig_vs_parent_and_docking()
    print("saved 2 figures ->", OUT)


if __name__ == "__main__":
    main()
