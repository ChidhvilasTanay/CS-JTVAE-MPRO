"""Summarise front-wide docking of the 6-obj res.opt front (docked copy).
Prints set-level Vina stats + surrogate-vs-physics Spearman + the dual headline
row + a paired-potency CI, for the v15 draft. Read-only on the docked copy."""
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("data/analogs_v3_6obj_docked.csv")
a = d["vina_kcal_per_mol"].astype(float)
p = d["orig_vina_kcal_per_mol"].astype(float)
delta = d["vina_delta"].astype(float)
n = len(d)
print(f"n={n}  docked={a.notna().sum()}")
print(f"analog mean Vina = {a.mean():.3f}  parent mean = {p.mean():.3f}  mean paired delta = {delta.mean():+.3f}")
# paired CI on delta
se = delta.std(ddof=1) / np.sqrt(n)
lo, hi = delta.mean() - 1.96 * se, delta.mean() + 1.96 * se
tval, pval = stats.ttest_rel(a, p)
print(f"delta 95% CI = [{lo:+.3f}, {hi:+.3f}]  paired t p = {pval:.2g}")
tighter = (delta < 0).sum()
print(f"tighter-or-equal (delta<=0): {(delta<=0).sum()}/{n} ({(delta<=0).mean()*100:.0f}%)  strictly tighter: {tighter}")
print(f"best absolute analog Vina = {a.min():.3f}")
# surrogate (pred_log10_ic50) vs Vina: higher potency = lower log10ic50 = lower (tighter) Vina => positive rho between them? we report rho(pred_pot, vina)
if "cand_pred_log10_ic50" in d.columns:
    m = d[["cand_pred_log10_ic50", "vina_kcal_per_mol"]].dropna()
    rho, pr = stats.spearmanr(m["cand_pred_log10_ic50"], m["vina_kcal_per_mol"])
    print(f"Spearman rho(pred_log10_ic50, Vina) = {rho:+.3f} (p={pr:.2g}); ",
          "lower ic50 (more potent) -> lower Vina expected => positive rho")
# potency surrogate paired ratio CI (analog vs parent ic50)
if {"cand_pred_log10_ic50","orig_pred_log10_ic50"}.issubset(d.columns):
    dd = (d["cand_pred_log10_ic50"] - d["orig_pred_log10_ic50"]).dropna()
    r = 10 ** dd.mean()
    sed = dd.std(ddof=1) / np.sqrt(len(dd))
    print(f"potency ratio (analog/parent IC50) = {r:.2f}x  95% CI [{10**(dd.mean()-1.96*sed):.2f}, {10**(dd.mean()+1.96*sed):.2f}]")
# dual headline row
DUAL = "CC(C)(C)C(NC(=O)C(F)(F)F)C(=O)N1CC(C#N)C(C(=O)NC(C#N)CC2CCNC2=O)C1"
row = d[d["cand_smi"] == DUAL]
if len(row):
    r = row.iloc[0]
    print(f"\nDUAL headline: Vina {r['vina_kcal_per_mol']:.3f} vs parent {r['orig_vina_kcal_per_mol']:.3f} "
          f"delta {r['vina_delta']:+.3f} | Tanimoto {r['tanimoto']:.3f} | warheads {r['cand_warheads']}")
