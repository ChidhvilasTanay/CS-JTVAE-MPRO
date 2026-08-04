"""Table 4 (per-anchor yield) + section 5.5 (yield-variance correlations) on the
NSGA-III feasible archive. From-CSV + junction-tree edge count. No ADMET/GPU."""
import sys
sys.path.insert(0, "src")
import pandas as pd
from scipy import stats
from charaka.vocab import smiles_to_tree

d = pd.read_csv("data/analogs_v3_6obj_feasible.csv").drop_duplicates("cand_smi")
yld = d.groupby("orig_smi").agg(n=("cand_smi", "nunique"),
                                wh=("orig_warheads", "first"),
                                pot=("orig_pred_log10_ic50", "first")).sort_values("n", ascending=False)

# edge count per anchor
def edges(smi):
    r = smiles_to_tree(smi)
    return len(r[1]) if r else None
yld["edges"] = [edges(s) for s in yld.index]

print("TABLE 4 per-anchor yield (top 8, NSGA-III archive):")
for smi, r in yld.head(8).iterrows():
    print(f"  n={int(r.n):4}  warheads={str(r.wh):22}  edges={r.edges}  ic50log={r.pot:.2f}")
print(f"  total {int(yld.n.sum())} across {len(yld)} anchors; range {int(yld.n.min())}-{int(yld.n.max())}")
print(f"  top-5 sum {int(yld.n.head(5).sum())}, rest {int(yld.n.iloc[5:].sum())}\n")

# section 5.5 correlations
v = yld.dropna(subset=["edges"])
rho_e, p_e = stats.spearmanr(v.edges, v.n)
# potency rank: lower log10(ic50) = more potent = rank 1
v = v.copy(); v["potrank"] = v.pot.rank()
rho_p, p_p = stats.spearmanr(v.potrank, v.n)
print("SECTION 5.5 yield-variance correlations (NSGA-III):")
print(f"  Spearman(edge count, yield)   = {rho_e:+.2f}  (p={p_e:.3f})")
print(f"  Spearman(potency rank, yield) = {rho_p:+.2f}  (p={p_p:.3f})")
# nirmatrelvir-class anchor (dual cf3+nitrile)
nirma = yld[yld.wh.fillna("").str.contains("cf3_amide")]
if len(nirma):
    r = nirma.iloc[0]
    print(f"  nirmatrelvir-class anchor: yield={int(r.n)}  edges={r.edges}")
