"""Generation-stage stats recomputed on the NSGA-III 6-obj FEASIBLE ARCHIVE
(the honest 'library' if the paper becomes one NSGA-III run). No ADMET needed
for these; pure from the feasible CSV. Prints numbers for Tables 2/4 and
sections 5.2.1 / 5.3, plus the warhead-class spread that decides Fig 2."""
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
RDLogger.DisableLog("rdApp.*")

d = pd.read_csv("data/analogs_v3_6obj_feasible.csv")
u = d.drop_duplicates("cand_smi").copy()  # 1013 unique
print(f"LIBRARY: {len(u)} unique feasible analogs across {u.orig_smi.nunique()} anchors\n")

# --- warhead retention (cf paper 5.2.1) ---
has_wh = u.orig_warheads.fillna("").str.len() > 0
full = (u.cand_warhead_score == 1.0)
print("WARHEAD RETENTION (5.2.1):")
print(f"  full retention w=1.0 over ALL:        {full.sum()}/{len(u)} ({full.mean()*100:.0f}%)")
print(f"  parents WITH a warhead (analogs):     {has_wh.sum()}")
print(f"  parents WITHOUT a warhead (analogs):  {(~has_wh).sum()}")
fw = u[has_wh]
print(f"  full retention over warhead-bearing:  {(fw.cand_warhead_score==1.0).sum()}/{len(fw)} ({(fw.cand_warhead_score==1.0).mean()*100:.0f}%)")
print(f"  mean warhead score (all):             {u.cand_warhead_score.mean():.3f}")
print(f"  mean warhead score (warhead-bearing): {fw.cand_warhead_score.mean():.3f}\n")

# --- analog quality (cf paper 5.3) ---
print("ANALOG QUALITY (5.3):")
print(f"  mean composite gain: {u.composite_gain.mean():+.3f}  max {u.composite_gain.max():+.3f}  "
      f"positive {(u.composite_gain>0).sum()}/{len(u)} ({(u.composite_gain>0).mean()*100:.1f}%)")
print(f"  mean Tanimoto: {u.tanimoto.mean():.3f}  >0.4: {(u.tanimoto>0.4).mean()*100:.1f}%  "
      f">=0.20 floor: {(u.tanimoto>=0.20).mean()*100:.0f}%")
print(f"  mean predicted potency (cand_potency_norm): {u.cand_potency_norm.mean():.3f}\n")

# --- per-anchor yield (cf Table 4) ---
yld = u.groupby("orig_smi").cand_smi.nunique().sort_values(ascending=False)
print("PER-ANCHOR YIELD (Table 4):")
print(f"  range {yld.min()} to {yld.max()}, total {yld.sum()}")
print(f"  top-5 anchor yields: {yld.head(5).tolist()}  rest sum: {yld.iloc[5:].sum()}\n")

# --- warhead-class spread (decides Fig 2) ---
print("WARHEAD-CLASS SPREAD in the 1,013 (Fig 2 needs >=15/class to be shown):")
cnt = {}
for ws in u.cand_warheads.fillna("-"):
    cnt[ws] = cnt.get(ws, 0) + 1
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"  {k:35} {v}")
