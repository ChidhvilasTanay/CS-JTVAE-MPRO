"""Regenerate Fig 2 (AMES by warhead class) on the NSGA-III 1,013 library,
matching the existing v3_ames_by_warhead.png style. Uses saved ADMET; no
re-prediction. Green = reversible (nitrile / non-covalent), red = irreversible."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "figures/v3_ames_by_warhead.png"
GREEN, RED = "#41ab5d", "#d7301f"
IRREV = ("chloro", "michael", "acryl", "enone", "aldehyde", "epoxide", "vinyl_sulfone")

pc = pd.read_csv("data/analogs_v3_6obj_gen_admet.csv", index_col=0)
d = pd.read_csv("data/analogs_v3_6obj_feasible.csv").drop_duplicates("cand_smi")
d = d[d.cand_smi.isin(pc.index)].copy()
d["AMES"] = d.cand_smi.map(pc["AMES"].to_dict())
d["wh"] = d.cand_warheads.fillna("-")

grp = d.groupby("wh").agg(n=("AMES", "size"), ames=("AMES", "mean"))
grp = grp[grp.n >= 15].sort_values("ames")
noncov = grp.loc["-", "ames"] if "-" in grp.index else d[d.wh == "-"].AMES.mean()

labels = [w.replace(",", "+\n") if w != "-" else "non-covalent" for w in grp.index]
colors = [GREEN if (w == "-" or not any(k in w for k in IRREV)) else RED for w in grp.index]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(grp))
ax.barh(y, grp.ames, color=colors, edgecolor="white")
for i, (a, n) in enumerate(zip(grp.ames, grp.n)):
    ax.text(a + 0.01, i, f"{a:.2f} (n={int(n)})", va="center", fontsize=9)
ax.axvline(noncov, ls="--", c="0.4", lw=1)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("mean predicted AMES mutagenicity (lower = cleaner)")
ax.set_xlim(0, 1.0)
ax.set_title("Mutagenicity is the warhead class, not covalency", fontsize=11)
ax.legend(handles=[Patch(color=GREEN, label="reversible"), Patch(color=RED, label="irreversible")],
          loc="lower right", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
print(f"classes>=15: {len(grp)} covering {int(grp.n.sum())} of {len(d)}; non-covalent mean {noncov:.2f}")
