"""Three-compound headline showcase for the nirmatrelvir anchor:
  panel 1: the anchor (nirmatrelvir)
  panel 2: the current headline analog (Tanimoto 0.65, safer 7/8)
  panel 3: a structurally divergent analog (Tanimoto 0.33, safer 5/8)
All three retain both electrophiles (w=1.0, highlighted). The point of the
figure is that the generator also produces significantly different structures,
so the conservative headline is a selection and not an objective hack.

Numbers are read from the saved feasible + ADMET CSVs (no re-prediction).
Output: figures/v3_headline_triptych.png
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
RDLogger.DisableLog("rdApp.*")

R = ""
FIG = "figures/"
OUT = "v3_headline_triptych.png"
EPS = ["AMES", "hERG", "DILI", "CYP1A2_Veith", "CYP2C9_Veith",
       "CYP2C19_Veith", "CYP2D6_Veith", "CYP3A4_Veith"]
SMARTS = ["[CX2]#N", "C(=O)C(F)(F)F"]          # nitrile + trifluoroacetyl cap

NIRMA = "CC(C)(C)[C@H](NC(=O)C(F)(F)F)C(=O)N1C[C@H]2[C@@H]([C@H]1C(=O)N[C@H](C#N)C[C@@H]1CCNC1=O)C2(C)C"
HEADLINE = "CC(C)(C)C(NC(=O)C(F)(F)F)C(=O)N1CC(C#N)C(C(=O)NC(C#N)CC2CCNC2=O)C1"   # Tan 0.65, 7/8
DIVERGENT = "CNC(=O)C1CN(C(=O)C(CC#N)NC(=O)C(F)(F)F)CC12CC2(C)C"                    # Tan 0.33, 5/8

feas = pd.read_csv(R + "data/analogs_v3_6obj_feasible.csv").drop_duplicates("cand_smi")
pc = pd.read_csv(R + "data/analogs_v3_6obj_gen_admet.csv", index_col=0)
pp = pd.read_csv(R + "data/analogs_v3_6obj_gen_parents_admet.csv", index_col=0)


def highlight(m):
    hits = []
    for s in SMARTS:
        p = Chem.MolFromSmarts(s)
        if p:
            for match in m.GetSubstructMatches(p):
                hits.extend(match)
    return list(set(hits))


def stats(cand):
    row = feas[feas.cand_smi == cand].iloc[0]
    tan, cg = float(row.tanimoto), float(row.composite_gain)
    nsafer = sum(1 for e in EPS if pc.loc[cand, e] < pp.loc[NIRMA, e])
    return tan, cg, nsafer


def img(smi):
    m = Chem.MolFromSmiles(smi)
    return np.asarray(Draw.MolToImage(m, size=(520, 440), highlightAtoms=highlight(m)))


def main():
    th, ch, sh = stats(HEADLINE)
    td, cd, sd = stats(DIVERGENT)
    panels = [
        (NIRMA, "nirmatrelvir (parent / anchor)"),
        (HEADLINE, f"headline analog (Tanimoto {th:.2f}, safer {sh}/8)"),
        (DIVERGENT, f"divergent analog (Tanimoto {td:.2f}, safer {sd}/8)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for ax, (smi, title) in zip(axes, panels):
        ax.imshow(img(smi)); ax.set_title(title, fontsize=11); ax.axis("off")
    fig.suptitle(
        "Nirmatrelvir-anchor analogs on the six-objective NSGA-III front, both electrophiles retained "
        "(w=1.0, highlighted).\nThe headline stays close to the parent by design and keeps the P1 lactam "
        f"(Tanimoto {th:.2f}, {sh}/8); the generator also yields a structurally divergent variant "
        f"(Tanimoto {td:.2f}, composite gain {cd:+.3f}, {sd}/8), so the headline is a selection, not a hack.",
        fontsize=9, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIG + OUT, dpi=160)
    plt.close(fig)
    print(f"saved {FIG}{OUT}")
    print(f"  headline : Tan={th:.3f}  compGain={ch:+.3f}  safer={sh}/8")
    print(f"  divergent: Tan={td:.3f}  compGain={cd:+.3f}  safer={sd}/8")


if __name__ == "__main__":
    main()
