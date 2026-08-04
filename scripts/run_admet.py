"""ADMET pass on the NSGA-III 6-obj feasible archive (1,013 cands + 20 parents).
Serves Table 5 'ours' column and Fig 2 (AMES by warhead). Saves per-molecule
ADMET so the figure/table steps don't re-predict. GPU, a few minutes."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
EPS = ["AMES","hERG","DILI","CYP1A2_Veith","CYP2C9_Veith","CYP2C19_Veith","CYP2D6_Veith","CYP3A4_Veith"]


def main():
    d = pd.read_csv("data/analogs_v3_6obj_feasible.csv").drop_duplicates("cand_smi")
    from admet_ai import ADMETModel
    m = ADMETModel()
    cands = d.cand_smi.tolist()
    pars = sorted(d.orig_smi.unique().tolist())
    print(f"predicting {len(cands)} cands + {len(pars)} parents ...", flush=True)
    pc = m.predict(smiles=cands); pp = m.predict(smiles=pars)
    pc.index = cands
    pc.to_csv("data/analogs_v3_6obj_gen_admet.csv")
    pp.index = pars
    pp.to_csv("data/analogs_v3_6obj_gen_parents_admet.csv")

    # --- Table 5 'ours' metrics ---
    par = {ep: pp[ep].to_dict() for ep in EPS}
    cad = {ep: pc[ep].to_dict() for ep in EPS}
    # AMES safer-than-parent (fraction of analogs with cand AMES < parent AMES)
    ames_safer = np.mean([cad["AMES"][c] < par["AMES"][d.set_index("cand_smi").orig_smi[c]] for c in cands])
    # ADMET 8-endpoint average of the raw cand liabilities (lower=better; paper reports as a score)
    admet8 = np.mean([np.mean([cad[ep][c] for ep in EPS]) for c in cands])
    print(f"\nTABLE 5 'ours' (NSGA-III, n={len(cands)}):")
    print(f"  AMES safer-than-parent fraction: {ames_safer:.2f}")
    print(f"  ADMET 8-endpoint mean liability: {admet8:.2f}  (1 - this = 'safety' style if paper uses that)")
    print(f"  mean warhead score: {d.cand_warhead_score.mean():.2f}")
    print(f"  mean predicted potency: {d.cand_potency_norm.mean():.2f}")

    # --- Fig 2: AMES by warhead class (>=15 analogs) ---
    d2 = d.copy(); d2["AMES"] = d2.cand_smi.map(cad["AMES"])
    print("\nFIG 2  AMES by warhead class (>=15 analogs):")
    grp = d2.groupby(d2.cand_warheads.fillna("-")).agg(n=("AMES","size"), ames=("AMES","mean"))
    for w, r in grp[grp.n >= 15].sort_values("ames").iterrows():
        print(f"  {w:35} n={int(r.n):4}  AMES={r.ames:.2f}")


if __name__ == "__main__":
    main()
