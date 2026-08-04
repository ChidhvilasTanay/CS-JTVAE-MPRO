"""Verify the dual-warhead headline vs nirmatrelvir on the 8 ADMET endpoints
(per-molecule) and list all nirmatrelvir-anchor analogs in the docked front."""
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
EPS = ["AMES","hERG","DILI","CYP1A2_Veith","CYP2C9_Veith","CYP2C19_Veith","CYP2D6_Veith","CYP3A4_Veith"]
NIRMA = "CC(C)(C)[C@H](NC(=O)C(F)(F)F)C(=O)N1C[C@H]2[C@@H]([C@H]1C(=O)N[C@H](C#N)C[C@@H]1CCNC1=O)C2(C)C"
DUAL  = "CC(C)(C)C(NC(=O)C(F)(F)F)C(=O)N1CC(C#N)C(C(=O)NC(C#N)CC2CCNC2=O)C1"

def main():
    d = pd.read_csv("data/analogs_v3_6obj_docked.csv")
    sub = d[d.orig_smi == NIRMA][["cand_smi","cand_warheads","tanimoto","composite_gain","vina_kcal_per_mol","vina_delta"]]
    print(f"nirmatrelvir-anchor analogs in front: {len(sub)}")
    for _, r in sub.iterrows():
        tag = " <-- DUAL HEADLINE" if r.cand_smi == DUAL else ""
        print(f"  Tan={r.tanimoto:.3f} comp_gain={r.composite_gain:+.3f} vina={r.vina_kcal_per_mol:.2f} d={r.vina_delta:+.2f} wh={r.cand_warheads}{tag}")
    from admet_ai import ADMETModel
    m = ADMETModel()
    pr = m.predict(smiles=[NIRMA, DUAL])
    par, cand = pr.iloc[0], pr.iloc[1]
    n = 0
    print("\nDUAL vs NIRMATRELVIR (per-molecule), lower=safer:")
    for ep in EPS:
        safer = cand[ep] < par[ep]; n += safer
        print(f"  {ep.replace('_Veith',''):8} nirma={par[ep]:.3f}  dual={cand[ep]:.3f}  d={cand[ep]-par[ep]:+.3f} {'SAFER' if safer else ''}")
    print(f"\nDUAL safer-than-nirmatrelvir on {n}/8 endpoints")

if __name__ == "__main__":
    main()
