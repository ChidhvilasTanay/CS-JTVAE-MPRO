"""MOO algorithm benchmark: same frozen pipeline, same objectives, same
anchors, same budget -- swap ONLY the optimiser.

For each (algorithm, anchor, seed) it runs the identical ``LatentProblem``
(4 objectives: QED-no-alerts, SA-norm, warhead, potency; 5 constraints)
under a matched evaluation budget (pop_size x generations), then scores the
accumulated feasible front by hypervolume. Encoder / decoder / scorer /
bioactivity ensemble are loaded frozen from the checkpoint directory and never
touched -- only the search operator differs.

Usage:
    python scripts/bench_moo.py --config configs/moo_catalogue_v3.yaml \
        --algos NSGA2,NSGA3,MOEAD,SMSEMOA,SPEA2,RVEA \
        --n-anchors 4 --seeds 3 --pop 24 --gen 12 --out results/moo_bench.csv

Smoke test (fast):
    python scripts/bench_moo.py --algos NSGA2,SMSEMOA --n-anchors 1 \
        --seeds 1 --pop 12 --gen 4 --out results/moo_smoke.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charaka.bioactivity import BioactivityEnsemble  # noqa: E402
from charaka.data import load_moonshot, split  # noqa: E402
from charaka.model import JTreeDecoder, JTreeEncoder, JTreeVAE  # noqa: E402
from charaka.nsga import (  # noqa: E402
    LatentProblem,
    NSGAConfig,
    build_anchor,
    make_decoder_evaluator,
)
from charaka.scorer import GINEAttachmentScorer  # noqa: E402
from charaka.tokenize import sequence_to_tree, tree_to_sequence  # noqa: E402
from charaka.vocab import Vocabulary, smiles_to_tree  # noqa: E402

from charaka.builders import _build_assembler, _build_potency_predictor  # noqa: E402

from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.rvea import RVEA


def make_algorithm(name: str, n_obj: int, pop: int, n_var: int,
                   sbx_eta: int, pm_eta: int):
    """Configure a pymoo algorithm with a population matched to ``pop``."""
    name = name.upper()
    sbx = SBX(prob=0.9, eta=sbx_eta)
    pm = PM(prob=1.0 / n_var, eta=pm_eta)
    if name == "NSGA2":
        return NSGA2(pop_size=pop, sampling=FloatRandomSampling(),
                     crossover=sbx, mutation=pm, eliminate_duplicates=True)
    if name == "SMSEMOA":
        return SMSEMOA(pop_size=pop, sampling=FloatRandomSampling(),
                       crossover=sbx, mutation=pm)
    if name == "SPEA2":
        return SPEA2(pop_size=pop, sampling=FloatRandomSampling(),
                     crossover=sbx, mutation=pm)
    # reference-direction algorithms: exactly ``pop`` directions => matched budget
    ref = get_reference_directions("energy", n_obj, pop, seed=1)
    if name == "NSGA3":
        return NSGA3(ref_dirs=ref, sampling=FloatRandomSampling(),
                     crossover=sbx, mutation=pm, eliminate_duplicates=True)
    if name == "CTAEA":  # constraint-native decomposition (replaces MOEAD here)
        return CTAEA(ref_dirs=ref, sampling=FloatRandomSampling(),
                     crossover=sbx, mutation=pm)
    if name == "MOEAD":  # NOTE: pymoo's MOEAD does NOT support constraints
        return MOEAD(ref_dirs=ref, n_neighbors=min(15, pop),
                     prob_neighbor_mating=0.7, crossover=sbx, mutation=pm)
    if name == "RVEA":
        return RVEA(ref_dirs=ref, sampling=FloatRandomSampling(),
                    crossover=sbx, mutation=pm)
    raise ValueError(f"Unknown algorithm: {name}")


def hypervolume(feasible, n_obj: int) -> tuple:
    """HV of the non-dominated front of the accumulated feasible set.

    Objectives are maximised in [0,1]; we map to minimisation loss
    ``1 - obj`` and use ref point [1,...,1]. Returns ``(hv, n_front)``.
    """
    if not feasible:
        return 0.0, 0
    rows = []
    for r in feasible:
        pot = r.potency if r.potency is not None else 0.0
        rows.append([r.qed_na, r.sa_norm, r.warhead, pot])
    obj = np.array(rows, dtype=float)
    loss = 1.0 - obj  # minimise; 0 = best, 1 = worst
    nd = NonDominatedSorting().do(loss, only_non_dominated_front=True)
    front = loss[nd]
    hv = HV(ref_point=np.ones(n_obj))(front)
    return float(hv), int(len(front))


def hv_from_F(F, n_obj: int) -> tuple:
    """HV of pymoo's FINAL returned non-dominated feasible front (res.F).

    This is the STANDARD MOO-benchmark metric (what the algorithm converged
    to), unlike ``hypervolume()`` which scores the accumulated feasible set
    (an analog-yield view that rewards exploration breadth). ``res.F`` holds
    the minimised objectives (-obj, obj in [0,1]); loss = 1 + F in [0,1],
    ref point = ones. Returns ``(hv, front_size)``.
    """
    if F is None:
        return 0.0, 0
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if F.size == 0:
        return 0.0, 0
    loss = np.clip(1.0 + F, 0.0, 1.0)
    return float(HV(ref_point=np.ones(n_obj))(loss)), int(len(F))


def run_one(anchor, decode_fn, nsga_cfg, algo_name, n_gen, seed):
    problem = LatentProblem(anchor, decode_fn, nsga_cfg)
    algo = make_algorithm(algo_name, problem.n_obj, nsga_cfg.pop_size,
                          problem.n_var, nsga_cfg.sbx_eta, nsga_cfg.pm_eta)
    t0 = time.time()
    res = pymoo_minimize(problem, algo, ("n_gen", n_gen), verbose=False, seed=seed)
    dt = time.time() - t0
    hv_final, front_final = hv_from_F(res.F, problem.n_obj)        # STANDARD: comparison
    hv_accum, front_accum = hypervolume(problem.feasible, problem.n_obj)  # analog-yield
    return {
        "n_feasible": len(problem.feasible),
        "hv_final": hv_final, "front_final": front_final,
        "hv_accum": hv_accum, "front_accum": front_accum,
        "runtime_s": dt,
    }


def load_pipeline(cfg, device, scorer_path=None):
    """Load the frozen pipeline (encoder, decoder, scorer) and anchors.

    ``scorer_path`` overrides which scorer weights to load (default:
    ``scorer.pth`` in the checkpoint directory).
    """
    ckpt = Path(cfg["paths"]["checkpoint_dir"])
    vocab = Vocabulary.load(ckpt / "vocab.txt")
    m = cfg["model"]
    encoder = JTreeEncoder(vocab_size=len(vocab), pad_idx=vocab.pad,
                           latent_dim=m["latent_dim"], emb_dim=m["emb_dim"],
                           n_layers=m["encoder_layers"], n_heads=m["n_heads"],
                           max_len=m["max_tree_len"])
    decoder = JTreeDecoder(vocab_size=len(vocab), pad_idx=vocab.pad,
                           sos_idx=vocab.sos, eos_idx=vocab.eos,
                           latent_dim=m["latent_dim"], emb_dim=m["emb_dim"],
                           n_layers=m["decoder_layers"], n_heads=m["n_heads"],
                           max_len=m["max_tree_len"])
    model = JTreeVAE(encoder, decoder).to(device)
    state = torch.load(ckpt / "jtvae_best.pth", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    scorer = GINEAttachmentScorer(latent_dim=m["latent_dim"]).to(device)
    sp = Path(scorer_path) if scorer_path else (ckpt / "scorer.pth")
    scorer.load_state_dict(torch.load(sp, map_location=device))
    scorer.eval()
    print(f"scorer: {sp}")

    ens = BioactivityEnsemble(n_members=cfg["bioactivity"]["n_members"],
                              latent_dim=m["latent_dim"],
                              hidden=cfg["bioactivity"]["hidden_dim"],
                              dropout=cfg["bioactivity"]["dropout"])
    ens.load_members([ckpt / "bioactivity" / f"member_{i}.pth"
                      for i in range(cfg["bioactivity"]["n_members"])], device)
    ens.to(device).eval()

    predict = _build_potency_predictor(model.encoder, ens, vocab,
                                       m["max_tree_len"], device)
    assembler = _build_assembler(scorer, vocab, device,
                                 cfg["nsga"]["scorer_max_attempts"], use_scorer=True)
    decode_fn = make_decoder_evaluator(
        model=model,
        sequence_to_tree=lambda t: sequence_to_tree(t, vocab),
        assemble_tree=assembler, bioactivity_predictor=predict,
        device=device, temperature=cfg["nsga"]["temperature"])

    df = load_moonshot(cfg["dataset"]["csv_path"])
    train_df, _ = split(df, test_size=cfg["dataset"]["val_fraction"], seed=cfg["seed"])
    measured = train_df[train_df["value"].notna()]
    anchors_df = measured.sort_values("value").head(cfg["nsga"]["top_k"])
    return model, vocab, predict, decode_fn, anchors_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/moo_catalogue_v3.yaml",
                    help="catalogue config (with a 'catalogue:' block), or a "
                         "plain pipeline config for legacy CLI-driven use")
    ap.add_argument("--algos", default=None)
    ap.add_argument("--n-anchors", type=int, default=None)
    ap.add_argument("--anchor-indices", default=None,
                    help="comma-sep anchor indices (potency rank); overrides --n-anchors")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--pop", type=int, default=None, help="override pop_size")
    ap.add_argument("--gen", type=int, default=None, help="override generations")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="override scorer assembly attempts (speed knob)")
    ap.add_argument("--scorer-path", default=None,
                    help="override scorer weights")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Config resolution: a catalogue config carries a 'catalogue:' block and
    # points at the frozen-pipeline config; a plain pipeline config is the
    # legacy CLI-driven path. Precedence for every param: CLI > config > default.
    with open(args.config, "r", encoding="utf-8") as f:
        top = yaml.safe_load(f)
    cat = top.get("catalogue", {})
    if cat:
        with open(top.get("pipeline_config", "configs/moo_catalogue_v3.yaml"),
                  "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = top

    def pick(cli, key, default):
        return cli if cli is not None else cat.get(key, default)
    algos_str = pick(args.algos, "algos", "NSGA2,NSGA3,CTAEA,SMSEMOA,SPEA2,RVEA")
    algos_str = ",".join(algos_str) if isinstance(algos_str, list) else algos_str
    anchor_ind = args.anchor_indices or (
        ",".join(map(str, cat["anchor_indices"])) if cat.get("anchor_indices") else None)
    seeds = int(pick(args.seeds, "seeds", 3))
    n_anchors = int(pick(args.n_anchors, "n_anchors", 4))
    out_path = pick(args.out, "out", "results/moo_bench.csv")
    scorer_path = pick(args.scorer_path, "scorer_path", None)
    if pick(args.pop, "pop_size", None):
        cfg["nsga"]["pop_size"] = int(pick(args.pop, "pop_size", None))
    if pick(args.gen, "generations", None):
        cfg["nsga"]["generations"] = int(pick(args.gen, "generations", None))
    if pick(args.max_attempts, "max_attempts", None):
        cfg["nsga"]["scorer_max_attempts"] = int(pick(args.max_attempts, "max_attempts", None))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  pop={cfg['nsga']['pop_size']}  "
          f"gen={cfg['nsga']['generations']}")
    model, vocab, predict, decode_fn, anchors_df = load_pipeline(
        cfg, device, scorer_path=scorer_path)

    nsga_cfg = NSGAConfig(
        pop_size=cfg["nsga"]["pop_size"], generations=cfg["nsga"]["generations"],
        radius=cfg["nsga"]["radius"], sbx_eta=cfg["nsga"]["sbx_eta"],
        pm_eta=cfg["nsga"]["pm_eta"], pot_floor_abs=cfg["nsga"]["pot_floor_abs"],
        pot_floor_rel=cfg["nsga"]["pot_floor_rel"],
        sa_floor_base=cfg["nsga"]["sa_floor_base"],
        sa_floor_rel_delta=cfg["nsga"]["sa_floor_rel_delta"],
        tanimoto_min=cfg["nsga"]["tanimoto_min"],
        uncertainty_global=cfg["nsga"]["uncertainty_global"],
        uncertainty_margin=cfg["nsga"]["uncertainty_margin"])

    algos = [a.strip() for a in algos_str.split(",") if a.strip()]
    idxs = ([int(x) for x in anchor_ind.split(",")] if anchor_ind
            else list(range(min(n_anchors, len(anchors_df)))))
    rows = []
    for ai in idxs:
        arow = anchors_df.iloc[ai]
        smi = arow["canonical"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        pred_v, pot, std = predict(smi)
        res = smiles_to_tree(smi)
        if res is None:
            continue
        cs, edges, _ = res
        seq = torch.tensor(tree_to_sequence(cs, edges, vocab, cfg["model"]["max_tree_len"]),
                           dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            _, mu, _ = model.encoder(seq)
        anchor = build_anchor(smi=smi, mu=mu[0].cpu().numpy(),
                              pred_log_ic50=pred_v, potency=pot, pred_std=std)
        for algo in algos:
            for s in range(seeds):
                try:
                    r = run_one(anchor, decode_fn, nsga_cfg, algo,
                                nsga_cfg.generations, seed=s)
                except Exception as e:  # one algo/anchor failing must not kill the run
                    print(f"  anchor {ai+1} {algo} seed {s} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                r.update(anchor_idx=ai, algo=algo, seed=s)
                rows.append(r)
                pd.DataFrame(rows).to_csv(out_path, index=False)  # checkpoint each run
                print(f"  anchor {ai} ({idxs.index(ai)+1}/{len(idxs)}) {algo:<8} seed {s}  "
                      f"HVfin={r['hv_final']:.4f} (front {r['front_final']:>2})  "
                      f"HVacc={r['hv_accum']:.4f}  feas={r['n_feasible']:>4}  "
                      f"{r['runtime_s']:.0f}s", flush=True)

    if not rows:
        print("No results.")
        return 1
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== MOO algorithm comparison (mean over anchors x seeds) ===")
    print("    HV_final = standard metric (res.F); HV_accum = analog-yield view")
    summ = (df.groupby("algo")
              .agg(HV_final=("hv_final", "mean"), HVfin_std=("hv_final", "std"),
                   HV_accum=("hv_accum", "mean"), feasible=("n_feasible", "mean"),
                   front_final=("front_final", "mean"), runtime_s=("runtime_s", "mean"))
              .sort_values("HV_final", ascending=False))
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nsaved per-run rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
