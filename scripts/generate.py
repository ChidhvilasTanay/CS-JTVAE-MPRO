"""Latent-space multi-objective search: generate analogs for each anchor.

For every top-active anchor this loads the frozen JT-VAE, the GINEConv
attachment scorer, and the bioactivity ensemble, runs a constrained
multi-objective search over the latent box around the anchor, and keeps the
final-population Pareto front (``res.opt``). The optimiser (NSGA-III for the
reported run), the objectives, and the constraints are all set by the config.
From the same run it also reports the feasible archive and the non-dominated
front over that archive, so the two notions can be compared.

Reproducibility note. The decoder samples with temperature, so a fresh run is
not byte-identical to a saved archive; this script seeds numpy and torch per
anchor so its own output is reproducible.

Usage
  python scripts/generate.py --config configs/nsga_v3_6obj.yaml
  python scripts/generate.py --quick --max-anchors 1   # mechanics smoke test
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from rdkit import Chem
from rdkit.Chem.QED import qed as compute_qed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Reuse the EXACT pymoo wiring + problem class from the existing module.
from charaka import nsga as _nsga  # noqa: E402
from charaka.bioactivity import (  # noqa: E402
    BioactivityEnsemble,
    POTENCY_CENTRE,
    POTENCY_SLOPE,
)
from charaka.data import load_moonshot, split  # noqa: E402
from charaka.model import JTreeDecoder, JTreeEncoder, JTreeVAE  # noqa: E402
from charaka.nsga import NSGAConfig, build_anchor, make_decoder_evaluator  # noqa: E402
from charaka.objectives import (  # noqa: E402
    CompositeWeights,
    composite,
    detect_warheads,
)
from charaka.scorer import GINEAttachmentScorer, assemble_with_scorer  # noqa: E402
from charaka.tokenize import sequence_to_tree, tree_to_sequence  # noqa: E402
from charaka.assemble import assemble_tree  # noqa: E402
from charaka.vocab import Vocabulary, smiles_to_tree  # noqa: E402

# The script writes only its own *_resopt.csv / *_summary.csv outputs; guard
# against accidentally naming an output over an input dataset.
_PROTECTED = {"covid_moonshot_processed.csv", "covid_moonshot_augmented_v3.csv"}

# Live-tracking log handle (set in main). _emit streams to stdout and the log.
_LOGF = None


def _emit(msg):
    print(msg, flush=True)
    if _LOGF is not None:
        _LOGF.write(msg + "\n")
        _LOGF.flush()


class _Progress(Callback):
    """Streams one line per search generation for the current anchor."""

    def __init__(self, label, n_gen, problem, t0):
        super().__init__()
        self.label, self.n_gen, self.problem, self.t0 = label, n_gen, problem, t0

    def notify(self, algorithm):
        _emit(
            f"    [{self.label}] gen {algorithm.n_gen:>2}/{self.n_gen}  "
            f"feasible_so_far={len(self.problem.feasible):>4}  "
            f"+{time.time() - self.t0:5.0f}s"
        )


# --- pipeline builders: the latent-space potency predictor and the
#     scorer-guided assembler --------------------------------------------
def _build_potency_predictor(encoder, ensemble, vocab, max_len, device):
    def predict(smi):
        result = smiles_to_tree(smi)
        if result is None:
            return None, 0.5, 0.0
        cs, edges, _ = result
        seq = torch.tensor(
            tree_to_sequence(cs, edges, vocab, max_len),
            dtype=torch.long, device=device,
        ).unsqueeze(0)
        encoder.eval()
        with torch.no_grad():
            _, mu, _ = encoder(seq)
            pred, std = ensemble(mu)
        pred_val = float(pred.item())
        norm = float(torch.sigmoid(-(pred - POTENCY_CENTRE) * POTENCY_SLOPE).item())
        return pred_val, norm, float(std.item())

    return predict


def _build_assembler(scorer, vocab, device, max_attempts, use_scorer=True):
    def _finalise(rwmol):
        try:
            Chem.SanitizeMol(rwmol)
            smi = Chem.MolToSmiles(rwmol, canonical=True, isomericSmiles=True)
            return smi if Chem.MolFromSmiles(smi) is not None else None
        except Exception:
            return None

    def assemble(cs, edges, z=None):
        if use_scorer and z is not None:
            out = assemble_with_scorer(
                cs, edges, scorer, z, device,
                max_attempts=max_attempts, finaliser=_finalise,
            )
            if out is not None:
                return out
        return assemble_tree(cs, edges)

    return assemble


def _run_capture(anchor, decode_fn, cfg, seed, label, t0):
    """Runs one anchor's search like the base loop but also returns the
    pymoo result (for ``res.opt``) and streams per-generation progress."""
    problem = _nsga.LatentProblem(anchor, decode_fn, cfg)
    sbx = _nsga.SBX(prob=0.9, eta=cfg.sbx_eta)
    pm = _nsga.PM(prob=1.0 / problem.n_var, eta=cfg.pm_eta)
    if getattr(cfg, "algorithm", "NSGA2").upper() == "NSGA3":
        # reference-direction algorithm for many-objective (>=5) fronts
        ref = get_reference_directions("energy", problem.n_obj, cfg.pop_size, seed=1)
        algorithm = NSGA3(ref_dirs=ref, sampling=_nsga.FloatRandomSampling(),
                          crossover=sbx, mutation=pm, eliminate_duplicates=True)
    else:
        algorithm = _nsga.NSGA2(
            pop_size=cfg.pop_size, sampling=_nsga.FloatRandomSampling(),
            crossover=sbx, mutation=pm, eliminate_duplicates=True)
    res = _nsga.pymoo_minimize(
        problem, algorithm, ("n_gen", cfg.generations), verbose=False, seed=seed,
        callback=_Progress(label, cfg.generations, problem, t0),
    )
    return problem.feasible, res


def _nondominated_idx(M):
    """Boolean mask of non-dominated rows of M (all columns maximised)."""
    n = len(M)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and np.all(M[j] >= M[i]) and np.any(M[j] > M[i]):
                keep[i] = False
                break
    return keep


def _row(anchor, mol, orig_smi, orig_composite, r, weights):
    cand_comp = composite(r.qed_na, r.sa_norm, r.warhead, r.potency, weights)
    return {
        "orig_smi": orig_smi,
        "orig_warheads": ",".join(sorted(anchor.warheads)) or "-",
        "orig_qed": compute_qed(mol),
        "orig_qed_noalerts": anchor.qed_na,
        "orig_sa_norm": anchor.sa_norm,
        "orig_sa_raw": anchor.sa_raw,
        "orig_potency_norm": anchor.potency,
        "orig_pred_log10_ic50": anchor.pred_log_ic50,
        "orig_pred_v_std": anchor.pred_std,
        "orig_composite": orig_composite,
        "cand_smi": r.cand_smi,
        "cand_warheads": ",".join(sorted(detect_warheads(r.cand_mol))) or "-",
        "cand_qed": compute_qed(r.cand_mol),
        "cand_qed_noalerts": r.qed_na,
        "cand_sa_norm": r.sa_norm,
        "cand_sa_raw": r.sa_raw,
        "cand_warhead_score": r.warhead,
        "cand_potency_norm": r.potency,
        "cand_pred_log10_ic50": r.pred_log_ic50,
        "cand_pred_v_std": r.pred_std,
        "cand_composite": cand_comp,
        "composite_gain": cand_comp - orig_composite,
        "tanimoto": r.tanimoto,
    }


def _keep_awake():
    """Inhibit system sleep for the duration (long runs stalled overnight when
    the laptop slept). Additive, Windows-only, resets when the process exits."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def main(config_path, out_path, max_anchors, quick, feasible_out=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_path = Path(out_path)
    if out_path.name in _PROTECTED:
        raise SystemExit(f"refusing to write over protected file: {out_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if feasible_out is not None and Path(feasible_out).name in _PROTECTED:
        raise SystemExit(f"refusing to write over protected file: {feasible_out}")
    _keep_awake()

    global _LOGF
    log_dir = Path("logs/pareto_front")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    _LOGF = open(log_dir / f"{stamp}.log", "a", encoding="utf-8")
    _emit(f"=== run_pareto_front | start {stamp} | config={config_path} | out={out_path.name} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    vocab = Vocabulary.load(ckpt_dir / "vocab.txt")
    encoder = JTreeEncoder(
        vocab_size=len(vocab), pad_idx=vocab.pad,
        latent_dim=cfg["model"]["latent_dim"], emb_dim=cfg["model"]["emb_dim"],
        n_layers=cfg["model"]["encoder_layers"], n_heads=cfg["model"]["n_heads"],
        max_len=cfg["model"]["max_tree_len"],
    )
    decoder = JTreeDecoder(
        vocab_size=len(vocab), pad_idx=vocab.pad,
        sos_idx=vocab.sos, eos_idx=vocab.eos,
        latent_dim=cfg["model"]["latent_dim"], emb_dim=cfg["model"]["emb_dim"],
        n_layers=cfg["model"]["decoder_layers"], n_heads=cfg["model"]["n_heads"],
        max_len=cfg["model"]["max_tree_len"],
    )
    model = JTreeVAE(encoder, decoder).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "jtvae_best.pth", map_location=device)["model"])
    model.eval()

    scorer = GINEAttachmentScorer(latent_dim=cfg["model"]["latent_dim"]).to(device)
    scorer.load_state_dict(torch.load(ckpt_dir / "scorer.pth", map_location=device))
    scorer.eval()

    ensemble = BioactivityEnsemble(
        n_members=cfg["bioactivity"]["n_members"],
        latent_dim=cfg["model"]["latent_dim"],
        hidden=cfg["bioactivity"]["hidden_dim"],
        dropout=cfg["bioactivity"]["dropout"],
    )
    ensemble.load_members(
        [ckpt_dir / "bioactivity" / f"member_{i}.pth"
         for i in range(cfg["bioactivity"]["n_members"])],
        device,
    )
    ensemble.to(device).eval()

    df = load_moonshot(cfg["dataset"]["csv_path"])
    train_df, _ = split(df, test_size=cfg["dataset"]["val_fraction"], seed=cfg["seed"])
    measured = train_df[train_df["value"].notna()]
    anchors = measured.sort_values("value").head(cfg["nsga"]["top_k"])
    if max_anchors:
        anchors = anchors.head(max_anchors)

    pop = 12 if quick else cfg["nsga"]["pop_size"]
    gen = 5 if quick else cfg["nsga"]["generations"]
    nsga_cfg = NSGAConfig(
        pop_size=pop, generations=gen,
        radius=cfg["nsga"]["radius"], sbx_eta=cfg["nsga"]["sbx_eta"], pm_eta=cfg["nsga"]["pm_eta"],
        pot_floor_abs=cfg["nsga"]["pot_floor_abs"], pot_floor_rel=cfg["nsga"]["pot_floor_rel"],
        sa_floor_base=cfg["nsga"]["sa_floor_base"], sa_floor_rel_delta=cfg["nsga"]["sa_floor_rel_delta"],
        tanimoto_min=cfg["nsga"]["tanimoto_min"], uncertainty_global=cfg["nsga"]["uncertainty_global"],
        uncertainty_margin=cfg["nsga"]["uncertainty_margin"],
        use_logp=cfg["nsga"].get("use_logp", False),
        logp_relative=cfg["nsga"].get("logp_relative", False),
        use_genotox=cfg["nsga"].get("use_genotox", False),
        algorithm=cfg["nsga"].get("algorithm", "NSGA2"),
        logp_constraint=cfg["nsga"].get("logp_constraint", False),
        logp_margin=cfg["nsga"].get("logp_margin", 0.0),
        reversible_only=cfg["nsga"].get("reversible_only", False),
    )
    weights = CompositeWeights(
        qed=cfg["composite"]["qed"], sa=cfg["composite"]["sa"],
        warhead=cfg["composite"]["warhead"], potency=cfg["composite"]["potency"],
    )

    predict_potency = _build_potency_predictor(
        model.encoder, ensemble, vocab, cfg["model"]["max_tree_len"], device,
    )
    assembler = _build_assembler(
        scorer, vocab, device, cfg["nsga"]["scorer_max_attempts"],
        use_scorer=cfg["nsga"].get("use_scorer_assembler", True),
    )
    decode_fn = make_decoder_evaluator(
        model=model,
        sequence_to_tree=lambda t: sequence_to_tree(t, vocab),
        assemble_tree=assembler,
        bioactivity_predictor=predict_potency,
        device=device,
        temperature=cfg["nsga"]["temperature"],
    )

    rows, summary, feas_rows = [], [], []
    t_start = time.time()
    n_anchors = len(anchors)
    i_anchor = 0
    for ai, row in anchors.iterrows():
        i_anchor += 1
        # reproducible per anchor; affects only this process
        np.random.seed(int(ai))
        torch.manual_seed(int(ai))
        t_anchor = time.time()
        label = f"{i_anchor}/{n_anchors}"
        _emit(f"[{label}] anchor idx={int(ai)} starting")

        orig_smi = row["canonical"]
        mol = Chem.MolFromSmiles(orig_smi)
        if mol is None:
            continue
        pred_v, pot, std = predict_potency(orig_smi)
        result = smiles_to_tree(orig_smi)
        if result is None:
            continue
        cs, edges, _ = result
        seq = torch.tensor(
            tree_to_sequence(cs, edges, vocab, cfg["model"]["max_tree_len"]),
            dtype=torch.long, device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            _, mu, _ = model.encoder(seq)
        anchor = build_anchor(
            smi=orig_smi, mu=mu[0].cpu().numpy(),
            pred_log_ic50=pred_v, potency=pot, pred_std=std,
        )
        orig_composite = composite(anchor.qed_na, anchor.sa_norm, 1.0, anchor.potency, weights)

        feasible, res = _run_capture(
            anchor, decode_fn, nsga_cfg, seed=int(ai), label=label, t0=t_anchor,
        )

        # feasible archive molecules (same run) -- only when requested
        if feasible_out is not None:
            for r in feasible:
                feas_rows.append(_row(anchor, mol, orig_smi, orig_composite, r, weights))

        # archive objective matrix + archive Pareto front (same run)
        if feasible:
            March = np.array(
                [[r.qed_na, r.sa_norm, r.warhead, r.potency] for r in feasible], float,
            )
            Xarch = np.array([np.asarray(r.x, float) for r in feasible])
            arch_pareto = int(_nondominated_idx(March).sum())
        else:
            Xarch = np.empty((0, anchor.mu.shape[0]))
            arch_pareto = 0

        # res.opt = final-population non-dominated front; map back to molecules
        opt_rows = 0
        if res is not None and getattr(res, "opt", None) is not None and len(feasible):
            Xopt = np.atleast_2d(np.array(res.opt.get("X"), float))
            for xo in Xopt:
                d = np.linalg.norm(Xarch - xo, axis=1)
                k = int(np.argmin(d))
                if d[k] > 1e-6:
                    continue  # res.opt member not in feasible archive (e.g. infeasible)
                rows.append(_row(anchor, mol, orig_smi, orig_composite, feasible[k], weights))
                opt_rows += 1

        summary.append({
            "anchor_idx": int(ai), "orig_smi": orig_smi,
            "feasible_archive": len(feasible),
            "archive_pareto": arch_pareto,
            "resopt_front": opt_rows,
        })
        dt = time.time() - t_anchor
        eta_min = (time.time() - t_start) / i_anchor * (n_anchors - i_anchor) / 60.0
        _emit(
            f"[{label}] DONE idx={int(ai)}  feasible={len(feasible):>4}  "
            f"archive_pareto={arch_pareto:>3}  res.opt={opt_rows:>3}  "
            f"({dt:.0f}s, ETA {eta_min:.0f} min)"
        )

    if rows:
        pd.DataFrame(rows).drop_duplicates(
            subset=["orig_smi", "cand_smi"],
        ).to_csv(out_path, index=False)
    sm = pd.DataFrame(summary)
    sum_path = out_path.with_name(out_path.stem + "_summary.csv")
    sm.to_csv(sum_path, index=False)

    if feasible_out is not None and feas_rows:
        fp = Path(feasible_out)
        fp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(feas_rows).drop_duplicates(
            subset=["orig_smi", "cand_smi"]).to_csv(fp, index=False)
        _emit(f"feasible archive       -> {fp}  ({len(feas_rows)} rows pre-dedup)")

    _emit(f"\nres.opt analogs        -> {out_path}")
    _emit(f"per-anchor summary     -> {sum_path}")
    if len(sm):
        _emit(
            f"TOTALS  feasible={int(sm.feasible_archive.sum())}  "
            f"archive_pareto={int(sm.archive_pareto.sum())}  "
            f"res.opt={int(sm.resopt_front.sum())}"
        )
    _emit(f"total wall time {(time.time() - t_start) / 60.0:.1f} min")
    if _LOGF is not None:
        _LOGF.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nsga_v3_6obj.yaml")
    ap.add_argument("--out", default="data/analogs_v3_6obj_resopt.csv")
    ap.add_argument("--max-anchors", type=int, default=0,
                    help="limit to the first N anchors (0 = all)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny pop/gen for a fast mechanics smoke test")
    ap.add_argument("--feasible-out", default=None,
                    help="also dump the feasible archive molecules from the SAME "
                         "run to this path (for feasible-vs-res.opt comparison)")
    a = ap.parse_args()
    raise SystemExit(main(a.config, a.out, a.max_anchors, a.quick, a.feasible_out))
