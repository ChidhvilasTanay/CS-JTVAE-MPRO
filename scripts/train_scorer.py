"""Jointly train the GINEConv attachment scorer and JT-VAE encoder.

The schedule has two phases:

* Phase A (warm-up): encoder frozen, scorer-only cross-entropy.
* Phase B (joint): encoder unfrozen at a small learning rate, combined
  loss ``scorer_CE + lambda_recon * recon_CE + beta * KL``. The decoder
  remains frozen throughout.

The scorer dataset is constructed by walking each training molecule's
junction tree edge-by-edge and recording the ground-truth alignment
plus the candidate set produced by :func:`charaka.scorer.candidate_alignments`.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml
from rdkit import Chem
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charaka.data import load_moonshot, split  # noqa: E402
from charaka.model import JTreeDecoder, JTreeEncoder, JTreeVAE  # noqa: E402
from charaka.scorer import (  # noqa: E402
    GINEAttachmentScorer,
    apply_alignment,
    candidate_alignments,
    candidate_graph,
)
from charaka.tokenize import MAX_TREE_LEN, tree_to_sequence  # noqa: E402
from charaka.training import JointConfig, joint_phase, joint_phase_batched  # noqa: E402
from charaka.vocab import Vocabulary, get_clusters, smiles_to_tree  # noqa: E402


def _submol_with_mapping(
    mol: Chem.Mol, atom_indices,
) -> Tuple[Optional[str], Optional[dict]]:
    # Kekulise first (clear aromatic flags) so a fragment that contains aromatic
    # ring atoms but not the whole ring (e.g. a c-Cl or c-N bond cluster) extracts
    # cleanly. This matches cluster_smiles() (the tokenizer that built the vocab),
    # which kekulises before fragmenting; without it the substructure match below
    # fails and the entire molecule is silently dropped from scorer training.
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
        mol = kek
    except Exception:
        pass
    rw = Chem.RWMol()
    orig_to_sub = {}
    for o in atom_indices:
        a = mol.GetAtomWithIdx(o)
        new = Chem.Atom(a.GetSymbol())
        new.SetFormalCharge(a.GetFormalCharge())
        new.SetIsAromatic(a.GetIsAromatic())
        new.SetNoImplicit(False)
        orig_to_sub[o] = rw.AddAtom(new)

    seen = set()
    for o in atom_indices:
        a = mol.GetAtomWithIdx(o)
        for bond in a.GetBonds():
            other = bond.GetOtherAtomIdx(o)
            if other in orig_to_sub:
                key = frozenset((o, other))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    rw.AddBond(orig_to_sub[o], orig_to_sub[other],
                               bond.GetBondType())
                except Exception:
                    return None, None
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        try:
            Chem.SanitizeMol(
                rw, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE,
            )
        except Exception:
            pass
    try:
        smi = Chem.MolToSmiles(rw, canonical=True)
    except Exception:
        return None, None
    canonical = Chem.MolFromSmiles(smi)
    if canonical is None:
        return None, None
    match = canonical.GetSubstructMatch(rw)
    if not match or len(match) != rw.GetNumAtoms():
        return None, None
    sub_to_canonical = {si: match[si] for si in range(rw.GetNumAtoms())}
    return smi, {o: sub_to_canonical[orig_to_sub[o]] for o in atom_indices}


def build_scorer_dataset(df, vocab: Vocabulary, max_len: int) -> List:
    """Build ``(candidate_graphs, gt_index, token_sequence)`` examples."""
    examples: List = []
    for smi in tqdm(list(df["canonical"]), desc="build scorer dataset", unit="mol",
                    mininterval=15.0, miniters=100):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            clusters_orig = get_clusters(mol)
        except Exception:
            continue
        if len(clusters_orig) < 2:
            continue
        result = smiles_to_tree(smi)
        if result is None:
            continue
        cs, _, _ = result
        token_seq = tree_to_sequence(cs, [], vocab, max_len)

        cluster_info = []
        ok = True
        for atom_set in clusters_orig:
            smi_c, mapping = _submol_with_mapping(mol, atom_set)
            if smi_c is None:
                ok = False
                break
            cluster_info.append((set(atom_set), smi_c, mapping))
        if not ok:
            continue

        adjacency = defaultdict(dict)
        for i in range(len(cluster_info)):
            for j in range(i + 1, len(cluster_info)):
                inter = cluster_info[i][0] & cluster_info[j][0]
                if inter:
                    adjacency[i][j] = inter
                    adjacency[j][i] = inter
        if not adjacency:
            continue

        visited = {0}
        bfs = [0]
        tree_edges = []
        while bfs:
            u = bfs.pop(0)
            for v in adjacency[u]:
                if v not in visited:
                    visited.add(v)
                    bfs.append(v)
                    tree_edges.append((u, v, adjacency[u][v]))

        parent_mol = Chem.MolFromSmiles(cluster_info[0][1])
        if parent_mol is None:
            continue
        rwmol = Chem.RWMol(parent_mol)
        atom_map = {0: list(range(parent_mol.GetNumAtoms()))}

        for (par, node, shared_orig) in tree_edges:
            if par not in atom_map:
                continue
            parent_smi = cluster_info[par][1]
            child_smi = cluster_info[node][1]
            parent_map = cluster_info[par][2]
            child_map = cluster_info[node][2]
            pm = Chem.MolFromSmiles(parent_smi)
            cm = Chem.MolFromSmiles(child_smi)
            if pm is None or cm is None:
                break

            shared_p = sorted(parent_map[s] for s in shared_orig if s in parent_map)
            shared_c = sorted(child_map[s] for s in shared_orig if s in child_map)
            if (len(shared_p) != len(shared_c)
                    or not shared_p or len(shared_p) > 2):
                continue

            cands = candidate_alignments(pm, cm)
            par_atoms = atom_map[par]
            translated = []
            for c in cands:
                if max(c["p_atoms"]) >= len(par_atoms):
                    continue
                tc = dict(c)
                tc["p_atoms"] = tuple(par_atoms[pi] for pi in c["p_atoms"])
                translated.append((c, tc))
            if not translated:
                continue

            gt_p_rw = sorted(par_atoms[ci] for ci in shared_p)
            gt_idx = -1
            for i, (orig_c, tc) in enumerate(translated):
                if (sorted(tc["p_atoms"]) == gt_p_rw
                        and sorted(orig_c["c_atoms"]) == shared_c):
                    gt_idx = i
                    break
            if gt_idx < 0:
                continue

            graphs = [candidate_graph(rwmol, cm, tc) for _, tc in translated]
            if any(g is None for g in graphs):
                continue

            examples.append((graphs, gt_idx, token_seq))

            _, gt_tc = translated[gt_idx]
            new_indices = apply_alignment(rwmol, cm, gt_tc)
            if new_indices is None:
                break
            atom_map[node] = [new_indices[ci] for ci in range(cm.GetNumAtoms())]
    return examples


def main(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    import datetime, time as _time
    t_start = _time.time()
    print(f"=== scorer joint training | start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
          f"| device={device} | ckpt_dir={ckpt_dir} ===", flush=True)

    vocab = Vocabulary.load(ckpt_dir / "vocab.txt")
    df = load_moonshot(cfg["dataset"]["csv_path"])
    train_df, _ = split(
        df, test_size=cfg["dataset"]["val_fraction"], seed=cfg["seed"],
    )

    max_mols = cfg["dataset"].get("max_molecules")
    if max_mols and max_mols < len(train_df):
        train_df = train_df.sample(n=max_mols, random_state=cfg["seed"])
        print(f"[subset] training on {len(train_df)} molecules "
              f"(max_molecules={max_mols})", flush=True)

    print("Building scorer dataset...", flush=True)
    dataset = build_scorer_dataset(train_df, vocab, cfg["model"]["max_tree_len"])
    print(f"Scorer training examples: {len(dataset)}")

    # Restore VAE
    encoder = JTreeEncoder(
        vocab_size=len(vocab), pad_idx=vocab.pad,
        latent_dim=cfg["model"]["latent_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        n_layers=cfg["model"]["encoder_layers"],
        n_heads=cfg["model"]["n_heads"],
        max_len=cfg["model"]["max_tree_len"],
    )
    decoder = JTreeDecoder(
        vocab_size=len(vocab), pad_idx=vocab.pad,
        sos_idx=vocab.sos, eos_idx=vocab.eos,
        latent_dim=cfg["model"]["latent_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        n_layers=cfg["model"]["decoder_layers"],
        n_heads=cfg["model"]["n_heads"],
        max_len=cfg["model"]["max_tree_len"],
    )
    model = JTreeVAE(encoder, decoder).to(device)
    state = torch.load(ckpt_dir / "jtvae_best.pth", map_location=device)
    model.load_state_dict(state["model"])

    # Freeze decoder throughout
    for p in model.decoder.parameters():
        p.requires_grad = False

    scorer = GINEAttachmentScorer(latent_dim=cfg["model"]["latent_dim"]).to(device)
    scorer_opt = torch.optim.Adam(
        scorer.parameters(), lr=cfg["joint"]["scorer_lr"],
    )
    encoder_opt = torch.optim.Adam(
        model.encoder.parameters(), lr=cfg["joint"]["encoder_lr"],
    )

    joint_cfg = JointConfig(
        warmup_epochs=cfg["joint"]["warmup_epochs"],
        joint_epochs=cfg["joint"]["joint_epochs"],
        scorer_lr=cfg["joint"]["scorer_lr"],
        encoder_lr=cfg["joint"]["encoder_lr"],
        lambda_recon=cfg["joint"]["lambda_recon"],
        beta_kl=cfg["joint"]["beta_kl"],
        patience=cfg["joint"]["patience"],
    )

    indices = list(range(len(dataset)))
    val_size = max(1, int(cfg["joint"]["val_fraction"] * len(indices)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    scorer_path = ckpt_dir / "scorer.pth"
    encoder_path = ckpt_dir / "encoder_finetuned.pth"
    best_val = -1.0

    batch_size = int(cfg["joint"].get("batch_size", 1))
    use_batched = batch_size > 1
    print(f"\n=== Phase A: scorer warm-up (encoder frozen)"
          f"{f' | minibatched bs={batch_size}' if use_batched else ''} ===")
    for p in model.encoder.parameters():
        p.requires_grad = False
    for ep in range(1, joint_cfg.warmup_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release fragmented cache each epoch (Windows has no expandable_segments)
        if use_batched:
            t_sc, _, _, t_acc = joint_phase_batched(
                scorer, model.encoder, dataset, train_indices, device,
                vocab.pad, joint_cfg, True, scorer_opt, batch_size=batch_size)
            v_sc, _, _, v_acc = joint_phase_batched(
                scorer, model.encoder, dataset, val_indices, device,
                vocab.pad, joint_cfg, False, scorer_opt, batch_size=batch_size)
        else:
            t_sc, _, _, t_acc = joint_phase(
                scorer, model.encoder, model.decoder, dataset, train_indices,
                device, vocab.pad, joint_cfg, train=True, train_encoder=False,
                scorer_opt=scorer_opt,
            )
            v_sc, _, _, v_acc = joint_phase(
                scorer, model.encoder, model.decoder, dataset, val_indices,
                device, vocab.pad, joint_cfg, train=False, train_encoder=False,
                scorer_opt=scorer_opt,
            )
        tag = ""
        if v_acc > best_val:
            best_val = v_acc
            torch.save(scorer.state_dict(), scorer_path)
            tag = "  [saved]"
        print(
            f"[A {ep:02d}/{joint_cfg.warmup_epochs}] "
            f"trn loss={t_sc:.3f} acc={t_acc:.3f} | "
            f"val loss={v_sc:.3f} acc={v_acc:.3f}{tag}"
        )

    print("\n=== Phase B: joint training (encoder unfrozen) ===")
    for p in model.encoder.parameters():
        p.requires_grad = True
    no_improve = 0
    for ep in range(1, joint_cfg.joint_epochs + 1):
        t_sc, t_rec, t_kl, t_acc = joint_phase(
            scorer, model.encoder, model.decoder, dataset, train_indices,
            device, vocab.pad, joint_cfg, train=True, train_encoder=True,
            scorer_opt=scorer_opt, encoder_opt=encoder_opt,
        )
        v_sc, v_rec, v_kl, v_acc = joint_phase(
            scorer, model.encoder, model.decoder, dataset, val_indices,
            device, vocab.pad, joint_cfg, train=False, train_encoder=True,
            scorer_opt=scorer_opt, encoder_opt=encoder_opt,
        )
        tag = ""
        if v_acc > best_val:
            best_val = v_acc
            torch.save(scorer.state_dict(), scorer_path)
            torch.save(model.encoder.state_dict(), encoder_path)
            tag = "  [saved]"
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"[B {ep:02d}/{joint_cfg.joint_epochs}] "
            f"trn sc={t_sc:.3f} rec={t_rec:.3f} kl={t_kl:.3f} acc={t_acc:.3f} | "
            f"val sc={v_sc:.3f} rec={v_rec:.3f} kl={v_kl:.3f} acc={v_acc:.3f}{tag}"
        )
        if no_improve >= joint_cfg.patience:
            print("Early stop.")
            break

    print(f"\nBest validation top-1: {best_val:.3f}")
    print(f"Scorer:  {scorer_path}")
    print(f"Encoder: {encoder_path}")
    print(f"=== done in {(_time.time() - t_start)/60:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/joint_full_v3_aug.yaml")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
