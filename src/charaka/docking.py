"""AutoDock Vina wrapper for paired ligand-vs-anchor docking.

The wrapper expects a prepared receptor PDBQT file (e.g. via Meeko or
OpenBabel) and a Vina executable on ``PATH``. Ligands are embedded with
RDKit ETKDGv3 + MMFF94 and converted to PDBQT via OpenBabel.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class DockingBox:
    """Vina search box. Coordinates are angstroms."""
    center: tuple = (10.8, -18.0, 20.4)
    size: tuple = (22.0, 22.0, 22.0)


@dataclass
class DockingConfig:
    vina_executable: str = "vina"
    receptor_pdbqt: str = "7L0D_receptor.pdbqt"
    box: DockingBox = None
    exhaustiveness: int = 8
    num_modes: int = 5
    seed: Optional[int] = 42
    embed_seed: int = 42

    def __post_init__(self):
        if self.box is None:
            self.box = DockingBox()


def _binary_available(path: str, args: Sequence[str] = ("--version",)) -> bool:
    try:
        result = subprocess.run(
            [path, *args], capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def check_environment(cfg: DockingConfig) -> Dict[str, bool]:
    """Verify Vina, OpenBabel and the receptor file are all available."""
    return {
        "vina": _binary_available(cfg.vina_executable),
        "obabel": _binary_available("obabel", ("-V",)),
        "receptor": Path(cfg.receptor_pdbqt).exists(),
    }


def smiles_to_pdbqt(smi: str, out_path: str, cfg: DockingConfig) -> bool:
    """Embed ``smi`` to 3D and write a Gasteiger-charged PDBQT."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=cfg.embed_seed) != 0:
            return False
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        return False
    pdb_path = out_path.replace(".pdbqt", ".pdb")
    Chem.MolToPDBFile(mol, pdb_path)
    try:
        subprocess.run(
            ["obabel", pdb_path, "-O", out_path,
             "--partialcharge", "gasteiger"],
            capture_output=True, timeout=30, check=True,
        )
        os.remove(pdb_path)
        return os.path.exists(out_path)
    except Exception:
        return False


def dock_one(ligand_pdbqt: str, cfg: DockingConfig) -> Optional[float]:
    """Dock a single PDBQT and return the mode-1 affinity (kcal/mol)."""
    out_pdbqt = ligand_pdbqt.replace(".pdbqt", "_out.pdbqt")
    cmd = [
        cfg.vina_executable,
        "--receptor", cfg.receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", out_pdbqt,
        "--center_x", str(cfg.box.center[0]),
        "--center_y", str(cfg.box.center[1]),
        "--center_z", str(cfg.box.center[2]),
        "--size_x", str(cfg.box.size[0]),
        "--size_y", str(cfg.box.size[1]),
        "--size_z", str(cfg.box.size[2]),
        "--exhaustiveness", str(cfg.exhaustiveness),
        "--num_modes", str(cfg.num_modes),
    ]
    if cfg.seed is not None:
        cmd += ["--seed", str(cfg.seed)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, check=True,
        )
    except Exception:
        return None
    finally:
        if os.path.exists(out_pdbqt):
            try:
                os.remove(out_pdbqt)
            except OSError:
                pass
    for line in result.stdout.split("\n"):
        ls = line.strip()
        if ls.startswith("1 ") or ls.startswith("   1 "):
            parts = ls.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return None


def dock_many(
    smiles: Iterable[str],
    cfg: DockingConfig,
) -> Dict[str, Optional[float]]:
    """Dock a batch of SMILES sequentially. Returns ``{smi: score}``."""
    scores: Dict[str, Optional[float]] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, smi in enumerate(smiles):
            ligand = os.path.join(tmpdir, f"lig_{i}.pdbqt")
            score = None
            if smiles_to_pdbqt(smi, ligand, cfg):
                score = dock_one(ligand, cfg)
                if os.path.exists(ligand):
                    try:
                        os.remove(ligand)
                    except OSError:
                        pass
            scores[smi] = score
    return scores
