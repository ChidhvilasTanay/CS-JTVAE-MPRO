# Chemotype-Symmetric Junction-Tree VAE for SARS-CoV-2 M<sup>pro</sup> Inhibitor Design

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21364056.svg)](https://doi.org/10.5281/zenodo.21364056)

Code for the paper *Chemotype-Symmetric Junction-Tree VAE for SARS-CoV-2 M<sup>pro</sup> Inhibitor Design*. The framework generates close analogs of known SARS-CoV-2 main protease (M<sup>pro</sup>) inhibitors that are predicted safer while preserving the covalent warhead and target affinity.

## Overview

Each molecule is represented as a junction tree of chemical fragments and encoded into a continuous latent space by a Transformer junction-tree VAE. A latent-conditioned GINEConv attachment scorer reassembles the fused-ring chemistry at decoding time. A constrained six-objective search over the latent space, optimised with NSGA-III, trades off drug-likeness, synthetic accessibility, warhead retention, predicted potency (from a deep-ensemble surrogate), lipophilicity, and warhead reversibility, with an anchor-relative uncertainty cap that rejects candidates far from familiar chemistry. Surviving analogs are verified with ADMET-AI and AutoDock Vina.

## Results

Trained on 20,057 M<sup>pro</sup> inhibitors from the COVID Moonshot dataset and evaluated using the twenty most potent known inhibitors as generation anchors.

- **Faithful reconstruction.** The latent-conditioned assembler reconstructs 19 of the 20 most potent inhibitors faithfully, including the edge-fused ring systems where SMILES-based models most often produce invalid structures.
- **Warhead-preserving generation.** Across all twenty anchors the framework produces a de-duplicated library of 1,013 analogs and keeps the source covalent warhead in 85% of the parents that carry one.
- **Safer without erasing the mechanism.** The final Pareto front of 138 analogs is predicted as safe as or safer than each analog's own source inhibitor on four of eight ADMET endpoints, at parity on predicted mutagenicity, with the warhead retained in 94% of the front and docked affinity within AutoDock Vina's scoring error of the parents.
- **Mutagenicity tracks the warhead class, not covalency.** The reversible nitrile of the marketed inhibitors scores as clean as non-covalent chemistry, while the irreversible chloroacetamide and Michael warheads carry the predicted liability.
- **Lead candidate.** A dual-warhead analog of nirmatrelvir is predicted safer than the drug itself on seven of eight ADMET endpoints while keeping both warheads.

See the paper (linked above) for the full tables, the per-warhead analysis, and the multi-objective optimiser benchmark.

## Repository layout

```
src/charaka/          the package
  vocab.py            junction-tree decomposition and chemotype-symmetric cluster vocabulary
  tokenize.py         tree <-> token-sequence serialisation
  model.py            Transformer JT-VAE (encoder and decoder)
  scorer.py           latent-conditioned GINEConv attachment scorer
  assemble.py         tree-to-molecule assembly
  bioactivity.py      deep-ensemble potency surrogate
  objectives.py       objective functions and the composite ranking score
  nsga.py             latent-space search problem, constraints, anchor-relative uncertainty cap
  data.py, dataset.py dataset loading
  training.py         training loops
  admet.py            ADMET-AI wrapper
  docking.py          AutoDock Vina wrapper
scripts/              one script per pipeline stage
configs/              per-stage configuration
run.py                single-entry command-line runner
```

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10 or newer. A CUDA GPU is needed for the training and generation stages; the ADMET, figure, and statistics stages run on CPU. The docking stage additionally uses AutoDock Vina and Open Babel.

## Data and trained weights

The trained weights (JT-VAE, attachment scorer, and bioactivity ensemble) are committed in this repository under `checkpoints_full_v3_aug/`, so a fresh clone is ready to run. They are also archived on Zenodo with a citable DOI:

https://doi.org/10.5281/zenodo.21364056

The processed and augmented dataset is not committed here; download it from the same Zenodo record and place it at the repository root as `data/`. The weights already sit at the root as:

```
checkpoints_full_v3_aug/
  vocab.txt
  jtvae_best.pth            JT-VAE encoder and decoder
  encoder_joint.pth         jointly-trained encoder
  scorer_joint.pth          GINEConv attachment scorer
  bioactivity/member_0..4.pth
  bioactivity_summary.yaml
```

## Reproducing the results

List the stages and pipelines:

```
python run.py list
```

With the weights and data in place, reproduce the analogs, verification, figures, and reported numbers:

```
python run.py all          # generate -> admet -> docking -> figures -> stats
```

To rebuild everything from the raw dataset (GPU, several hours):

```
python run.py train-all
```

Individual stages can be run directly, for example

```
python run.py generate --config configs/nsga_v3_6obj.yaml
python run.py moo-catalogue
```

Every stage writes a timestamped log under `logs/`.

## Citation

If you use this code, please cite the paper through the Zenodo record above. Machine-readable metadata is in `CITATION.cff`.

## License

See [`LICENSE`](LICENSE).
