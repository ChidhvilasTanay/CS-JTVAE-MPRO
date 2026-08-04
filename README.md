# Chemotype-Symmetric Junction-Tree VAE for SARS-CoV-2 M<sup>pro</sup> Inhibitor Design

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21364056.svg)](https://doi.org/10.5281/zenodo.21364056)

Code for the paper *Chemotype-Symmetric Junction-Tree VAE for SARS-CoV-2 M<sup>pro</sup> Inhibitor Design*. The framework generates close analogs of known SARS-CoV-2 main protease (M<sup>pro</sup>) inhibitors that are predicted safer while preserving the covalent warhead and target affinity.

## Overview

Each molecule is represented as a junction tree of chemical fragments and encoded into a continuous latent space by a Transformer junction-tree VAE. A latent-conditioned GINEConv attachment scorer reassembles the fused-ring chemistry at decoding time. A constrained six-objective search over the latent space, optimised with NSGA-III, trades off drug-likeness, synthetic accessibility, warhead retention, predicted potency (from a deep-ensemble surrogate), lipophilicity, and warhead reversibility, with an anchor-relative uncertainty cap that rejects candidates far from familiar chemistry. Surviving analogs are verified with ADMET-AI and AutoDock Vina.

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

The processed dataset and the trained weights (JT-VAE, attachment scorer, and bioactivity ensemble) are archived on Zenodo:

https://doi.org/10.5281/zenodo.21364056

Download and place them at the repository root as:

```
data/                       processed and augmented datasets, plus the analog result CSVs
checkpoints_full_v3_aug/
  vocab.txt
  jtvae_best.pth            JT-VAE encoder and decoder
  scorer.pth                GINEConv attachment scorer
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
