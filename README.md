# Mimic

[![PyPI version](https://img.shields.io/pypi/v/poseigen_mimic)](https://pypi.org/project/poseigen_mimic/)
[![Python versions](https://img.shields.io/pypi/pyversions/poseigen_mimic)](https://pypi.org/project/poseigen_mimic/)
[![License](https://img.shields.io/pypi/l/poseigen_mimic)](https://pypi.org/project/poseigen_mimic/)
[![bioRxiv preprint](https://img.shields.io/badge/preprint-bioRxiv-B31B1B)](https://www.biorxiv.org/content/10.64898/2026.10.26.761901)

Mimic is a configurable neural-network architecture for genomic sequence modeling. It separates a model into *k*-mer embedding, sequence embedding, and output mapping, making it possible to compare sequence-embedding strategies within a shared architecture.

Part of the [Poseigen ecosystem](https://github.com/Husam94/poseigen).

See the [Mimic preprint](https://www.biorxiv.org/content/10.64898/2026.10.26.761901). The study evaluates recurrent and convolutional sequence-embedding strategies on protein-DNA interaction and enhancer-activity datasets. It reports that Mimic significantly outperformed BiChrom on protein-DNA interactions and DeepSTARR on enhancer activity in the evaluated datasets.

## Installation

Requires Python 3.12 or newer. Install from PyPI:

```bash
python -m pip install poseigen_mimic
```

## Package

- **`Mimic`** is a single-input architecture with three configurable stages: a convolutional *k*-mer embedding, a sequence-embedding component, and an output mapping. The sequence-embedding component uses Trident's `Prong_W` by default and can also use recurrent layers such as GRU or LSTM, convolutional layers, or compatible custom pooling modules.
- **`Dual_Mimic`** applies separate Mimic branches to two inputs. The branches can have different input dimensions and *k*-mer settings. With `late_comb=True`, their outputs are summed; with `late_comb=False`, their sequence representations are concatenated before a shared output mapping.

Mimic's neural-network components are built with [poseigen_trident](https://github.com/Husam94/poseigen_trident).

## Case studies

- **Protein-DNA interactions**
  - [CTCF case study](https://github.com/Husam94/CS-CTCF)
  - [YY1 case study](https://github.com/Husam94/CS-YY1)
- **Enhancer activity**
  - [Enhancer activity case study](https://github.com/Husam94/CS-EnhAct)

## Usage

For the default input dimensions, Mimic expects a batch shaped `(batch, 1, sequence_length, 4)`. This example uses a GRU for sequence embedding:

```python
import torch
import torch.nn as nn
from poseigen_mimic import Mimic

model = Mimic(dim_i=(1, 249, 4), PR_layer=nn.GRU)
sequence_batch = torch.randn(8, 1, 249, 4)
predictions = model(sequence_batch)
```
