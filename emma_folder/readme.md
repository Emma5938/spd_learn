# Project Structure

This repository contains the implementation of an SPD neural network model with custom spectral and element-wise activation functions.

---
## ⚠️ Unmaintained Repository

This fork is no longer active and will not be updated anymore.

Please fork and use the official repository at the following address:  
https://github.com/spdlearn/spd_learn

The files required for this project are available in this repository **emma_folder** directories.

```text
emma_folder
├── activation/
│   ├── spectral.py
│   ├── utils_spectral.py
│   ├── elementwise.py
│   └── utils_elementwise.py
├── models_/
│   └── model_SPD.py
├── scripts/
│   └── script.py
├── tests/
│   ├── test_stat.py
│   ├── test_stat_ju.jl
│   ├── test_expP.py
│   ├── test_coshP.py
│   └── test_expT.py
└── utils/
    └── monitoring.py
```
## Activation Functions (`activation/`)

### `spectral.py`

Implements the spectral activation functions:

- `PowerEig`
- `TanhEig`
- `SpAEig`

### `utils_spectral.py`

Contains the custom backward implementations used by the spectral activation functions.

### `elementwise.py`

Implements element-wise activation functions:

- `coshP`
- `sinhP`
- `expP`
- `expT`
- `polynomialActivation`

### `utils_elementwise.py`

Contains the custom backward implementations used by the element-wise activation functions.

---

## Models (`models_/`)

### `model_SPD.py`

Implementation of the SPD neural network model.

---

## Scripts (`scripts/`)

### `script.py`

Main pipeline used to train and evaluate the model on a selected dataset.

---

## Tests (`tests/`)

### `test_stat.py`

Runs the model using 5 different random seeds and collects performance metrics for statistical analysis.

### `test_stat_ju.jl`

Julia script used to perform statistical analysis:

- **Omnibus statistical tests**: `AnovaTestRM`
- **Post-hoc statistical tests**: `studentMcTestRM`

### `test_expP.py`

Test for the backward implementation of the parametric exponential activation `expP`.

### `test_coshP.py`

Test for the backward implementation of the parametric hyperbolic cosine activation `coshP`.

### `test_expT.py`

Test for the backward implementation of the truncated exponential activation `expT`.

---

## Utils (`utils/`)

### `monitoring.py`

Functions allowing the monitoring of the activation parameters during training.