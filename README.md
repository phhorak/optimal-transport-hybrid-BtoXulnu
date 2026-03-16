# Optimal transport hybrid model for B → Xᵤℓν

This repository accompanies the paper

> *Mapping quark-level kinematics to hadrons in a new hybrid model of semileptonic B meson decays*
> P. Horak, R. Kowalewski, T. Martinov (2025)

and provides an implementation of the optimal transport (OT) hybrid method for constructing simulation models of inclusive semileptonic B → Xᵤℓν decays.

## Background

Precision measurements of |V_ub| from inclusive B → Xᵤℓν decays rely on combining two types of simulation: an inclusive prediction from the Heavy Quark Expansion (HQE), and separate exclusive simulations of the low-mass resonances (π, ρ, ω, ...) whose branching fractions have been measured independently. Merging them into a single "hybrid" model has traditionally been done bin-by-bin, which introduces discontinuities in kinematic spectra and can produce negative weights.

The OT hybrid replaces the bin-by-bin subtraction with an optimal transport problem in the 2D light-cone variable space (P+, P-). Rather than adjusting each bin independently, it finds a global redistribution of probability mass from the inclusive prediction to the resonant component that minimizes the aggregate displacement in phase space. The result is smooth kinematic distributions with better preservation of HQE spectral moments.

## Installation

```bash
pip install -r requirements.txt
```

The main dependency for the OT solver is [POT](https://pythonot.github.io/) (Python Optimal Transport).

## Quick start

1. Edit `config.yaml` — set the paths to your simulation files and adjust the branching fractions if needed.
2. Run the script:
   ```bash
   python compute_weights.py --config config.yaml
   ```
3. This produces `hybrid_weights.csv` — a table of OT hybrid weights indexed by (P+, P-) bin edges. Pass `--plot` to also generate kinematic distribution and moment comparison plots.

See `example_notebook.ipynb` for a step-by-step walkthrough.

## Reproducing the paper results

The simulation samples used in the paper are archived on Zenodo:

> **Simulation samples for "Mapping quark-level kinematics to hadrons in a new hybrid model of semileptonic B meson decays"**
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19055004.svg)](https://doi.org/10.5281/zenodo.19055004)

The EvtGen decay files used to generate all samples in the paper (B⁺ and B⁰, DFN and BLNP) are included in the Zenodo record alongside the simulation samples.

Download the parquet files, set the paths in `config.yaml`:

```yaml
input:
  inclusive: /path/to/BplusToXuenu_central.pq
  resonant:  /path/to/BplusToExclenu.pq
```

then either run the CLI:

```bash
python compute_weights.py --config config.yaml --plot
```

or open `example_notebook.ipynb`, set `PATH_INCLUSIVE` and `PATH_RESONANT` in the configuration cell, and run all cells in order. The moment comparison table and plots will match the paper exactly.

## Input format

The script expects two simulation samples — an inclusive HQE sample and a combined resonant sample — in ROOT or parquet format. The following columns must be present (names are configurable in `config.yaml`):

| Column | Description |
|--------|-------------|
| `genPplus` | P+ = E_X + \|p_X\| in the B rest frame [GeV] |
| `genPminus` | P- = E_X − \|p_X\| in the B rest frame [GeV] |
| `X_gen_PDG` | PDG code of the hadronic system |
| `input_weight` | Optional per-event weight (e.g. form-factor corrections). Column name is configurable; omit entirely if not needed. |
| `genMx` | Hadronic system invariant mass [GeV] |
| `genq2` | Lepton-neutrino invariant mass squared [GeV²] |
| `gen_lep_E_B` | Lepton energy in B rest frame [GeV] |
| `genCosThetaL` | Lepton helicity angle cos θ_ℓ |

## Output

The primary output is `hybrid_weights.csv` — a weight table with one row per (P+, P-) bin:

| Column | Description |
|--------|-------------|
| `pplus_low` | Lower edge of the P+ bin [GeV] |
| `pplus_high` | Upper edge of the P+ bin [GeV] |
| `pminus_low` | Lower edge of the P- bin [GeV] |
| `pminus_high` | Upper edge of the P- bin [GeV] |
| `ot_hybrid_weight` | Fraction of inclusive events in this bin that remain after resonances are placed |

To apply the weights to your events, find each event's (P+, P-) bin and multiply its existing weight by `ot_hybrid_weight`. Values close to 1 mean the bin is unaffected by resonances; values close to 0 mean it is fully covered. The resonant events keep their own weights unchanged.

## Config reference

Key parameters in `config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `mode` | `charged` (B+) or `neutral` (B0) |
| `optimal_transport.bin_width` | Grid spacing in (P+, P-) [GeV]. 0.08 GeV is the default. |
| `optimal_transport.window` | Spatial window limiting transport connections. `null` = no restriction. |
| `optimal_transport.lambda_reg` | Entropy regularization (0 = exact EMD, recommended). |
| `branching_fractions` | PDG code → absolute branching fraction for each exclusive mode. |
| `inclusive_branching_fraction` | B(B → Xᵤℓν) for the full phase space. |
| `conventional_hybrid.binning` | Bin edges for the bin-by-bin method (for comparison). |

## Using the library directly

The `hybrid` package can also be imported in your own scripts or notebooks:

```python
from hybrid import (
    build_phase_space_grid,
    solve_ot,
    extract_reweights,
    apply_ot_weights,
    compute_normalization,
)

# Build 2D histograms
Pinc, Pres = build_phase_space_grid(df_incl, df_excl, bin_width=0.08,
                                     pminus_range=(0, 5.31), pplus_range=(0, 3.01))

# Solve the transport problem
G, src_coords, src_shape = solve_ot(Pinc, Pres, bin_width=0.08, verbose=True)

# Extract per-bin reweights and apply to events
reweight_map = extract_reweights(G, src_coords, src_shape, Pinc)
df_incl["ot_hybrid_weight"] = apply_ot_weights(df_incl, reweight_map, bin_width=0.08)
```

## Citation

If you use this code in your work, please cite:

```
@article{horak2025hybrid,
  title   = {Mapping quark-level kinematics to hadrons in a new hybrid model
             of semileptonic $B$ meson decays},
  author  = {Horak, Philipp and Kowalewski, Robert and Martinov, Tommy},
  year    = {2025},
}
```
