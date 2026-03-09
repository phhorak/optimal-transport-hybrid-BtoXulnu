# Optimal transport hybrid model for B → Xᵤℓν

This repository accompanies the paper

> *Mapping quark-level kinematics to hadrons in a new hybrid model of semileptonic B meson decays*
> P. Horak, R. Kowalewski, T. Martinov (2025)

and provides a clean implementation of the optimal transport (OT) hybrid method for constructing simulation models of inclusive semileptonic B → Xᵤℓν decays.

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
3. This produces `hybrid_weights.parquet` containing per-event `ot_hybrid_weight` and `conventional_hybrid_weight` columns for your inclusive sample. Pass `--plot` to also generate kinematic distribution and moment comparison plots.

See `example_notebook.ipynb` for a step-by-step walkthrough.

## Input format

The script expects two simulation samples — an inclusive HQE sample and a combined resonant sample — in ROOT or parquet format. The following columns must be present (names are configurable in `config.yaml`):

| Column | Description |
|--------|-------------|
| `genPplus` | P+ = E_X + \|p_X\| in the B rest frame [GeV] |
| `genPminus` | P- = E_X − \|p_X\| in the B rest frame [GeV] |
| `X_gen_PDG` | PDG code of the hadronic system |
| `FF_weight` | Form-factor weight (set to 1 if not applicable) |
| `genMx` | Hadronic system invariant mass [GeV] |
| `genq2` | Lepton-neutrino invariant mass squared [GeV²] |
| `gen_lep_E_B` | Lepton energy in B rest frame [GeV] |
| `genCosThetaL` | Lepton helicity angle cos θ_ℓ |

## Output

The output parquet file contains one row per inclusive event with these columns added:

| Column | Description |
|--------|-------------|
| `ot_hybrid_weight` | OT hybrid reweight — multiply by `total_weight` to get the final hybrid weight |
| `conventional_hybrid_weight` | Bin-by-bin hybrid reweight (for comparison) |
| `total_weight` | Normalization weight = `branching_fraction_weight × FF_weight` |

The final event weight for the hybrid sample is `total_weight × ot_hybrid_weight`. The resonant events keep their `total_weight` unchanged.

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
