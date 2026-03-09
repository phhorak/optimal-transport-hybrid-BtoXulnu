#!/usr/bin/env python3
"""
compute_weights.py — compute OT and conventional hybrid weights for B → Xᵤℓν.

Usage:
    python compute_weights.py --config config.yaml [--plot]

Reads inclusive and resonant simulation samples, computes normalization weights,
then runs both the optimal transport hybrid and the conventional bin-by-bin hybrid.
Per-event weight columns are appended to the inclusive DataFrame and saved as a
parquet file. Pass --plot to also produce kinematic distribution and moment plots.
"""

import argparse
import os

import numpy as np
import pandas as pd
import uproot
import yaml

from hybrid import (
    build_phase_space_grid,
    solve_ot,
    extract_reweights,
    apply_ot_weights,
    compute_normalization,
    compute_conventional_weights,
    apply_bin_weights,
    compute_all_moments,
    plot_moment_comparison,
    print_moment_errors,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataframe(path, columns=None):
    """Read a ROOT file (any tree named 'tree' or 'events') or parquet file."""
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path, columns=columns)
    # ROOT: try common tree names
    with uproot.open(path) as f:
        tree_name = next(
            (k.split(";")[0] for k in f.keys() if not k.startswith("/")), None
        )
        if tree_name is None:
            raise ValueError(f"No tree found in {path}")
        return f[tree_name].arrays(columns, library="pd")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Plotting (only imported / used when --plot is passed)
# ---------------------------------------------------------------------------

def plot_distributions(df_incl, df_excl, mode, incl_model, plots_dir):
    """Stack plot of kinematic variables with hybrid / DFN ratio panel."""
    from plothist import get_color_palette
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(plots_dir, exist_ok=True)

    variables = [
        ("genMx",       np.linspace(0, 3.51, 50),  r"$M_X$ [GeV]"),
        ("gen_lep_E_B", np.linspace(0, 2.65, 50),  r"$E_\ell^B$ [GeV]"),
        ("genq2",       np.linspace(0, 25.01, 50), r"$q^2$ [GeV$^2$]"),
        ("genCosThetaL",np.linspace(-1, 1, 50),    r"$\cos\theta_\ell$"),
    ]

    if mode == "charged":
        masks = {
            r"$\pi^0$":  df_excl["X_gen_PDG"].abs() == 111,
            r"$\eta$":   df_excl["X_gen_PDG"].abs() == 221,
            r"$\rho^0$": df_excl["X_gen_PDG"].abs() == 113,
            r"$\omega$": df_excl["X_gen_PDG"].abs() == 223,
            r"$\eta'$":  df_excl["X_gen_PDG"].abs() == 331,
        }
    else:
        masks = {
            r"$\pi^\pm$": df_excl["X_gen_PDG"].abs() == 211,
            r"$\rho^\pm$":df_excl["X_gen_PDG"].abs() == 213,
        }

    palette = get_color_palette("cubehelix", len(masks) + 1)[::-1]
    def rgba(k):
        r, g, b = palette[k][:3]
        return (r, g, b, 0.85)

    all_labels = ["Incl leftover"] + list(masks.keys())
    all_colors = [rgba(0)] + [rgba(j + 1) for j in range(len(masks))]

    ncols = 2
    nrows = (len(variables) + 1) // ncols
    fig = plt.figure(figsize=(14, 6 * nrows))
    gs  = GridSpec(2 * nrows, ncols, height_ratios=[4, 1] * nrows,
                   hspace=0.12, wspace=0.28, figure=fig)

    for i, (var, bins, xlabel) in enumerate(variables):
        ax  = fig.add_subplot(gs[2 * (i // ncols), i % ncols])
        axr = fig.add_subplot(gs[2 * (i // ncols) + 1, i % ncols], sharex=ax)

        h_dfn  = np.histogram(df_incl[var], bins=bins,
                               weights=df_incl["total_weight"])[0]
        h_incl = np.histogram(df_incl[var], bins=bins,
                               weights=df_incl["total_weight"] * df_incl["ot_hybrid_weight"])[0]
        h_excl_list = [
            np.histogram(df_excl[var][m], bins=bins,
                         weights=df_excl["total_weight"][m])[0]
            for m in masks.values()
        ]

        bottom = np.zeros_like(h_incl, dtype=float)
        for hdat, lab, col in zip([h_incl] + h_excl_list, all_labels, all_colors):
            ax.bar(bins[:-1], hdat, width=np.diff(bins), bottom=bottom,
                   align="edge", color=col, edgecolor="none", label=lab)
            bottom += hdat

        mode_label = r"$B^\pm$" if mode == "charged" else r"$B^0$"
        ax.step(bins[:-1], bottom, where="post", color="k", lw=1.2, label="Hybrid")
        ax.step(bins[:-1], h_dfn,  where="post", color="r", lw=1.1, label=incl_model)
        if i == 0:
            ax.legend(fontsize=14)
        ax.text(0.05, 0.95, mode_label, transform=ax.transAxes,
                va="top", ha="left", fontsize=16)
        ax.set_ylabel("Events")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False)

        ratio = np.divide(bottom, h_dfn, out=np.ones_like(bottom), where=h_dfn > 0)
        axr.plot(0.5 * (bins[:-1] + bins[1:]), ratio, "k.", markersize=4)
        axr.axhline(1.0, color="r", linestyle="--", lw=1)
        axr.set_ylim(0.85, 1.15)
        axr.set_ylabel("Hybrid/DFN")
        axr.set_xlabel(xlabel)
        axr.grid(alpha=0.3)

    plt.savefig(os.path.join(plots_dir, f"distributions_{mode}.pdf"),
                bbox_inches="tight", facecolor="white")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML configuration file (default: config.yaml)")
    parser.add_argument("--plot", action="store_true",
                        help="Produce kinematic and moment comparison plots")
    args = parser.parse_args()

    cfg = load_config(args.config)

    mode        = cfg["mode"]
    incl_model  = cfg.get("inclusive_model", "inclusive")
    in_cfg      = cfg["input"]
    ot_cfg      = cfg["optimal_transport"]
    conv_cfg    = cfg["conventional_hybrid"]
    out_cfg     = cfg["output"]

    bfs         = {int(k): v for k, v in cfg["branching_fractions"][mode].items()}
    incl_bf     = cfg["inclusive_branching_fraction"]

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading inclusive sample ...")
    df_incl = load_dataframe(in_cfg["inclusive"])

    print("Loading resonant sample ...")
    df_excl = load_dataframe(in_cfg["resonant"])

    pplus_col  = in_cfg.get("pplus_col",  "genPplus")
    pminus_col = in_cfg.get("pminus_col", "genPminus")
    pdg_col    = in_cfg.get("pdg_col",    "X_gen_PDG")
    ff_col     = in_cfg.get("ff_weight_col", "FF_weight")

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    print("Computing normalization weights ...")
    compute_normalization(
        df_incl, df_excl,
        branching_fractions=bfs,
        inclusive_bf=incl_bf,
        pdg_col=pdg_col,
        ff_weight_col=ff_col,
    )

    # ------------------------------------------------------------------
    # OT hybrid
    # ------------------------------------------------------------------
    print("Building phase-space grid ...")
    Pinc, Pres = build_phase_space_grid(
        df_incl, df_excl,
        bin_width    = ot_cfg["bin_width"],
        pminus_range = tuple(ot_cfg["pminus_range"]),
        pplus_range  = tuple(ot_cfg["pplus_range"]),
        pplus_col    = pplus_col,
        pminus_col   = pminus_col,
    )

    print("Solving optimal transport problem ...")
    G, src_coords, src_shape = solve_ot(
        Pinc, Pres,
        bin_width  = ot_cfg["bin_width"],
        window     = ot_cfg.get("window"),
        lambda_reg = ot_cfg.get("lambda_reg", 0),
        verbose    = True,
    )

    reweight_map = extract_reweights(G, src_coords, src_shape, Pinc)
    df_incl["ot_hybrid_weight"] = apply_ot_weights(
        df_incl, reweight_map, ot_cfg["bin_width"],
        pplus_col=pplus_col, pminus_col=pminus_col,
    )

    # ------------------------------------------------------------------
    # Conventional hybrid
    # ------------------------------------------------------------------
    print("Computing conventional bin-by-bin weights ...")
    binning = {k: np.array(v) for k, v in conv_cfg["binning"].items()}
    conv_weights_arr = compute_conventional_weights(
        df_incl, df_excl, binning, weight_col="total_weight",
    )
    df_incl["conventional_hybrid_weight"] = apply_bin_weights(
        df_incl, binning, conv_weights_arr,
    )

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_cfg["weights_file"]) or ".", exist_ok=True)
    out_cols = ["ot_hybrid_weight", "conventional_hybrid_weight", "total_weight"]
    out_cols = [c for c in out_cols if c in df_incl.columns]
    df_incl[out_cols].to_parquet(out_cfg["weights_file"])
    print(f"Saved weights to {out_cfg['weights_file']}")

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if args.plot:
        plots_dir = out_cfg.get("plots_dir", "plots/")
        os.makedirs(plots_dir, exist_ok=True)

        print("Producing distribution plots ...")
        plot_distributions(df_incl, df_excl, mode, incl_model, plots_dir)

        print("Computing moments ...")
        varlist = ["genMxSq", "genq2", "gen_lep_E_B"]
        # genMxSq may not exist yet
        for df in (df_incl, df_excl):
            if "genMxSq" not in df.columns:
                df["genMxSq"] = df["genMx"] ** 2
        df_excl["_null"] = 0.0

        kwargs = dict(df_incl=df_incl, df_excl=df_excl, varlist=varlist, n=4)

        mom_ref     = compute_all_moments(**kwargs, incl_weight_col="total_weight",
                                          excl_weight_col="_null", central=False)
        mom_ref_cen = compute_all_moments(**kwargs, incl_weight_col="total_weight",
                                          excl_weight_col="_null", central=True)

        mom_ot      = compute_all_moments(**kwargs, incl_weight_col="ot_hybrid_weight",
                                          excl_weight_col="total_weight", central=False)
        mom_ot_cen  = compute_all_moments(**kwargs, incl_weight_col="ot_hybrid_weight",
                                          excl_weight_col="total_weight", central=True)

        # conventional: reuse the existing total_weight × conventional_hybrid_weight product
        df_incl["_conv_wt"] = df_incl["total_weight"] * df_incl["conventional_hybrid_weight"]
        mom_conv    = compute_all_moments(**kwargs, incl_weight_col="_conv_wt",
                                          excl_weight_col="total_weight", central=False)
        mom_conv_cen= compute_all_moments(**kwargs, incl_weight_col="_conv_wt",
                                          excl_weight_col="total_weight", central=True)

        print_moment_errors(mom_ot, mom_conv, mom_ref,
                            mom_ot_cen, mom_conv_cen, mom_ref_cen, varlist)

        plot_moment_comparison(
            mom_ot, mom_conv, mom_ref,
            mom_ot_cen, mom_conv_cen, mom_ref_cen,
            varlist, mode,
            savefig=os.path.join(plots_dir, f"moments_{mode}.pdf"),
        )


if __name__ == "__main__":
    main()
