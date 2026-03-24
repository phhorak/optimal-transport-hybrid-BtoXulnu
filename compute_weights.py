#!/usr/bin/env python3
"""
compute_weights.py — compute OT and conventional hybrid weights for B → Xᵤℓν.

Usage:
    python compute_weights.py --config config.yaml [--plot]

Reads inclusive and resonant simulation samples, computes normalization weights,
then runs both the optimal transport hybrid and the conventional bin-by-bin hybrid.

The primary output is a CSV weight table with one row per (P+, P-) bin:

    pplus_low, pplus_high, pminus_low, pminus_high, ot_hybrid_weight

To apply the weights to your events, look up each event's (P+, P-) bin and
multiply its existing weight by the corresponding ot_hybrid_weight value.

Pass --plot to also produce kinematic distribution and moment comparison plots.
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

def _plot_ratio(axr, bottom, h_dfn, bins, i):
    """Ratio panel: grey shading on first plot, red arrows for out-of-range points."""
    ratio = np.divide(bottom, h_dfn, out=np.ones_like(bottom), where=h_dfn > 0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    if i == 0:
        axr.axvspan(bins[0], 1.25, color="grey", alpha=0.3, zorder=0)
        mask = bins[:-1] >= 1.25
        axr.plot(bin_centers[mask], ratio[mask], "k.", markersize=4)
    else:
        axr.plot(bin_centers, ratio, "k.", markersize=4)
        out_range = (ratio < 0.85) | (ratio > 1.15)
        for x, y in zip(bin_centers[out_range], ratio[out_range]):
            if y > 1.15:
                axr.annotate("", xy=(x, 1.15), xytext=(x, 1.05),
                             arrowprops=dict(arrowstyle="Simple,head_length=.3,head_width=.2,tail_width=.05",
                                             color="r", alpha=0.5, lw=1))
            else:
                axr.annotate("", xy=(x, 0.85), xytext=(x, 0.95),
                             arrowprops=dict(arrowstyle="Simple,head_length=.3,head_width=.2,tail_width=.05",
                                             color="r", alpha=0.5, lw=1))
    axr.axhline(1.0, color="r", linestyle="--", lw=1)
    axr.set_ylim(0.85, 1.15)
    axr.set_ylabel("Hybrid/DFN")
    axr.grid(alpha=0.3)


def _set_axis_spacing(axes_main, axes_ratio, ncols):
    """Custom tight positioning matching paper layout."""
    n_rows = (len(axes_main) + ncols - 1) // ncols
    vspace, hspace = 0.10, 0.07
    left_margin, right_margin = 0.06, 0.03
    top_margin, bottom_margin = 0.05, 0.05
    total_h = 1 - top_margin - bottom_margin
    main_height  = total_h * 0.65
    ratio_height = total_h * 0.15
    gap          = total_h * 0.05
    avail_w = 1 - left_margin - right_margin
    subplot_w = (avail_w - hspace * (ncols - 1)) / ncols
    row_h = main_height + gap + ratio_height
    for row in range(n_rows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= len(axes_main):
                break
            top_row = 1 - top_margin - row * (row_h + vspace)
            left_col = left_margin + col * (subplot_w + hspace)
            axes_main[idx].set_position([left_col, top_row - main_height, subplot_w, main_height])
            axes_ratio[idx].set_position([left_col, top_row - main_height - gap - ratio_height,
                                          subplot_w, ratio_height])


def plot_distributions(df_incl, df_excl, mode, incl_model, plots_dir):
    """Stack plot of kinematic variables with hybrid / DFN ratio panel."""
    from plothist import get_color_palette
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    variables = [
        ("genMx",        np.linspace(0, 3.50, 36), r"$M_X$ [GeV]"),
        ("gen_lep_E_B",  np.linspace(0, 2.65, 54), r"$E_\ell^B$ [GeV]"),
        ("genq2",        np.linspace(0, 25,   51), r"$q^2$ [GeV$^2$]"),
        ("genCosThetaL", np.linspace(-1, 1,   50), r"$\cos\theta_\ell$"),
    ]

    if mode == "charged":
        masks = {
            r"$\pi^0$":  df_excl["X_gen_PDG"] == 111,
            r"$\eta$":   df_excl["X_gen_PDG"] == 221,
            r"$\rho^0$": df_excl["X_gen_PDG"] == 113,
            r"$\omega$": df_excl["X_gen_PDG"] == 223,
            r"$\eta'$":  df_excl["X_gen_PDG"] == 331,
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

    ncols = 4
    fig = plt.figure(figsize=(23, 5))
    axes_main, axes_ratio = [], []

    for i, (var, bins, xlabel) in enumerate(variables):
        ax  = fig.add_subplot(len(variables) * 2, ncols, i + 1)
        axr = fig.add_subplot(len(variables) * 2, ncols, i + 1 + ncols, sharex=ax)
        axes_main.append(ax)
        axes_ratio.append(axr)

        h_dfn  = np.histogram(df_incl[var], bins=bins,
                               weights=df_incl["norm_weight"])[0].astype(float)
        h_incl = np.histogram(df_incl[var], bins=bins,
                               weights=df_incl["norm_weight"] * df_incl["ot_hybrid_weight"])[0].astype(float)
        h_excl_list = [
            np.histogram(df_excl[var][m], bins=bins,
                         weights=df_excl["norm_weight"][m])[0].astype(float)
            for m in masks.values()
        ]

        bottom = np.zeros(len(bins) - 1)
        for hdat, lab, col in zip([h_incl] + h_excl_list, all_labels, all_colors):
            ax.bar(bins[:-1], hdat, width=np.diff(bins), bottom=bottom,
                   align="edge", color=col, edgecolor="none", label=lab)
            bottom += hdat

        mode_label = r"$B^\pm$" if mode == "charged" else r"$B^0$"
        ax.stairs(bottom, bins, color="#A0A0A0", lw=1.7, label="Hybrid")
        ax.stairs(h_dfn,  bins, color="#000000", lw=1.7, label=incl_model,
                  linestyle=(0, (5, 1)))

        ax.text(0.05, 0.95, mode_label, transform=ax.transAxes,
                va="top", ha="left", fontsize=20)
        ax.set_ylabel("Events")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False)
        lo, hi = ax.get_ylim()
        if i == 0:
            ax.legend(fontsize=11, ncol=2)
            ax.set_ylim(lo, hi * 1.5)
        else:
            ax.set_ylim(lo, hi * 1.2)

        _plot_ratio(axr, bottom, h_dfn, bins, i)
        axr.set_xlabel(xlabel)

    _set_axis_spacing(axes_main, axes_ratio, ncols)

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

    pplus_col  = in_cfg.get("pplus_col",  "genPplus")
    pminus_col = in_cfg.get("pminus_col", "genPminus")
    pdg_col    = in_cfg.get("pdg_col",    "X_gen_PDG")
    input_weight_col = in_cfg.get("input_weight_col", "input_weight")

    # ------------------------------------------------------------------
    # Load data — only the columns actually needed
    # ------------------------------------------------------------------
    signal_col = in_cfg.get("signal_filter_col")

    needed_cols = list(dict.fromkeys(filter(None, [
        signal_col,
        pplus_col,
        pminus_col,
        pdg_col,
        *conv_cfg["binning"].keys(),
        *(["genCosThetaL"] if args.plot else []),
    ])))

    print("Loading inclusive sample ...")
    df_incl = load_dataframe(in_cfg["inclusive"], columns=needed_cols)

    print("Loading resonant sample ...")
    df_excl = load_dataframe(in_cfg["resonant"], columns=needed_cols)

    if signal_col:
        df_incl = df_incl[df_incl[signal_col] > 0].reset_index(drop=True)
        df_excl = df_excl[df_excl[signal_col] > 0].reset_index(drop=True)
        print(f"After signal filter ({signal_col} > 0): "
              f"{len(df_incl):,} inclusive, {len(df_excl):,} resonant")

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    print("Computing normalization weights ...")
    compute_normalization(
        df_incl, df_excl,
        branching_fractions=bfs,
        inclusive_bf=incl_bf,
        pdg_col=pdg_col,
        input_weight_col=input_weight_col,
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
        df_incl, df_excl, binning, weight_col="norm_weight",
    )
    df_incl["conventional_hybrid_weight"] = apply_bin_weights(
        df_incl, binning, conv_weights_arr,
    )

    # ------------------------------------------------------------------
    # Save output: 2D weight map as CSV (one row per bin)
    # ------------------------------------------------------------------
    bw = ot_cfg["bin_width"]
    pp_edges = np.arange(ot_cfg["pplus_range"][0],  ot_cfg["pplus_range"][1],  bw)
    pm_edges = np.arange(ot_cfg["pminus_range"][0], ot_cfg["pminus_range"][1], bw)

    rows = []
    for i, pp_lo in enumerate(pp_edges):
        for j, pm_lo in enumerate(pm_edges):
            if i < reweight_map.shape[0] and j < reweight_map.shape[1]:
                rows.append({
                    "pplus_low":        round(pp_lo, 6),
                    "pplus_high":       round(pp_lo + bw, 6),
                    "pminus_low":       round(pm_lo, 6),
                    "pminus_high":      round(pm_lo + bw, 6),
                    "ot_hybrid_weight": reweight_map[i, j],
                })

    weight_table = pd.DataFrame(rows)
    out_path = out_cfg["weights_file"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    weight_table.to_csv(out_path, index=False)
    print(f"Saved {len(weight_table)} bin weights to {out_path}")

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

        mom_ref     = compute_all_moments(**kwargs, incl_weight_col="norm_weight",
                                          excl_weight_col="_null", central=False)
        mom_ref_cen = compute_all_moments(**kwargs, incl_weight_col="norm_weight",
                                          excl_weight_col="_null", central=True)

        mom_ot      = compute_all_moments(**kwargs, incl_weight_col="ot_hybrid_weight",
                                          excl_weight_col="norm_weight", central=False)
        mom_ot_cen  = compute_all_moments(**kwargs, incl_weight_col="ot_hybrid_weight",
                                          excl_weight_col="norm_weight", central=True)

        # conventional: reuse the existing norm_weight × conventional_hybrid_weight product
        df_incl["_conv_wt"] = df_incl["norm_weight"] * df_incl["conventional_hybrid_weight"]
        mom_conv    = compute_all_moments(**kwargs, incl_weight_col="_conv_wt",
                                          excl_weight_col="norm_weight", central=False)
        mom_conv_cen= compute_all_moments(**kwargs, incl_weight_col="_conv_wt",
                                          excl_weight_col="norm_weight", central=True)

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
