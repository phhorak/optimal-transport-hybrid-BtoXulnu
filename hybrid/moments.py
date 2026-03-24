import numpy as np
import matplotlib.pyplot as plt


def compute_moment(data, weights, n, central=True):
    """
    Compute the n-th weighted moment of a distribution.

    Parameters
    ----------
    data : array-like
    weights : array-like or None
    n : int
        Moment order.
    central : bool
        If True, compute central moment (subtract the mean first).
        If False, compute raw moment.
    """
    data    = np.asarray(data,    dtype=float)
    weights = np.ones_like(data) if weights is None else np.asarray(weights, dtype=float)

    norm = weights.sum()
    if norm == 0:
        return np.nan

    if central and n > 1:
        mean = np.dot(weights, data) / norm
        return np.dot(weights, (data - mean) ** n) / norm
    else:
        return np.dot(weights, data ** n) / norm


def compute_all_moments(df_incl, df_excl, varlist,
                        incl_weight_col, excl_weight_col,
                        n=4, central=False):
    """
    Compute raw or central moments up to order n for a list of variables,
    combining inclusive and exclusive samples with their respective weights.

    Returns a dict: variable -> 1D array of n moment values.
    """
    w_incl = df_incl[incl_weight_col].values
    w_excl = df_excl[excl_weight_col].values

    moments = {}
    for var in varlist:
        data    = np.concatenate([df_incl[var].values, df_excl[var].values])
        weights = np.concatenate([w_incl, w_excl])
        moments[var] = np.array([
            compute_moment(data, weights, k, central=central)
            for k in range(1, n + 1)
        ])
    return moments


# Variable display labels
_VAR_LABELS = {
    "genMxSq":     r"$M_X^2$",
    "genMx":       r"$M_X$",
    "genq2":       r"$q^2$",
    "gen_lep_E_B": r"$E_\ell^B$",
}


def plot_moment_comparison(mom_ot, mom_conv, mom_ref,
                           mom_ot_central, mom_conv_central, mom_ref_central,
                           varlist, mode, savefig=None):
    """
    Plot ratios of hybrid / reference moments for both OT and conventional methods,
    showing raw (top) and central (bottom) moments side by side.

    Parameters
    ----------
    mom_ot, mom_conv, mom_ref : dict
        Raw moment dicts (variable -> array of n values).
    mom_ot_central, mom_conv_central, mom_ref_central : dict
        Central moment dicts.
    varlist : list of str
        Variables to plot. Each gets one column.
    mode : str
        "charged" or "neutral", used for axis labels.
    savefig : str or None
        File path to save the figure.
    """
    color_ot   = "#10B981"
    color_conv = "#EAB308"
    mode_label = r"$B^\pm$" if mode == "charged" else r"$B^0$"
    orders = np.array([1, 2, 3, 4])

    ncols = len(varlist)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for i, var in enumerate(varlist):
        label = _VAR_LABELS.get(var, var)

        for row, (mom_ot_d, mom_conv_d, mom_ref_d, title_suffix) in enumerate([
            (mom_ot, mom_conv, mom_ref, "raw moments"),
            (mom_ot_central, mom_conv_central, mom_ref_central, "central moments"),
        ]):
            ax = axes[row, i]

            ratio_ot   = mom_ot_d[var]   / mom_ref_d[var]
            ratio_conv = mom_conv_d[var] / mom_ref_d[var]

            ax.plot(orders, ratio_ot,   "o-",  color=color_ot,   label="Optimal transport")
            ax.plot(orders, ratio_conv, "s--", color=color_conv, label="Bin-by-bin")
            ax.axhline(1.0, color="k", linestyle="--", alpha=0.4)

            ax.set_xlim(0.8, 4.2)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xlabel(r"$n^\mathrm{th}$ moment", fontsize=14)
            ax.set_ylabel("Ratio (hybrid / reference)", fontsize=14)
            ax.set_title(f"{label} {title_suffix}", fontsize=16)
            ax.text(0.05, 0.95, mode_label, transform=ax.transAxes,
                    va="top", ha="left", fontsize=16)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=11)

        # Adjust y-limits per variable/mode to match paper
        if mode == "neutral":
            axes[0, i].set_ylim(0.7, 1.3)
            axes[1, i].set_ylim(0.6, 1.4)
        else:
            axes[0, i].set_ylim(0.9, 1.1)
            axes[1, i].set_ylim(0.8, 1.2)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, bbox_inches="tight", facecolor="white")
    return fig


def print_moment_errors(mom_ot, mom_conv, mom_ref,
                        mom_ot_central, mom_conv_central, mom_ref_central,
                        varlist):
    """
    Print the average absolute relative error (AARE) over the first four moments
    for each variable, for both OT and conventional hybrids.
    """
    def aare(ref, test):
        ref  = np.asarray(ref[:4])
        test = np.asarray(test[:4])
        return np.mean(np.abs((test - ref) / ref))

    print("Average absolute relative error (first 4 moments)\n")
    for kind, (ot_d, conv_d, ref_d) in [
        ("raw",     (mom_ot, mom_conv, mom_ref)),
        ("central", (mom_ot_central, mom_conv_central, mom_ref_central)),
    ]:
        print(f"{kind.upper()} moments:")
        ot_vals, conv_vals = [], []
        for var in varlist:
            e_ot   = aare(ref_d[var], ot_d[var])
            e_conv = aare(ref_d[var], conv_d[var])
            ot_vals.append(e_ot)
            conv_vals.append(e_conv)
            print(f"  {var:20s}  OT: {e_ot:.2%}   conventional: {e_conv:.2%}")
        print(f"  {'average':20s}  OT: {np.mean(ot_vals):.2%}   conventional: {np.mean(conv_vals):.2%}")
        print()
