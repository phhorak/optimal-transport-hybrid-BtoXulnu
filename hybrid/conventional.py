import numpy as np


def compute_conventional_weights(df_incl, df_excl, binning, weight_col="total_weight"):
    """
    Compute bin-by-bin hybrid weights as (H_inc - H_exc) / H_inc in N-dimensional
    phase space. This is the standard method used in existing analyses.

    Negative weights (where exclusive locally exceeds inclusive) are clipped to zero.
    The result is an N-dimensional array aligned with the provided binning.

    Parameters
    ----------
    df_incl : DataFrame
        Inclusive simulation sample.
    df_excl : DataFrame
        Combined exclusive (resonant) simulation sample.
    binning : dict
        Ordered mapping of column name -> bin edges. E.g.:
        {"genq2": [...], "gen_lep_E_B": [...], "genMx": [...]}
    weight_col : str
        Column to use as event weight (same column used for both samples).

    Returns
    -------
    weights_arr : ndarray
        N-dimensional weight array with the same axis ordering as binning.
    """
    variables = list(binning.keys())
    bins = list(binning.values())

    H_exc, _ = np.histogramdd(
        df_excl[variables].values,
        bins=bins,
        weights=df_excl[weight_col].values,
    )
    H_inc, _ = np.histogramdd(
        df_incl[variables].values,
        bins=bins,
        weights=df_incl[weight_col].values,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        weights_arr = np.where(H_inc > 0, (H_inc - H_exc) / H_inc, 0.0)

    weights_arr = np.clip(weights_arr, 0.0, 10.0)
    return weights_arr


def apply_bin_weights(df, binning, weights_arr):
    """
    Look up per-event weights from an N-dimensional weight array.

    Events outside the binning range receive weight 1.

    Parameters
    ----------
    df : DataFrame
        Events to reweight. Must contain all columns in binning.
    binning : dict
        Same binning dict passed to compute_conventional_weights.
    weights_arr : ndarray
        N-dimensional weight array returned by compute_conventional_weights.

    Returns
    -------
    ndarray, shape (n_events,)
        Per-event weights.
    """
    variables = list(binning.keys())
    bins = list(binning.values())

    indices = np.stack(
        [np.digitize(df[var].values, b) - 1 for var, b in zip(variables, bins)],
        axis=1,
    )  # shape (n_events, n_dims)

    shape = np.array(weights_arr.shape)
    in_bounds = np.all((indices >= 0) & (indices < shape), axis=1)

    result = np.ones(len(df), dtype=float)
    if in_bounds.any():
        result[in_bounds] = weights_arr[tuple(indices[in_bounds].T)]
    return result
