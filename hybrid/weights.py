import numpy as np


def compute_normalization(df_incl, df_excl, branching_fractions, inclusive_bf,
                          pdg_col="X_gen_PDG", ff_weight_col="FF_weight"):
    """
    Compute per-event normalization weights so that the resonant and inclusive
    samples can be combined into a single hybrid model.

    The normalization follows the standard hybrid construction:

        excl_norm = (N_incl / N_excl) * (1 - BF_Xu / (sum_BF_excl + BF_Xu))

    This factor is applied uniformly to all exclusive events (independent of
    PDG code, since the exclusive sample is pre-mixed by EvtGen). Inclusive
    events get weight 1 from this factor. Both samples are further multiplied
    by their FF_weight column (set to 1 if form-factor corrections are absent).

    The resulting column "total_weight" is added in-place to both DataFrames.

    Parameters
    ----------
    df_incl : DataFrame
        Inclusive simulation sample.
    df_excl : DataFrame
        Combined exclusive (resonant) simulation sample.
    branching_fractions : dict
        PDG code (int) -> branching fraction (float, absolute). Only used to
        compute the total exclusive branching fraction for the normalization.
    inclusive_bf : float
        Inclusive B(B -> Xu l nu) branching fraction.
    pdg_col : str
        Column name for the hadronic system PDG code.
    ff_weight_col : str
        Column name for form-factor weights (1 if not applicable).
    """
    sum_excl_bf = sum(branching_fractions.values())
    denom = sum_excl_bf + inclusive_bf

    n_incl = len(df_incl)
    n_excl = len(df_excl)
    excl_norm = (n_incl / n_excl) * (1.0 - inclusive_bf / denom)

    df_incl["branching_fraction_weight"] = 1.0
    df_excl["branching_fraction_weight"] = excl_norm

    for df in (df_incl, df_excl):
        if ff_weight_col not in df.columns:
            df[ff_weight_col] = 1.0
        df["total_weight"] = df["branching_fraction_weight"] * df[ff_weight_col]


def apply_ot_weights(df, reweight_map, bin_width,
                     pplus_col="genPplus", pminus_col="genPminus"):
    """
    Assign per-event OT hybrid weights by looking up each event's (P+, P-) bin
    in the 2D reweight map produced by extract_reweights.

    Parameters
    ----------
    df : DataFrame
        Inclusive events.
    reweight_map : 2D array
        Weight map of shape (n_pplus_bins, n_pminus_bins).
    bin_width : float
        Bin size used when building the grid (same value as in solve_ot).
    pplus_col, pminus_col : str
        Column names for P+ and P-.

    Returns
    -------
    ndarray, shape (n_events,)
        Per-event OT hybrid weights.
    """
    ix = (df[pplus_col].values  / bin_width).astype(int).clip(0, reweight_map.shape[0] - 1)
    iy = (df[pminus_col].values / bin_width).astype(int).clip(0, reweight_map.shape[1] - 1)
    return reweight_map[ix, iy]
