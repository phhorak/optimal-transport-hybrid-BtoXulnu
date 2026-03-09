import numpy as np


def compute_normalization(df_incl, df_excl, branching_fractions, inclusive_bf,
                          pdg_col="X_gen_PDG", input_weight_col="input_weight"):
    """
    Compute per-event normalization weights so that the resonant and inclusive
    samples can be combined into a single hybrid model.

    The normalization follows the standard hybrid construction:

        excl_norm = (N_incl / N_excl) * (1 - BF_Xu / (sum_BF_excl + BF_Xu))

    This factor is applied uniformly to all exclusive events (independent of
    PDG code, since the exclusive sample is pre-mixed by EvtGen). Inclusive
    events get weight 1 from this factor. Both samples are further multiplied
    by their input_weight column (e.g. form-factor corrections; set to 1 if absent).

    The resulting column "norm_weight" is added in-place to both DataFrames.

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
    input_weight_col : str
        Optional per-event weight column already present in the DataFrames
        (e.g. form-factor corrections). If the column does not exist, it is
        treated as 1 for all events.
    """
    sum_excl_bf = sum(branching_fractions.values())
    denom = sum_excl_bf + inclusive_bf

    n_incl = len(df_incl)
    n_excl = len(df_excl)
    excl_norm = (n_incl / n_excl) * (1.0 - inclusive_bf / denom)

    for df, bf_factor in ((df_incl, 1.0), (df_excl, excl_norm)):
        input_w = df[input_weight_col].values if input_weight_col in df.columns else 1.0
        df["norm_weight"] = bf_factor * input_w


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
