import time

import numpy as np
import matplotlib.pyplot as plt
import ot


def build_phase_space_grid(df_incl, df_excl, bin_width, pminus_range, pplus_range,
                           incl_weight_col="norm_weight", excl_weight_col="norm_weight",
                           pplus_col="genPplus", pminus_col="genPminus"):
    """
    Histogram inclusive and resonant samples onto a 2D (P+, P-) grid.

    Returns two 2D arrays with shape (n_pplus_bins, n_pminus_bins).
    """
    pm_bins = np.arange(pminus_range[0], pminus_range[1], bin_width)
    pp_bins = np.arange(pplus_range[0],  pplus_range[1],  bin_width)

    def _hist2d(df, weight_col):
        counts, _, _ = np.histogram2d(
            df[pminus_col].values,
            df[pplus_col].values,
            bins=[pm_bins, pp_bins],
            weights=df[weight_col].values,
        )
        return counts.T  # shape: (pplus_bins, pminus_bins)

    Pinc = _hist2d(df_incl, incl_weight_col)
    Pres = _hist2d(df_excl,  excl_weight_col)
    return Pinc, Pres


def solve_ot(Psource, Ptarget, bin_width, window=None, lambda_reg=0, verbose=False):
    """
    Solve the optimal transport problem between a source and target distribution
    in 2D (P+, P-) space, with a sink node to absorb unmatched source mass.

    Parameters
    ----------
    Psource : 2D array
        Source distribution (inclusive HQE prediction), binned in (P+, P-).
    Ptarget : 2D array
        Target distribution (sum of known resonances), same binning as Psource.
    bin_width : float
        Bin size in GeV, used to convert the spatial window into bin offsets.
    window : tuple or None
        (dP+_min, dP+_max, dP-_min, dP-_max) in GeV. Restricts which source-target
        bin pairs are allowed connections. Set to None to allow all pairs.
    lambda_reg : float
        Entropy regularization strength. 0 uses the exact EMD (network simplex),
        values > 0 use the Sinkhorn algorithm and produce smoother transport plans.
    verbose : bool
        Print timing and convergence info.

    Returns
    -------
    G : 2D array
        Transport plan of shape (n_src, n_tgt + 1). The last column corresponds
        to the sink node.
    src_coords : 2D array
        (row, col) bin indices of non-empty source bins.
    Psource_shape : tuple
        Shape of the original Psource array, needed to reconstruct the weight map.
    """
    if Psource.shape != Ptarget.shape:
        raise ValueError("Psource and Ptarget must have the same shape")

    min_content = 1

    src_coords, src_weights = [], []
    tgt_coords, tgt_weights = [], []

    for ir in range(Psource.shape[0]):
        for ic in range(Psource.shape[1]):
            if Psource[ir, ic] >= min_content:
                src_coords.append((ir, ic))
                src_weights.append(Psource[ir, ic])
            if Ptarget[ir, ic] >= min_content:
                tgt_coords.append((ir, ic))
                tgt_weights.append(Ptarget[ir, ic])

    src_coords  = np.array(src_coords,  dtype=float)
    tgt_coords  = np.array(tgt_coords,  dtype=float)
    src_weights = np.array(src_weights, dtype=float)
    tgt_weights = np.array(tgt_weights, dtype=float)

    supply   = src_weights.sum()
    demand   = tgt_weights.sum()
    mismatch = supply - demand

    if mismatch < 0:
        raise ValueError(
            "Target has more total mass than source. "
            "Check that the inclusive branching fraction is larger than the sum of exclusive ones."
        )

    # Append a sink node to absorb the excess inclusive mass.
    # Coordinate 500 places it far outside the physical grid.
    _SINK_COORD = 500
    tgt_coords  = np.vstack([tgt_coords, [_SINK_COORD, _SINK_COORD]])
    tgt_weights = np.append(tgt_weights, mismatch)

    # Build cost matrix: Euclidean distance in bin-index space.
    # _INFEASIBLE_COST is used to block connections outside the window.
    _INFEASIBLE_COST = 9_999_999.0
    n_src = len(src_coords)
    n_tgt = len(tgt_coords)
    M = np.full((n_src, n_tgt), _INFEASIBLE_COST)

    src_r = src_coords[:, 0].astype(int)
    src_c = src_coords[:, 1].astype(int)
    tgt_r = tgt_coords[:-1, 0].astype(int)
    tgt_c = tgt_coords[:-1, 1].astype(int)

    dr = src_r[:, None] - tgt_r
    dc = src_c[:, None] - tgt_c

    if window is None:
        M[:, :-1] = np.sqrt(dr**2 + dc**2)
    else:
        left   = int(window[2] / bin_width)
        right  = int(window[3] / bin_width)
        bottom = int(window[0] / bin_width)
        top    = int(window[1] / bin_width)
        if verbose:
            print(f"Window in bin offsets: bottom={bottom}, top={top}, left={left}, right={right}")
        valid = (
            (tgt_r >= src_r[:, None] + bottom) & (tgt_r <= src_r[:, None] + top) &
            (tgt_c >= src_c[:, None] + left)   & (tgt_c <= src_c[:, None] + right)
        )
        M[:, :-1] = np.where(valid, np.sqrt(dr**2 + dc**2), _INFEASIBLE_COST)

    M[:, -1] = 0.0  # sink is free to receive

    total_mass = src_weights.sum()
    a = src_weights / total_mass
    b = tgt_weights / total_mass

    assert abs(a.sum() - b.sum()) < 1e-9, "Supply/demand normalisation mismatch"

    t0 = time.time()
    if lambda_reg == 0:
        G = ot.emd(a, b, M)
    else:
        G = ot.sinkhorn(a, b, M, lambda_reg, numItermax=10_000)
    G *= total_mass

    if verbose:
        print(f"OT converged in {time.time() - t0:.1f} s")

    return G, src_coords, Psource.shape


def _extract_sink_mass(G, src_coords, shape):
    """Return a 2D array of mass routed to the sink node for each source bin."""
    remaining = np.zeros(shape)
    for i, (r, c) in enumerate(src_coords.astype(int)):
        remaining[r, c] = G[i, -1]
    return remaining


def extract_reweights(G, src_coords, Psource_shape, Psource):
    """
    Extract the per-bin reweight map from the transport solution.

    For each source bin, the fraction of mass routed to the sink gives the
    weight assigned to that bin's inclusive events. Bins fully covered by
    resonances receive weight 0; bins with no resonance coverage get weight 1.

    Returns a 2D array of the same shape as Psource.
    """
    remaining = _extract_sink_mass(G, src_coords, Psource_shape)
    return remaining / np.maximum(Psource, 1)


def plot_sink_distribution(G, src_coords, Psource_shape, ax=None):
    """
    Diagnostic plot of the mass routed to the sink (i.e. the inclusive leftover
    before normalization). Useful for checking the transport solution.
    """
    remaining = _extract_sink_mass(G, src_coords, Psource_shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(remaining, origin="lower", cmap="magma", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mass routed to sink")
    ax.set_title("Inclusive leftover (sink mass)")
    ax.set_xlabel("$P^-$ bin")
    ax.set_ylabel("$P^+$ bin")
    return ax
