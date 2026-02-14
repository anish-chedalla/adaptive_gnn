"""MWPM decoder via PyMatching for CSS code-capacity decoding.

PyMatching requires at most 2 non-zeros per column in the check matrix
(each edge connects at most 2 syndrome nodes). For LDPC codes with
higher-weight columns, we decompose each column into weight-2 edges
by chaining adjacent check nodes, and add boundary edges to ensure
perfect matching always exists.

This is APPROXIMATE for LDPC codes -- MWPM is optimal only for codes
with purely weight-2 columns (surface codes). For bivariate bicycle
codes with weight-3 columns, the decomposition introduces approximation.

Usage:
    from gnn_pipeline.matching_decoder import build_mwpm_css, run_mwpm_css

    matcher_z, matcher_x, emap_z, emap_x = build_mwpm_css(hx, hz, pz, px)
    z_errors, x_errors = run_mwpm_css(x_syn, z_syn, matcher_z, matcher_x, n,
                                       edge_map_z=emap_z, edge_map_x=emap_x)
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _decompose_pcm_for_matching(pcm, weights):
    """Decompose a PCM with high-weight columns into weight-<=2 columns.

    For each column with k > 2 non-zero entries at rows [r0, r1, ..., r_{k-1}],
    we create (k-1) weight-2 columns chaining the rows:
        (r0, r1), (r1, r2), ..., (r_{k-2}, r_{k-1})

    Each new edge gets weight = original_weight / (k-1) to preserve total cost.
    Columns with weight <= 2 are kept as-is.

    Additionally, adds one boundary edge per check node (weight-1 columns)
    with high weight, so that odd-parity syndromes can always find a match.

    Returns:
        (new_pcm_sparse, new_weights, edge_map)
        edge_map[i] = original qubit index, or -1 for boundary edges
    """
    m, n = pcm.shape
    new_cols = []
    new_weights = []
    edge_map = []

    for j in range(n):
        col = pcm[:, j]
        rows_nonzero = np.where(col > 0)[0]
        k = len(rows_nonzero)

        if k == 0:
            continue
        elif k <= 2:
            new_col = np.zeros(m, dtype=np.uint8)
            new_col[rows_nonzero] = 1
            new_cols.append(new_col)
            new_weights.append(weights[j])
            edge_map.append(j)
        else:
            edge_weight = weights[j] / (k - 1)
            for i in range(k - 1):
                new_col = np.zeros(m, dtype=np.uint8)
                new_col[rows_nonzero[i]] = 1
                new_col[rows_nonzero[i + 1]] = 1
                new_cols.append(new_col)
                new_weights.append(edge_weight)
                edge_map.append(j)

    # Add boundary edges: one weight-1 column per check node.
    # These connect each check to a virtual boundary, allowing
    # matching when syndrome has odd parity in a component.
    boundary_weight = 100.0  # high cost discourages spurious boundary matches
    for row in range(m):
        boundary_col = np.zeros(m, dtype=np.uint8)
        boundary_col[row] = 1
        new_cols.append(boundary_col)
        new_weights.append(boundary_weight)
        edge_map.append(-1)  # boundary, not a real qubit

    if not new_cols:
        return csr_matrix((m, 0)), np.array([]), []

    new_pcm = np.column_stack(new_cols)
    return csr_matrix(new_pcm), np.array(new_weights, dtype=np.float64), edge_map


def _fold_correction(correction, edge_map, n):
    """Fold decomposed correction back to n qubits via XOR."""
    result = np.zeros(n, dtype=np.int64)
    for edge_idx, qubit_idx in enumerate(edge_map):
        if correction[edge_idx] and qubit_idx >= 0:
            result[qubit_idx] ^= 1
    return result


def build_mwpm_css(hx, hz, pz, px):
    """Build MWPM matchers for separate CSS decoding.

    Args:
        hx: (mx, n) X parity check matrix (detects Z errors)
        hz: (mz, n) Z parity check matrix (detects X errors)
        pz: Z-error probability per qubit
        px: X-error probability per qubit

    Returns:
        (matcher_z, matcher_x, edge_map_z, edge_map_x)
    """
    import pymatching

    w_z = np.log((1 - pz) / pz) if 0 < pz < 1 else 20.0
    w_x = np.log((1 - px) / px) if 0 < px < 1 else 20.0
    weights_z = np.full(hx.shape[1], w_z)
    weights_x = np.full(hz.shape[1], w_x)

    hx_dec, wz_dec, emap_z = _decompose_pcm_for_matching(hx, weights_z)
    hz_dec, wx_dec, emap_x = _decompose_pcm_for_matching(hz, weights_x)

    matcher_z = pymatching.Matching(hx_dec, weights=wz_dec)
    matcher_x = pymatching.Matching(hz_dec, weights=wx_dec)

    return matcher_z, matcher_x, emap_z, emap_x


def run_mwpm_css(x_syndrome, z_syndrome, matcher_z, matcher_x, n,
                 edge_map_z=None, edge_map_x=None):
    """Decode CSS syndromes using MWPM.

    Args:
        x_syndrome: (mx,) X-check syndrome (detects Z errors)
        z_syndrome: (mz,) Z-check syndrome (detects X errors)
        matcher_z: pymatching.Matching for Z-error decoding
        matcher_x: pymatching.Matching for X-error decoding
        n: number of data qubits
        edge_map_z: mapping from decomposed edges to original qubits (for hx)
        edge_map_x: mapping from decomposed edges to original qubits (for hz)

    Returns:
        (z_errors, x_errors) -- each (n,) integer arrays
    """
    if x_syndrome.sum() > 0:
        try:
            corr_z = matcher_z.decode(x_syndrome.astype(np.uint8))
            if edge_map_z is not None:
                z_errors = _fold_correction(corr_z, edge_map_z, n)
            else:
                z_errors = corr_z[:n].astype(np.int64)
        except ValueError:
            z_errors = np.zeros(n, dtype=np.int64)
    else:
        z_errors = np.zeros(n, dtype=np.int64)

    if z_syndrome.sum() > 0:
        try:
            corr_x = matcher_x.decode(z_syndrome.astype(np.uint8))
            if edge_map_x is not None:
                x_errors = _fold_correction(corr_x, edge_map_x, n)
            else:
                x_errors = corr_x[:n].astype(np.int64)
        except ValueError:
            x_errors = np.zeros(n, dtype=np.int64)
    else:
        x_errors = np.zeros(n, dtype=np.int64)

    return z_errors, x_errors
