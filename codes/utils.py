# utils.py - Linear algebra over GF(2): RREF, kernels, rank, inverses for code analysis
from __future__ import annotations

import numpy as np
from typing import List, Tuple


def _gf2(A) -> np.ndarray:
    A = np.array(A, dtype=np.uint8, copy=True)
    if A.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    return A & 1


def row_echelon(A: np.ndarray) -> Tuple[np.ndarray, int, List[int], List[int]]:
    """Reduced Row Echelon Form (RREF) over GF(2).
    Returns: (RREF matrix, rank, pivot row indices, pivot column indices).
    """
    R = _gf2(A)
    m, n = R.shape

    pivot_rows: List[int] = []
    pivot_cols: List[int] = []

    r = 0
    for c in range(n):
        if r >= m:
            break

        # Find a 1 in column c starting from row r
        piv = None
        for rr in range(r, m):
            if R[rr, c] == 1:
                piv = rr
                break
        if piv is None:
            continue

        # Swap pivot row to position r
        if piv != r:
            R[[r, piv]] = R[[piv, r]]

        pivot_rows.append(r)
        pivot_cols.append(c)

        # Zero out all other 1s in column (makes it RREF, not just REF)
        for rr in range(m):
            if rr != r and R[rr, c] == 1:
                R[rr] ^= R[r]  # XOR = addition in GF(2)

        r += 1

    return R, len(pivot_cols), pivot_rows, pivot_cols


def rank(A: np.ndarray) -> int:
    """Rank over GF(2)."""
    return int(row_echelon(A)[1])


def kernel(A: np.ndarray) -> Tuple[np.ndarray, int, List[int]]:
    """Compute nullspace basis over GF(2): all vectors x where A·x = 0.
    Returns: (basis rows, rank, pivot rows).
    """
    A = _gf2(A)
    m, n = A.shape

    R, rnk, pivot_rows, pivot_cols = row_echelon(A)

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]

    # Trivial kernel if no free variables
    if not free_cols:
        return np.zeros((0, n), dtype=np.uint8), int(rnk), pivot_rows

    # For each free variable, construct a basis vector by setting it to 1
    # and solving for dependent variables via back-substitution
    basis = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1  # Set this free variable

        # Back-substitute to find dependent variable values
        for i, pc in enumerate(pivot_cols):
            val = 0
            for ff in free_cols:
                if R[i, ff] & x[ff]:
                    val ^= 1  # XOR contributions
            x[pc] = val

        basis.append(x)

    return np.stack(basis, axis=0).astype(np.uint8), int(rnk), pivot_rows


def inverse(A: np.ndarray) -> np.ndarray:
    """Compute matrix inverse over GF(2) using Gauss-Jordan elimination."""
    A = _gf2(A)
    n, m = A.shape
    if n != m:
        raise ValueError("inverse() requires a square matrix")

    I = np.eye(n, dtype=np.uint8)
    Aug = np.hstack([A, I])  # [A | I]

    R, rnk, _, pivot_cols = row_echelon(Aug)

    # A is invertible iff rank is n and pivots cover the first n columns
    if rnk < n or any(pc >= n for pc in pivot_cols[:n]):
        raise ValueError("Matrix is not invertible over GF(2)")

    left = R[:, :n]
    if not np.array_equal(left, np.eye(n, dtype=np.uint8)):
        raise ValueError("Matrix is not invertible over GF(2)")

    return R[:, n:]


def int2bin(x: int, n: int) -> np.ndarray:
    """Convert integer to binary vector (LSB-first, bit i represents 2^i)."""
    if x < 0:
        raise ValueError("int2bin expects nonnegative x")
    return np.array([(x >> i) & 1 for i in range(n)], dtype=np.uint8)


def compute_code_distance(*args, **kwargs) -> int:
    """Placeholder: distance computation is expensive for BB codes and not needed for sampling."""
    return 0
