
from __future__ import annotations

from typing import Dict, List


# parameters verified empirically via create_bivariate_bicycle_codes()
CODE_CATALOG: Dict[str, dict] = {
    "72_12_6": {
        "l": 6,
        "m": 6,
        "A_x_pows": [3],
        "A_y_pows": [1, 2],
        "B_x_pows": [1, 2],
        "B_y_pows": [3],
    },
    "144_12_12": {
        "l": 12,
        "m": 6,
        "A_x_pows": [3],
        "A_y_pows": [1, 2],
        "B_x_pows": [1, 2],
        "B_y_pows": [3],
    },
    # Bravyi et al. "two-gross" code: l=12, m=12
    # A = x^3 + y^2 + y^7,  B = y^3 + x + x^2
    "288_12_18": {
        "l": 12,
        "m": 12,
        "A_x_pows": [3],
        "A_y_pows": [2, 7],
        "B_x_pows": [1, 2],
        "B_y_pows": [3],
    },
}


def get_code_params(code_name: str) -> dict:
    """Get construction parameters for a named code.

    Args:
        code_name: key in CODE_CATALOG (e.g. '72_12_6')

    Returns:
        dict with keys l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows

    Raises:
        KeyError: if code_name is not in the catalog
    """
    if code_name not in CODE_CATALOG:
        available = ", ".join(sorted(CODE_CATALOG.keys()))
        raise KeyError(f"Unknown code {code_name!r}. Available: {available}")
    return dict(CODE_CATALOG[code_name])  # return a copy


def list_codes() -> List[str]:
    """Return sorted list of available code names."""
    return sorted(CODE_CATALOG.keys())
