
# astra_stim: Biased noise injection for QLDPC/CSS codes with Stim
from .qldpc_circuit import CSSCodeSpec, build_qldpc_memory_circuit_text
from .biased_noise import (
    apply_biased_data_noise,
    apply_biased_circuit_noise,
    BiasedCircuitNoiseSpec,
)

__all__ = [
    "CSSCodeSpec",
    "build_qldpc_memory_circuit_text",
    "apply_biased_data_noise",
    "apply_biased_circuit_noise",
    "BiasedCircuitNoiseSpec",
]

if __name__ == "__main__":
    raise SystemExit(
        "Do not run astra_stim/__init__.py directly.\n"
        "Use:\n"
        "  python -c \"import astra_stim; print('astra_stim OK')\""
    )
