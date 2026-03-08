# Hoisting key classes and functions for cleaner code
from .architectures import GeneralMLP, GeneralCNN, ResearchCNN
from .ht_library import init_heavy_tailed, apply_heavy_tailed_init
from .analysis import (
    get_singular_values,
    marcenkoPastur,
    fit_marcenkoPastur,
    pdf_from_spectrum,
    get_layer_fingerprint,
    evaluate_spectral_perturbation,
    get_layer_from_checkpoint,
    run_spectral_analysis,
    GaussBroadening,
)
__all__ = [
    "GeneralMLP",
    "GeneralCNN",
    "ResearchCNN",
    "init_heavy_tailed",
    "apply_heavy_tailed_init",
    "get_singular_values",
    "marcenkoPastur",
    "fit_marcenkoPastur",
    "pdf_from_spectrum",
    "get_layer_fingerprint",
    "evaluate_spectral_perturbation",
    "get_layer_from_checkpoint",
    "run_spectral_analysis",
    "GaussBroadening",
]
