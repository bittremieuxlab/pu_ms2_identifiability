# Transformer models for spectral quality assessment
from .model_bce_loss_one_hot import SimpleSpectraTransformer as BCEModel
from .model_nn_pu_loss_detach_diff_polarity import SimpleSpectraTransformer as nnPUModel

__all__ = ["BCEModel", "nnPUModel"]
