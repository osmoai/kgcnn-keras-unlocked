"""Add-GNN: A Dual-Representation Fusion Molecular Property Prediction Based on Graph Neural Networks with Additive Attention.

Reference:
    Zhou, R., Zhang, Y., He, K., & Liu, H. (2025). Add-GNN: A Dual-Representation Fusion Molecular Property Prediction 
    Based on Graph Neural Networks with Additive Attention. Symmetry, 17(6), 873.
"""

from ._make import make_model, model_default
from ._make import make_contrastive_addgnn_model
from ._make import make_addgnn_pna_model, model_addgnn_pna_default
from ._addgnn_conv import AddGNNConv
from ._addgnn_pna_conv import AddGNPPNALayer

__all__ = [
    "make_model",
    "model_default",
    "make_contrastive_addgnn_model",
    "make_addgnn_pna_model",
    "model_addgnn_pna_default",
    "AddGNPPNALayer"
] 