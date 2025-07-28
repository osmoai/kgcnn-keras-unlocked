"""Add-GNN: A Dual-Representation Fusion Molecular Property Prediction Based on Graph Neural Networks with Additive Attention.

Reference:
    Zhou, R., Zhang, Y., He, K., & Liu, H. (2025). Add-GNN: A Dual-Representation Fusion Molecular Property Prediction 
    Based on Graph Neural Networks with Additive Attention. Symmetry, 17(6), 873.
"""

from ._make import make_model
from ._addgnn_conv import AddGNNConv

__all__ = ["make_model", "AddGNNConv"] 