"""DHTNN: Double-Head Transformer Neural Network.

This module implements DHTNN, which enhances DMPNN with "double-head" attention blocks
that combine local and global attention mechanisms for superior molecular property prediction.

References:
- Double-Head Attention: Enhanced attention mechanism for molecular graphs
- DMPNN Enhancement: Improved directed message passing with attention
- Graph Transformer: https://arxiv.org/abs/2006.06247
"""

from ._dhtnn_conv import DHTNNConv, DoubleHeadAttention
from ._make import make_model

__all__ = [
    "DHTNNConv",
    "DoubleHeadAttention", 
    "make_model"
] 