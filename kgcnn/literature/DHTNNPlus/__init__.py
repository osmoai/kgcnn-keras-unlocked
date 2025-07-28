"""DHTNNPlus: Enhanced Double-Head Transformer Neural Network with Collaboration.

This module implements DHTNNPlus, which combines DHTNNConv with collaboration mechanisms
and GRU updates to achieve the best of both worlds - double-head attention with
collaborative intelligence and sequential processing.

References:
- DHTNN: Double-Head Transformer Neural Network
- CoAttentiveFP: Collaborative Attentive Fingerprint
- GRU Updates: Sequential processing for molecular graphs
"""

from ._dhtnnplus_conv import DHTNNPlusConv, EnhancedDoubleHeadAttention
from ._make import make_model

__all__ = [
    "DHTNNPlusConv",
    "EnhancedDoubleHeadAttention", 
    "make_model"
] 