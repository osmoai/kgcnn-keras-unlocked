"""Multi-Order Directed Message Passing Neural Network (MoDMPNN).

This module implements MoDMPNN, which combines:
- Multiple DMPNN layers with directed message passing
- Multi-order message aggregation
- RMS normalization throughout
- Descriptor fusion capabilities
"""

from ._make import make_model
from ._modmpnn_conv import MoDMPNNLayer

__all__ = [
    "make_model",
    "MoDMPNNLayer"
]
