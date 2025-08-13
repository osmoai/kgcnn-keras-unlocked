"""Multi-Order Directed Graph Attention Network v2 (MoDGATv2).

This module implements MoDGATv2, which combines:
- Multiple DGAT layers with directed attention (forward/backward)
- Multi-order message passing and aggregation
- RMS normalization throughout
- Descriptor fusion capabilities
"""

from ._make import make_model
from ._modgatv2_conv import MoDGATv2Layer

__all__ = [
    "make_model",
    "MoDGATv2Layer"
]
