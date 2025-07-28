"""KA-GAT: Kolmogorov-Arnold Graph Attention Network with Fourier-KAN.

This module implements KA-GAT, which replaces traditional MLPs with Fourier-KAN
(Fourier Kolmogorov-Arnold Networks) to boost attention-based GNN performance.

References:
- Kolmogorov-Arnold Networks: https://arxiv.org/abs/2404.19756
- Fourier-KAN: Enhanced Kolmogorov-Arnold Networks with Fourier basis
- Graph Attention Networks: https://arxiv.org/abs/1710.10903
"""

from ._kagat_conv import KAGATConv
from ._make import make_model

__all__ = [
    "KAGATConv",
    "make_model"
] 