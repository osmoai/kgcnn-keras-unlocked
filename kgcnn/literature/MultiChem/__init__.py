"""MultiChem GNN implementation for kgcnn.

MultiChem is a multi-modal chemical modeling framework that supports:
- Directed and undirected graph representations
- Dual node and edge features
- Multi-scale attention mechanisms
- Chemical-specific inductive biases
"""

from ._multichem_conv import MultiChemLayer, MultiChemAttention
from ._make import make_model

__all__ = [
    "MultiChemLayer",
    "MultiChemAttention", 
    "make_model"
] 