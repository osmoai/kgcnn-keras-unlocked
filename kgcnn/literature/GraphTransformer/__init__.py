"""Graph Transformer Neural Network implementation.

This module implements a Graph Transformer that combines transformer architecture
with graph structure, incorporating positional encodings and graph-aware attention.

Reference:
    Graph Transformer: A Generalization of Transformers to Graphs
    https://arxiv.org/abs/2012.09699
"""

from ._make import make_model, make_crystal_model
from ._graph_transformer_conv import GraphTransformerLayer, MultiHeadGraphAttention

__all__ = [
    "make_model",
    "make_crystal_model", 
    "GraphTransformerLayer",
    "MultiHeadGraphAttention"
] 