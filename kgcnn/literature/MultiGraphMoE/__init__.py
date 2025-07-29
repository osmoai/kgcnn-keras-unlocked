"""Multi-Graph MoE: Mixture of Experts with Multiple Graph Representations.

This module implements a Multi-Graph MoE that creates multiple graph representations
and uses different GNN experts (GIN, GAT, etc.) to improve ensemble performance
and reduce variance on small datasets.

Key Features:
- Multiple graph representations (different edge weights, node features, etc.)
- Different GNN experts (GIN, GAT, GCN, etc.)
- MoE routing for expert selection
- Ensemble combination for final prediction
- Variance reduction through multiple representations

References:
- Mixture of Experts: "Outrageously Large Neural Networks" (Shazeer et al., 2017)
- Multi-Graph Learning: "Multi-Graph Convolutional Networks" (Monti et al., 2017)
- Ensemble Methods: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
"""

from ._multigraph_moe_conv import MultiGraphMoEConv, GraphRepresentationLayer, ExpertRoutingLayer
from ._make import make_model, model_default
from ._make import make_contrastive_moe_model

__all__ = [
    "MultiGraphMoEConv",
    "GraphRepresentationLayer", 
    "ExpertRoutingLayer",
    "make_model",
    "model_default",
    "make_contrastive_moe_model"
] 