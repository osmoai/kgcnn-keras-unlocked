"""Contrastive GNN module for KGCNN.

This module provides contrastive learning implementations for popular GNN architectures
including GIN, GAT, DMPNN, etc. with integrated contrastive learning capabilities.
"""

from ._make import (
    make_contrastive_gnn_model,
    make_contrastive_gin_model,
    make_contrastive_gat_model,
    make_contrastive_dmpnn_model,
    compile_contrastive_gnn_model,
    contrastive_gnn_models
)

from ._contrastive_gin_conv import (
    ContrastiveGINConv,
    ContrastiveGINLayer,
    GraphViewGenerator
)

from ._contrastive_losses import (
    ContrastiveGNNLoss,
    ContrastiveGNNTripletLoss,
    ContrastiveGNNDiversityLoss,
    ContrastiveGNNAlignmentLoss,
    ContrastiveGNNMetric,
    create_contrastive_gnn_metrics,
    compute_contrastive_quality_score
)

__all__ = [
    # Model factories
    "make_contrastive_gnn_model",
    "make_contrastive_gin_model", 
    "make_contrastive_gat_model",
    "make_contrastive_dmpnn_model",
    "compile_contrastive_gnn_model",
    "contrastive_gnn_models",
    
    # Layers
    "ContrastiveGINConv",
    "ContrastiveGINLayer",
    "GraphViewGenerator",
    
    # Losses
    "ContrastiveGNNLoss",
    "ContrastiveGNNTripletLoss", 
    "ContrastiveGNNDiversityLoss",
    "ContrastiveGNNAlignmentLoss",
    
    # Metrics
    "ContrastiveGNNMetric",
    "create_contrastive_gnn_metrics",
    "compute_contrastive_quality_score"
] 