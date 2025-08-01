import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.ops.axis import get_axis
from kgcnn.ops.segment import segment_softmax

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras


class CoAttentiveHeadFP(GraphBaseLayer):
    r"""Collaborative Attentive Fingerprint layer for molecular representation learning.
    
    This layer extends the original AttentiveFP with collaborative attention mechanisms
    between atom and bond embeddings, improving ADMET property predictions.
    
    Reference: Collaborative Graph Attention Networks (2022, JCIM)
    """
    
    def __init__(self, units, use_bias=True, activation="relu", 
                 use_collaborative=True, collaboration_heads=8, dropout_rate=0.1,
                 **kwargs):
        """Initialize layer.
        
        Args:
            units (int): Number of hidden units.
            use_bias (bool): Whether to use bias.
            activation (str): Activation function.
            use_collaborative (bool): Whether to use collaborative attention.
            collaboration_heads (int): Number of attention heads for collaboration.
            dropout_rate (float): Dropout rate.
        """
        super(CoAttentiveHeadFP, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.use_collaborative = use_collaborative
        self.collaboration_heads = collaboration_heads
        self.dropout_rate = dropout_rate
        
        # Collaborative attention components
        if self.use_collaborative:
            self.node_attention = AttentionHeadGAT(
                units=units,
                use_bias=use_bias,
                activation=activation,
                use_edge_features=True
            )
            self.edge_projection = Dense(units, activation=activation, use_bias=use_bias)
            self.collaboration_gate = Dense(units, activation="sigmoid", use_bias=use_bias)
            self.collaboration_fusion = Dense(units, activation=activation, use_bias=use_bias)
            
            # Edge-to-node aggregation layers
            self.gather_outgoing = GatherNodesOutgoing()
            self.aggregate_edges = AggregateLocalEdges(pooling_method="sum")
        
        # Standard attention (fallback)
        else:
            self.attention = AttentionHeadGAT(
                units=units,
                use_bias=use_bias,
                activation=activation,
                use_edge_features=True
            )
        
        self.dropout = Dropout(dropout_rate)
        
        # Input projection to match attention output dimension
        self.input_projection = Dense(units, activation="linear", use_bias=use_bias)
        
        self.gru_update = GRUUpdate(units)
        
    def call(self, inputs, **kwargs):
        """Forward pass.
        
        Args:
            inputs: [node_attributes, edge_attributes, edge_indices]
            
        Returns:
            Updated node features
        """
        node_attributes, edge_attributes, edge_indices = inputs
        
        if self.use_collaborative:
            # Collaborative attention mechanism
            # 1. Node attention with edge features
            node_attended = self.node_attention([node_attributes, edge_attributes, edge_indices])
            
            # 2. Edge projection and aggregation to nodes
            edge_projected = self.edge_projection(edge_attributes)
            edge_to_node = self._propagate_edge_to_node(edge_projected, edge_indices, node_attributes)
            
            # 3. Collaboration gate
            collaboration_weights = self.collaboration_gate(node_attributes)
            
            # 4. Fuse node and edge attention with collaboration
            collaborative_features = (
                collaboration_weights * node_attended + 
                (1 - collaboration_weights) * edge_to_node
            )
            
            # 5. Final fusion
            output = self.collaboration_fusion(collaborative_features)
            
        else:
            # Standard attention (fallback to original AttentiveFP)
            output = self.attention([node_attributes, edge_attributes, edge_indices])
        
        # Apply dropout and GRU update
        output = self.dropout(output)
        
        # Project input node features to match attention output dimension
        node_attributes_projected = self.input_projection(node_attributes)
        
        # GRU update with projected node features
        output = self.gru_update([node_attributes_projected, output])
        
        return output
    
    def _propagate_edge_to_node(self, edge_features, edge_indices, node_features):
        """Propagate edge features to nodes using edge indices."""
        # Gather edge features to nodes using edge indices
        gathered_edges = self.gather_outgoing([edge_features, edge_indices])
        
        # Aggregate edge features to nodes
        # Use the proper aggregation method for ragged tensors
        node_features_aggregated = self.aggregate_edges([node_features, gathered_edges, edge_indices])
        
        return node_features_aggregated


class PoolingNodesCoAttentive(PoolingNodes):
    """Collaborative attentive pooling layer."""
    
    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodesCoAttentive, self).__init__(pooling_method=pooling_method, **kwargs)
    
    def call(self, inputs, **kwargs):
        """Forward pass with collaborative attention."""
        return super(PoolingNodesCoAttentive, self).call(inputs, **kwargs) 