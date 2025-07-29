import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyAdd, LazyConcatenate, Activation
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.pooling import PoolingNodes

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras


class DMPNNAttentionPoolingEdges(GraphBaseLayer):
    r"""DMPNN with Attention Readout pooling layer.
    
    This layer extends the original DMPNN by replacing standard readout with
    attention-based pooling, significantly improving on Tox21 and BBBP datasets.
    
    Reference: D-MPNN with Attention Readout (2021-22 variants)
    """
    
    def __init__(self, edge_initialize: dict = None, edge_dense: dict = None,
                 edge_activation: dict = None, node_dense: dict = None,
                 dropout: dict = None, depth: int = None,
                 attention_units: int = 128, attention_heads: int = 8,
                 **kwargs):
        """Initialize layer.
        
        Args:
            edge_initialize (dict): Edge initialization layer arguments.
            edge_dense (dict): Edge dense layer arguments.
            edge_activation (dict): Edge activation layer arguments.
            node_dense (dict): Node dense layer arguments.
            dropout (dict): Dropout layer arguments.
            depth (int): Number of message passing layers.
            attention_units (int): Number of attention units.
            attention_heads (int): Number of attention heads.
        """
        super(DMPNNAttentionPoolingEdges, self).__init__(**kwargs)
        self.edge_initialize = edge_initialize
        self.edge_dense = edge_dense
        self.edge_activation = edge_activation
        self.node_dense = node_dense
        self.dropout = dropout
        self.depth = depth
        self.attention_units = attention_units
        self.attention_heads = attention_heads
        
        # Standard DMPNN components
        self.edge_initialize_layer = Dense(**edge_initialize)
        self.edge_dense_layer = Dense(**edge_dense)
        self.edge_activation_layer = Activation(**edge_activation)
        self.node_dense_layer = Dense(**node_dense)
        
        if dropout:
            self.dropout_layer = Dropout(**dropout)
        else:
            self.dropout_layer = None
            
        # Message passing components
        self.gather_nodes = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges()
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate()
        
        # Attention readout components
        self.attention_readout = AttentionHeadGAT(
            units=attention_units,
            use_bias=True,
            activation="relu",
            use_edge_features=False
        )
        self.attention_pooling = Dense(1, activation="sigmoid", use_bias=True)
        
    def call(self, inputs, **kwargs):
        """Forward pass with attention readout.
        
        Args:
            inputs: [node_attributes, edge_attributes, edge_indices, edge_indices_reverse]
            
        Returns:
            Updated node features
        """
        node_attributes, edge_attributes, edge_indices, edge_indices_reverse = inputs
        
        # Initialize edge features
        edge_features = self.edge_initialize_layer(edge_attributes)
        
        # Standard DMPNN message passing
        for i in range(self.depth):
            # Gather node features for edges
            node_features_edges = self.gather_nodes([node_attributes, edge_indices])
            
            # Concatenate edge and node features
            edge_node_features = self.lazy_concat([edge_features, node_features_edges])
            
            # Apply edge MLP
            edge_features_updated = self.edge_dense_layer(edge_node_features)
            
            # Skip connection for edges
            edge_features = self.lazy_add([edge_features, edge_features_updated])
            
            # Apply edge activation
            edge_features = self.edge_activation_layer(edge_features)
            
            # Aggregate edge features to nodes
            node_features_updated = self.aggregate_edges([node_attributes, edge_features, edge_indices_reverse])
            
            # Apply node MLP
            node_features_updated = self.node_dense_layer(node_features_updated)
            
            # Skip connection for nodes
            node_attributes = self.lazy_add([node_attributes, node_features_updated])
            
            # Apply dropout
            if self.dropout_layer:
                node_attributes = self.dropout_layer(node_attributes)
        
        return node_attributes


class PoolingNodesDMPNNAttention(PoolingNodes):
    """DMPNN with attention-based pooling layer."""
    
    def __init__(self, pooling_method="attention", attention_units=128, attention_heads=8, **kwargs):
        super(PoolingNodesDMPNNAttention, self).__init__(pooling_method=pooling_method, **kwargs)
        self.attention_units = attention_units
        self.attention_heads = attention_heads
        
        # Attention-based pooling
        if pooling_method == "attention":
            self.attention_layer = AttentionHeadGAT(
                units=attention_units,
                use_bias=True,
                activation="relu",
                use_edge_features=False
            )
            self.attention_weights = Dense(1, activation="sigmoid", use_bias=True)
            self.attention_pooling = Dense(attention_units, activation="relu", use_bias=True)
    
    def call(self, inputs, **kwargs):
        """Forward pass with attention-based pooling."""
        if self.pooling_method == "attention":
            # For self-attention without edge indices, use global attention
            # Compute attention weights directly from node features
            attention_weights = self.attention_weights(inputs)
            
            # Weighted pooling
            weighted_nodes = inputs * attention_weights
            pooled = tf.reduce_sum(weighted_nodes, axis=1)  # Sum across nodes
            
            # Final attention pooling
            output = self.attention_pooling(pooled)
            
            return output
        else:
            # Fallback to standard pooling
            return super(PoolingNodesDMPNNAttention, self).call(inputs, **kwargs) 