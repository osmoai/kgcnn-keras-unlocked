import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyConcatenate, Activation, LazyAdd
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.attention import AttentionHeadGAT

ks = tf.keras

class DMPNNAttentionPoolingEdges(GraphBaseLayer):
    r"""DMPNN with Attention Pooling following standard DMPNN pattern.
    
    This implementation follows the same pattern as the standard DMPNN:
    1. Initialize edge features by concatenating node and edge features
    2. Use standard DMPNN message passing
    3. Apply attention-based pooling at the end
    """
    def __init__(self, edge_initialize=None, edge_dense=None, edge_activation=None, node_dense=None,
                 dropout=None, depth=None, attention_units=128, attention_heads=8,
                 use_attention_in_message_passing=True, **kwargs):
        super().__init__(**kwargs)
        self.edge_initialize_layer = Dense(**edge_initialize)
        self.edge_dense_layer = Dense(**edge_dense)
        self.edge_activation_layer = Activation(**edge_activation)
        self.node_dense_layer = Dense(**node_dense)
        self.dropout_layer = Dropout(**dropout) if dropout else None
        self.depth = depth
        self.gather_nodes_outgoing = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges(pooling_method="sum")
        self.lazy_concat = LazyConcatenate(axis=-1)
        self.lazy_add = LazyAdd()
        self.use_attention_in_message_passing = use_attention_in_message_passing
        if use_attention_in_message_passing:
            self.attention_pooling = AttentionHeadGAT(
                units=attention_units,
                use_bias=True,
                activation="relu",
                use_edge_features=False
            )

    def call(self, inputs, **kwargs):
        if len(inputs) == 4:
            node_attributes, edge_attributes, edge_indices, edge_indices_reverse = inputs
        else:
            node_attributes, edge_attributes, edge_indices = inputs
            edge_indices_reverse = edge_indices

        # Initialize edge features following DMPNN pattern
        node_features_outgoing = self.gather_nodes_outgoing([node_attributes, edge_indices])
        edge_features = self.lazy_concat([node_features_outgoing, edge_attributes])
        edge_features = self.edge_initialize_layer(edge_features)
        
        # Store initial edge features for skip connections
        edge_features_initial = edge_features
        
        # DMPNN message passing
        for _ in range(self.depth):
            # Aggregate edge features to nodes
            node_features_updated = self.aggregate_edges([node_attributes, edge_features, edge_indices])
            
            # Gather updated node features for edges
            node_features_outgoing = self.gather_nodes_outgoing([node_features_updated, edge_indices])
            
            # Concatenate with original edge features
            edge_node_features = self.lazy_concat([node_features_outgoing, edge_attributes])
            
            # Apply edge MLP
            edge_features_updated = self.edge_dense_layer(edge_node_features)
            
            # Skip connection (like in standard DMPNN)
            edge_features = self.lazy_add([edge_features_updated, edge_features_initial])
            
            # Apply activation
            edge_features = self.edge_activation_layer(edge_features)
            
            if self.dropout_layer:
                edge_features = self.dropout_layer(edge_features)

        # Final aggregation to nodes
        node_features_final = self.aggregate_edges([node_attributes, edge_features, edge_indices])
        
        # Apply node MLP
        node_attributes = self.node_dense_layer(node_features_final)

        # Apply attention-based pooling if enabled
        if self.use_attention_in_message_passing:
            node_attributes = self.attention_pooling([node_attributes, edge_attributes, edge_indices])
            
        return node_attributes 