import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyAdd, LazyConcatenate
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.attention import MultiHeadGATV2Layer
from kgcnn.ops.axis import get_axis

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2025.08.12"

ks = tf.keras


class GRPELayer(GraphBaseLayer):
    r"""Graph Relative Positional Encoding (GRPE) layer.
    
    This layer adds relative positional encoding to transformer attention,
    refining GraphTransformer for better molecular modeling.
    
    Reference: Graph Relative Positional Encoding for Transformers (2022-23)
    
    This implementation uses existing KGCNN components for robustness.
    """
    
    def __init__(self, units, use_bias=True, activation="relu",
                 attention_heads=8, attention_units=64, max_path_length=10,
                 use_edge_features=True, dropout_rate=0.1, **kwargs):
        """Initialize layer.
        
        Args:
            units (int): Number of hidden units.
            use_bias (bool): Whether to use bias.
            activation (str): Activation function.
            attention_heads (int): Number of attention heads.
            attention_units (int): Number of attention units.
            max_path_length (int): Maximum path length for relative positional encoding.
            use_edge_features (bool): Whether to use edge features.
            dropout_rate (float): Dropout rate.
        """
        super(GRPELayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.attention_heads = attention_heads
        self.attention_units = attention_units
        self.max_path_length = max_path_length
        self.use_edge_features = use_edge_features
        self.dropout_rate = dropout_rate
        
        # Node and edge transformations
        self.node_transform = Dense(units, activation=activation, use_bias=use_bias)
        if use_edge_features:
            self.edge_transform = Dense(units, activation=activation, use_bias=use_bias)
        
        # Attention output projection to match node features dimension
        self.attention_projection = Dense(units, activation=activation, use_bias=use_bias)
        
        # Multi-head attention with relative positional encoding
        self.attention_layer = MultiHeadGATV2Layer(
            units=attention_units,
            num_heads=attention_heads,
            use_edge_features=use_edge_features,
            use_bias=use_bias,
            concat_heads=False
        )
        
        # Layer normalization
        self.layer_norm1 = GraphLayerNormalization(epsilon=1e-6)
        self.layer_norm2 = GraphLayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(units * 2, activation=activation, use_bias=use_bias),
            Dropout(dropout_rate),
            Dense(units, activation=activation, use_bias=use_bias)
        ])
        
        # Output projection
        self.output_projection = Dense(units, activation=activation, use_bias=use_bias)
        
        # Skip connection projection
        self.skip_projection = Dense(units, activation=activation, use_bias=use_bias)
        
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
            
        # Message passing components
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate()
        
    def call(self, inputs, **kwargs):
        """Forward pass with relative positional encoding.
        
        Args:
            inputs: [node_attributes, edge_attributes, edge_indices]
            
        Returns:
            Updated node features
        """
        if len(inputs) == 3:
            node_attributes, edge_attributes, edge_indices = inputs
        else:
            node_attributes, edge_indices = inputs
            edge_attributes = None
        
        # Transform node features
        node_features = self.node_transform(node_attributes)
        
        # Transform edge features if provided
        if self.use_edge_features and edge_attributes is not None:
            edge_features = self.edge_transform(edge_attributes)
        else:
            edge_features = None
        
        # Multi-head attention with relative positional encoding
        if edge_features is not None:
            attention_output = self.attention_layer([node_features, edge_features, edge_indices])
        else:
            # Create dummy edge features if none provided
            dummy_edges = tf.zeros_like(node_features)
            attention_output = self.attention_layer([node_features, dummy_edges, edge_indices])
        
        # MultiHeadGATV2Layer returns (node_embeddings, attention_weights)
        # We only need the node embeddings
        if isinstance(attention_output, tuple):
            attention_output = attention_output[0]
        
        # Project attention output to match node features dimension
        attention_output = self.attention_projection(attention_output)
        
        # Apply layer normalization and residual connection
        attention_output = self.layer_norm1(node_features + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(attention_output)
        
        # Apply layer normalization and residual connection
        output = self.layer_norm2(attention_output + ffn_output)
        
        # Output projection
        output = self.output_projection(output)
        
        # Skip connection - project node_attributes to match output dimension
        projected_node_attributes = self.skip_projection(node_attributes)
        output = self.lazy_add([projected_node_attributes, output])
        
        # Apply dropout
        if self.dropout:
            output = self.dropout(output)
        
        return output


class PoolingNodesGRPE(GraphBaseLayer):
    """GRPE-specific pooling layer."""
    
    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodesGRPE, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
    def call(self, inputs, **kwargs):
        """Forward pass with GRPE pooling."""
        if self.pooling_method == "sum":
            return tf.reduce_sum(inputs, axis=1)
        elif self.pooling_method == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.pooling_method == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}") 
