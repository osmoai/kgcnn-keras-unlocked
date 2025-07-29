import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyAdd, LazyConcatenate
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.attention import MultiHeadGATV2Layer
from kgcnn.ops.axis import get_axis
from kgcnn.ops.segment import segment_softmax

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras


class TransformerGATLayer(GraphBaseLayer):
    r"""Transformer-enhanced GAT layer using existing KGCNN attention mechanisms.
    
    This layer combines local node-level attention (GATv2-style) with global transformer attention layers,
    integrating local and global graph contexts using well-tested KGCNN components.
    
    Reference: Local-Global Graph Attention Networks (2023)
    """
    
    def __init__(self, units, use_bias=True, activation="relu",
                 attention_heads=8, attention_units=64, transformer_heads=8,
                 use_edge_features=True, dropout_rate=0.1, **kwargs):
        """Initialize layer.
        
        Args:
            units (int): Number of hidden units.
            use_bias (bool): Whether to use bias.
            activation (str): Activation function.
            attention_heads (int): Number of local attention heads.
            attention_units (int): Number of attention units.
            transformer_heads (int): Number of global transformer attention heads.
            use_edge_features (bool): Whether to use edge features in attention.
            dropout_rate (float): Dropout rate.
        """
        super(TransformerGATLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.attention_heads = attention_heads
        self.attention_units = attention_units
        self.transformer_heads = transformer_heads
        self.use_edge_features = use_edge_features
        self.dropout_rate = dropout_rate
        
        # Local GAT attention using existing KGCNN MultiHeadGATV2Layer
        self.local_gat = MultiHeadGATV2Layer(
            units=attention_units,
            num_heads=attention_heads,
            activation=activation,
            use_bias=use_bias,
            concat_heads=False  # We'll handle concatenation ourselves
        )
        
        # Global transformer attention
        self.global_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=transformer_heads,
            key_dim=attention_units // transformer_heads,
            value_dim=attention_units // transformer_heads,
            dropout=dropout_rate
        )
        
        # Layer normalization
        self.layer_norm1 = GraphLayerNormalization(epsilon=1e-6)
        self.layer_norm2 = GraphLayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(attention_units * 2, activation=activation, use_bias=use_bias),
            Dropout(dropout_rate),
            Dense(attention_units, activation=activation, use_bias=use_bias)
        ])
        
        # Output projection
        self.output_projection = Dense(units, activation=activation, use_bias=use_bias)
        
        # Input projection to match attention output dimensions
        self.input_projection = Dense(attention_units, activation=activation, use_bias=use_bias)
        
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
        """Forward pass with local-global attention.
        
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
        
        # Project input to match attention output dimensions
        projected_node_attributes = self.input_projection(node_attributes)
        
        # Local GAT attention using existing KGCNN layer
        if edge_attributes is not None:
            local_output = self.local_gat([projected_node_attributes, edge_attributes, edge_indices])
        else:
            # Create dummy edge features if none provided
            dummy_edges = tf.zeros_like(projected_node_attributes)
            local_output = self.local_gat([projected_node_attributes, dummy_edges, edge_indices])
        
        # MultiHeadGATV2Layer returns (node_embeddings, attention_weights)
        # We only need the node embeddings
        if isinstance(local_output, tuple):
            local_output = local_output[0]
        
        # Apply layer normalization and residual connection
        local_output = self.layer_norm1(projected_node_attributes + local_output)
        
        # Global transformer attention
        # Convert ragged to dense for transformer
        if hasattr(local_output, 'to_tensor'):
            global_input = local_output.to_tensor()
            row_lengths = local_output.row_lengths()
        else:
            global_input = local_output
            row_lengths = None
        
        # Apply global attention
        global_output = self.global_attention(
            query=global_input,
            value=global_input,
            key=global_input
        )
        
        # Convert back to ragged tensor if needed
        if row_lengths is not None:
            global_output = tf.RaggedTensor.from_tensor(global_output, lengths=row_lengths)
        
        # Apply layer normalization and residual connection
        global_output = self.layer_norm2(local_output + global_output)
        
        # Feed-forward network
        ffn_output = self.ffn(global_output)
        
        # Final residual connection
        output = global_output + ffn_output
        
        # Output projection
        output = self.output_projection(output)
        
        # Skip connection - project node_attributes to match output dimension
        projected_node_attributes = self.skip_projection(node_attributes)
        output = self.lazy_add([projected_node_attributes, output])
        
        # Apply dropout
        if self.dropout:
            output = self.dropout(output)
        
        return output


class PoolingNodesTransformerGAT(GraphBaseLayer):
    """TransformerGAT-specific pooling layer."""
    
    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodesTransformerGAT, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
    def call(self, inputs, **kwargs):
        """Forward pass with TransformerGAT pooling."""
        if self.pooling_method == "sum":
            return tf.reduce_sum(inputs, axis=1)
        elif self.pooling_method == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.pooling_method == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}") 