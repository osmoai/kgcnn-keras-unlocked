"""Graph Transformer Convolution Layer.

This module implements the core Graph Transformer layer that combines
transformer architecture with graph structure awareness.
"""

import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, LazyAdd, LazyConcatenate, Activation, LazyAverage
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.set2set import PoolingSet2SetEncoder
from kgcnn.layers.mlp import MLP
from kgcnn.layers.attention import MultiHeadGATV2Layer
from kgcnn.model.utils import update_model_kwargs

ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='MultiHeadGraphAttention')
class MultiHeadGraphAttention(GraphBaseLayer):
    """Multi-head graph attention mechanism for Graph Transformer using existing KGCNN attention.
    
    This layer wraps the existing MultiHeadGATV2Layer from KGCNN for robust graph attention.
    
    Args:
        units: Hidden dimension of the layer
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_edge_features: Whether to use edge features
        use_positional_encoding: Whether to use positional encodings
        positional_encoding_dim: Dimension of positional encodings
        attention_dropout: Dropout rate for attention
        kernel_initializer: Kernel initializer
        bias_initializer: Bias initializer
        kernel_regularizer: Kernel regularizer
        bias_regularizer: Bias regularizer
        activity_regularizer: Activity regularizer
        kernel_constraint: Kernel constraint
        bias_constraint: Bias constraint
    """
    
    def __init__(self,
                 units: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_edge_features: bool = True,
                 use_positional_encoding: bool = True,
                 positional_encoding_dim: int = 64,
                 attention_dropout: float = 0.1,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiHeadGraphAttention, self).__init__(**kwargs)
        
        self.units = int(units)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.use_edge_features = bool(use_edge_features)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.positional_encoding_dim = int(positional_encoding_dim)
        self.attention_dropout = float(attention_dropout)
        
        # Use existing KGCNN multi-head attention
        self.attention_layer = MultiHeadGATV2Layer(
            units=self.units,
            num_heads=self.num_heads,
            activation="kgcnn>leaky_relu",
            use_bias=True,
            concat_heads=False  # Don't concatenate heads, we'll handle projection separately
        )
        
        # Projection layer to combine attention heads
        self.head_projection = Dense(self.units, activation="linear", use_bias=True)
        
        # Positional encoding projection
        if self.use_positional_encoding:
            self.pos_encoding_dense = Dense(self.units, activation="linear", use_bias=True)
        
        # Dropout layer
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        
        # Projection layer for graph descriptors
        self.graph_projection = Dense(units=self.units, activation="linear")
        
    def build(self, input_shape):
        """Build the layer weights."""
        super(MultiHeadGraphAttention, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        """Forward pass of multi-head graph attention.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices, positional_encoding, graph_descriptors]
                - node_features: Node feature matrix [N, F] (ragged)
                - edge_features: Edge feature matrix [E, F_e] (ragged)
                - edge_indices: Edge index matrix [E, 2] (ragged)
                - positional_encoding: Positional encoding [N, P] (ragged, optional)
                - graph_descriptors: Graph descriptor features [G] (optional)
                
        Returns:
            Updated node features [N, units] (ragged)
        """
        node_features = inputs[0]
        edge_features = inputs[1] if len(inputs) > 1 else None
        edge_indices = inputs[2] if len(inputs) > 2 else None
        positional_encoding = inputs[3] if len(inputs) > 3 and self.use_positional_encoding else None
        
        # Apply positional encoding if provided
        if positional_encoding is not None and self.use_positional_encoding:
            pos_encoded = self.pos_encoding_dense(positional_encoding)
            node_features = LazyAdd()([node_features, pos_encoded])
        
        # Use existing KGCNN attention mechanism
        # The MultiHeadGATV2Layer expects [node_features, edge_features, edge_indices]
        attention_inputs = [node_features]
        if edge_features is not None and self.use_edge_features:
            attention_inputs.append(edge_features)
        else:
            # Create dummy edge features if not provided
            dummy_edges = tf.ones_like(node_features.values[:, :1]) if hasattr(node_features, 'values') else tf.ones_like(node_features[:, :1])
            if hasattr(node_features, 'row_splits'):
                dummy_edges = tf.RaggedTensor.from_row_splits(dummy_edges, node_features.row_splits)
            attention_inputs.append(dummy_edges)
        
        attention_inputs.append(edge_indices)
        
        # Apply attention
        attention_result = self.attention_layer(attention_inputs, **kwargs)
        
        # MultiHeadGATV2Layer returns (output, attention_weights)
        if isinstance(attention_result, tuple):
            output = attention_result[0]  # Take only the node embeddings
        else:
            output = attention_result
        
        # Project attention heads to final dimension
        output = self.head_projection(output)
        
        # Apply dropout
        output = self.dropout_layer(output)
        
        return output
    
    def get_config(self):
        """Update layer config."""
        config = super(MultiHeadGraphAttention, self).get_config()
        config.update({
            "units": self.units,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "use_edge_features": self.use_edge_features,
            "use_positional_encoding": self.use_positional_encoding,
            "positional_encoding_dim": self.positional_encoding_dim,
            "attention_dropout": self.attention_dropout
        })
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GraphTransformerLayer')
class GraphTransformerLayer(GraphBaseLayer):
    """Complete Graph Transformer layer with attention and feed-forward network.
    
    This layer implements a full transformer block for graphs, including:
    - Multi-head graph attention
    - Layer normalization
    - Feed-forward network
    - Residual connections
    """
    
    def __init__(self,
                 units: int,
                 num_heads: int = 8,
                 ff_units: int = None,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_edge_features: bool = True,
                 use_positional_encoding: bool = True,
                 positional_encoding_dim: int = 64,
                 activation: str = "relu",
                 layer_norm_epsilon: float = 1e-6,
                 **kwargs):
        """Initialize GraphTransformerLayer.
        
        Args:
            units: Hidden dimension of the layer
            num_heads: Number of attention heads
            ff_units: Units in feed-forward network (default: 4 * units)
            dropout: Dropout rate
            attention_dropout: Dropout rate for attention
            use_edge_features: Whether to use edge features
            use_positional_encoding: Whether to use positional encodings
            positional_encoding_dim: Dimension of positional encodings
            activation: Activation function for feed-forward network
            layer_norm_epsilon: Epsilon for layer normalization
        """
        super(GraphTransformerLayer, self).__init__(**kwargs)
        
        self.units = int(units)
        self.num_heads = int(num_heads)
        self.ff_units = int(ff_units) if ff_units is not None else 4 * self.units
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.use_edge_features = bool(use_edge_features)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.positional_encoding_dim = int(positional_encoding_dim)
        self.activation = activation
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        
        # Multi-head attention
        self.attention = MultiHeadGraphAttention(
            units=self.units,
            num_heads=self.num_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            use_edge_features=self.use_edge_features,
            use_positional_encoding=self.use_positional_encoding,
            positional_encoding_dim=self.positional_encoding_dim
        )
        
        # Layer normalization
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ff_norm = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        
        # Feed-forward network
        self.ff_network = MLP(
            units=[self.ff_units, self.units],
            activation=[self.activation, "linear"],
            use_bias=[True, True],
            use_normalization=False
        )
        
        # Dropout
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        
        # Residual connections
        self.residual_add = LazyAdd()
        
        # Projection layer for graph descriptors
        self.graph_projection = Dense(units=self.units, activation="linear")
        
    def build(self, input_shape):
        """Build the layer weights."""
        super(GraphTransformerLayer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        """Forward pass of Graph Transformer layer.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices, positional_encoding, graph_descriptors]
                - node_features: Node feature matrix [N, F] (ragged)
                - edge_features: Edge feature matrix [E, F_e] (ragged)
                - edge_indices: Edge index matrix [E, 2] (ragged)
                - positional_encoding: Positional encoding [N, P] (ragged, optional)
                - graph_descriptors: Graph descriptor features [G] (optional)
                
        Returns:
            Updated node features [N, units] (ragged)
        """
        node_features = inputs[0]
        edge_features = inputs[1] if len(inputs) > 1 else None
        edge_indices = inputs[2] if len(inputs) > 2 else None
        positional_encoding = inputs[3] if len(inputs) > 3 else None
        graph_descriptors = inputs[4] if len(inputs) > 4 else None
        
        # Incorporate graph descriptors into node features if provided
        if graph_descriptors is not None:
            # Use GatherState to properly distribute graph descriptors to nodes
            from kgcnn.layers.gather import GatherState
            from kgcnn.layers.modules import LazyConcatenate
            
            # Gather graph state for each node
            graph_state_node = GatherState()([graph_descriptors, node_features])
            
            # Add graph descriptors to node features
            node_features = LazyConcatenate(axis=-1)([node_features, graph_state_node])
            
            # Project to the expected dimension for transformer
            node_features = self.graph_projection(node_features)
        
        # Prepare attention inputs
        attention_inputs = [node_features]
        if edge_features is not None:
            attention_inputs.append(edge_features)
        if edge_indices is not None:
            attention_inputs.append(edge_indices)
        if positional_encoding is not None:
            attention_inputs.append(positional_encoding)
        
        # Self-attention with residual connection
        attention_output = self.attention(attention_inputs, **kwargs)
        attention_output = self.dropout_layer(attention_output)
        
        # Residual connection and normalization
        node_features = self.residual_add([node_features, attention_output])
        node_features = self.attention_norm(node_features)
        
        # Feed-forward network with residual connection
        ff_output = self.ff_network(node_features, **kwargs)
        ff_output = self.dropout_layer(ff_output)
        
        # Residual connection and normalization
        node_features = self.residual_add([node_features, ff_output])
        node_features = self.ff_norm(node_features)
        
        return node_features
    
    def get_config(self):
        """Update layer config."""
        config = super(GraphTransformerLayer, self).get_config()
        config.update({
            "units": self.units,
            "num_heads": self.num_heads,
            "ff_units": self.ff_units,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "use_edge_features": self.use_edge_features,
            "use_positional_encoding": self.use_positional_encoding,
            "positional_encoding_dim": self.positional_encoding_dim,
            "activation": self.activation,
            "layer_norm_epsilon": self.layer_norm_epsilon
        })
        return config 