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
from kgcnn.model.utils import update_model_kwargs

ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='MultiHeadGraphAttention')
class MultiHeadGraphAttention(GraphBaseLayer):
    """Multi-head attention mechanism for graph transformer.
    
    This layer implements scaled dot-product attention with graph structure awareness,
    incorporating edge features and positional encodings.
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
        """Initialize MultiHeadGraphAttention layer.
        
        Args:
            units: Output dimension of the layer
            num_heads: Number of attention heads
            dropout: Dropout rate for the output
            use_edge_features: Whether to incorporate edge features in attention
            use_positional_encoding: Whether to use positional encodings
            positional_encoding_dim: Dimension of positional encodings
            attention_dropout: Dropout rate for attention weights
            kernel_initializer: Initializer for kernel weights
            bias_initializer: Initializer for bias weights
            kernel_regularizer: Regularizer for kernel weights
            bias_regularizer: Regularizer for bias weights
            activity_regularizer: Regularizer for activity
            kernel_constraint: Constraint for kernel weights
            bias_constraint: Constraint for bias weights
        """
        super(MultiHeadGraphAttention, self).__init__(**kwargs)
        
        self.units = int(units)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.use_edge_features = bool(use_edge_features)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.positional_encoding_dim = int(positional_encoding_dim)
        self.attention_dropout = float(attention_dropout)
        
        # Ensure units is divisible by num_heads
        if self.units % self.num_heads != 0:
            raise ValueError(f"units ({self.units}) must be divisible by num_heads ({self.num_heads})")
        
        self.head_dim = self.units // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Layer arguments
        kernel_args = {
            "kernel_regularizer": kernel_regularizer,
            "activity_regularizer": activity_regularizer,
            "bias_regularizer": bias_regularizer,
            "kernel_constraint": kernel_constraint,
            "bias_constraint": bias_constraint,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer,
            "use_bias": True
        }
        
        # Query, Key, Value projections
        self.query_dense = Dense(self.units, activation="linear", **kernel_args)
        self.key_dense = Dense(self.units, activation="linear", **kernel_args)
        self.value_dense = Dense(self.units, activation="linear", **kernel_args)
        
        # Output projection
        self.output_dense = Dense(self.units, activation="linear", **kernel_args)
        
        # Edge feature projection (if used)
        if self.use_edge_features:
            self.edge_dense = Dense(self.units, activation="linear", **kernel_args)
        
        # Positional encoding (if used)
        if self.use_positional_encoding:
            self.pos_encoding_dense = Dense(self.units, activation="linear", **kernel_args)
        
        # Dropout layers
        self.attention_dropout_layer = tf.keras.layers.Dropout(self.attention_dropout)
        self.output_dropout_layer = tf.keras.layers.Dropout(self.dropout)
        
        # Graph-specific layers
        self.gather_outgoing = GatherNodesOutgoing()
        self.gather_ingoing = GatherNodesIngoing()
        self.aggregate_edges = AggregateLocalEdges(pooling_method="sum")
        
    def build(self, input_shape):
        """Build the layer weights."""
        super(MultiHeadGraphAttention, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        """Forward pass of multi-head graph attention.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices, positional_encoding]
                - node_features: Node feature matrix [N, F] (ragged)
                - edge_features: Edge feature matrix [E, F_e] (ragged) 
                - edge_indices: Edge index matrix [E, 2] (ragged)
                - positional_encoding: Positional encoding [N, P] (ragged, optional)
                
        Returns:
            Updated node features [N, units] (ragged)
        """
        if len(inputs) < 3:
            raise ValueError("Expected at least 3 inputs: [node_features, edge_features, edge_indices]")
        
        node_features = inputs[0]
        edge_features = inputs[1] if len(inputs) > 1 else None
        edge_indices = inputs[2]
        positional_encoding = inputs[3] if len(inputs) > 3 and self.use_positional_encoding else None
        
        # Get input dimensions
        batch_size = tf.shape(node_features.row_splits)[0] - 1
        num_nodes = tf.shape(node_features.values)[0]
        
        # Apply positional encoding if provided
        if positional_encoding is not None and self.use_positional_encoding:
            pos_encoded = self.pos_encoding_dense(positional_encoding)
            node_features = LazyAdd()([node_features, pos_encoded])
        
        # Project to Q, K, V
        query = self.query_dense(node_features)  # [N, units]
        key = self.key_dense(node_features)      # [N, units]
        value = self.value_dense(node_features)  # [N, units]
        
        # Reshape for multi-head attention
        query = self._reshape_for_heads(query)  # [N, num_heads, head_dim]
        key = self._reshape_for_heads(key)      # [N, num_heads, head_dim]
        value = self._reshape_for_heads(value)  # [N, num_heads, head_dim]
        
        # Gather neighbor information for graph-aware attention
        neighbor_query = self.gather_outgoing([query, edge_indices])  # [E, num_heads, head_dim]
        neighbor_key = self.gather_outgoing([key, edge_indices])      # [E, num_heads, head_dim]
        neighbor_value = self.gather_outgoing([value, edge_indices])  # [E, num_heads, head_dim]
        
        # Incorporate edge features if available
        if edge_features is not None and self.use_edge_features:
            edge_projected = self.edge_dense(edge_features)  # [E, units]
            edge_projected = self._reshape_for_heads(edge_projected)  # [E, num_heads, head_dim]
            # Add edge information to key and value
            neighbor_key = LazyAdd()([neighbor_key, edge_projected])
            neighbor_value = LazyAdd()([neighbor_value, edge_projected])
        
        # Compute attention scores
        # For each node, compute attention with its neighbors
        attention_scores = self._compute_attention_scores(
            query, neighbor_key, edge_indices, num_nodes
        )  # [E, num_heads]
        
        # Apply attention dropout
        attention_scores = self.attention_dropout_layer(attention_scores)
        
        # Apply attention weights to values
        attended_values = neighbor_value * tf.expand_dims(attention_scores, axis=-1)  # [E, num_heads, head_dim]
        
        # Aggregate attended values for each node
        aggregated_values = self.aggregate_edges([
            node_features, attended_values, edge_indices
        ])  # [N, num_heads, head_dim]
        
        # Reshape back to original format
        output = self._reshape_from_heads(aggregated_values)  # [N, units]
        
        # Apply output projection
        output = self.output_dense(output)
        
        # Apply output dropout
        output = self.output_dropout_layer(output)
        
        return output
    
    def _reshape_for_heads(self, x):
        """Reshape tensor for multi-head attention."""
        # x: [N, units] -> [N, num_heads, head_dim]
        shape = tf.shape(x.values) if hasattr(x, 'values') else tf.shape(x)
        new_shape = tf.concat([shape[:-1], [self.num_heads, self.head_dim]], axis=0)
        
        if hasattr(x, 'values'):
            reshaped_values = tf.reshape(x.values, new_shape)
            return tf.RaggedTensor.from_row_splits(reshaped_values, x.row_splits)
        else:
            return tf.reshape(x, new_shape)
    
    def _reshape_from_heads(self, x):
        """Reshape tensor from multi-head attention."""
        # x: [N, num_heads, head_dim] -> [N, units]
        shape = tf.shape(x.values) if hasattr(x, 'values') else tf.shape(x)
        new_shape = tf.concat([shape[:-2], [self.units]], axis=0)
        
        if hasattr(x, 'values'):
            reshaped_values = tf.reshape(x.values, new_shape)
            return tf.RaggedTensor.from_row_splits(reshaped_values, x.row_splits)
        else:
            return tf.reshape(x, new_shape)
    
    def _compute_attention_scores(self, query, key, edge_indices, num_nodes):
        """Compute attention scores between query and key."""
        # query: [N, num_heads, head_dim]
        # key: [E, num_heads, head_dim] (neighbor keys)
        # edge_indices: [E, 2]
        
        # Gather query for each edge
        edge_query = self.gather_outgoing([query, edge_indices])  # [E, num_heads, head_dim]
        
        # Compute dot product attention
        # [E, num_heads, head_dim] * [E, num_heads, head_dim] -> [E, num_heads]
        attention_scores = tf.reduce_sum(edge_query * key, axis=-1)
        
        # Scale by sqrt(head_dim)
        attention_scores = attention_scores * self.scale
        
        # Apply softmax across neighbors for each node
        # We need to group by target nodes and apply softmax
        target_nodes = edge_indices.values[:, 1] if hasattr(edge_indices, 'values') else edge_indices[:, 1]
        
        # Create segment IDs for softmax
        segment_ids = target_nodes
        
        # Apply softmax within each segment (node's neighbors)
        # Use segment_softmax for proper grouping
        attention_scores = tf.math.segment_softmax(attention_scores, segment_ids)
        
        return attention_scores
    
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