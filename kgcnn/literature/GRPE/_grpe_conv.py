import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyAdd, LazyConcatenate, LayerNormalization
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.ops.axis import get_axis

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras


class GRPELayer(GraphBaseLayer):
    r"""Graph Relative Positional Encoding (GRPE) layer.
    
    This layer adds relative positional encoding (shortest path distances, graph structure)
    to transformer attention, refining GraphTransformer for better molecular modeling.
    
    Reference: Graph Relative Positional Encoding for Transformers (2022-23)
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
        
        # GRPE components
        self.node_transform = Dense(units, activation=activation, use_bias=use_bias)
        
        if use_edge_features:
            self.edge_transform = Dense(units, activation=activation, use_bias=use_bias)
        
        # Relative positional encoding
        self.relative_pos_encoding = Dense(attention_units, activation=activation, use_bias=use_bias)
        
        # Multi-head attention with relative positional encoding
        self.attention_heads = []
        for i in range(attention_heads):
            self.attention_heads.append(
                self._create_attention_head(attention_units // attention_heads)
            )
        
        # Layer normalization
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(units * 2, activation=activation, use_bias=use_bias),
            Dropout(dropout_rate),
            Dense(units, activation=activation, use_bias=use_bias)
        ])
        
        # Output projection
        self.output_projection = Dense(units, activation=activation, use_bias=use_bias)
        
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
            
        # Message passing components
        self.gather_nodes = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges()
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate()
        
    def _create_attention_head(self, head_dim):
        """Create a single attention head with relative positional encoding."""
        return {
            'query': Dense(head_dim, use_bias=self.use_bias),
            'key': Dense(head_dim, use_bias=self.use_bias),
            'value': Dense(head_dim, use_bias=self.use_bias),
            'relative_key': Dense(head_dim, use_bias=self.use_bias),
            'relative_value': Dense(head_dim, use_bias=self.use_bias)
        }
        
    def _compute_shortest_paths(self, edge_indices, num_nodes):
        """Compute shortest path distances between all pairs of nodes."""
        # Initialize distance matrix
        distances = tf.fill([num_nodes, num_nodes], float('inf'))
        
        # Set diagonal to 0
        distances = tf.tensor_scatter_nd_update(
            distances,
            tf.range(num_nodes)[:, None],
            tf.zeros(num_nodes)
        )
        
        # Set direct edges to 1
        for edge in edge_indices:
            i, j = edge[0], edge[1]
            distances = tf.tensor_scatter_nd_update(
                distances,
                [[i, j], [j, i]],
                [1.0, 1.0]
            )
        
        # Floyd-Warshall algorithm for shortest paths
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances = tf.tensor_scatter_nd_update(
                            distances,
                            [[i, j]],
                            [distances[i, k] + distances[k, j]]
                        )
        
        # Cap distances at max_path_length
        distances = tf.clip_by_value(distances, 0, self.max_path_length)
        
        return distances
        
    def _apply_relative_attention(self, query, key, value, relative_key, relative_value, attention_mask=None):
        """Apply attention with relative positional encoding."""
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        
        # Add relative positional encoding
        relative_scores = tf.matmul(query, relative_key, transpose_b=True)
        scores = scores + relative_scores
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply dropout
        if self.dropout_rate > 0:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout_rate)
        
        # Compute output
        output = tf.matmul(attention_weights, value)
        
        # Add relative positional encoding to output
        relative_output = tf.matmul(attention_weights, relative_value)
        output = output + relative_output
        
        return output
        
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
        
        # Compute shortest path distances for relative positional encoding
        num_nodes = tf.shape(node_features)[1]
        shortest_paths = self._compute_shortest_paths(edge_indices, num_nodes)
        
        # Create relative positional encoding
        relative_pos_encoding = self.relative_pos_encoding(shortest_paths)
        
        # Multi-head attention with relative positional encoding
        attention_outputs = []
        for head in self.attention_heads:
            # Project inputs
            query = head['query'](node_features)
            key = head['key'](node_features)
            value = head['value'](node_features)
            relative_key = head['relative_key'](relative_pos_encoding)
            relative_value = head['relative_value'](relative_pos_encoding)
            
            # Apply attention
            attended = self._apply_relative_attention(
                query, key, value, relative_key, relative_value
            )
            attention_outputs.append(attended)
        
        # Concatenate attention heads
        if len(attention_outputs) > 1:
            multi_head_output = self.lazy_concat(attention_outputs)
        else:
            multi_head_output = attention_outputs[0]
        
        # Apply layer normalization and residual connection
        multi_head_output = self.layer_norm1(node_attributes + multi_head_output)
        
        # Feed-forward network
        ffn_output = self.ffn(multi_head_output)
        
        # Apply layer normalization and residual connection
        output = self.layer_norm2(multi_head_output + ffn_output)
        
        # Output projection
        output = self.output_projection(output)
        
        # Skip connection
        output = self.lazy_add([node_attributes, output])
        
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