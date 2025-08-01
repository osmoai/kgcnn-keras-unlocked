import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyConcatenate, Activation, LazyAdd, LazyMultiply
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.mlp import MLP
from kgcnn.ops.axis import get_axis

ks = tf.keras

class MultiChemAttention(GraphBaseLayer):
    """Multi-scale attention mechanism for MultiChem.
    
    This layer implements attention that can handle both directed and undirected graphs
    with dual node and edge features.
    """
    
    def __init__(self, units, num_heads=8, use_directed=True, use_dual_features=True,
                 attention_dropout=0.1, **kwargs):
        """Initialize MultiChem attention layer.
        
        Args:
            units: Number of hidden units
            num_heads: Number of attention heads
            use_directed: Whether to use directed graph attention
            use_dual_features: Whether to use dual node/edge features
            attention_dropout: Dropout rate for attention
        """
        super(MultiChemAttention, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.use_directed = use_directed
        self.use_dual_features = use_dual_features
        self.attention_dropout = attention_dropout
        
        # Multi-head attention for different scales
        self.node_attention = AttentionHeadGAT(
            units=units // num_heads,
            use_bias=True,
            activation="relu",
            use_edge_features=True,
            num_heads=num_heads
        )
        
        # Edge attention for dual features
        if self.use_dual_features:
            self.edge_attention = AttentionHeadGAT(
                units=units // num_heads,
                use_bias=True,
                activation="relu",
                use_edge_features=True,
                num_heads=num_heads
            )
        
        # Directional attention for directed graphs
        if self.use_directed:
            self.directed_attention = AttentionHeadGAT(
                units=units // num_heads,
                use_bias=True,
                activation="relu",
                use_edge_features=True,
                num_heads=num_heads
            )
        
        # Feature fusion layers
        self.node_projection = Dense(units, activation="linear", use_bias=True)
        self.edge_projection = Dense(units, activation="linear", use_bias=True)
        self.fusion_gate = Dense(units, activation="sigmoid", use_bias=True)
        
        # Dropout
        self.dropout = Dropout(attention_dropout)
        
        # Aggregation layers
        self.gather_outgoing = GatherNodesOutgoing()
        self.gather_ingoing = GatherNodesIngoing()
        self.aggregate_edges = AggregateLocalEdges(pooling_method="sum")
        
    def call(self, inputs, **kwargs):
        """Forward pass with multi-scale attention.
        
        Args:
            inputs: [node_features, edge_features, edge_indices, edge_indices_reverse]
                   or [node_features, edge_features, edge_indices] for undirected
            
        Returns:
            Updated node features
        """
        if len(inputs) == 4:
            node_features, edge_features, edge_indices, edge_indices_reverse = inputs
        else:
            node_features, edge_features, edge_indices = inputs
            edge_indices_reverse = edge_indices
        
        # Node-level attention
        node_attended = self.node_attention([node_features, edge_features, edge_indices])
        
        # Edge-level attention (if using dual features)
        if self.use_dual_features:
            edge_attended = self.edge_attention([edge_features, node_features, edge_indices])
            # Aggregate edge features to nodes
            edge_to_node = self.aggregate_edges([node_features, edge_attended, edge_indices])
        else:
            edge_to_node = node_features
        
        # Directional attention (if using directed graphs)
        if self.use_directed:
            # Outgoing attention
            outgoing_features = self.gather_outgoing([node_features, edge_indices])
            outgoing_attended = self.directed_attention([outgoing_features, edge_features, edge_indices])
            
            # Incoming attention
            incoming_features = self.gather_ingoing([node_features, edge_indices_reverse])
            incoming_attended = self.directed_attention([incoming_features, edge_features, edge_indices_reverse])
            
            # Combine directional features
            directional_features = LazyAdd()([outgoing_attended, incoming_attended])
        else:
            directional_features = node_attended
        
        # Feature fusion with gating mechanism
        node_proj = self.node_projection(node_attended)
        edge_proj = self.edge_projection(edge_to_node)
        
        # Gating mechanism
        gate = self.fusion_gate(LazyConcatenate(axis=-1)([node_proj, edge_proj]))
        
        # Fused features
        fused_features = LazyAdd()([
            LazyMultiply()([gate, node_proj]),
            LazyMultiply()([1 - gate, edge_proj])
        ])
        
        # Add directional information
        output = LazyAdd()([fused_features, directional_features])
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class MultiChemLayer(GraphBaseLayer):
    """Main MultiChem layer combining attention and message passing.
    
    This layer implements the core MultiChem architecture with support for:
    - Multi-scale attention
    - Dual node/edge features
    - Directed/undirected graphs
    - Chemical-specific message passing
    """
    
    def __init__(self, units, num_heads=8, use_directed=True, use_dual_features=True,
                 dropout=0.1, attention_dropout=0.1, use_residual=True, **kwargs):
        """Initialize MultiChem layer.
        
        Args:
            units: Number of hidden units
            num_heads: Number of attention heads
            use_directed: Whether to use directed graph processing
            use_dual_features: Whether to use dual node/edge features
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            use_residual: Whether to use residual connections
        """
        super(MultiChemLayer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.use_directed = use_directed
        self.use_dual_features = use_dual_features
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_residual = use_residual
        
        # Multi-scale attention
        self.attention = MultiChemAttention(
            units=units,
            num_heads=num_heads,
            use_directed=use_directed,
            use_dual_features=use_dual_features,
            attention_dropout=attention_dropout
        )
        
        # Message passing MLP
        self.message_mlp = MLP(
            units=[units, units],
            activation=["relu", "linear"],
            use_bias=[True, True],
            use_normalization=False
        )
        
        # Node update MLP
        self.node_mlp = MLP(
            units=[units, units],
            activation=["relu", "linear"],
            use_bias=[True, True],
            use_normalization=False
        )
        
        # Edge update MLP (if using dual features)
        if self.use_dual_features:
            self.edge_mlp = MLP(
                units=[units, units],
                activation=["relu", "linear"],
                use_bias=[True, True],
                use_normalization=False
            )
        
        # Layer normalization
        self.node_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        if self.use_dual_features:
            self.edge_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout_layer = Dropout(dropout)
        
        # Residual connections
        if self.use_residual:
            self.node_residual = LazyAdd()
            if self.use_dual_features:
                self.edge_residual = LazyAdd()
        
        # Aggregation layers
        self.gather_outgoing = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges(pooling_method="sum")
        
    def call(self, inputs, **kwargs):
        """Forward pass of MultiChem layer.
        
        Args:
            inputs: [node_features, edge_features, edge_indices, edge_indices_reverse]
                   or [node_features, edge_features, edge_indices] for undirected
            
        Returns:
            Updated node and edge features
        """
        if len(inputs) == 4:
            node_features, edge_features, edge_indices, edge_indices_reverse = inputs
        else:
            node_features, edge_features, edge_indices = inputs
            edge_indices_reverse = edge_indices
        
        # Store original features for residual connections
        node_original = node_features
        edge_original = edge_features
        
        # Multi-scale attention
        attention_inputs = [node_features, edge_features, edge_indices]
        if self.use_directed:
            attention_inputs.append(edge_indices_reverse)
        
        attended_features = self.attention(attention_inputs, **kwargs)
        
        # Message passing
        # Gather node features to edges
        node_to_edge = self.gather_outgoing([attended_features, edge_indices])
        
        # Combine node and edge features for message computation
        if self.use_dual_features:
            message_input = LazyConcatenate(axis=-1)([node_to_edge, edge_features])
        else:
            message_input = node_to_edge
        
        # Compute messages
        messages = self.message_mlp(message_input, **kwargs)
        
        # Aggregate messages to nodes
        aggregated_messages = self.aggregate_edges([node_features, messages, edge_indices])
        
        # Update node features
        node_update_input = LazyConcatenate(axis=-1)([node_features, aggregated_messages])
        node_updated = self.node_mlp(node_update_input, **kwargs)
        
        # Update edge features (if using dual features)
        if self.use_dual_features:
            edge_update_input = LazyConcatenate(axis=-1)([edge_features, messages])
            edge_updated = self.edge_mlp(edge_update_input, **kwargs)
        else:
            edge_updated = edge_features
        
        # Apply normalization
        node_updated = self.node_norm(node_updated)
        if self.use_dual_features:
            edge_updated = self.edge_norm(edge_updated)
        
        # Apply dropout
        node_updated = self.dropout_layer(node_updated)
        if self.use_dual_features:
            edge_updated = self.dropout_layer(edge_updated)
        
        # Residual connections
        if self.use_residual:
            node_updated = self.node_residual([node_original, node_updated])
            if self.use_dual_features:
                edge_updated = self.edge_residual([edge_original, edge_updated])
        
        return node_updated, edge_updated


class PoolingNodesMultiChem(GraphBaseLayer):
    """MultiChem-specific pooling layer with support for dual features."""
    
    def __init__(self, pooling_method="sum", use_dual_features=True, **kwargs):
        super(PoolingNodesMultiChem, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.use_dual_features = use_dual_features
        
    def call(self, inputs, **kwargs):
        """Forward pass with MultiChem pooling.
        
        Args:
            inputs: [node_features, edge_features] or [node_features]
            
        Returns:
            Pooled graph features
        """
        if self.use_dual_features and len(inputs) == 2:
            node_features, edge_features = inputs
            # Pool both node and edge features
            node_pooled = tf.reduce_sum(node_features, axis=1)
            edge_pooled = tf.reduce_sum(edge_features, axis=1)
            # Concatenate pooled features
            return LazyConcatenate(axis=-1)([node_pooled, edge_pooled])
        else:
            node_features = inputs[0] if isinstance(inputs, list) else inputs
            if self.pooling_method == "sum":
                return tf.reduce_sum(node_features, axis=1)
            elif self.pooling_method == "mean":
                return tf.reduce_mean(node_features, axis=1)
            elif self.pooling_method == "max":
                return tf.reduce_max(node_features, axis=1)
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling_method}") 