"""Multi-Order Directed Message Passing Neural Network (MoDMPNN) Layer.

This layer implements the core MoDMPNN architecture combining:
- Multiple DMPNN layers with directed message passing
- Multi-order message aggregation
- RMS normalization throughout
"""

import tensorflow as tf
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.norm import RMSNormalization
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.modules import LazyConcatenate


class MoDMPNNLayer(tf.keras.layers.Layer):
    """Multi-Order Directed Message Passing Neural Network Layer.
    
    This layer stacks multiple DMPNN layers and aggregates their outputs
    using multi-order message fusion, similar to MoGATv2 but with
    directed message passing mechanisms.
    """
    
    def __init__(self, 
                 units=128,
                 depth=4,
                 dropout_rate=0.1,
                 use_rms_norm=True,
                 rms_norm_args=None,
                 **kwargs):
        """Initialize MoDMPNN Layer.
        
        Args:
            units: Number of hidden units
            depth: Number of DMPNN layers to stack
            dropout_rate: Dropout rate
            use_rms_norm: Whether to use RMS normalization
            rms_norm_args: Arguments for RMS normalization
        """
        super().__init__(**kwargs)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.use_rms_norm = use_rms_norm
        
        # Default RMS normalization arguments
        if rms_norm_args is None:
            rms_norm_args = {"epsilon": 1e-6, "scale": True}
        self.rms_norm_args = rms_norm_args
        
        # Create multiple DMPNN layers
        self.dmpnn_layers = []
        for i in range(depth):
            layer = DirectedDMPNNLayer(
                units=units,
                dropout_rate=dropout_rate,
                use_rms_norm=use_rms_norm,
                rms_norm_args=rms_norm_args,
                name=f"dmpnn_layer_{i+1}"
            )
            self.dmpnn_layers.append(layer)
        
        # Multi-order message fusion (like MoGATv2)
        self.multi_order_attention = tf.keras.layers.Dense(
            units, activation='tanh', name="multi_order_attention"
        )
        self.multi_order_weights = tf.keras.layers.Dense(
            1, activation=None, name="multi_order_weights"
        )
        
        # Final RMS normalization before output
        if use_rms_norm:
            self.final_rms_norm = RMSNormalization(**rms_norm_args)
    
    def call(self, inputs, training=None):
        """Forward pass through MoDMPNN.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices, edge_indices_reverse]
            training: Training mode flag
            
        Returns:
            Multi-order aggregated node representations
        """
        node_features, edge_features, edge_indices, edge_indices_reverse = inputs
        
        # Store multi-order representations
        multi_order_reprs = []
        current_node_features = node_features
        current_edge_features = edge_features
        
        # Pass through each DMPNN layer
        for i, dmpnn_layer in enumerate(self.dmpnn_layers):
            # Apply DMPNN layer
            current_node_features, current_edge_features = dmpnn_layer([
                current_node_features, current_edge_features, edge_indices, edge_indices_reverse
            ], training=training)
            
            # Store this order's representation
            multi_order_reprs.append(current_node_features)
            
            # Apply RMS normalization after each layer (except the last)
            if self.use_rms_norm and i < self.depth - 1:
                current_node_features = self.final_rms_norm(current_node_features)
        
        # Multi-order message fusion (like MoGATv2)
        if len(multi_order_reprs) > 1:
            # Stack all representations
            stacked_reprs = tf.stack(multi_order_reprs, axis=1)  # [batch, depth, nodes, features]
            
            # Compute attention weights for each order
            attention_input = self.multi_order_attention(stacked_reprs)  # [batch, depth, nodes, units]
            attention_weights = self.multi_order_weights(attention_input)  # [batch, depth, nodes, 1]
            attention_weights = tf.nn.softmax(attention_weights, axis=1)  # [batch, depth, nodes, 1]
            
            # Weighted combination
            weighted_reprs = stacked_reprs * attention_weights  # [batch, depth, nodes, features]
            fused_reprs = tf.reduce_sum(weighted_reprs, axis=1)  # [batch, nodes, features]
        else:
            # Single layer case
            fused_reprs = multi_order_reprs[0]
        
        # Final RMS normalization
        if self.use_rms_norm:
            fused_reprs = self.final_rms_norm(fused_reprs)
        
        return fused_reprs
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "depth": self.depth,
            "dropout_rate": self.dropout_rate,
            "use_rms_norm": self.use_rms_norm,
            "rms_norm_args": self.rms_norm_args
        })
        return config


class DirectedDMPNNLayer(tf.keras.layers.Layer):
    """Single Directed DMPNN Layer.
    
    This implements directed message passing by processing both
    forward and backward edge directions.
    """
    
    def __init__(self, 
                 units=128,
                 dropout_rate=0.1,
                 use_rms_norm=True,
                 rms_norm_args=None,
                 **kwargs):
        """Initialize Directed DMPNN Layer.
        
        Args:
            units: Number of hidden units
            dropout_rate: Dropout rate
            use_rms_norm: Whether to use RMS normalization
            rms_norm_args: Arguments for RMS normalization
        """
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_rms_norm = use_rms_norm
        
        # Default RMS normalization arguments
        if rms_norm_args is None:
            rms_norm_args = {"epsilon": 1e-6, "scale": True}
        self.rms_norm_args = rms_norm_args
        
        # Message passing networks for forward and backward
        self.forward_mpn = GraphMLP(
            units=[units, units],
            activation=['relu', 'relu'],
            use_bias=True,
            name="forward_mpn"
        )
        
        self.backward_mpn = GraphMLP(
            units=[units, units],
            activation=['relu', 'relu'],
            use_bias=True,
            name="backward_mpn"
        )
        
        # Node update network
        self.node_update = GraphMLP(
            units=[units, units],
            activation=['relu', 'relu'],
            use_bias=True,
            name="node_update"
        )
        
        # Edge update network
        self.edge_update = GraphMLP(
            units=[units, units],
            activation=['relu', 'relu'],
            use_bias=True,
            name="edge_update"
        )
        
        # RMS normalization
        if use_rms_norm:
            self.rms_norm = RMSNormalization(**rms_norm_args)
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        """Forward pass through Directed DMPNN Layer.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices, edge_indices_reverse]
            training: Training mode flag
            
        Returns:
            Updated node and edge representations
        """
        node_features, edge_features, edge_indices, edge_indices_reverse = inputs
        
        # Forward message passing (original direction)
        forward_messages = self._gather_messages(
            node_features, edge_features, edge_indices
        )
        forward_updated = self.forward_mpn(forward_messages)
        
        # Backward message passing (reverse direction)
        backward_messages = self._gather_messages(
            node_features, edge_features, edge_indices_reverse
        )
        backward_updated = self.backward_mpn(backward_messages)
        
        # Combine forward and backward messages
        combined_messages = forward_updated + backward_updated
        
        # Update nodes
        updated_nodes = self.node_update(
            tf.concat([node_features, combined_messages], axis=-1)
        )
        
        # Update edges
        updated_edges = self.edge_update(edge_features)
        
        # RMS normalization
        if self.use_rms_norm:
            updated_nodes = self.rms_norm(updated_nodes)
            updated_edges = self.rms_norm(updated_edges)
        
        # Dropout
        updated_nodes = self.dropout(updated_nodes, training=training)
        updated_edges = self.dropout(updated_edges, training=training)
        
        return updated_nodes, updated_edges
    
    def _gather_messages(self, node_features, edge_features, edge_indices):
        """Gather messages from neighboring nodes using proper graph operations."""
        from kgcnn.layers.gather import GatherNodesOutgoing
        from kgcnn.layers.aggr import AggregateLocalEdges
        
        # Gather neighboring node features for each edge
        neighbor_nodes = GatherNodesOutgoing()([node_features, edge_indices])
        
        # Ensure edge_features has the right shape for concatenation
        # edge_features: (None, 11, 128) -> (None, 128) by taking mean across feature dimension
        edge_features_flat = tf.reduce_mean(edge_features, axis=1)  # Shape: (None, 128)
        
        # Concatenate flattened edge features with neighbor node features
        from kgcnn.layers.modules import LazyConcatenate
        edge_node_features = LazyConcatenate(axis=-1)([edge_features_flat, neighbor_nodes])
        
        # Aggregate messages for each node
        messages = AggregateLocalEdges(pooling_method="sum")([node_features, edge_node_features, edge_indices])
        
        return messages
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate,
            "use_rms_norm": self.use_rms_norm,
            "rms_norm_args": self.rms_norm_args
        })
        return config


class PoolingNodesMoDMPNN(PoolingNodes):
    """Pooling layer specifically for MoDMPNN.
    
    This handles the pooling of node representations after
    multi-order directed message passing.
    """
    
    def __init__(self, **kwargs):
        """Initialize MoDMPNN pooling layer."""
        super().__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        """Pool node representations for MoDMPNN."""
        return super().call(inputs, **kwargs)
