"""Multi-Order Directed Graph Attention Network v2 (MoDGATv2) Layer.

This layer implements the core MoDGATv2 architecture combining:
- Multiple DGAT layers with directed attention
- Multi-order message passing and aggregation
- RMS normalization throughout
"""

import tensorflow as tf
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.norm import RMSNormalization
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import GraphMLP


class MoDGATv2Layer(tf.keras.layers.Layer):
    """Multi-Order Directed Graph Attention Network v2 Layer.
    
    This layer stacks multiple DGAT layers and aggregates their outputs
    using multi-order attention fusion, similar to MoGATv2 but with
    directed attention mechanisms.
    """
    
    def __init__(self, 
                 units=128,
                 depth=4,
                 attention_heads=8,
                 dropout_rate=0.1,
                 use_rms_norm=True,
                 rms_norm_args=None,
                 **kwargs):
        """Initialize MoDGATv2 Layer.
        
        Args:
            units: Number of hidden units
            depth: Number of DGAT layers to stack
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate
            use_rms_norm: Whether to use RMS normalization
            rms_norm_args: Arguments for RMS normalization
        """
        super().__init__(**kwargs)
        self.units = units
        self.depth = depth
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.use_rms_norm = use_rms_norm
        
        # Default RMS normalization arguments
        if rms_norm_args is None:
            rms_norm_args = {"epsilon": 1e-6, "scale": True}
        self.rms_norm_args = rms_norm_args
        
        # Create multiple DGAT layers
        self.dgat_layers = []
        for i in range(depth):
            layer = DirectedGATLayer(
                units=units,
                attention_heads=attention_heads,
                dropout_rate=dropout_rate,
                use_rms_norm=use_rms_norm,
                rms_norm_args=rms_norm_args,
                name=f"dgat_layer_{i+1}"
            )
            self.dgat_layers.append(layer)
        
        # Multi-order attention fusion (like MoGATv2)
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
        """Forward pass through MoDGATv2.
        
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
        
        # Pass through each DGAT layer
        for i, dgat_layer in enumerate(self.dgat_layers):
            # Apply DGAT layer
            current_node_features = dgat_layer([
                current_node_features, edge_features, edge_indices, edge_indices_reverse
            ], training=training)
            
            # Store this order's representation
            multi_order_reprs.append(current_node_features)
            
            # Apply RMS normalization after each layer (except the last)
            if self.use_rms_norm and i < self.depth - 1:
                current_node_features = self.final_rms_norm(current_node_features)
        
        # Multi-order attention fusion (like MoGATv2)
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
            "attention_heads": self.attention_heads,
            "dropout_rate": self.dropout_rate,
            "use_rms_norm": self.use_rms_norm,
            "rms_norm_args": self.rms_norm_args
        })
        return config


class DirectedGATLayer(tf.keras.layers.Layer):
    """Single Directed GAT Layer using AttentiveHeadGAT.
    
    This implements the custom DGAT logic (not the paper version)
    using AttentiveHeadGAT for forward and backward attention.
    """
    
    def __init__(self, 
                 units=128,
                 attention_heads=8,
                 dropout_rate=0.1,
                 use_rms_norm=True,
                 rms_norm_args=None,
                 **kwargs):
        """Initialize Directed GAT Layer.
        
        Args:
            units: Number of hidden units
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate
            use_rms_norm: Whether to use RMS normalization
            rms_norm_args: Arguments for RMS normalization
        """
        super().__init__(**kwargs)
        self.units = units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.use_rms_norm = use_rms_norm
        
        # Default RMS normalization arguments
        if rms_norm_args is None:
            rms_norm_args = {"epsilon": 1e-6, "scale": True}
        self.rms_norm_args = rms_norm_args
        
        # Forward attention (original edge direction)
        self.forward_attention = AttentionHeadGAT(
            units=units,
            attention_heads=attention_heads,
            use_bias=True,
            name="forward_attention"
        )
        
        # Backward attention (reverse edge direction)
        self.backward_attention = AttentionHeadGAT(
            units=units,
            attention_heads=attention_heads,
            use_bias=True,
            name="backward_attention"
        )
        
        # Output projection
        self.output_projection = tf.keras.layers.Dense(
            units, activation=None, use_bias=True, name="output_projection"
        )
        
        # RMS normalization
        if use_rms_norm:
            self.rms_norm = RMSNormalization(**rms_norm_args)
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        """Forward pass through Directed GAT Layer.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices, edge_indices_reverse]
            training: Training mode flag
            
        Returns:
            Updated node representations with directed attention
        """
        node_features, edge_features, edge_indices, edge_indices_reverse = inputs
        
        # Forward attention (original direction)
        forward_out = self.forward_attention([
            node_features, edge_features, edge_indices
        ], training=training)
        
        # Backward attention (reverse direction)
        backward_out = self.backward_attention([
            node_features, edge_features, edge_indices_reverse
        ], training=training)
        
        # Combine forward and backward
        combined = forward_out + backward_out
        
        # Output projection
        output = self.output_projection(combined)
        
        # Residual connection
        if output.shape[-1] == node_features.shape[-1]:
            output = output + node_features
        
        # RMS normalization
        if self.use_rms_norm:
            output = self.rms_norm(output)
        
        # Dropout
        output = self.dropout(output, training=training)
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "depth": self.depth,
            "attention_heads": self.attention_heads,
            "dropout_rate": self.dropout_rate,
            "use_rms_norm": self.use_rms_norm,
            "rms_norm_args": self.rms_norm_args
        })
        return config


class PoolingNodesMoDGATv2(PoolingNodes):
    """Pooling layer specifically for MoDGATv2.
    
    This handles the pooling of node representations after
    multi-order directed attention processing.
    """
    
    def __init__(self, **kwargs):
        """Initialize MoDGATv2 pooling layer."""
        super().__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        """Pool node representations for MoDGATv2."""
        return super().call(inputs, **kwargs)
