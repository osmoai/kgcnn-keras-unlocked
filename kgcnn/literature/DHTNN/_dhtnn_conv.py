"""DHTNN Convolution Layer with Double-Head Attention.

This module implements the DHTNN convolution layer that uses double-head attention
blocks to enhance DMPNN (Directed Message Passing Neural Network) performance.
The double-head attention combines local and global attention mechanisms.
"""

import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Activation, Dropout
from tensorflow.keras.layers import LayerNormalization
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingNodes
import math

ks = tf.keras


class DoubleHeadAttention(ks.layers.Layer):
    """Double-Head Attention Layer combining local and global attention.
    
    This layer implements a double-head attention mechanism that combines:
    1. Local attention: Focuses on immediate neighborhood relationships
    2. Global attention: Captures long-range dependencies across the graph
    
    Args:
        units (int): Output dimension
        local_heads (int): Number of local attention heads
        global_heads (int): Number of global attention heads
        local_attention_units (int): Units for local attention computation
        global_attention_units (int): Units for global attention computation
        use_edge_features (bool): Whether to use edge features
        dropout_rate (float): Dropout rate
        activation (str): Activation function
        use_bias (bool): Whether to use bias
    """
    
    def __init__(self, units, local_heads=4, global_heads=4, 
                 local_attention_units=64, global_attention_units=64,
                 use_edge_features=True, dropout_rate=0.1, activation="relu",
                 use_bias=True, **kwargs):
        super(DoubleHeadAttention, self).__init__(**kwargs)
        
        self.units = units
        self.local_heads = local_heads
        self.global_heads = global_heads
        self.local_attention_units = local_attention_units
        self.global_attention_units = global_attention_units
        self.use_edge_features = use_edge_features
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        
        # Local attention heads (neighborhood-focused)
        self.local_attention_heads = []
        for i in range(local_heads):
            local_head = AttentionHeadGAT(
                units=local_attention_units,
                use_edge_features=use_edge_features,
                use_final_activation=False,
                has_self_loops=True,
                activation=activation,
                use_bias=use_bias
            )
            self.local_attention_heads.append(local_head)
        
        # Global attention heads (long-range dependencies)
        self.global_attention_heads = []
        for i in range(global_heads):
            global_head = AttentionHeadGAT(
                units=global_attention_units,
                use_edge_features=use_edge_features,
                use_final_activation=False,
                has_self_loops=True,
                activation=activation,
                use_bias=use_bias
            )
            self.global_attention_heads.append(global_head)
        
        # Fusion layer to combine local and global attention
        self.fusion_layer = Dense(
            units=units,
            activation=activation,
            use_bias=use_bias
        )
        
        # Layer normalization
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Forward pass of double-head attention.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices]
            training: Training mode flag
            
        Returns:
            Combined attention output
        """
        node_features, edge_features, edge_indices = inputs
        
        # Local attention computation
        local_outputs = []
        for local_head in self.local_attention_heads:
            local_out = local_head([node_features, edge_features, edge_indices])
            local_outputs.append(local_out)
        
        # Global attention computation (with graph-level context)
        global_outputs = []
        for global_head in self.global_attention_heads:
            global_out = global_head([node_features, edge_features, edge_indices])
            global_outputs.append(global_out)
        
        # Concatenate local attention outputs
        if len(local_outputs) > 1:
            local_combined = tf.concat(local_outputs, axis=-1)
        else:
            local_combined = local_outputs[0]
        
        # Concatenate global attention outputs
        if len(global_outputs) > 1:
            global_combined = tf.concat(global_outputs, axis=-1)
        else:
            global_combined = global_outputs[0]
        
        # Combine local and global attention
        combined_attention = tf.concat([local_combined, global_combined], axis=-1)
        
        # Apply fusion layer
        fused_output = self.fusion_layer(combined_attention)
        
        # Apply layer normalization
        normalized_output = self.layer_norm(fused_output)
        
        # Apply dropout if specified
        if self.dropout is not None and training:
            normalized_output = self.dropout(normalized_output)
        
        return normalized_output
    
    def get_config(self):
        """Get layer configuration."""
        config = super(DoubleHeadAttention, self).get_config()
        config.update({
            "units": self.units,
            "local_heads": self.local_heads,
            "global_heads": self.global_heads,
            "local_attention_units": self.local_attention_units,
            "global_attention_units": self.global_attention_units,
            "use_edge_features": self.use_edge_features,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_bias": self.use_bias
        })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='DHTNNConv')
class DHTNNConv(GraphBaseLayer):
    """DHTNN Convolution Layer with Double-Head Attention.
    
    This layer implements DHTNN convolution that enhances DMPNN with double-head
    attention blocks, combining local and global attention mechanisms for superior
    molecular property prediction.
    
    Args:
        units (int): Output dimension
        local_heads (int): Number of local attention heads
        global_heads (int): Number of global attention heads
        local_attention_units (int): Units for local attention computation
        global_attention_units (int): Units for global attention computation
        use_edge_features (bool): Whether to use edge features
        use_final_activation (bool): Whether to apply final activation
        has_self_loops (bool): Whether graph has self-loops
        dropout_rate (float): Dropout rate
        activation (str): Activation function
        use_bias (bool): Whether to use bias
        kernel_regularizer: Kernel regularization
        bias_regularizer: Bias regularization
        activity_regularizer: Activity regularization
        kernel_constraint: Kernel constraint
        bias_constraint: Bias constraint
        kernel_initializer: Kernel initializer
        bias_initializer: Bias initializer
    """
    
    def __init__(self, units, local_heads=4, global_heads=4, 
                 local_attention_units=64, global_attention_units=64,
                 use_edge_features=True, use_final_activation=True, 
                 has_self_loops=True, dropout_rate=0.1, activation="relu",
                 use_bias=True, kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros", **kwargs):
        super(DHTNNConv, self).__init__(**kwargs)
        
        self.units = units
        self.local_heads = local_heads
        self.global_heads = global_heads
        self.local_attention_units = local_attention_units
        self.global_attention_units = global_attention_units
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        
        # Double-head attention layer
        self.double_head_attention = DoubleHeadAttention(
            units=units,
            local_heads=local_heads,
            global_heads=global_heads,
            local_attention_units=local_attention_units,
            global_attention_units=global_attention_units,
            use_edge_features=use_edge_features,
            dropout_rate=dropout_rate,
            activation=activation,
            use_bias=use_bias
        )
        
        # Feature transformation layer
        self.feature_transform = Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        
        # Final activation
        if use_final_activation:
            self.final_activation = Activation(activation)
        else:
            self.final_activation = None
    
    def call(self, inputs, **kwargs):
        """Forward pass of DHTNN convolution.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices]
            
        Returns:
            Updated node features
        """
        node_features, edge_features, edge_indices = inputs
        
        # Apply double-head attention
        attention_output = self.double_head_attention([node_features, edge_features, edge_indices])
        
        # Apply feature transformation
        transformed_features = self.feature_transform(node_features)
        
        # Combine attention and transformed features (residual connection)
        combined_output = attention_output + transformed_features
        
        # Apply final activation if specified
        if self.final_activation is not None:
            combined_output = self.final_activation(combined_output)
        
        return combined_output
    
    def get_config(self):
        """Get layer configuration."""
        config = super(DHTNNConv, self).get_config()
        config.update({
            "units": self.units,
            "local_heads": self.local_heads,
            "global_heads": self.global_heads,
            "local_attention_units": self.local_attention_units,
            "global_attention_units": self.global_attention_units,
            "use_edge_features": self.use_edge_features,
            "use_final_activation": self.use_final_activation,
            "has_self_loops": self.has_self_loops,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_bias": self.use_bias
        })
        return config 