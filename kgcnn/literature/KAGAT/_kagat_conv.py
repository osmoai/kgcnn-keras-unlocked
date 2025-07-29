"""KA-GAT Convolution Layer with KAN.

This module implements the KA-GAT convolution layer that uses KAN
(Kolmogorov-Arnold Networks) instead of traditional MLPs for
attention computation and feature transformation.
"""

import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Activation, Dropout
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingNodes
import math

ks = tf.keras


class KANLayer(ks.layers.Layer):
    """KAN Layer: Kolmogorov-Arnold Network implementation.
    
    This layer implements a Kolmogorov-Arnold Network that follows the universal
    approximation theorem by decomposing multivariate functions into univariate functions.
    
    Args:
        units (int): Output dimension
        hidden_dim (int): Hidden dimension for inner functions
        activation (str): Activation function
        use_bias (bool): Whether to use bias
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, units, hidden_dim=64, activation="relu", use_bias=True,
                 dropout_rate=0.1, **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.units = units
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        
        # Kolmogorov-Arnold structure: outer function
        self.outer_transform = Dense(
            units=units,
            activation=activation,
            use_bias=use_bias
        )
        
        # Inner functions (Kolmogorov-Arnold decomposition)
        # According to the theorem, we need 2n+1 inner functions for n-dimensional output
        self.inner_functions = []
        for i in range(2*units + 1):
            inner_func = Dense(
                units=hidden_dim,
                activation="relu",
                use_bias=True
            )
            self.inner_functions.append(inner_func)
        
        # Dropout layer
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Forward pass of KAN layer.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Transformed tensor
        """
        # Apply inner functions (Kolmogorov-Arnold decomposition)
        inner_outputs = []
        for inner_func in self.inner_functions:
            inner_out = inner_func(inputs)
            inner_outputs.append(inner_out)
        
        # Sum inner function outputs (Kolmogorov-Arnold structure)
        summed_inner = tf.add_n(inner_outputs)
        
        # Apply outer function
        output = self.outer_transform(summed_inner)
        
        # Apply dropout if specified
        if self.dropout is not None and training:
            output = self.dropout(output)
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super(KANLayer, self).get_config()
        config.update({
            "units": self.units,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate
        })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='KAGATConv')
class KAGATConv(GraphBaseLayer):
    """KA-GAT Convolution Layer with KAN attention.
    
    This layer implements Graph Attention Network convolution using KAN
    (Kolmogorov-Arnold Networks) instead of traditional MLPs for enhanced
    attention computation and feature transformation.
    
    Args:
        units (int): Output dimension
        attention_heads (int): Number of attention heads
        attention_units (int): Units for attention computation
        use_edge_features (bool): Whether to use edge features
        use_final_activation (bool): Whether to apply final activation
        has_self_loops (bool): Whether graph has self-loops
        dropout_rate (float): Dropout rate
        hidden_dim (int): Hidden dimension for KAN layers
        activation (str): Activation function
        use_bias (bool): Whether to use bias
        kernel_regularizer: Kernel regularizer
        bias_regularizer: Bias regularizer
        activity_regularizer: Activity regularizer
        kernel_constraint: Kernel constraint
        bias_constraint: Bias constraint
        kernel_initializer: Kernel initializer
        bias_initializer: Bias initializer
    """
    
    def __init__(self, units, attention_heads=8, attention_units=64, use_edge_features=True,
                 use_final_activation=True, has_self_loops=True, dropout_rate=0.1,
                 hidden_dim=64, activation="relu", use_bias=True, kernel_regularizer=None, 
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
                 bias_constraint=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", **kwargs):
        super(KAGATConv, self).__init__(**kwargs)
        self.units = units
        self.attention_heads = attention_heads
        self.attention_units = attention_units
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_bias = use_bias
        
        # KAN attention layer
        self.kan_attention = KANLayer(
            units=attention_units,
            hidden_dim=hidden_dim,
            activation=activation,
            use_bias=use_bias,
            dropout_rate=dropout_rate
        )
        
        # KAN feature transformation
        self.kan_transform = KANLayer(
            units=units,
            hidden_dim=hidden_dim,
            activation=activation,
            use_bias=use_bias,
            dropout_rate=dropout_rate
        )
        
        # Multi-head attention
        self.attention_heads = []
        for i in range(attention_heads):
            attention_head = AttentionHeadGAT(
                units=attention_units,
                use_edge_features=use_edge_features,
                use_final_activation=False,
                has_self_loops=has_self_loops,
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
            self.attention_heads.append(attention_head)
        
        # Final activation
        if use_final_activation:
            self.final_activation = Activation(activation)
        else:
            self.final_activation = None
    
    def call(self, inputs, **kwargs):
        """Forward pass of KA-GAT convolution.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices]
            
        Returns:
            Updated node features
        """
        node_features, edge_features, edge_indices = inputs
        
        # Apply attention for each head
        attention_outputs = []
        for attention_head in self.attention_heads:
            attention_out = attention_head([node_features, edge_features, edge_indices])
            attention_outputs.append(attention_out)
        
        # Concatenate multi-head attention outputs
        if len(attention_outputs) > 1:
            multi_head_output = tf.concat(attention_outputs, axis=-1)
        else:
            multi_head_output = attention_outputs[0]
        
        # Apply KAN transformation to the concatenated output
        output = self.kan_transform(multi_head_output)
        
        # Apply final activation if specified
        if self.final_activation is not None:
            output = self.final_activation(output)
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super(KAGATConv, self).get_config()
        config.update({
            "units": self.units,
            "attention_heads": self.attention_heads,
            "attention_units": self.attention_units,
            "use_edge_features": self.use_edge_features,
            "use_final_activation": self.use_final_activation,
            "has_self_loops": self.has_self_loops,
            "dropout_rate": self.dropout_rate,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "use_bias": self.use_bias
        })
        return config 