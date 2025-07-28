"""KA-GAT Convolution Layer with Fourier-KAN.

This module implements the KA-GAT convolution layer that uses Fourier-KAN
(Fourier Kolmogorov-Arnold Networks) instead of traditional MLPs for
attention computation and feature transformation.
"""

import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Activation, Dropout
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.geom import PositionEncodingBasisLayer
import math

ks = tf.keras


class FourierKANLayer(ks.layers.Layer):
    """Fourier-KAN Layer: Enhanced Kolmogorov-Arnold Network with Fourier basis.
    
    This layer implements a Fourier-enhanced Kolmogorov-Arnold Network that uses
    Fourier basis functions for better approximation of complex functions.
    
    Args:
        units (int): Output dimension
        fourier_dim (int): Dimension of Fourier basis expansion
        activation (str): Activation function
        use_bias (bool): Whether to use bias
        dropout_rate (float): Dropout rate
        fourier_freq_min (float): Minimum frequency for Fourier expansion
        fourier_freq_max (float): Maximum frequency for Fourier expansion
    """
    
    def __init__(self, units, fourier_dim=32, activation="relu", use_bias=True,
                 dropout_rate=0.1, fourier_freq_min=1.0, fourier_freq_max=100.0, **kwargs):
        super(FourierKANLayer, self).__init__(**kwargs)
        self.units = units
        self.fourier_dim = fourier_dim
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.fourier_freq_min = fourier_freq_min
        self.fourier_freq_max = fourier_freq_max
        
        # Fourier basis layer
        self.fourier_basis = PositionEncodingBasisLayer(
            dim_half=fourier_dim//2,
            wave_length_min=fourier_freq_min,
            num_mult=fourier_freq_max/fourier_freq_min,
            include_frequencies=True,
            interleave_sin_cos=True
        )
        
        # Kolmogorov-Arnold structure: outer function
        self.outer_transform = Dense(
            units=units,
            activation=activation,
            use_bias=use_bias
        )
        
        # Inner functions (Kolmogorov-Arnold decomposition)
        self.inner_functions = []
        for i in range(2*units + 1):  # Kolmogorov-Arnold theorem requirement
            inner_func = Dense(
                units=fourier_dim,
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
        """Forward pass of Fourier-KAN layer.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Transformed tensor
        """
        # Expand input to Fourier basis
        fourier_expanded = self.fourier_basis(inputs)
        
        # Apply inner functions (Kolmogorov-Arnold decomposition)
        inner_outputs = []
        for inner_func in self.inner_functions:
            inner_out = inner_func(fourier_expanded)
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
        config = super(FourierKANLayer, self).get_config()
        config.update({
            "units": self.units,
            "fourier_dim": self.fourier_dim,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "fourier_freq_min": self.fourier_freq_min,
            "fourier_freq_max": self.fourier_freq_max
        })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='KAGATConv')
class KAGATConv(GraphBaseLayer):
    """KA-GAT Convolution Layer with Fourier-KAN attention.
    
    This layer implements Graph Attention Network convolution using Fourier-KAN
    for attention computation and feature transformation, providing enhanced
    expressiveness and performance.
    
    Args:
        units (int): Output dimension
        attention_heads (int): Number of attention heads
        attention_units (int): Units for attention computation
        use_edge_features (bool): Whether to use edge features
        use_final_activation (bool): Whether to apply final activation
        has_self_loops (bool): Whether graph has self-loops
        dropout_rate (float): Dropout rate
        fourier_dim (int): Dimension of Fourier basis expansion
        fourier_freq_min (float): Minimum frequency for Fourier expansion
        fourier_freq_max (float): Maximum frequency for Fourier expansion
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
    
    def __init__(self, units, attention_heads=8, attention_units=64, use_edge_features=True,
                 use_final_activation=True, has_self_loops=True, dropout_rate=0.1,
                 fourier_dim=32, fourier_freq_min=1.0, fourier_freq_max=100.0,
                 activation="relu", use_bias=True, kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros", **kwargs):
        super(KAGATConv, self).__init__(**kwargs)
        
        self.units = units
        self.attention_heads = attention_heads
        self.attention_units = attention_units
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.dropout_rate = dropout_rate
        self.fourier_dim = fourier_dim
        self.fourier_freq_min = fourier_freq_min
        self.fourier_freq_max = fourier_freq_max
        self.activation = activation
        self.use_bias = use_bias
        
        # Fourier-KAN attention layer
        self.fourier_attention = FourierKANLayer(
            units=attention_units,
            fourier_dim=fourier_dim,
            activation=activation,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            fourier_freq_min=fourier_freq_min,
            fourier_freq_max=fourier_freq_max
        )
        
        # Fourier-KAN feature transformation
        self.fourier_transform = FourierKANLayer(
            units=units,
            fourier_dim=fourier_dim,
            activation=activation,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            fourier_freq_min=fourier_freq_min,
            fourier_freq_max=fourier_freq_max
        )
        
        # Multi-head attention
        self.attention_heads = []
        for i in range(attention_heads):
            attention_head = AttentionHeadGAT(
                units=attention_units,
                use_edge_features=use_edge_features,
                use_final_activation=False,
                has_self_loops=has_self_loops,
                dropout_rate=dropout_rate,
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
        
        # Apply Fourier-KAN attention for each head
        attention_outputs = []
        for attention_head in self.attention_heads:
            # Use Fourier-KAN for attention computation
            attention_out = attention_head([node_features, edge_features, edge_indices])
            attention_outputs.append(attention_out)
        
        # Concatenate multi-head attention outputs
        if len(attention_outputs) > 1:
            multi_head_output = tf.concat(attention_outputs, axis=-1)
        else:
            multi_head_output = attention_outputs[0]
        
        # Apply Fourier-KAN transformation
        output = self.fourier_transform(multi_head_output)
        
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
            "fourier_dim": self.fourier_dim,
            "fourier_freq_min": self.fourier_freq_min,
            "fourier_freq_max": self.fourier_freq_max,
            "activation": self.activation,
            "use_bias": self.use_bias
        })
        return config 