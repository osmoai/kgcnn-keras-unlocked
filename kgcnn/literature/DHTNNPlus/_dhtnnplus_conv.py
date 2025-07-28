"""DHTNNPlus Convolution Layer with Enhanced Double-Head Attention and Collaboration.

This module implements DHTNNPlusConv, which combines the best of DHTNNConv and CoAttentiveFP:
- Double-head attention (local + global)
- Collaboration mechanisms between node and edge features
- GRU updates for sequential processing
- Enhanced fusion strategies
"""

import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Activation, Dropout, LayerNormalization
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.update import GRUUpdate
import math

ks = tf.keras


class EnhancedDoubleHeadAttention(ks.layers.Layer):
    """Enhanced Double-Head Attention with Collaboration.
    
    This layer extends the original DoubleHeadAttention with collaboration mechanisms
    between node and edge features, inspired by CoAttentiveFP.
    
    Args:
        units (int): Output dimension
        local_heads (int): Number of local attention heads
        global_heads (int): Number of global attention heads
        local_attention_units (int): Units for local attention computation
        global_attention_units (int): Units for global attention computation
        use_edge_features (bool): Whether to use edge features
        use_collaboration (bool): Whether to use collaboration mechanisms
        collaboration_heads (int): Number of collaboration heads
        dropout_rate (float): Dropout rate
        activation (str): Activation function
        use_bias (bool): Whether to use bias
    """
    
    def __init__(self, units, local_heads=4, global_heads=4, 
                 local_attention_units=64, global_attention_units=64,
                 use_edge_features=True, use_collaboration=True, collaboration_heads=4,
                 dropout_rate=0.1, activation="relu", use_bias=True, **kwargs):
        super(EnhancedDoubleHeadAttention, self).__init__(**kwargs)
        
        self.units = units
        self.local_heads = local_heads
        self.global_heads = global_heads
        self.local_attention_units = local_attention_units
        self.global_attention_units = global_attention_units
        self.use_edge_features = use_edge_features
        self.use_collaboration = use_collaboration
        self.collaboration_heads = collaboration_heads
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
                dropout_rate=dropout_rate,
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
                dropout_rate=dropout_rate,
                activation=activation,
                use_bias=use_bias
            )
            self.global_attention_heads.append(global_head)
        
        # Collaboration mechanisms (inspired by CoAttentiveFP)
        if self.use_collaboration:
            # Node-specific attention for collaboration
            self.node_collaboration_heads = []
            for i in range(collaboration_heads):
                node_head = AttentionHeadGAT(
                    units=units // collaboration_heads,
                    use_edge_features=use_edge_features,
                    use_final_activation=False,
                    has_self_loops=True,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    use_bias=use_bias
                )
                self.node_collaboration_heads.append(node_head)
            
            # Edge-specific attention for collaboration
            self.edge_collaboration_heads = []
            for i in range(collaboration_heads):
                edge_head = AttentionHeadGAT(
                    units=units // collaboration_heads,
                    use_edge_features=use_edge_features,
                    use_final_activation=False,
                    has_self_loops=True,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    use_bias=use_bias
                )
                self.edge_collaboration_heads.append(edge_head)
            
            # Collaboration gate (learns optimal node/edge balance)
            self.collaboration_gate = Dense(units, activation="sigmoid", use_bias=use_bias)
            
            # Enhanced fusion layer
            self.collaboration_fusion = Dense(units, activation=activation, use_bias=use_bias)
        
        # Standard fusion layer for non-collaborative mode
        self.fusion_layer = Dense(units, activation=activation, use_bias=use_bias)
        
        # Layer normalization
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Forward pass of enhanced double-head attention with collaboration.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices]
            training: Training mode flag
            
        Returns:
            Enhanced attention output with collaboration
        """
        node_features, edge_features, edge_indices = inputs
        
        # Local attention computation
        local_outputs = []
        for local_head in self.local_attention_heads:
            local_out = local_head([node_features, edge_features, edge_indices])
            local_outputs.append(local_out)
        
        # Global attention computation
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
        
        if self.use_collaboration:
            # Collaboration mechanism (inspired by CoAttentiveFP)
            
            # Node collaboration attention
            node_collaboration_outputs = []
            for node_head in self.node_collaboration_heads:
                node_collab_out = node_head([node_features, edge_features, edge_indices])
                node_collaboration_outputs.append(node_collab_out)
            
            # Edge collaboration attention (using node features as context)
            edge_collaboration_outputs = []
            for edge_head in self.edge_collaboration_heads:
                edge_collab_out = edge_head([edge_features, node_features, edge_indices])
                edge_collaboration_outputs.append(edge_collab_out)
            
            # Concatenate collaboration outputs
            if len(node_collaboration_outputs) > 1:
                node_collab_combined = tf.concat(node_collaboration_outputs, axis=-1)
            else:
                node_collab_combined = node_collaboration_outputs[0]
            
            if len(edge_collaboration_outputs) > 1:
                edge_collab_combined = tf.concat(edge_collaboration_outputs, axis=-1)
            else:
                edge_collab_combined = edge_collaboration_outputs[0]
            
            # Collaboration gate (learns optimal balance)
            collaboration_weights = self.collaboration_gate(node_features)
            
            # Fuse node and edge collaboration with learned weights
            collaborative_features = (
                collaboration_weights * node_collab_combined + 
                (1 - collaboration_weights) * edge_collab_combined
            )
            
            # Enhanced fusion: combine attention and collaboration
            enhanced_combined = tf.concat([combined_attention, collaborative_features], axis=-1)
            fused_output = self.collaboration_fusion(enhanced_combined)
            
        else:
            # Standard fusion (no collaboration)
            fused_output = self.fusion_layer(combined_attention)
        
        # Apply layer normalization
        normalized_output = self.layer_norm(fused_output)
        
        # Apply dropout if specified
        if self.dropout is not None and training:
            normalized_output = self.dropout(normalized_output)
        
        return normalized_output
    
    def get_config(self):
        """Get layer configuration."""
        config = super(EnhancedDoubleHeadAttention, self).get_config()
        config.update({
            "units": self.units,
            "local_heads": self.local_heads,
            "global_heads": self.global_heads,
            "local_attention_units": self.local_attention_units,
            "global_attention_units": self.global_attention_units,
            "use_edge_features": self.use_edge_features,
            "use_collaboration": self.use_collaboration,
            "collaboration_heads": self.collaboration_heads,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_bias": self.use_bias
        })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='DHTNNPlusConv')
class DHTNNPlusConv(GraphBaseLayer):
    """DHTNNPlus Convolution Layer: Enhanced DHTNNConv with Collaboration and GRU Updates.
    
    This layer combines the best of DHTNNConv and CoAttentiveFP:
    - Double-head attention (local + global)
    - Collaboration mechanisms between node and edge features
    - GRU updates for sequential processing
    - Enhanced fusion strategies
    
    Args:
        units (int): Output dimension
        local_heads (int): Number of local attention heads
        global_heads (int): Number of global attention heads
        local_attention_units (int): Units for local attention computation
        global_attention_units (int): Units for global attention computation
        use_edge_features (bool): Whether to use edge features
        use_collaboration (bool): Whether to use collaboration mechanisms
        collaboration_heads (int): Number of collaboration heads
        use_gru_updates (bool): Whether to use GRU updates
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
                 use_edge_features=True, use_collaboration=True, collaboration_heads=4,
                 use_gru_updates=True, use_final_activation=True, 
                 has_self_loops=True, dropout_rate=0.1, activation="relu",
                 use_bias=True, kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros", **kwargs):
        super(DHTNNPlusConv, self).__init__(**kwargs)
        
        self.units = units
        self.local_heads = local_heads
        self.global_heads = global_heads
        self.local_attention_units = local_attention_units
        self.global_attention_units = global_attention_units
        self.use_edge_features = use_edge_features
        self.use_collaboration = use_collaboration
        self.collaboration_heads = collaboration_heads
        self.use_gru_updates = use_gru_updates
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        
        # Enhanced double-head attention with collaboration
        self.enhanced_attention = EnhancedDoubleHeadAttention(
            units=units,
            local_heads=local_heads,
            global_heads=global_heads,
            local_attention_units=local_attention_units,
            global_attention_units=global_attention_units,
            use_edge_features=use_edge_features,
            use_collaboration=use_collaboration,
            collaboration_heads=collaboration_heads,
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
        
        # GRU updates for sequential processing (inspired by CoAttentiveFP)
        if use_gru_updates:
            self.gru_update = GRUUpdate(units)
        else:
            self.gru_update = None
        
        # Final activation
        if use_final_activation:
            self.final_activation = Activation(activation)
        else:
            self.final_activation = None
    
    def call(self, inputs, **kwargs):
        """Forward pass of DHTNNPlus convolution.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices]
            
        Returns:
            Updated node features with enhanced attention and collaboration
        """
        node_features, edge_features, edge_indices = inputs
        
        # Apply enhanced double-head attention with collaboration
        attention_output = self.enhanced_attention([node_features, edge_features, edge_indices])
        
        # Apply feature transformation
        transformed_features = self.feature_transform(node_features)
        
        # Combine attention and transformed features (residual connection)
        combined_output = attention_output + transformed_features
        
        # Apply GRU updates if enabled (sequential processing)
        if self.gru_update is not None:
            combined_output = self.gru_update(combined_output)
        
        # Apply final activation if specified
        if self.final_activation is not None:
            combined_output = self.final_activation(combined_output)
        
        return combined_output
    
    def get_config(self):
        """Get layer configuration."""
        config = super(DHTNNPlusConv, self).get_config()
        config.update({
            "units": self.units,
            "local_heads": self.local_heads,
            "global_heads": self.global_heads,
            "local_attention_units": self.local_attention_units,
            "global_attention_units": self.global_attention_units,
            "use_edge_features": self.use_edge_features,
            "use_collaboration": self.use_collaboration,
            "collaboration_heads": self.collaboration_heads,
            "use_gru_updates": self.use_gru_updates,
            "use_final_activation": self.use_final_activation,
            "has_self_loops": self.has_self_loops,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_bias": self.use_bias
        })
        return config 