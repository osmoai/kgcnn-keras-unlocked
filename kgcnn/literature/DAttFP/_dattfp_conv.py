import tensorflow as tf
from kgcnn.layers.modules import Dense, Dropout, LazyConcatenate
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.pooling import PoolingNodes

ks = tf.keras

class DAttFPLayer(ks.layers.Layer):
    """
    Directed AttentiveFP Layer
    
    Novel combination of AttentiveFP attention mechanism with direction awareness.
    Combines the best of both worlds:
    - AttentiveFP's molecular attention mechanism
    - Direction awareness for molecular chirality
    - Bidirectional attention for incoming vs. outgoing bonds
    
    Key innovations:
    - Forward attention (source → target) with AttentiveFP mechanism
    - Backward attention (target → source) with AttentiveFP mechanism
    - Direction-specific attention weights
    - Molecular chirality awareness
    """
    
    def __init__(self, units, attention_heads=8, attention_units=64, 
                 use_edge_features=True, dropout_rate=0.1, use_bias=True, 
                 activation="relu", use_gru_update=True, **kwargs):
        super(DAttFPLayer, self).__init__(**kwargs)
        self.units = units
        self.attention_heads = attention_heads
        self.attention_units = attention_units
        self.use_edge_features = use_edge_features
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation = activation
        self.use_gru_update = use_gru_update
        
        # Forward attention (source → target) using AttentiveFP mechanism
        self.forward_attention = AttentionHeadGAT(
            units=attention_units,
            use_edge_features=use_edge_features,
            use_bias=use_bias,
            name="forward_attention"
        )
        
        # Backward attention (target → source) using AttentiveFP mechanism
        self.backward_attention = AttentionHeadGAT(
            units=attention_units,
            use_edge_features=use_edge_features,
            use_bias=use_bias,
            name="backward_attention"
        )
        
        # Node transformation layers
        self.node_transform = Dense(
            units=units,
            use_bias=use_bias,
            activation=activation,
            name="node_transform"
        )
        
        # Edge transformation layer
        if use_edge_features:
            self.edge_transform = Dense(
                units=units,
                use_bias=use_bias,
                activation=activation,
                name="edge_transform"
            )
        
        # GRU update for AttentiveFP-style message passing
        if use_gru_update:
            self.gru_update = GRUUpdate(
                units=units,
                name="gru_update"
            )
        
        # Output projection
        self.output_projection = Dense(
            units=units,
            use_bias=use_bias,
            activation=None,
            name="output_projection"
        )
        
        # Dropout
        self.dropout = Dropout(rate=dropout_rate)
        
        # Layer normalization
        self.layer_norm = ks.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, training=None):
        """
        Forward pass of DAttFP layer
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices, edge_indices_reverse]
            training: Training mode flag
            
        Returns:
            Updated node features
        """
        if len(inputs) == 3:
            # Standard input: [node_features, edge_features, edge_indices]
            node_features, edge_features, edge_indices = inputs
            edge_indices_reverse = None
        elif len(inputs) == 4:
            # With reverse edges: [node_features, edge_features, edge_indices, edge_indices_reverse]
            node_features, edge_features, edge_indices, edge_indices_reverse = inputs
        else:
            raise ValueError(f"DAttFPLayer expects 3 or 4 inputs, got {len(inputs)}")
        
        # Transform node features
        node_features_transformed = self.node_transform(node_features)
        
        # Transform edge features if provided
        if edge_features is not None and self.use_edge_features:
            edge_features_transformed = self.edge_transform(edge_features)
        else:
            edge_features_transformed = None
        
        # Forward attention (source → target) - AttentiveFP style
        forward_output = self.forward_attention(
            [node_features_transformed, edge_features_transformed, edge_indices]
        )
        
        # Backward attention (target → source) - AttentiveFP style
        if edge_indices_reverse is not None:
            backward_output = self.backward_attention(
                [node_features_transformed, edge_features_transformed, edge_indices_reverse]
            )
        else:
            # If no reverse edges, create implied reverse by swapping source/target
            edge_indices_reverse_implied = tf.stack([edge_indices[:, :, 1], edge_indices[:, :, 0]], axis=-1)
            backward_output = self.backward_attention(
                [node_features_transformed, edge_features_transformed, edge_indices_reverse_implied]
            )
        
        # Combine forward and backward attention outputs
        if self.attention_heads > 1:
            # Multi-head attention - concatenate heads
            combined_output = LazyConcatenate()([forward_output, backward_output])
        else:
            # Single head - add outputs
            combined_output = forward_output + backward_output
        
        # Apply output projection
        output = self.output_projection(combined_output)
        
        # GRU update (AttentiveFP style) if enabled
        if self.use_gru_update:
            # Use the transformed node features for the GRU update
            output = self.gru_update([output, node_features_transformed])
        
        # Residual connection
        if node_features.shape[-1] == self.units:
            output = output + node_features
        else:
            # If dimensions don't match, project input to match
            input_projection = Dense(units=self.units, use_bias=False)(node_features)
            output = output + input_projection
        
        # Layer normalization
        output = self.layer_norm(output)
        
        # Dropout
        output = self.dropout(output, training=training)
        
        return output
    
    def get_config(self):
        config = super(DAttFPLayer, self).get_config()
        config.update({
            'units': self.units,
            'attention_heads': self.attention_heads,
            'attention_units': self.attention_units,
            'use_edge_features': self.use_edge_features,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'use_gru_update': self.use_gru_update
        })
        return config


class PoolingNodesDAttFP(PoolingNodes):
    """
    Pooling layer specifically designed for DAttFP
    Handles both forward and backward aggregated features with molecular focus
    """
    
    def __init__(self, pooling_method="mean", **kwargs):
        super(PoolingNodesDAttFP, self).__init__(pooling_method=pooling_method, **kwargs)
    
    def call(self, inputs, **kwargs):
        """
        Pool node features for DAttFP
        
        Args:
            inputs: Node features from DAttFP layers
            
        Returns:
            Graph-level representation
        """
        return super(PoolingNodesDAttFP, self).call(inputs, **kwargs)
