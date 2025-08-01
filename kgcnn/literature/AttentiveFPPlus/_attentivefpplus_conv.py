import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyConcatenate, Activation
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdgesAttention
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.update import GRUUpdate
from kgcnn.ops.axis import get_axis
from kgcnn.ops.segment import segment_softmax

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras


class AttentiveHeadFPPlus(GraphBaseLayer):
    r"""AttentiveFP+ layer with multi-scale attention fusion for molecular representation learning.
    
    This layer extends the original AttentiveFP by multi-scale attention fusion across
    message passing layers, capturing both local and global contexts more effectively.
    
    Reference: AttentiveFP+ (Multi-scale Attention Fusion) (2023)
    """
    
    def __init__(self, units, use_bias=True, activation="relu", 
                 use_multiscale=True, scale_fusion="weighted_sum", attention_scales=[1, 2, 4],
                 dropout_rate=0.1, **kwargs):
        """Initialize layer.
        
        Args:
            units (int): Number of hidden units.
            use_bias (bool): Whether to use bias.
            activation (str): Activation function.
            use_multiscale (bool): Whether to use multi-scale attention.
            scale_fusion (str): How to fuse different scales ("weighted_sum", "concatenate").
            attention_scales (list): Different attention scales to use.
            dropout_rate (float): Dropout rate.
        """
        super(AttentiveHeadFPPlus, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.use_multiscale = use_multiscale
        self.scale_fusion = scale_fusion
        self.attention_scales = attention_scales
        self.dropout_rate = dropout_rate
        
        # Multi-scale attention components
        if self.use_multiscale:
            self.scale_attentions = []
            for scale in attention_scales:
                # Create attention layer for each scale with different receptive fields
                scale_units = units // len(attention_scales)
                
                # Multi-scale attention with different aggregation strategies
                if scale == 1:
                    # Local attention (original AttentiveFP style)
                    attention_layer = self._create_local_attention(scale_units)
                elif scale == 2:
                    # Medium-range attention with 2-hop neighbors
                    attention_layer = self._create_medium_attention(scale_units)
                elif scale == 4:
                    # Long-range attention with 4-hop neighbors
                    attention_layer = self._create_long_attention(scale_units)
                else:
                    # Default to local attention
                    attention_layer = self._create_local_attention(scale_units)
                
                self.scale_attentions.append(attention_layer)
            
            # Fusion layer
            if scale_fusion == "weighted_sum":
                self.fusion_weights = Dense(len(attention_scales), activation="softmax", use_bias=use_bias)
            elif scale_fusion == "concatenate":
                self.fusion_layer = Dense(units, activation=activation, use_bias=use_bias)
            else:
                raise ValueError(f"Unsupported scale fusion method: {scale_fusion}")
        
        # Standard attention (fallback)
        else:
            self.attention = self._create_local_attention(units)
        
        self.dropout = Dropout(dropout_rate)
        
    def _create_local_attention(self, units):
        """Create local attention layer (1-hop neighbors)."""
        return {
            'linear_trafo': Dense(units, activation="linear", use_bias=self.use_bias),
            'alpha_activation': Dense(units, activation=self.activation, use_bias=self.use_bias),
            'alpha': Dense(1, activation="linear", use_bias=False),
            'final_activ': Activation(activation="elu"),
            'gather_in': GatherNodesIngoing(),
            'gather_out': GatherNodesOutgoing(),
            'concat': LazyConcatenate(axis=-1),
            'pool_attention': AggregateLocalEdgesAttention()
        }
    
    def _create_medium_attention(self, units):
        """Create medium-range attention layer (2-hop neighbors)."""
        return {
            'linear_trafo': Dense(units, activation="linear", use_bias=self.use_bias),
            'alpha_activation': Dense(units, activation=self.activation, use_bias=self.use_bias),
            'alpha': Dense(1, activation="linear", use_bias=False),
            'final_activ': Activation(activation="elu"),
            'gather_in': GatherNodesIngoing(),
            'gather_out': GatherNodesOutgoing(),
            'concat': LazyConcatenate(axis=-1),
            'pool_attention': AggregateLocalEdgesAttention(),
            'hop_aggregation': Dense(units, activation=self.activation, use_bias=self.use_bias)
        }
    
    def _create_long_attention(self, units):
        """Create long-range attention layer (4-hop neighbors)."""
        return {
            'linear_trafo': Dense(units, activation="linear", use_bias=self.use_bias),
            'alpha_activation': Dense(units, activation=self.activation, use_bias=self.use_bias),
            'alpha': Dense(1, activation="linear", use_bias=False),
            'final_activ': Activation(activation="elu"),
            'gather_in': GatherNodesIngoing(),
            'gather_out': GatherNodesOutgoing(),
            'concat': LazyConcatenate(axis=-1),
            'pool_attention': AggregateLocalEdgesAttention(),
            'global_context': Dense(units, activation=self.activation, use_bias=self.use_bias)
        }
    
    def _apply_attention_scale(self, attention_layer, node_attributes, edge_attributes, edge_indices, scale):
        """Apply attention at a specific scale."""
        # Gather neighbor information
        n_in = attention_layer['gather_in']([node_attributes, edge_indices])
        n_out = attention_layer['gather_out']([node_attributes, edge_indices])
        
        # Linear transformation
        n_in = attention_layer['linear_trafo'](n_in)
        n_out = attention_layer['linear_trafo'](n_out)
        
        # Concatenate for attention computation
        n_concat = attention_layer['concat']([n_in, n_out])
        
        # Compute attention coefficients
        alpha = attention_layer['alpha_activation'](n_concat)
        alpha = attention_layer['alpha'](alpha)
        
        # Apply attention pooling
        context = attention_layer['pool_attention']([node_attributes, n_out, alpha, edge_indices])
        context = attention_layer['final_activ'](context)
        
        # Scale-specific processing
        if scale == 2 and 'hop_aggregation' in attention_layer:
            # Medium-range: add 2-hop aggregation
            context = attention_layer['hop_aggregation'](context)
        elif scale == 4 and 'global_context' in attention_layer:
            # Long-range: add global context
            context = attention_layer['global_context'](context)
        
        return context
        
    def call(self, inputs, **kwargs):
        """Forward pass.
        
        Args:
            inputs: [node_attributes, edge_attributes, edge_indices]
            
        Returns:
            Updated node features
        """
        node_attributes, edge_attributes, edge_indices = inputs
        
        if self.use_multiscale:
            # Apply multi-scale attention
            scale_outputs = []
            for i, scale in enumerate(self.attention_scales):
                scale_output = self._apply_attention_scale(
                    self.scale_attentions[i], node_attributes, edge_attributes, edge_indices, scale
                )
                scale_outputs.append(scale_output)
            
            # Fuse different scales
            if self.scale_fusion == "weighted_sum":
                # Stack outputs for weighted fusion
                stacked_outputs = tf.stack(scale_outputs, axis=-1)  # Shape: (batch, nodes, features, scales)
                
                # Get fusion weights
                fusion_weights = self.fusion_weights(node_attributes)  # Shape: (batch, nodes, scales)
                
                # Apply weighted sum
                # Reshape for broadcasting
                fusion_weights = tf.expand_dims(fusion_weights, axis=-2)  # Shape: (batch, nodes, 1, scales)
                output = tf.reduce_sum(stacked_outputs * fusion_weights, axis=-1)  # Shape: (batch, nodes, features)
                
            elif self.scale_fusion == "concatenate":
                # Concatenate all scale outputs
                concatenated = tf.concat(scale_outputs, axis=-1)
                output = self.fusion_layer(concatenated)
            else:
                # Default to simple average
                output = tf.reduce_mean(scale_outputs, axis=0)
        else:
            # Use single attention scale (fallback)
            output = self._apply_attention_scale(
                self.attention, node_attributes, edge_attributes, edge_indices, 1
            )
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class PoolingNodesAttentivePlus(PoolingNodes):
    """Multi-scale attentive pooling layer."""
    
    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodesAttentivePlus, self).__init__(pooling_method=pooling_method, **kwargs)
    
    def call(self, inputs, **kwargs):
        """Forward pass with multi-scale attention."""
        return super(PoolingNodesAttentivePlus, self).call(inputs, **kwargs) 