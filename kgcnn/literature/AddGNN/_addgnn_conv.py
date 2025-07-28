"""Add-GNN convolution layer with additive attention mechanism.

Reference:
    Zhou, R., Zhang, Y., He, K., & Liu, H. (2025). Add-GNN: A Dual-Representation Fusion Molecular Property Prediction 
    Based on Graph Neural Networks with Additive Attention. Symmetry, 17(6), 873.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.aggr import AggregateLocalEdges


class AddGNNConv(GraphBaseLayer):
    """Add-GNN convolution layer with additive attention mechanism.
    
    This layer implements the Add-GNN message passing mechanism as described in the paper:
    - Additive attention mechanism (Equations 4-6)
    - Message aggregation (Equation 7)
    - Node update with ReLU activation (Equation 8)
    
    Args:
        units (int): Number of output units/features
        heads (int): Number of attention heads
        activation (str): Activation function for the layer
        use_bias (bool): Whether to use bias in linear transformations
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias weights
        activity_regularizer: Regularizer for activity
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias weights
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias weights
    """
    
    def __init__(self,
                 units,
                 heads=1,
                 activation="relu",
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 **kwargs):
        super(AddGNNConv, self).__init__(**kwargs)
        
        self.units = units
        self.heads = heads
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        # Attention weight matrix W ∈ R^(heads × d) (Equation 4)
        self.attention_weights = None
        self.attention_bias = None
        
        # MLP for node feature transformation
        self.node_mlp = None
        
        # Edge feature transformation layer
        self.edge_dense = None
        
    def build(self, input_shape):
        """Build the layer weights."""
        super(AddGNNConv, self).build(input_shape)
        
        # Get input dimensions
        node_shape = input_shape[0]
        edge_shape = input_shape[1]
        
        # Attention weight matrix W ∈ R^(heads × d) (Equation 4)
        self.attention_weights = self.add_weight(
            shape=(self.heads, self.units),
            initializer=self.kernel_initializer,
            name="attention_weights",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        
        if self.use_bias:
            self.attention_bias = self.add_weight(
                shape=(self.heads,),
                initializer=self.bias_initializer,
                name="attention_bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        
        # MLP for node feature transformation (used in readout)
        self.node_mlp = DenseEmbedding(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer
        )
        
        # Edge feature transformation layer
        self.edge_dense = DenseEmbedding(
            units=self.units,
            activation="linear",
            use_bias=False,
            kernel_initializer=self.kernel_initializer
        )
        
    def call(self, inputs, **kwargs):
        """Forward pass of Add-GNN convolution.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices]
                - node_features: Node feature matrix [N, F] (ragged)
                - edge_features: Edge feature matrix [E, F_e] (ragged)
                - edge_indices: Edge index matrix [E, 2] (ragged)
                
        Returns:
            Updated node features [N, units] (ragged)
        """
        node_features, edge_features, edge_indices = inputs
        
        # Import required layers
        from kgcnn.layers.gather import GatherNodesOutgoing
        from kgcnn.layers.aggr import AggregateLocalEdges
        from kgcnn.layers.modules import LazyAdd, DenseEmbedding
        
        # Gather target node features (outgoing edges)
        target_features = GatherNodesOutgoing()([node_features, edge_indices])
        
        # Transform edge features to match node feature dimensions
        edge_features_transformed = self.edge_dense(edge_features)
        
        # Equation 4: α_vw = softmax(W · tanh(h_w + e_vw))
        # Add neighbor features and edge features
        neighbor_edge_sum = LazyAdd()([target_features, edge_features_transformed])  # [E, F]
        
        # Apply tanh activation
        tanh_output = tf.tanh(neighbor_edge_sum)  # [E, F]
        
        # Apply attention weights W ∈ R^(heads × d)
        # For ragged tensors, we need to work with the values
        tanh_values = tanh_output.values if hasattr(tanh_output, 'values') else tanh_output
        
        # Apply attention weights to the values
        # Reshape for multi-head attention
        tanh_reshaped = tf.expand_dims(tanh_values, axis=1)  # [E, 1, F]
        attention_weights_reshaped = tf.expand_dims(self.attention_weights, axis=0)  # [1, heads, F]
        
        # Compute attention scores
        attention_scores = tf.reduce_sum(
            attention_weights_reshaped * tanh_reshaped, axis=-1
        )  # [E, heads]
        
        if self.use_bias:
            attention_scores += self.attention_bias
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=0)  # [E, heads]
        
        # Equation 5: value_vw = max(h_w, e_vw)
        # Element-wise maximum between neighbor features and edge features
        value_vw = tf.maximum(target_features, edge_features_transformed)  # [E, F]
        value_values = value_vw.values if hasattr(value_vw, 'values') else value_vw
        
        # Equation 6: att_out_vw = (value_vw)' ⊙ α'_vw
        # Reshape for multi-head attention
        value_reshaped = tf.expand_dims(value_values, axis=1)  # [E, 1, F]
        attention_weights_reshaped = tf.expand_dims(attention_weights, axis=-1)  # [E, heads, 1]
        
        # Element-wise multiplication
        att_out = value_reshaped * attention_weights_reshaped  # [E, heads, F]
        
        # Average across heads if multiple heads
        if self.heads > 1:
            att_out = tf.reduce_mean(att_out, axis=1)  # [E, F]
        else:
            att_out = tf.squeeze(att_out, axis=1)  # [E, F]
        
        # Convert back to ragged tensor if needed
        if hasattr(value_vw, 'row_splits'):
            att_out = tf.RaggedTensor.from_row_splits(att_out, value_vw.row_splits)
        
        # Equation 7: m^(t+1)_v = Σ_(w∈N(v)) att_out^t_vw
        # Aggregate messages from neighbors
        aggregated_messages = AggregateLocalEdges(
            pooling_method="sum"
        )([node_features, att_out, edge_indices])
        
        # Equation 8: h^(t+1)_v = Relu(m^(t+1)_v)
        # Apply ReLU activation to get updated node features
        updated_features = tf.nn.relu(aggregated_messages)
        
        return updated_features
    
    def get_config(self):
        """Get layer configuration."""
        config = super(AddGNNConv, self).get_config()
        config.update({
            "units": self.units,
            "heads": self.heads,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer
        })
        return config 