import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyAdd, LazyConcatenate
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.ops.axis import get_axis
from kgcnn.ops.segment import segment_softmax

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras


class EGATLayer(GraphBaseLayer):
    r"""Edge-Guided Graph Attention (EGAT) layer.
    
    This layer explicitly incorporates edge features into attention computation,
    allowing bond types to directly guide attention weights.
    
    Reference: Edge Feature Guided Graph Attention Networks (2021, NeurIPS)
    """
    
    def __init__(self, units, use_bias=True, activation="relu",
                 attention_heads=8, attention_units=64, use_edge_features=True,
                 dropout_rate=0.1, **kwargs):
        """Initialize layer.
        
        Args:
            units (int): Number of hidden units.
            use_bias (bool): Whether to use bias.
            activation (str): Activation function.
            attention_heads (int): Number of attention heads.
            attention_units (int): Number of attention units.
            use_edge_features (bool): Whether to use edge features in attention.
            dropout_rate (float): Dropout rate.
        """
        super(EGATLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.attention_heads = attention_heads
        self.attention_units = attention_units
        self.use_edge_features = use_edge_features
        self.dropout_rate = dropout_rate
        
        # EGAT components
        self.node_transform = Dense(units, activation=activation, use_bias=use_bias)
        
        if use_edge_features:
            self.edge_transform = Dense(units, activation=activation, use_bias=use_bias)
        
        # Edge-guided attention
        self.attention = AttentionHeadGAT(
            units=attention_units,
            use_bias=use_bias,
            use_edge_features=use_edge_features,
            activation=activation
        )
        
        # Multi-head attention
        self.multi_head_attention = []
        for i in range(attention_heads):
            self.multi_head_attention.append(
                AttentionHeadGAT(
                    units=attention_units // attention_heads,
                    use_bias=use_bias,
                    use_edge_features=use_edge_features,
                    activation=activation
                )
            )
        
        # Output projection
        self.output_projection = Dense(units, activation=activation, use_bias=use_bias)
        
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
            
        # Message passing components
        self.gather_nodes = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges()
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate()
        
    def call(self, inputs, **kwargs):
        """Forward pass with edge-guided attention.
        
        Args:
            inputs: [node_attributes, edge_attributes, edge_indices]
            
        Returns:
            Updated node features
        """
        if len(inputs) == 3:
            node_attributes, edge_attributes, edge_indices = inputs
        else:
            node_attributes, edge_indices = inputs
            edge_attributes = None
        
        # Transform node features
        node_features = self.node_transform(node_attributes)
        
        # Transform edge features if provided
        if self.use_edge_features and edge_attributes is not None:
            edge_features = self.edge_transform(edge_attributes)
        else:
            edge_features = None
        
        # Gather neighbor features
        neighbor_features = self.gather_nodes([node_features, edge_indices])
        
        # Multi-head edge-guided attention
        attention_outputs = []
        for attention_head in self.multi_head_attention:
            if edge_features is not None:
                attended = attention_head([node_features, neighbor_features, edge_features, edge_indices])
            else:
                attended = attention_head([node_features, neighbor_features, edge_indices])
            attention_outputs.append(attended)
        
        # Concatenate attention heads
        if len(attention_outputs) > 1:
            multi_head_output = self.lazy_concat(attention_outputs)
        else:
            multi_head_output = attention_outputs[0]
        
        # Output projection
        output = self.output_projection(multi_head_output)
        
        # Skip connection
        output = self.lazy_add([node_attributes, output])
        
        # Apply dropout
        if self.dropout:
            output = self.dropout(output)
        
        return output


class PoolingNodesEGAT(GraphBaseLayer):
    """EGAT-specific pooling layer."""
    
    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodesEGAT, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
    def call(self, inputs, **kwargs):
        """Forward pass with EGAT pooling."""
        if self.pooling_method == "sum":
            return tf.reduce_sum(inputs, axis=1)
        elif self.pooling_method == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.pooling_method == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}") 