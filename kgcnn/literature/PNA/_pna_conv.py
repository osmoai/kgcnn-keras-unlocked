import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyAdd, LazyConcatenate
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.mlp import GraphMLP
from kgcnn.ops.axis import get_axis

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras


class PNALayer(GraphBaseLayer):
    r"""Principal Neighborhood Aggregation (PNA) layer.
    
    This layer aggregates neighbor information using multiple aggregators (mean, max, min, std)
    with degree-scaling, capturing richer local structural information than GIN.
    
    Reference: Principal Neighbourhood Aggregation for Graph Nets (Corso et al., NeurIPS 2020)
    """
    
    def __init__(self, units, use_bias=True, activation="relu",
                 aggregators=["mean", "max", "min", "std"], scalers=["identity", "amplification", "attenuation"],
                 delta=1.0, dropout_rate=0.1, **kwargs):
        """Initialize layer.
        
        Args:
            units (int): Number of hidden units.
            use_bias (bool): Whether to use bias.
            activation (str): Activation function.
            aggregators (list): List of aggregator functions to use.
            scalers (list): List of degree scaling functions to use.
            delta (float): Delta parameter for degree scaling.
            dropout_rate (float): Dropout rate.
        """
        super(PNALayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.dropout_rate = dropout_rate
        
        # PNA components
        self.mlp = GraphMLP(
            units=[units, units],
            use_bias=use_bias,
            activation=activation,
            use_normalization=True,
            normalization_technique="graph_batch"
        )
        
        # Degree scaling parameters
        self.degree_embedding = Dense(units, activation=activation, use_bias=use_bias)
        
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
            
        # Message passing components
        self.gather_nodes = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges()
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate()
        
        # Initialize projection layer for skip connection (will be built when needed)
        self.node_projection = None
        
    def call(self, inputs, **kwargs):
        """Forward pass with PNA aggregation.
        
        Args:
            inputs: [node_attributes, edge_indices]
            
        Returns:
            Updated node features
        """
        node_attributes, edge_indices = inputs
        
        # Gather neighbor features using edge indices
        neighbor_features = self.gather_nodes([node_attributes, edge_indices])
        
        # Apply multiple aggregators
        aggregated_features = []
        for aggregator in self.aggregators:
            if aggregator == "mean":
                # Create dummy edge features (zeros) since PNA doesn't use edge features
                dummy_edge_features = tf.zeros_like(neighbor_features)
                agg_feat = self.aggregate_edges([neighbor_features, dummy_edge_features, edge_indices])
            elif aggregator == "max":
                # Use AggregateLocalEdges with max pooling to maintain shape consistency
                dummy_edge_features = tf.zeros_like(neighbor_features)
                agg_feat = AggregateLocalEdges(pooling_method="max")([neighbor_features, dummy_edge_features, edge_indices])
            elif aggregator == "min":
                # Use AggregateLocalEdges with min pooling to maintain shape consistency
                dummy_edge_features = tf.zeros_like(neighbor_features)
                agg_feat = AggregateLocalEdges(pooling_method="min")([neighbor_features, dummy_edge_features, edge_indices])
            elif aggregator == "std":
                # For std, we need to compute it manually since AggregateLocalEdges doesn't support std
                # First compute mean, then compute std
                dummy_edge_features = tf.zeros_like(neighbor_features)
                mean_feat = AggregateLocalEdges(pooling_method="mean")([neighbor_features, dummy_edge_features, edge_indices])
                # Compute squared differences and then mean
                # Use map_flat_values to avoid nested ragged tensors
                squared_diff = tf.ragged.map_flat_values(
                    lambda x: tf.square(x - mean_feat.flat_values), 
                    neighbor_features
                )
                dummy_edge_features_sq = tf.zeros_like(squared_diff)
                variance = AggregateLocalEdges(pooling_method="mean")([squared_diff, dummy_edge_features_sq, edge_indices])
                agg_feat = tf.sqrt(variance + 1e-8)  # Add small epsilon to avoid sqrt(0)
            else:
                raise ValueError(f"Unsupported aggregator: {aggregator}")
            
            aggregated_features.append(agg_feat)
        
        # Concatenate all aggregated features
        if len(aggregated_features) > 1:
            aggregated = self.lazy_concat(aggregated_features)
        else:
            aggregated = aggregated_features[0]
        
        # Apply degree scaling
        # Calculate node degrees
        num_nodes = tf.cast(tf.shape(node_attributes.flat_values)[0], tf.int32)
        node_degrees = tf.ragged.map_flat_values(
            lambda x, n=num_nodes: tf.math.bincount(tf.cast(x[:, 0], tf.int32), minlength=n),
            edge_indices
        )
        
        # Apply scalers
        scaled_features = []
        for scaler in self.scalers:
            if scaler == "identity":
                scaled = aggregated
            elif scaler == "amplification":
                # log(degree + 1) / log(delta + 1)
                degree_factor = tf.math.log(tf.cast(node_degrees, tf.float32) + 1.0) / tf.math.log(self.delta + 1.0)
                scaled = aggregated * tf.expand_dims(degree_factor, axis=-1)
            elif scaler == "attenuation":
                # log(degree + 1) / log(delta + 1) but inverted
                degree_factor = tf.math.log(self.delta + 1.0) / tf.math.log(tf.cast(node_degrees, tf.float32) + 1.0)
                scaled = aggregated * tf.expand_dims(degree_factor, axis=-1)
            else:
                raise ValueError(f"Unsupported scaler: {scaler}")
            
            scaled_features.append(scaled)
        
        # Concatenate scaled features
        if len(scaled_features) > 1:
            final_features = self.lazy_concat(scaled_features)
        else:
            final_features = scaled_features[0]
        
        # Apply MLP
        output = self.mlp(final_features)
        
        # Skip connection
        # Project node_attributes to match output shape if needed
        if node_attributes.shape[-1] != output.shape[-1]:
            # Create projection layer if not already created
            if self.node_projection is None:
                self.node_projection = Dense(output.shape[-1], use_bias=False)
            node_proj_out = self.node_projection(node_attributes)
            output = self.lazy_add([node_proj_out, output])
        else:
        output = self.lazy_add([node_attributes, output])
        
        # Apply dropout
        if self.dropout:
            output = self.dropout(output)
        
        return output


class PoolingNodesPNA(GraphBaseLayer):
    """PNA-specific pooling layer."""
    
    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodesPNA, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
    def call(self, inputs, **kwargs):
        """Forward pass with PNA pooling."""
        if self.pooling_method == "sum":
            return tf.reduce_sum(inputs, axis=1)
        elif self.pooling_method == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.pooling_method == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}") 