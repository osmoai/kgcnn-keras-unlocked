"""Contrastive PNA convolution layer.

This module implements contrastive learning with PNA (Principal Neighbourhood Aggregation) architecture,
combining powerful aggregation functions with contrastive learning capabilities.
"""

import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.modules import OptionalInputEmbedding
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from kgcnn.layers.norm import GraphBatchNormalization
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate


class ContrastivePNALayer(tf.keras.layers.Layer):
    """
    A single PNA layer for processing graph views.
    
    Args:
        units (int): Number of units
        use_bias (bool): Whether to use bias
        activation (str): Activation function
        aggregators (list): List of aggregation functions
        scalers (list): List of scaling functions
        delta (float): Delta parameter for scaling
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, units=128, use_bias=True, activation="relu", 
                 aggregators=["mean", "max", "sum"], scalers=["identity", "amplification", "attenuation"],
                 delta=1.0, dropout_rate=0.1, name="contrastive_pna_layer", **kwargs):
        super(ContrastivePNALayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.dropout_rate = dropout_rate
        
        # PNA aggregation functions
        self.aggregation_functions = {
            "mean": tf.reduce_mean,
            "max": tf.reduce_max,
            "min": tf.reduce_min,
            "sum": tf.reduce_sum,
            "std": tf.math.reduce_std
        }
        
        # PNA scaling functions
        self.scaling_functions = {
            "identity": lambda x, d: x,
            "amplification": lambda x, d: x * (tf.math.log(tf.cast(d, tf.float32) + 1) / tf.math.log(10)),
            "attenuation": lambda x, d: x * (tf.math.log(tf.cast(d, tf.float32) + 1) / tf.math.log(10))
        }
        
        # Node update MLP
        self.node_mlp = MLP(
            units=[units],
            use_bias=use_bias,
            activation=activation
        )
        
        # Normalization
        self.node_norm = GraphBatchNormalization()
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        """
        Process a single graph view.
        
        Args:
            inputs: [node_attributes, edge_indices]
            training: Whether in training mode
            
        Returns:
            Updated node features
        """
        node_attributes, edge_indices = inputs
        
        # Gather neighbor features
        neighbor_features = GatherNodesOutgoing()([node_attributes, edge_indices])
        
        # Get node degrees for scaling
        node_degrees = tf.reduce_sum(tf.one_hot(edge_indices[:, 0], tf.shape(node_attributes)[0]), axis=0)
        node_degrees = tf.maximum(node_degrees, 1)  # Avoid division by zero
        
        # Apply PNA aggregations
        aggregated_features = []
        
        for aggregator_name in self.aggregators:
            if aggregator_name in self.aggregation_functions:
                # Aggregate neighbors
                agg_func = self.aggregation_functions[aggregator_name]
                aggregated = AggregateLocalEdges(pooling_method=aggregator_name)([node_attributes, neighbor_features, edge_indices])
                
                # Apply scalers
                for scaler_name in self.scalers:
                    if scaler_name in self.scaling_functions:
                        scaled = self.scaling_functions[scaler_name](aggregated, node_degrees)
                        aggregated_features.append(scaled)
        
        # Concatenate all aggregated features
        if aggregated_features:
            concatenated = tf.concat(aggregated_features, axis=-1)
        else:
            concatenated = node_attributes
        
        # Node update
        node_updated = self.node_mlp(concatenated)
        node_updated = self.node_norm(node_updated, training=training)
        node_updated = self.dropout(node_updated, training=training)
        
        return node_updated
    
    def get_config(self):
        config = super(ContrastivePNALayer, self).get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "aggregators": self.aggregators,
            "scalers": self.scalers,
            "delta": self.delta,
            "dropout_rate": self.dropout_rate
        })
        return config


class ContrastivePNAConv(tf.keras.layers.Layer):
    """
    Contrastive PNA convolution layer.
    
    Implements contrastive learning with PNA architecture,
    combining powerful aggregation functions with contrastive learning capabilities.
    
    Args:
        units (int): Number of units
        depth (int): Number of layers
        use_bias (bool): Whether to use bias
        activation (str): Activation function
        aggregators (list): List of aggregation functions
        scalers (list): List of scaling functions
        delta (float): Delta parameter for scaling
        dropout_rate (float): Dropout rate
        num_views (int): Number of contrastive views
        edge_drop_rate (float): Edge dropping rate for augmentation
        node_mask_rate (float): Node masking rate for augmentation
        feature_noise_std (float): Feature noise standard deviation
        use_contrastive_loss (bool): Whether to use contrastive loss
        contrastive_loss_type (str): Type of contrastive loss
        temperature (float): Temperature for contrastive loss
    """
    
    def __init__(self, units=128, depth=3, use_bias=True, activation="relu",
                 aggregators=["mean", "max", "sum"], scalers=["identity", "amplification", "attenuation"],
                 delta=1.0, dropout_rate=0.1, num_views=2, edge_drop_rate=0.1, node_mask_rate=0.1, 
                 feature_noise_std=0.01, use_contrastive_loss=True, 
                 contrastive_loss_type="regression_aware", temperature=0.1,
                 name="contrastive_pna_conv", **kwargs):
        super(ContrastivePNAConv, self).__init__(name=name, **kwargs)
        self.units = units
        self.depth = depth
        self.use_bias = use_bias
        self.activation = activation
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.dropout_rate = dropout_rate
        self.num_views = num_views
        self.edge_drop_rate = edge_drop_rate
        self.node_mask_rate = node_mask_rate
        self.feature_noise_std = feature_noise_std
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_loss_type = contrastive_loss_type
        self.temperature = temperature
        
        # PNA layers
        self.pna_layers = []
        for i in range(depth):
            self.pna_layers.append(ContrastivePNALayer(
                units=units,
                use_bias=use_bias,
                activation=activation,
                aggregators=aggregators,
                scalers=scalers,
                delta=delta,
                dropout_rate=dropout_rate
            ))
        
        # Projection for contrastive learning
        self.contrastive_projection = MLP(
            units=[units, units // 2],
            use_bias=use_bias,
            activation=activation
        )
        
        # Store embeddings for contrastive loss
        self.embeddings = None
        self.view_embeddings = None
    
    def call(self, inputs, training=None):
        """
        Forward pass with contrastive learning.
        
        Args:
            inputs: [node_attributes, edge_indices]
            training: Whether in training mode
            
        Returns:
            Graph embeddings and view embeddings
        """
        node_attributes, edge_indices = inputs
        
        # Generate multiple views
        views = self._generate_views([node_attributes, edge_indices], training=training)
        
        # Process main graph
        node_features = node_attributes
        
        for layer in self.pna_layers:
            node_features = layer([node_features, edge_indices], training=training)
        
        # Pool to graph-level representation
        graph_embedding = tf.reduce_mean(node_features, axis=1)
        graph_embedding = self.contrastive_projection(graph_embedding)
        
        # Process views for contrastive learning
        view_embeddings = []
        for view in views:
            view_nodes, view_indices = view
            
            # Process through layers
            for layer in self.pna_layers:
                view_nodes = layer([view_nodes, view_indices], training=training)
            
            # Pool to graph-level
            view_embedding = tf.reduce_mean(view_nodes, axis=1)
            view_embedding = self.contrastive_projection(view_embedding)
            view_embeddings.append(view_embedding)
        
        # Store embeddings for contrastive loss
        if training and self.use_contrastive_loss:
            self.embeddings = graph_embedding
            self.view_embeddings = tf.stack(view_embeddings, axis=1)
        
        return graph_embedding, view_embeddings
    
    def _generate_views(self, inputs, training=None):
        """Generate multiple augmented views of the input graph."""
        node_attributes, edge_indices = inputs
        views = []
        
        for i in range(self.num_views):
            # Create augmented view
            aug_node_attr = node_attributes
            aug_edge_idx = edge_indices
            
            if training:
                # Add feature noise
                if self.feature_noise_std > 0:
                    noise = tf.random.normal(tf.shape(aug_node_attr), 
                                           stddev=self.feature_noise_std)
                    aug_node_attr = aug_node_attr + noise
                
                # Node masking
                if self.node_mask_rate > 0:
                    mask = tf.random.uniform(tf.shape(aug_node_attr)[:1]) > self.node_mask_rate
                    mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
                    aug_node_attr = aug_node_attr * mask
                
                # Edge dropping
                if self.edge_drop_rate > 0:
                    edge_mask = tf.random.uniform(tf.shape(edge_indices)[:1]) > self.edge_drop_rate
                    edge_mask = tf.cast(edge_mask, tf.int32)
                    aug_edge_idx = tf.boolean_mask(edge_indices, edge_mask)
            
            views.append([aug_node_attr, aug_edge_idx])
        
        return views
    
    def get_config(self):
        config = super(ContrastivePNAConv, self).get_config()
        config.update({
            "units": self.units,
            "depth": self.depth,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "aggregators": self.aggregators,
            "scalers": self.scalers,
            "delta": self.delta,
            "dropout_rate": self.dropout_rate,
            "num_views": self.num_views,
            "edge_drop_rate": self.edge_drop_rate,
            "node_mask_rate": self.node_mask_rate,
            "feature_noise_std": self.feature_noise_std,
            "use_contrastive_loss": self.use_contrastive_loss,
            "contrastive_loss_type": self.contrastive_loss_type,
            "temperature": self.temperature
        })
        return config 