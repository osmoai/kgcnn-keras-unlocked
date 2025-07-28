"""Contrastive DGIN convolution layer.

This module implements contrastive learning with DGIN architecture,
combining directed graph isomorphism networks with contrastive learning capabilities.
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


class ContrastiveDGINLayer(tf.keras.layers.Layer):
    """
    A single DGIN layer for processing graph views.
    
    Args:
        units (int): Number of units
        use_bias (bool): Whether to use bias
        activation (str): Activation function
        use_normalization (bool): Whether to use normalization
        normalization_technique (str): Normalization technique
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, units=128, use_bias=True, activation="relu", 
                 use_normalization=True, normalization_technique="graph_batch",
                 dropout_rate=0.1, name="contrastive_dgin_layer", **kwargs):
        super(ContrastiveDGINLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.use_normalization = use_normalization
        self.normalization_technique = normalization_technique
        self.dropout_rate = dropout_rate
        
        # GIN MLP
        self.gin_mlp = MLP(
            units=[units, units],
            use_bias=use_bias,
            activation=activation,
            use_normalization=use_normalization,
            normalization_technique=normalization_technique
        )
        
        # Edge MLP for directed edges
        self.edge_mlp = MLP(
            units=[units],
            use_bias=use_bias,
            activation=activation
        )
        
        # Edge activation
        self.edge_activation = tf.keras.layers.Activation(activation)
        
        # Node update
        self.node_update = MLP(
            units=[units],
            use_bias=use_bias,
            activation=activation
        )
        
        # Normalization
        self.node_norm = GraphBatchNormalization()
        self.edge_norm = GraphBatchNormalization()
    
    def call(self, inputs, training=None):
        """
        Process a single graph view.
        
        Args:
            inputs: [node_attributes, edge_attributes, edge_indices, edge_indices_reverse]
            training: Whether in training mode
            
        Returns:
            Updated node and edge features
        """
        node_attributes, edge_attributes, edge_indices, edge_indices_reverse = inputs
        
        # Gather edge features
        edge_in = GatherNodesOutgoing()([node_attributes, edge_indices])
        edge_out = GatherNodesIngoing()([node_attributes, edge_indices])
        
        # Update edge features
        edge_concat = tf.concat([edge_in, edge_out, edge_attributes], axis=-1)
        edge_updated = self.edge_mlp(edge_concat)
        edge_updated = self.edge_norm(edge_updated, training=training)
        edge_updated = self.edge_activation(edge_updated)
        
        # Aggregate incoming edges
        node_aggregated = AggregateLocalEdges()([node_attributes, edge_indices_reverse, edge_updated])
        
        # GIN update
        node_concat = tf.concat([node_attributes, node_aggregated], axis=-1)
        node_updated = self.gin_mlp(node_concat)
        node_updated = self.node_norm(node_updated, training=training)
        
        # Node update
        node_final = self.node_update(node_updated)
        
        return node_final, edge_updated
    
    def get_config(self):
        config = super(ContrastiveDGINLayer, self).get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "use_normalization": self.use_normalization,
            "normalization_technique": self.normalization_technique,
            "dropout_rate": self.dropout_rate
        })
        return config


class ContrastiveDGINConv(tf.keras.layers.Layer):
    """
    Contrastive DGIN convolution layer.
    
    Implements contrastive learning with DGIN architecture,
    combining directed graph isomorphism networks with contrastive learning capabilities.
    
    Args:
        units (int): Number of units
        depth (int): Number of layers
        use_bias (bool): Whether to use bias
        activation (str): Activation function
        use_normalization (bool): Whether to use normalization
        normalization_technique (str): Normalization technique
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
                 use_normalization=True, normalization_technique="graph_batch",
                 dropout_rate=0.1, num_views=2, edge_drop_rate=0.1, node_mask_rate=0.1, 
                 feature_noise_std=0.01, use_contrastive_loss=True, 
                 contrastive_loss_type="regression_aware", temperature=0.1,
                 name="contrastive_dgin_conv", **kwargs):
        super(ContrastiveDGINConv, self).__init__(name=name, **kwargs)
        self.units = units
        self.depth = depth
        self.use_bias = use_bias
        self.activation = activation
        self.use_normalization = use_normalization
        self.normalization_technique = normalization_technique
        self.dropout_rate = dropout_rate
        self.num_views = num_views
        self.edge_drop_rate = edge_drop_rate
        self.node_mask_rate = node_mask_rate
        self.feature_noise_std = feature_noise_std
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_loss_type = contrastive_loss_type
        self.temperature = temperature
        
        # DGIN layers
        self.dgin_layers = []
        for i in range(depth):
            self.dgin_layers.append(ContrastiveDGINLayer(
                units=units,
                use_bias=use_bias,
                activation=activation,
                use_normalization=use_normalization,
                normalization_technique=normalization_technique,
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
            inputs: [node_attributes, edge_attributes, edge_indices, edge_indices_reverse]
            training: Whether in training mode
            
        Returns:
            Graph embeddings and view embeddings
        """
        node_attributes, edge_attributes, edge_indices, edge_indices_reverse = inputs
        
        # Generate multiple views
        views = self._generate_views([node_attributes, edge_attributes, edge_indices, edge_indices_reverse], training=training)
        
        # Process main graph
        node_features = node_attributes
        edge_features = edge_attributes
        
        for layer in self.dgin_layers:
            node_features, edge_features = layer([node_features, edge_features, edge_indices, edge_indices_reverse], training=training)
        
        # Pool to graph-level representation
        graph_embedding = tf.reduce_mean(node_features, axis=1)
        graph_embedding = self.contrastive_projection(graph_embedding)
        
        # Process views for contrastive learning
        view_embeddings = []
        for view in views:
            view_nodes, view_edges, view_indices, view_indices_reverse = view
            
            # Process through layers
            for layer in self.dgin_layers:
                view_nodes, view_edges = layer([view_nodes, view_edges, view_indices, view_indices_reverse], training=training)
            
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
        node_attributes, edge_attributes, edge_indices, edge_indices_reverse = inputs
        views = []
        
        for i in range(self.num_views):
            # Create augmented view
            aug_node_attr = node_attributes
            aug_edge_attr = edge_attributes
            aug_edge_idx = edge_indices
            aug_edge_idx_reverse = edge_indices_reverse
            
            if training:
                # Add feature noise
                if self.feature_noise_std > 0:
                    node_noise = tf.random.normal(tf.shape(aug_node_attr), 
                                                stddev=self.feature_noise_std)
                    edge_noise = tf.random.normal(tf.shape(aug_edge_attr), 
                                                stddev=self.feature_noise_std)
                    aug_node_attr = aug_node_attr + node_noise
                    aug_edge_attr = aug_edge_attr + edge_noise
                
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
                    aug_edge_attr = tf.boolean_mask(edge_attributes, edge_mask)
                    aug_edge_idx_reverse = tf.boolean_mask(edge_indices_reverse, edge_mask)
            
            views.append([aug_node_attr, aug_edge_attr, aug_edge_idx, aug_edge_idx_reverse])
        
        return views
    
    def get_config(self):
        config = super(ContrastiveDGINConv, self).get_config()
        config.update({
            "units": self.units,
            "depth": self.depth,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "use_normalization": self.use_normalization,
            "normalization_technique": self.normalization_technique,
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