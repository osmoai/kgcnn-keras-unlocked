"""Contrastive AttentiveFP convolution layer.

This module implements contrastive learning with AttentiveFP architecture,
combining graph attention with contrastive learning capabilities.
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


class GraphViewGenerator(tf.keras.layers.Layer):
    """
    Generate multiple augmented views of a graph for contrastive learning.
    
    Args:
        num_views (int): Number of views to generate
        edge_drop_rate (float): Probability of dropping edges
        node_mask_rate (float): Probability of masking node features
        feature_noise_std (float): Standard deviation of feature noise
    """
    
    def __init__(self, num_views=2, edge_drop_rate=0.1, node_mask_rate=0.1, 
                 feature_noise_std=0.01, name="graph_view_generator", **kwargs):
        super(GraphViewGenerator, self).__init__(name=name, **kwargs)
        self.num_views = num_views
        self.edge_drop_rate = edge_drop_rate
        self.node_mask_rate = node_mask_rate
        self.feature_noise_std = feature_noise_std
    
    def call(self, inputs, training=None):
        """
        Generate multiple views of the input graph.
        
        Args:
            inputs: List of [node_attributes, edge_attributes, edge_indices]
            training: Whether in training mode
            
        Returns:
            List of augmented graph views
        """
        node_attributes, edge_attributes, edge_indices = inputs
        views = []
        
        for i in range(self.num_views):
            # Create augmented view
            aug_node_attr = node_attributes
            aug_edge_attr = edge_attributes
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
                    aug_edge_attr = tf.boolean_mask(edge_attributes, edge_mask)
            
            views.append([aug_node_attr, aug_edge_attr, aug_edge_idx])
        
        return views
    
    def get_config(self):
        config = super(GraphViewGenerator, self).get_config()
        config.update({
            "num_views": self.num_views,
            "edge_drop_rate": self.edge_drop_rate,
            "node_mask_rate": self.node_mask_rate,
            "feature_noise_std": self.feature_noise_std
        })
        return config


class ContrastiveAttFPLayer(tf.keras.layers.Layer):
    """
    A single AttentiveFP layer for processing graph views.
    
    Args:
        units (int): Number of units
        use_bias (bool): Whether to use bias
        activation (str): Activation function
        attention_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, units=128, use_bias=True, activation="relu", 
                 attention_heads=8, dropout_rate=0.1, name="contrastive_attfp_layer", **kwargs):
        super(ContrastiveAttFPLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        
        # Attention mechanism
        self.attention = AttentionHeadGAT(
            units=units,
            use_bias=use_bias,
            activation=activation,
            attention_heads=attention_heads,
            dropout_rate=dropout_rate
        )
        
        # Node update
        self.node_update = MLP(
            units=[units],
            use_bias=use_bias,
            activation=activation
        )
        
        # Edge update
        self.edge_update = MLP(
            units=[units],
            use_bias=use_bias,
            activation=activation
        )
        
        # GRU for state updates
        self.gru_update = GRUUpdate(units=units)
        
        # Normalization
        self.node_norm = GraphBatchNormalization()
        self.edge_norm = GraphBatchNormalization()
    
    def call(self, inputs, training=None):
        """
        Process a single graph view.
        
        Args:
            inputs: [node_attributes, edge_attributes, edge_indices]
            training: Whether in training mode
            
        Returns:
            Updated node and edge features
        """
        node_attributes, edge_attributes, edge_indices = inputs
        
        # Gather edge features
        edge_in = GatherNodesOutgoing()([node_attributes, edge_indices])
        edge_out = GatherNodesIngoing()([node_attributes, edge_indices])
        
        # Update edge features
        edge_concat = tf.concat([edge_in, edge_out, edge_attributes], axis=-1)
        edge_updated = self.edge_update(edge_concat)
        edge_updated = self.edge_norm(edge_updated, training=training)
        
        # Attention mechanism
        node_attended = self.attention([node_attributes, edge_indices, edge_updated])
        
        # Node update
        node_updated = self.node_update(node_attributes)
        node_updated = self.node_norm(node_updated, training=training)
        
        # GRU update
        node_final = self.gru_update([node_updated, node_attended])
        
        return node_final, edge_updated
    
    def get_config(self):
        config = super(ContrastiveAttFPLayer, self).get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "attention_heads": self.attention_heads,
            "dropout_rate": self.dropout_rate
        })
        return config


class ContrastiveAttFPConv(tf.keras.layers.Layer):
    """
    Contrastive AttentiveFP convolution layer.
    
    Implements contrastive learning with AttentiveFP architecture,
    combining graph attention with contrastive learning capabilities.
    
    Args:
        units (int): Number of units
        depth (int): Number of layers
        use_bias (bool): Whether to use bias
        activation (str): Activation function
        attention_heads (int): Number of attention heads
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
                 attention_heads=8, dropout_rate=0.1, num_views=2, 
                 edge_drop_rate=0.1, node_mask_rate=0.1, feature_noise_std=0.01,
                 use_contrastive_loss=True, contrastive_loss_type="regression_aware",
                 temperature=0.1, name="contrastive_attfp_conv", **kwargs):
        super(ContrastiveAttFPConv, self).__init__(name=name, **kwargs)
        self.units = units
        self.depth = depth
        self.use_bias = use_bias
        self.activation = activation
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.num_views = num_views
        self.edge_drop_rate = edge_drop_rate
        self.node_mask_rate = node_mask_rate
        self.feature_noise_std = feature_noise_std
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_loss_type = contrastive_loss_type
        self.temperature = temperature
        
        # View generator
        self.view_generator = GraphViewGenerator(
            num_views=num_views,
            edge_drop_rate=edge_drop_rate,
            node_mask_rate=node_mask_rate,
            feature_noise_std=feature_noise_std
        )
        
        # AttentiveFP layers
        self.attfp_layers = []
        for i in range(depth):
            self.attfp_layers.append(ContrastiveAttFPLayer(
                units=units,
                use_bias=use_bias,
                activation=activation,
                attention_heads=attention_heads,
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
            inputs: [node_attributes, edge_attributes, edge_indices]
            training: Whether in training mode
            
        Returns:
            Graph embeddings and view embeddings
        """
        node_attributes, edge_attributes, edge_indices = inputs
        
        # Generate multiple views
        views = self.view_generator([node_attributes, edge_attributes, edge_indices], training=training)
        
        # Process main graph
        node_features = node_attributes
        edge_features = edge_attributes
        
        for layer in self.attfp_layers:
            node_features, edge_features = layer([node_features, edge_features, edge_indices], training=training)
        
        # Pool to graph-level representation
        graph_embedding = tf.reduce_mean(node_features, axis=1)
        graph_embedding = self.contrastive_projection(graph_embedding)
        
        # Process views for contrastive learning
        view_embeddings = []
        for view in views:
            view_nodes, view_edges, view_indices = view
            
            # Process through layers
            for layer in self.attfp_layers:
                view_nodes, view_edges = layer([view_nodes, view_edges, view_indices], training=training)
            
            # Pool to graph-level
            view_embedding = tf.reduce_mean(view_nodes, axis=1)
            view_embedding = self.contrastive_projection(view_embedding)
            view_embeddings.append(view_embedding)
        
        # Store embeddings for contrastive loss
        if training and self.use_contrastive_loss:
            self.embeddings = graph_embedding
            self.view_embeddings = tf.stack(view_embeddings, axis=1)
        
        return graph_embedding, view_embeddings
    
    def get_config(self):
        config = super(ContrastiveAttFPConv, self).get_config()
        config.update({
            "units": self.units,
            "depth": self.depth,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "attention_heads": self.attention_heads,
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