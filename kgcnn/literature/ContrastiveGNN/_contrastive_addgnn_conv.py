"""Contrastive AddGNN convolution layer.

This module implements contrastive learning with AddGNN architecture,
combining additive graph neural networks with contrastive learning capabilities.
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
from kgcnn.layers.set2set import PoolingSet2SetEncoder


class ContrastiveAddGNNLayer(tf.keras.layers.Layer):
    """
    A single AddGNN layer for processing graph views.
    
    Args:
        units (int): Number of units
        heads (int): Number of attention heads
        use_bias (bool): Whether to use bias
        activation (str): Activation function
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, units=128, heads=4, use_bias=True, activation="relu", 
                 dropout_rate=0.1, name="contrastive_addgnn_layer", **kwargs):
        super(ContrastiveAddGNNLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.heads = heads
        self.use_bias = use_bias
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.attention_heads = []
        for i in range(heads):
            self.attention_heads.append(AttentionHeadGAT(
                units=units // heads,
                use_bias=use_bias,
                activation=activation,
                attention_heads=1,
                dropout_rate=dropout_rate
            ))
        
        # Node update
        self.node_update = MLP(
            units=[units],
            use_bias=use_bias,
            activation=activation
        )
        
        # Normalization
        self.node_norm = GraphBatchNormalization()
    
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
        
        # Multi-head attention
        attention_outputs = []
        for attention_head in self.attention_heads:
            # Create dummy edge features for attention
            dummy_edge_features = tf.ones([tf.shape(edge_indices)[0], 1])
            attention_output = attention_head([node_attributes, edge_indices, dummy_edge_features])
            attention_outputs.append(attention_output)
        
        # Concatenate attention outputs
        if len(attention_outputs) > 1:
            attended_features = tf.concat(attention_outputs, axis=-1)
        else:
            attended_features = attention_outputs[0]
        
        # Node update
        node_updated = self.node_update(node_attributes)
        node_updated = self.node_norm(node_updated, training=training)
        
        # Additive combination
        node_final = node_updated + attended_features
        
        return node_final
    
    def get_config(self):
        config = super(ContrastiveAddGNNLayer, self).get_config()
        config.update({
            "units": self.units,
            "heads": self.heads,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        })
        return config


class ContrastiveAddGNNConv(tf.keras.layers.Layer):
    """
    Contrastive AddGNN convolution layer.
    
    Implements contrastive learning with AddGNN architecture,
    combining additive graph neural networks with contrastive learning capabilities.
    
    Args:
        units (int): Number of units
        depth (int): Number of layers
        heads (int): Number of attention heads
        use_bias (bool): Whether to use bias
        activation (str): Activation function
        dropout_rate (float): Dropout rate
        num_views (int): Number of contrastive views
        edge_drop_rate (float): Edge dropping rate for augmentation
        node_mask_rate (float): Node masking rate for augmentation
        feature_noise_std (float): Feature noise standard deviation
        use_contrastive_loss (bool): Whether to use contrastive loss
        contrastive_loss_type (str): Type of contrastive loss
        temperature (float): Temperature for contrastive loss
        use_set2set (bool): Whether to use Set2Set pooling
        set2set_args (dict): Set2Set arguments
    """
    
    def __init__(self, units=128, depth=3, heads=4, use_bias=True, activation="relu",
                 dropout_rate=0.1, num_views=2, edge_drop_rate=0.1, node_mask_rate=0.1, 
                 feature_noise_std=0.01, use_contrastive_loss=True, 
                 contrastive_loss_type="regression_aware", temperature=0.1,
                 use_set2set=True, set2set_args=None, name="contrastive_addgnn_conv", **kwargs):
        super(ContrastiveAddGNNConv, self).__init__(name=name, **kwargs)
        self.units = units
        self.depth = depth
        self.heads = heads
        self.use_bias = use_bias
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.num_views = num_views
        self.edge_drop_rate = edge_drop_rate
        self.node_mask_rate = node_mask_rate
        self.feature_noise_std = feature_noise_std
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_loss_type = contrastive_loss_type
        self.temperature = temperature
        self.use_set2set = use_set2set
        
        # Default Set2Set arguments
        if set2set_args is None:
            set2set_args = {
                "channels": units,
                "T": 3,
                "pooling_method": "sum",
                "init_qstar": "0"
            }
        self.set2set_args = set2set_args
        
        # AddGNN layers
        self.addgnn_layers = []
        for i in range(depth):
            self.addgnn_layers.append(ContrastiveAddGNNLayer(
                units=units,
                heads=heads,
                use_bias=use_bias,
                activation=activation,
                dropout_rate=dropout_rate
            ))
        
        # Set2Set pooling
        if use_set2set:
            self.set2set_pooling = PoolingSet2SetEncoder(**set2set_args)
        
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
        
        for layer in self.addgnn_layers:
            node_features = layer([node_features, edge_indices], training=training)
        
        # Pool to graph-level representation
        if self.use_set2set:
            graph_embedding = self.set2set_pooling(node_features)
        else:
            graph_embedding = tf.reduce_mean(node_features, axis=1)
        
        graph_embedding = self.contrastive_projection(graph_embedding)
        
        # Process views for contrastive learning
        view_embeddings = []
        for view in views:
            view_nodes, view_indices = view
            
            # Process through layers
            for layer in self.addgnn_layers:
                view_nodes = layer([view_nodes, view_indices], training=training)
            
            # Pool to graph-level
            if self.use_set2set:
                view_embedding = self.set2set_pooling(view_nodes)
            else:
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
        config = super(ContrastiveAddGNNConv, self).get_config()
        config.update({
            "units": self.units,
            "depth": self.depth,
            "heads": self.heads,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "num_views": self.num_views,
            "edge_drop_rate": self.edge_drop_rate,
            "node_mask_rate": self.node_mask_rate,
            "feature_noise_std": self.feature_noise_std,
            "use_contrastive_loss": self.use_contrastive_loss,
            "contrastive_loss_type": self.contrastive_loss_type,
            "temperature": self.temperature,
            "use_set2set": self.use_set2set,
            "set2set_args": self.set2set_args
        })
        return config 