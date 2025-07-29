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
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.modules import LazyConcatenate, LazyAdd
from kgcnn.layers.mlp import MLP
from kgcnn.layers.norm import GraphBatchNormalization
from tensorflow.keras.layers import Activation
from kgcnn.layers.modules import Dense, Dropout


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DGINPPoolingEdgesDirected')
class DGINPPoolingEdgesDirected(GraphBaseLayer):
    """Directed Graph Isomorphism Network (DGIN) layer.
    
    Unlike DMPNN which focuses on edge feature updates, DGIN focuses on node feature updates
    using GIN-style aggregation. DGIN combines the directed graph structure with the 
    Graph Isomorphism Network's approach to node feature learning.
    
    Key differences from DMPNN:
    - DMPNN: Updates edge features and returns edge embeddings
    - DGIN: Updates node features and returns node embeddings
    - DMPNN: Uses edge pair subtraction for message passing
    - DGIN: Uses GIN-style node aggregation with MLP updates
    """

    def __init__(self, units=128, use_bias=True, activation="relu", **kwargs):
        """Initialize layer."""
        super(DGINPPoolingEdgesDirected, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        
        # Edge processing layers
        self.edge_mlp = Dense(units=units, use_bias=use_bias, activation="linear")
        self.edge_activation = Activation(activation=activation)
        
        # Node aggregation - use segment operations which are more robust to invalid indices
        self.node_aggregate = AggregateLocalEdges(pooling_method="sum", has_unconnected=True, is_sorted=False)
        
        # Node update layers (GIN-style)
        self.node_mlp = MLP(units=[units, units], use_bias=use_bias, activation=[activation, "linear"])
        
        # Learnable embedding for single nodes (nodes with no incoming edges)
        self.single_node_embedding = Dense(units=units, use_bias=use_bias, activation=activation)
        
    def build(self, input_shape):
        """Build layer."""
        super(DGINPPoolingEdgesDirected, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index, edge_indices_reverse]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)
                - edge_indices_reverse (tf.RaggedTensor): Reverse edge indices of shape (batch, [M], 1)

        Returns:
            tf.RaggedTensor: Updated node embeddings of shape (batch, [N], F)
        """
        n, e, ed, ed_reverse = inputs
        
        # DGIN: Direct Graph Isomorphism Network approach
        # Unlike DMPNN which updates edge features, DGIN focuses on node feature updates
        
        # Step 1: Gather neighbor node features using edge indices
        # Get source and target node features for each edge
        edge_in = GatherNodesOutgoing()([n, ed])  # Source nodes
        edge_out = GatherNodesIngoing()([n, ed])  # Target nodes
        
        # Step 2: Create edge messages by combining source, target, and edge features
        edge_messages = tf.concat([edge_in, edge_out, e], axis=-1)
        edge_messages = self.edge_mlp(edge_messages)
        edge_messages = self.edge_activation(edge_messages)
        
        # Step 2.5: Add self-loops for all nodes to ensure every node has incoming edges
        # This is crucial for single nodes and improves the overall graph representation
        
        # Step 2.5: Add self-loops for single nodes using a learnable approach
        # Instead of physically adding edges, we'll handle single nodes in the aggregation
        
        # Create a learnable single node representation
        single_node_rep = self.single_node_embedding(n)
        
        # We'll use this in the aggregation step to handle single nodes
        

        
        # Step 3: Aggregate messages to target nodes using reverse indices
        # This is the key difference from DMPNN - we aggregate to nodes, not update edges
        
        # Fix 1-indexed edge indices (common data issue)
        # If edge indices are 1-indexed, convert them to 0-indexed
        num_nodes = tf.shape(n)[0]
        
        # Check if indices are 1-indexed (if max index equals num_nodes)
        # This is a heuristic to detect 1-indexed vs 0-indexed
        if hasattr(ed_reverse, 'values'):
            max_index = tf.reduce_max(ed_reverse.values)
        else:
            max_index = tf.reduce_max(ed_reverse)
            
        # If max_index == num_nodes, indices are likely 1-indexed
        is_one_indexed = tf.equal(max_index, tf.cast(num_nodes, max_index.dtype))
        
        # Convert 1-indexed to 0-indexed if needed
        if is_one_indexed:
            ed_reverse_fixed = ed_reverse - 1
        else:
            ed_reverse_fixed = ed_reverse
            
        # Ensure indices are within valid range [0, num_nodes-1]
        ed_reverse_fixed = tf.clip_by_value(ed_reverse_fixed, 0, tf.cast(num_nodes, ed_reverse_fixed.dtype) - 1)
        
        node_aggregated = self.node_aggregate([n, edge_messages, ed_reverse_fixed])
        
        # Handle single nodes by adding their learnable representation
        # This ensures every node gets meaningful aggregated features
        node_aggregated = node_aggregated + single_node_rep
        
        # Step 4: GIN-style node update: h_v = MLP(h_v + Σ neighbor_messages)
        # This is the core of DGIN - combining original node features with aggregated messages
        node_concat = tf.concat([n, node_aggregated], axis=-1)
        node_updated = self.node_mlp(node_concat)
        
        return node_updated


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
        
        # Ensure edge_attributes has the correct shape for directed graph processing
        # Handle potential nested ragged tensor issues
        if hasattr(edge_attributes, 'values') and hasattr(edge_attributes.values, 'values'):
            # If edge_attributes is a nested ragged tensor, flatten it
            edge_attributes = edge_attributes.values
        
        # Gather edge features for directed graph
        # edge_in: features of nodes that are the source of edges (outgoing)
        edge_in = GatherNodesOutgoing()([node_attributes, edge_indices])
        # edge_out: features of nodes that are the target of edges (ingoing)
        edge_out = GatherNodesIngoing()([node_attributes, edge_indices])
        
        # Update edge features by concatenating source, target, and edge features
        edge_concat = tf.concat([edge_in, edge_out, edge_attributes], axis=-1)
        edge_updated = self.edge_mlp(edge_concat)
        edge_updated = self.edge_norm(edge_updated, training=training)
        edge_updated = self.edge_activation(edge_updated)
        
        # Aggregate incoming edges using reverse indices (directed graph aggregation)
        # AggregateLocalEdges expects [nodes, edges, edge_indices] in that order
        node_aggregated = AggregateLocalEdges()([node_attributes, edge_updated, edge_indices_reverse])
        
        # GIN update: combine original node features with aggregated neighbor features
        node_concat = tf.concat([node_attributes, node_aggregated], axis=-1)
        node_updated = self.gin_mlp(node_concat)
        node_updated = self.node_norm(node_updated, training=training)
        
        # Final node update
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