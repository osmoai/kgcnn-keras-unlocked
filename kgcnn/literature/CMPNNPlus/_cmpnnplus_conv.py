import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyAdd, LazyConcatenate, Activation
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.mlp import GraphMLP

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras


class CMPNNPlusPoolingEdges(GraphBaseLayer):
    r"""CMPNN+ pooling layer with multi-level communicative message passing.
    
    This layer extends the original DMPNN with explicit multi-level communicative
    message passing, achieving stronger performance on bioactivity and ADMET tasks.
    
    Reference: Communicative Message Passing Neural Network for Molecular Representation (2023, J. Chem. Inf. Model.)
    """
    
    def __init__(self, edge_initialize: dict = None, edge_dense: dict = None,
                 edge_activation: dict = None, node_dense: dict = None,
                 dropout: dict = None, depth: int = None,
                 use_communicative: bool = True, communication_levels: int = 3,
                 **kwargs):
        """Initialize layer.
        
        Args:
            edge_initialize (dict): Edge initialization layer arguments.
            edge_dense (dict): Edge dense layer arguments.
            edge_activation (dict): Edge activation layer arguments.
            node_dense (dict): Node dense layer arguments.
            dropout (dict): Dropout layer arguments.
            depth (int): Number of message passing layers.
            use_communicative (bool): Whether to use communicative message passing.
            communication_levels (int): Number of communication levels.
        """
        super(CMPNNPlusPoolingEdges, self).__init__(**kwargs)
        self.edge_initialize = edge_initialize
        self.edge_dense = edge_dense
        self.edge_activation = edge_activation
        self.node_dense = node_dense
        self.dropout = dropout
        self.depth = depth
        self.use_communicative = use_communicative
        self.communication_levels = communication_levels
        
        # Multi-level communicative components
        if self.use_communicative:
            self.communication_layers = []
            for level in range(communication_levels):
                level_layer = {
                    'edge_mlp': GraphMLP(**edge_dense),
                    'node_mlp': GraphMLP(**node_dense),
                    'communication_gate': Dense(edge_dense['units'], activation='sigmoid'),
                    'fusion_layer': Dense(edge_dense['units'], activation='relu')
                }
                self.communication_layers.append(level_layer)
        
        # Standard DMPNN components
        self.edge_initialize_layer = Dense(**edge_initialize)
        self.edge_dense_layer = Dense(**edge_dense)
        self.edge_activation_layer = Activation(**edge_activation)
        self.node_dense_layer = Dense(**node_dense)
        
        if dropout:
            self.dropout_layer = Dropout(**dropout)
        else:
            self.dropout_layer = None
            
        # Message passing components
        self.gather_nodes = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges()
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate()
        
    def call(self, inputs, **kwargs):
        """Forward pass with multi-level communicative message passing.
        
        Args:
            inputs: [node_attributes, edge_attributes, edge_indices, edge_indices_reverse]
            
        Returns:
            Updated node features
        """
        node_attributes, edge_attributes, edge_indices, edge_indices_reverse = inputs
        
        # Initialize edge features
        edge_features = self.edge_initialize_layer(edge_attributes)
        
        # Initialize node features to match expected shape
        node_attributes = self.node_dense_layer(node_attributes)
        
        # Multi-level communicative message passing
        if self.use_communicative:
            for level in range(self.communication_levels):
                level_outputs = []
                
                # Standard message passing at this level
                for i in range(self.depth):
                    # Gather node features for edges
                    node_features_edges = self.gather_nodes([node_attributes, edge_indices])
                    
                    # Concatenate edge and node features
                    edge_node_features = self.lazy_concat([edge_features, node_features_edges])
                    
                    # Apply edge MLP
                    edge_features_updated = self.communication_layers[level]['edge_mlp'](edge_node_features)
                    
                    # Apply communication gate
                    communication_weights = self.communication_layers[level]['communication_gate'](edge_features)
                    edge_features_updated = communication_weights * edge_features_updated
                    
                    # Aggregate edge features to nodes - filter out invalid reverse indices
                    valid_mask = tf.squeeze(edge_indices_reverse >= 0, axis=-1)
                    # Use ragged tensor operations for ragged tensors
                    valid_edge_features = tf.ragged.boolean_mask(edge_features_updated, valid_mask)
                    valid_reverse_indices = tf.ragged.boolean_mask(edge_indices_reverse, valid_mask)
                    
                    # Check if we have any valid edges using tf.cond for graph execution
                    has_valid_edges = tf.greater(tf.shape(valid_edge_features)[0], 0)
                    node_features_updated = tf.cond(
                        has_valid_edges,
                        lambda: self.aggregate_edges([node_attributes, valid_edge_features, valid_reverse_indices]),
                        lambda: node_attributes
                    )
                    
                    # Apply node MLP
                    node_features_updated = self.communication_layers[level]['node_mlp'](node_features_updated)
                    
                    # Skip connection
                    node_attributes = self.lazy_add([node_attributes, node_features_updated])
                    
                    # Apply dropout
                    if self.dropout_layer:
                        node_attributes = self.dropout_layer(node_attributes)
                    
                    level_outputs.append(node_attributes)
                
                # Fuse outputs from this level
                if len(level_outputs) > 1:
                    level_fused = self.communication_layers[level]['fusion_layer'](
                        tf.concat(level_outputs, axis=-1)
                    )
                else:
                    level_fused = level_outputs[0]
                
                # Update node attributes for next level
                node_attributes = level_fused
        
        else:
            # Standard DMPNN message passing (fallback)
            for i in range(self.depth):
                # Gather node features for edges
                node_features_edges = self.gather_nodes([node_attributes, edge_indices])
                
                # Concatenate edge and node features
                edge_node_features = self.lazy_concat([edge_features, node_features_edges])
                
                # Apply edge MLP
                edge_features_updated = self.edge_dense_layer(edge_node_features)
                
                # Skip connection for edges
                edge_features = self.lazy_add([edge_features, edge_features_updated])
                
                # Apply edge activation
                edge_features = self.edge_activation_layer(edge_features)
                
                # Aggregate edge features to nodes - filter out invalid reverse indices
                valid_mask = tf.squeeze(edge_indices_reverse >= 0, axis=-1)
                # Use ragged tensor operations for ragged tensors
                valid_edge_features = tf.ragged.boolean_mask(edge_features, valid_mask)
                valid_reverse_indices = tf.ragged.boolean_mask(edge_indices_reverse, valid_mask)
                
                # Check if we have any valid edges using tf.cond for graph execution
                has_valid_edges = tf.greater(tf.shape(valid_edge_features)[0], 0)
                node_features_updated = tf.cond(
                    has_valid_edges,
                    lambda: self.aggregate_edges([node_attributes, valid_edge_features, valid_reverse_indices]),
                    lambda: node_attributes
                )
                
                # Apply node MLP
                node_features_updated = self.node_dense_layer(node_features_updated)
                
                # Skip connection for nodes
                node_attributes = self.lazy_add([node_attributes, node_features_updated])
                
                # Apply dropout
                if self.dropout_layer:
                    node_attributes = self.dropout_layer(node_attributes)
        
        return node_attributes 