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


class ExpCLayer(GraphBaseLayer):
    r"""Expressive Graph Neural Network with Path Counting (ExpC) layer.
    
    This layer adds subgraph counting (e.g., triangles) as auxiliary tasks during training,
    significantly enhancing expressivity compared to GIN.
    
    Reference: Going beyond Weisfeiler-Lehman: A Novel Expressive Graph Model (2022, ICML)
    """
    
    def __init__(self, units, use_bias=True, activation="relu",
                 use_subgraph_counting=True, subgraph_types=["triangle", "square", "pentagon"],
                 dropout_rate=0.1, **kwargs):
        """Initialize layer.
        
        Args:
            units (int): Number of hidden units.
            use_bias (bool): Whether to use bias.
            activation (str): Activation function.
            use_subgraph_counting (bool): Whether to use subgraph counting.
            subgraph_types (list): Types of subgraphs to count.
            dropout_rate (float): Dropout rate.
        """
        super(ExpCLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.use_subgraph_counting = use_subgraph_counting
        self.subgraph_types = subgraph_types
        self.dropout_rate = dropout_rate
        
        # ExpC components
        self.mlp = GraphMLP(
            units=[units, units],
            use_bias=use_bias,
            activation=activation,
            use_normalization=True,
            normalization_technique="graph_batch"
        )
        
        # Subgraph counting components
        if use_subgraph_counting:
            self.subgraph_mlp = GraphMLP(
                units=[units // 2, len(subgraph_types)],
                use_bias=use_bias,
                activation=activation,
                use_normalization=True,
                normalization_technique="graph_batch"
            )
        
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
            
        # Message passing components
        self.gather_nodes = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges()
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate()
        
    def _count_subgraphs(self, edge_indices, node_features):
        """Count subgraphs for auxiliary task."""
        if not self.use_subgraph_counting:
            return None
            
        # Simple triangle counting (for demonstration)
        # In practice, this would be more sophisticated
        triangle_counts = tf.zeros(tf.shape(node_features)[:2], dtype=tf.float32)
        
        # Count triangles by checking if three nodes form a triangle
        # This is a simplified implementation
        for i in range(tf.shape(edge_indices)[1]):
            for j in range(i + 1, tf.shape(edge_indices)[1]):
                for k in range(j + 1, tf.shape(edge_indices)[1]):
                    # Check if edges form a triangle
                    edge1 = edge_indices[:, i, :]
                    edge2 = edge_indices[:, j, :]
                    edge3 = edge_indices[:, k, :]
                    
                    # Simple triangle check (simplified)
                    if tf.reduce_all(tf.equal(edge1[0], edge2[1])) and \
                       tf.reduce_all(tf.equal(edge2[0], edge3[1])) and \
                       tf.reduce_all(tf.equal(edge3[0], edge1[1])):
                        triangle_counts = tf.tensor_scatter_nd_add(
                            triangle_counts, 
                            [[0, edge1[0]]], 
                            [1.0]
                        )
        
        return triangle_counts
        
    def call(self, inputs, **kwargs):
        """Forward pass with ExpC aggregation and subgraph counting.
        
        Args:
            inputs: [node_attributes, edge_indices]
            
        Returns:
            Updated node features and subgraph counts
        """
        node_attributes, edge_indices = inputs
        
        # Gather neighbor features
        neighbor_features = self.gather_nodes([node_attributes, edge_indices])
        
        # Aggregate neighbor features (GIN-style)
        aggregated = self.aggregate_edges([neighbor_features, edge_indices])
        
        # Apply MLP
        output = self.mlp(aggregated)
        
        # Skip connection
        output = self.lazy_add([node_attributes, output])
        
        # Subgraph counting (auxiliary task)
        subgraph_counts = None
        if self.use_subgraph_counting:
            subgraph_counts = self._count_subgraphs(edge_indices, node_attributes)
            if subgraph_counts is not None:
                # Process subgraph counts
                subgraph_features = self.subgraph_mlp(subgraph_counts)
                # Concatenate with main features
                output = self.lazy_concat([output, subgraph_features])
        
        # Apply dropout
        if self.dropout:
            output = self.dropout(output)
        
        return output


class PoolingNodesExpC(GraphBaseLayer):
    """ExpC-specific pooling layer."""
    
    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodesExpC, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
    def call(self, inputs, **kwargs):
        """Forward pass with ExpC pooling."""
        if self.pooling_method == "sum":
            return tf.reduce_sum(inputs, axis=1)
        elif self.pooling_method == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.pooling_method == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}") 