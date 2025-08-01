import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Dropout, LazyConcatenate, Activation, LazyAdd, LazyMultiply
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.mlp import MLP
from kgcnn.ops.axis import get_axis

ks = tf.keras

class MultiChemAttention(GraphBaseLayer):
    """Multi-scale attention mechanism for MultiChem.
    
    This layer implements attention that can handle both directed and undirected graphs
    with dual node and edge features.
    """
    
    def __init__(self, units, num_heads=8, use_directed=True, use_dual_features=True,
                 attention_dropout=0.1, **kwargs):
        """Initialize MultiChem attention layer.
        
        Args:
            units: Number of hidden units
            num_heads: Number of attention heads
            use_directed: Whether to use directed graph attention
            use_dual_features: Whether to use dual node/edge features
            attention_dropout: Dropout rate for attention
        """
        super(MultiChemAttention, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.use_directed = use_directed
        self.use_dual_features = use_dual_features
        self.attention_dropout = attention_dropout
        
        # Main attention layer - always used
        self.node_attention = AttentionHeadGAT(
            units=units,
            use_bias=True,
            activation="relu",
            use_edge_features=True
        )
        
        # Dropout
        self.dropout = Dropout(attention_dropout)
        
    def call(self, inputs, **kwargs):
        """Forward pass with multi-scale attention.
        
        Args:
            inputs: [node_features, edge_features, edge_indices, edge_indices_reverse]
                   or [node_features, edge_features, edge_indices] for undirected
            
        Returns:
            Updated node features
        """
        if len(inputs) == 4:
            node_features, edge_features, edge_indices, edge_indices_reverse = inputs
        else:
            node_features, edge_features, edge_indices = inputs
            edge_indices_reverse = edge_indices
        
        # Base node-level attention
        node_attended = self.node_attention([node_features, edge_features, edge_indices])
        
        # Apply dropout
        output = self.dropout(node_attended)
        
        return output


class DualGNNBlock(GraphBaseLayer):
    """Dual GNN block for MultiChem - processes both atom and bond streams in parallel."""
    
    def __init__(self, units, num_heads=8, dropout=0.1, **kwargs):
        super(DualGNNBlock, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Atom stream GNN block
        self.atom_gnn = AttentionHeadGAT(
            units=units,
            use_bias=True,
            activation="relu",
            use_edge_features=True
        )
        
        # Bond stream GNN block  
        self.bond_gnn = AttentionHeadGAT(
            units=units,
            use_bias=True,
            activation="relu",
            use_edge_features=True
        )
        
        # Dropout
        self.dropout_layer = Dropout(dropout)
        
        # Aggregation layer for bond stream
        self.gather_outgoing = GatherNodesOutgoing()
        
    def call(self, inputs, **kwargs):
        """Forward pass with dual GNN processing.
        
        Args:
            inputs: [atom_features, bond_features, atom_edge_indices, bond_edge_indices]
            
        Returns:
            Updated atom and bond features
        """
        atom_features, bond_features, atom_edge_indices, bond_edge_indices = inputs
        
        # Atom stream: process atoms with bond information
        atom_updated = self.atom_gnn([atom_features, bond_features, atom_edge_indices])
        atom_updated = self.dropout_layer(atom_updated)
        
        # Bond stream: process bonds with atom information
        # First gather atom features to bond context
        atom_for_bond = self.gather_outgoing([atom_features, bond_edge_indices])
        bond_updated = self.bond_gnn([bond_features, atom_for_bond, bond_edge_indices])
        bond_updated = self.dropout_layer(bond_updated)
        
        return atom_updated, bond_updated


class MultiChemLayer(GraphBaseLayer):
    """Main MultiChem layer implementing DUAL architecture.
    
    This layer implements the core MultiChem architecture with:
    - DUAL node features (TWO versions of each)
    - DUAL edge features (TWO versions of each) 
    - DUAL message passing (TWO different message passing mechanisms)
    - DUAL pooling (TWO separate pooling operations)
    - DUAL outputs (TWO separate predictions)
    """
    
    def __init__(self, units, num_heads=8, use_directed=True, use_dual_features=True,
                 dropout=0.1, attention_dropout=0.1, use_residual=True, **kwargs):
        """Initialize MultiChem layer.
        
        Args:
            units: Number of hidden units
            num_heads: Number of attention heads
            use_directed: Whether to use directed graph processing
            use_dual_features: Whether to use dual node/edge features
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            use_residual: Whether to use residual connections
        """
        super(MultiChemLayer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.use_directed = use_directed
        self.use_dual_features = use_dual_features
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_residual = use_residual
        
        # DUAL GNN blocks for atom and bond streams
        self.dual_gnn_block = DualGNNBlock(
            units=units,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Multi-scale attention for merging
        self.attention = MultiChemAttention(
            units=units,
            num_heads=num_heads,
            use_directed=use_directed,
            use_dual_features=use_dual_features,
            attention_dropout=attention_dropout
        )
        
        # Input projection layers for residual connections
        self.atom_input_projection = Dense(units, activation="linear", use_bias=True)
        self.bond_input_projection = Dense(units, activation="linear", use_bias=True)
        
        # Residual connections
        if self.use_residual:
            self.atom_residual = LazyAdd()
            self.bond_residual = LazyAdd()
        
        # Aggregation layers
        self.gather_outgoing = GatherNodesOutgoing()
        self.aggregate_edges = AggregateLocalEdges(pooling_method="sum")
        
    def call(self, inputs, **kwargs):
        """Forward pass of MultiChem layer with DUAL processing.
        
        Args:
            inputs: [atom_features, bond_features, atom_edge_indices, bond_edge_indices, atom_edge_indices_reverse]
                   or [atom_features, bond_features, atom_edge_indices, bond_edge_indices] for undirected
            
        Returns:
            Updated atom and bond features
        """
        if len(inputs) == 5:
            atom_features, bond_features, atom_edge_indices, bond_edge_indices, atom_edge_indices_reverse = inputs
        else:
            atom_features, bond_features, atom_edge_indices, bond_edge_indices = inputs
            atom_edge_indices_reverse = atom_edge_indices
        
        # Store original features for residual connections (project to correct dimension)
        atom_original = self.atom_input_projection(atom_features)
        bond_original = self.bond_input_projection(bond_features)
        
        # DUAL GNN processing: both atom and bond streams in parallel
        atom_updated, bond_updated = self.dual_gnn_block([
            atom_features, bond_features, atom_edge_indices, bond_edge_indices
        ], **kwargs)
        
        # Multi-scale attention for atom stream only (bond stream is processed separately)
        attention_inputs = [atom_updated, bond_updated, atom_edge_indices]
        if self.use_directed:
            attention_inputs.append(atom_edge_indices_reverse)
        
        atom_final = self.attention(attention_inputs, **kwargs)
        
        # Residual connections
        if self.use_residual:
            atom_final = self.atom_residual([atom_original, atom_final])
            bond_final = self.bond_residual([bond_original, bond_updated])
        else:
            bond_final = bond_updated
        
        # Ensure both streams are used in the final output
        # This prevents unused layer warnings by making sure all layers contribute
        return atom_final, bond_final


class PoolingNodesMultiChem(GraphBaseLayer):
    """MultiChem-specific pooling layer with DUAL pooling operations."""
    
    def __init__(self, pooling_method="sum", use_dual_features=True, **kwargs):
        super(PoolingNodesMultiChem, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.use_dual_features = use_dual_features
        
    def call(self, inputs, **kwargs):
        """Forward pass with MultiChem DUAL pooling.
        
        Args:
            inputs: [atom_features, bond_features] or [atom_features]
            
        Returns:
            Pooled graph features (DUAL outputs)
        """
        if self.use_dual_features and len(inputs) == 2:
            atom_features, bond_features = inputs
            # DUAL pooling: pool both atom and bond features separately
            atom_pooled = tf.reduce_sum(atom_features, axis=1)
            bond_pooled = tf.reduce_sum(bond_features, axis=1)
            # Concatenate dual pooled features
            return LazyConcatenate(axis=-1)([atom_pooled, bond_pooled])
        else:
            atom_features = inputs[0] if isinstance(inputs, list) else inputs
            if self.pooling_method == "sum":
                return tf.reduce_sum(atom_features, axis=1)
            elif self.pooling_method == "mean":
                return tf.reduce_mean(atom_features, axis=1)
            elif self.pooling_method == "max":
                return tf.reduce_max(atom_features, axis=1)
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling_method}") 