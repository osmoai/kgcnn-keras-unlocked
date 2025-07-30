"""Model factory for Contrastive GNN architectures.

This module provides functions to create contrastive versions of popular GNN architectures
including GIN, GAT, DMPNN, etc. with integrated contrastive learning capabilities.
"""

import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.modules import OptionalInputEmbedding
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from kgcnn.layers.norm import GraphBatchNormalization
from kgcnn.layers.geom import NodeDistanceEuclidean
from kgcnn.layers.attention import AttentionHeadGAT, AttentionHeadGATV2
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer
from kgcnn.layers.set2set import PoolingSet2SetEncoder
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.modules import LazyAdd, Activation, Dense

from ._contrastive_gin_conv import ContrastiveGINConv
from ._contrastive_attfp_conv import ContrastiveAttFPConv
from ._contrastive_addgnn_conv import ContrastiveAddGNNConv
from ._contrastive_dgin_conv import ContrastiveDGINConv
from ._contrastive_pna_conv import ContrastivePNAConv
from ._contrastive_losses import (
    ContrastiveGNNLoss,
    ContrastiveGNNTripletLoss,
    ContrastiveGNNDiversityLoss,
    ContrastiveGNNAlignmentLoss,
    RegressionContrastiveGNNLoss,
    RegressionContrastiveGNNTripletLoss,
    create_contrastive_gnn_metrics
)
from ._contrastive_augmentation import create_augmentation_layer


def make_contrastive_gnn_model(
    inputs,
    input_embedding=None,
    gnn_type="gin",
    contrastive_args=None,
    depth=3,
    units=128,
    use_set2set=True,
    set2set_args=None,
    use_graph_state=True,
    output_embedding="graph",
    output_to_tensor=True,
    output_mlp=None,
    **kwargs
):
    # Create input tensors from configuration
    if isinstance(inputs[0], dict):
        # Inputs are configuration dictionaries, create actual input tensors
        node_input = ks.layers.Input(**inputs[0])
        
        # Determine input structure based on number of inputs
        if len(inputs) == 3:
            # ContrastiveGIN: [node_attributes, edge_indices, graph_descriptors]
            edge_index_input = ks.layers.Input(**inputs[1])
            graph_descriptors_input = ks.layers.Input(**inputs[2])
            input_tensors = [node_input, edge_index_input, graph_descriptors_input]
            # Create dummy edge features for GIN
            edge_input = None
        elif len(inputs) == 4:
            # ContrastiveGAT: [node_attributes, edge_attributes, edge_indices, graph_descriptors]
            edge_input = ks.layers.Input(**inputs[1])
            edge_index_input = ks.layers.Input(**inputs[2])
            graph_descriptors_input = ks.layers.Input(**inputs[3])
            input_tensors = [node_input, edge_input, edge_index_input, graph_descriptors_input]
        elif len(inputs) == 5:
            # ContrastiveDMPNN: [node_attributes, edge_attributes, edge_indices, edge_indices_reverse, graph_descriptors]
            edge_input = ks.layers.Input(**inputs[1])
            edge_index_input = ks.layers.Input(**inputs[2])
            edge_reverse_input = ks.layers.Input(**inputs[3])
            graph_descriptors_input = ks.layers.Input(**inputs[4])
            input_tensors = [node_input, edge_input, edge_index_input, edge_reverse_input, graph_descriptors_input]
        else:
            # Handle other cases (DGIN, etc.)
            edge_input = ks.layers.Input(**inputs[1])
            edge_index_input = ks.layers.Input(**inputs[2])
            input_tensors = [node_input, edge_input, edge_index_input]
            
            # Add edge_indices_reverse if provided (for DGIN, etc.)
            if len(inputs) > 3:
                edge_reverse_input = ks.layers.Input(**inputs[3])
                input_tensors.append(edge_reverse_input)
                
                # Add graph_descriptors if provided
                if len(inputs) > 4:
                    graph_descriptors_input = ks.layers.Input(**inputs[4])
                    input_tensors.append(graph_descriptors_input)
            else:
                # Only graph_descriptors provided (for GAT, GATv2, etc.)
                if len(inputs) > 3:
                    graph_descriptors_input = ks.layers.Input(**inputs[3])
                    input_tensors.append(graph_descriptors_input)
    else:
        # Inputs are already tensors
        input_tensors = inputs
    """
    Create a contrastive GNN model.
    
    Args:
        inputs: Model inputs configuration
        input_embedding: Input embedding configuration
        gnn_type: Type of GNN ('gin', 'gat', 'dmpnn', 'gcn', etc.)
        contrastive_args: Contrastive learning arguments
        depth: Number of GNN layers
        units: Number of units per layer
        use_set2set: Whether to use Set2Set pooling
        set2set_args: Set2Set arguments
        use_graph_state: Whether to use graph state (descriptors)
        output_embedding: Output embedding type
        output_to_tensor: Whether to convert output to tensor
        output_mlp: Output MLP configuration
        **kwargs: Additional arguments
        
    Returns:
        Contrastive GNN model
    """
    
    # Default contrastive arguments
    if contrastive_args is None:
        contrastive_args = {
            "num_views": 2,
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "augmentation_type": "molclr",  # Default augmentation strategy
            "augmentation_args": {},  # Arguments for augmentation
            "temperature": 0.1,
            "use_diversity_loss": True,
            "use_auxiliary_loss": True,
            "edge_drop_rate": 0.1,
            "node_mask_rate": 0.1
        }
    
    # Default Set2Set arguments
    if set2set_args is None:
        set2set_args = {
            "channels": units,
            "T": 3,
            "pooling_method": "sum",
            "init_qstar": "0"
        }
    
    # Default output MLP
    if output_mlp is None:
        output_mlp = {
            "use_bias": [True, True, True],
            "units": [units, units // 2, 2],
            "activation": ["relu", "relu", "linear"]
        }
    
    # Input embeddings
    if input_embedding and "node" in input_embedding:
        node_embedding_config = input_embedding["node"].copy()
        # Use DenseEmbedding for float inputs (node attributes)
        n = DenseEmbedding(units=node_embedding_config["output_dim"])(input_tensors[0])
    else:
        n = input_tensors[0]
    
    # Handle edge features - create simple constant edge features since we don't have them
    from kgcnn.layers.base import GraphBaseLayer
    
    class ConstantEdgeFeatures(GraphBaseLayer):
        """Create constant edge features."""
        def __init__(self, units=128, **kwargs):
            super(ConstantEdgeFeatures, self).__init__(**kwargs)
            self.units = units
            
        def call(self, inputs, training=None):
            # Create constant edge features
            edge_indices = inputs
            if isinstance(edge_indices, tf.RaggedTensor):
                # Handle ragged tensors
                flat_values = edge_indices.flat_values
                # Create constant features for flat values
                edge_features = tf.ones_like(flat_values, dtype=tf.float32)
                # Expand to match the units dimension
                edge_features = tf.expand_dims(edge_features, axis=-1)
                edge_features = tf.tile(edge_features, [1, 1, self.units])
                # Reconstruct ragged tensor
                return tf.RaggedTensor.from_nested_row_splits(
                    edge_features, edge_indices.nested_row_splits
                )
            else:
                # Handle regular tensors
                edge_features = tf.ones_like(edge_indices, dtype=tf.float32)
                # Expand to match the units dimension
                edge_features = tf.expand_dims(edge_features, axis=-1)
                edge_features = tf.tile(edge_features, [1, 1, self.units])
                return edge_features
    
    # Create edge features
    if gnn_type.lower() in ["gat", "gatv2", "dgin", "dmpnn"]:
        # For GAT/GATv2/DGIN/DMPNN, use actual edge attributes
        if input_embedding and "edge" in input_embedding:
            edge_embedding_config = input_embedding["edge"].copy()
            # Use DenseEmbedding for float inputs (edge attributes)
            e = DenseEmbedding(units=edge_embedding_config["output_dim"])(input_tensors[1])
        else:
            e = input_tensors[1]
        ed = input_tensors[2]  # edge_indices
    else:
        # For other GNNs (GIN, etc.), create constant edge features
        # Determine edge_indices position based on input structure
        if len(inputs) == 3:
            # ContrastiveGIN: edge_indices is at position 1
            ed = input_tensors[1]  # edge_indices
        else:
            # Other models: edge_indices is at position 2
            ed = input_tensors[2]  # edge_indices
        e = ConstantEdgeFeatures(units=units)(ed)  # Use edge_indices for constant features
    
    # Graph state (descriptors) - handle the case where graph_descriptors is at different positions
    graph_embedding = None
    if use_graph_state:
        if len(inputs) == 3:
            # ContrastiveGIN: graph_descriptors is at position 2
            if input_embedding and "graph" in input_embedding:
                graph_embedding_config = input_embedding["graph"].copy()
                # Use regular Dense layer for graph descriptors (they are regular tensors, not ragged)
                graph_embedding = ks.layers.Dense(units=graph_embedding_config["output_dim"])(input_tensors[2])
            else:
                graph_embedding = input_tensors[2]
        elif len(inputs) == 4:
            # ContrastiveGAT: graph_descriptors is at position 3
            if input_embedding and "graph" in input_embedding:
                graph_embedding_config = input_embedding["graph"].copy()
                # Use regular Dense layer for graph descriptors (they are regular tensors, not ragged)
                graph_embedding = ks.layers.Dense(units=graph_embedding_config["output_dim"])(input_tensors[3])
            else:
                graph_embedding = input_tensors[3]
        elif len(inputs) == 5:
            # ContrastiveDMPNN: graph_descriptors is at position 4
            if input_embedding and "graph" in input_embedding:
                graph_embedding_config = input_embedding["graph"].copy()
                # Use regular Dense layer for graph descriptors (they are regular tensors, not ragged)
                graph_embedding = ks.layers.Dense(units=graph_embedding_config["output_dim"])(input_tensors[4])
            else:
                graph_embedding = input_tensors[4]
        elif len(input_tensors) > 4:
            # We have separate graph_descriptors input (for DGIN, etc.)
            if input_embedding and "graph" in input_embedding:
                graph_embedding_config = input_embedding["graph"].copy()
                # Use regular Dense layer for graph descriptors (they are regular tensors, not ragged)
                graph_embedding = ks.layers.Dense(units=graph_embedding_config["output_dim"])(input_tensors[4])
            else:
                graph_embedding = input_tensors[4]
        elif len(input_tensors) > 3:
            # We have separate graph_descriptors input (for GAT, GATv2, etc.)
            if input_embedding and "graph" in input_embedding:
                graph_embedding_config = input_embedding["graph"].copy()
                # Use DenseEmbedding for float inputs (graph descriptors)
                graph_embedding = DenseEmbedding(units=graph_embedding_config["output_dim"])(input_tensors[3])
            else:
                graph_embedding = input_tensors[3]
    
    # Create contrastive GNN layers based on type
    if gnn_type.lower() == "gin":
        gnn_layers = []
        for i in range(depth):
            layer = ContrastiveGINConv(
                units=units,
                num_views=contrastive_args["num_views"],
                depth=2,  # Internal GIN depth
                use_contrastive_loss=contrastive_args["use_contrastive_loss"],
                contrastive_loss_type=contrastive_args["contrastive_loss_type"],
                temperature=contrastive_args["temperature"],
                use_diversity_loss=contrastive_args["use_diversity_loss"],
                use_auxiliary_loss=contrastive_args["use_auxiliary_loss"]
            )
            gnn_layers.append(layer)
        
        # Apply GNN layers
        for layer in gnn_layers:
            n = layer([n, e, ed])
    
    elif gnn_type.lower() == "attfp":
        # Contrastive AttentiveFP implementation
        gnn_layers = []
        for i in range(depth):
            layer = ContrastiveAttFPConv(
                units=units,
                depth=2,  # Internal AttFP depth
                attention_heads=contrastive_args.get("attention_heads", 8),
                dropout_rate=contrastive_args.get("dropout_rate", 0.1),
                num_views=contrastive_args["num_views"],
                edge_drop_rate=contrastive_args.get("edge_drop_rate", 0.1),
                node_mask_rate=contrastive_args.get("node_mask_rate", 0.1),
                feature_noise_std=contrastive_args.get("feature_noise_std", 0.01),
                use_contrastive_loss=contrastive_args["use_contrastive_loss"],
                contrastive_loss_type=contrastive_args["contrastive_loss_type"],
                temperature=contrastive_args["temperature"]
            )
            gnn_layers.append(layer)
        
        # Apply AttFP layers
        for layer in gnn_layers:
            n, _ = layer([n, e, ed])
    
    elif gnn_type.lower() == "addgnn":
        # Contrastive AddGNN implementation
        gnn_layers = []
        for i in range(depth):
            layer = ContrastiveAddGNNConv(
                units=units,
                depth=2,  # Internal AddGNN depth
                heads=contrastive_args.get("heads", 4),
                dropout_rate=contrastive_args.get("dropout_rate", 0.1),
                num_views=contrastive_args["num_views"],
                edge_drop_rate=contrastive_args.get("edge_drop_rate", 0.1),
                node_mask_rate=contrastive_args.get("node_mask_rate", 0.1),
                feature_noise_std=contrastive_args.get("feature_noise_std", 0.01),
                use_contrastive_loss=contrastive_args["use_contrastive_loss"],
                contrastive_loss_type=contrastive_args["contrastive_loss_type"],
                temperature=contrastive_args["temperature"],
                use_set2set=use_set2set,
                set2set_args=set2set_args
            )
            gnn_layers.append(layer)
        
        # Apply AddGNN layers
        for layer in gnn_layers:
            n, _ = layer([n, ed])
    
    elif gnn_type.lower() == "dgin":
        # Contrastive DGIN implementation using real DGIN layers (like DMPNN)
        # DGIN requires edge_indices_reverse
        if len(input_tensors) < 4:
            raise ValueError("DGIN requires edge_indices_reverse input")
        ed_reverse = input_tensors[3]
        
        gnn_layers = []
        
        # Import the DGIN pooling layer
        from kgcnn.literature.ContrastiveGNN._contrastive_dgin_conv import DGINPPoolingEdgesDirected
        
        for i in range(depth):
            # Create multiple DGIN views
            dgin_views = []
            for view_idx in range(contrastive_args["num_views"]):
                # Use real DGIN layer
                dgin_layer = DGINPPoolingEdgesDirected(
                    units=units,
                    use_bias=kwargs.get("use_bias", True),
                    activation=kwargs.get("activation", "relu")
                )
                
                # Apply DGIN layer
                hv = dgin_layer([n, e, ed, ed_reverse])
                
                # Apply dropout if specified
                if kwargs.get("dropout") is not None:
                    hv = Dropout(**kwargs["dropout"])(hv)
                
                dgin_views.append(hv)
            
            # Store the DGIN views for contrastive loss computation
            gnn_layers.append(dgin_views)
            
            # Combine view outputs
            n = tf.add_n(dgin_views) / len(dgin_views)
    
    elif gnn_type.lower() == "gat":
        # Contrastive GAT implementation
        gnn_layers = []
        for i in range(depth):
            # Create multiple GAT views
            gat_views = []
            for view_idx in range(contrastive_args["num_views"]):
                gat_layer = AttentionHeadGAT(
                    units=units,
                    use_edge_features=True,
                    use_final_activation=True,
                    has_self_loops=True,
                    activation="relu",
                    use_bias=True
                )
                gat_views.append(gat_layer)
            
            # Store the GAT views for contrastive loss computation
            gnn_layers.append(gat_views)
            
            # Apply GAT views and combine
            view_outputs = []
            for gat_layer in gat_views:
                view_output = gat_layer([n, e, ed])
                view_outputs.append(view_output)
            
            # Combine view outputs
            n = tf.add_n(view_outputs) / len(view_outputs)
    
    elif gnn_type.lower() == "gatv2":
        # Contrastive GATv2 implementation
        gnn_layers = []
        for i in range(depth):
            # Create multiple GATv2 views
            gatv2_views = []
            for view_idx in range(contrastive_args["num_views"]):
                gatv2_layer = AttentionHeadGATV2(
                    units=units,
                    use_edge_features=True,
                    use_final_activation=True,
                    has_self_loops=True,
                    activation="relu",
                    use_bias=True
                )
                gatv2_views.append(gatv2_layer)
            
            # Store the GATv2 views for contrastive loss computation
            gnn_layers.append(gatv2_views)
            
            # Apply GATv2 views and combine
            view_outputs = []
            for gatv2_layer in gatv2_views:
                view_output = gatv2_layer([n, e, ed])
                view_outputs.append(view_output)
            
            # Combine view outputs
            n = tf.add_n(view_outputs) / len(view_outputs)
    
    elif gnn_type.lower() == "gin":
        # Contrastive GIN implementation with flexible augmentation
        gnn_layers = []
        
        # Create augmentation layer
        augmentation_layer = create_augmentation_layer(
            augmentation_type=contrastive_args.get("augmentation_type", "molclr"),
            **contrastive_args.get("augmentation_args", {})
        )
        
        for i in range(depth):
            # Single GIN layer
            gin_mlp = MLP(
                units=[units, units],
                use_bias=True,
                activation="relu"
            )
            
            # Gather and aggregate
            gather = GatherNodes()
            aggregate = AggregateLocalEdges(pooling_method="sum")
            
            # Apply augmentation to create multiple views
            view_outputs = []
            for view_idx in range(contrastive_args["num_views"]):
                # Apply augmentation to inputs
                aug_inputs = augmentation_layer([n, e, ed])
                aug_n, aug_e, aug_ed = aug_inputs
                
                # Process nodes with augmented inputs
                neighbor_features = gather([aug_n, aug_ed])
                aggregated = aggregate([aug_n, neighbor_features, aug_ed])
                gin_output = gin_mlp(aggregated)
                
                view_outputs.append(gin_output)
            
            # Store views for contrastive loss computation
            gnn_layers.append(view_outputs)
            
            # Combine view outputs (average)
            n = tf.add_n(view_outputs) / len(view_outputs)
    
    elif gnn_type.lower() == "dmpnn":
        # Contrastive DMPNN implementation using real DMPNN layers
        from kgcnn.literature.DMPNN._dmpnn_conv import DMPNNPPoolingEdgesDirected
        
        gnn_layers = []
        
        # Make first edge hidden h0 (same for all views)
        h_n0 = GatherNodesOutgoing()([n, ed])
        h0 = LazyConcatenate(axis=-1)([h_n0, e])
        h0 = Dense(**kwargs.get("edge_initialize", {"units": units, "use_bias": True, "activation": "relu"}))(h0)
        
        # One Dense layer for all message steps
        edge_dense_all = Dense(**kwargs.get("edge_dense", {"units": units, "use_bias": True, "activation": "linear"}))
        
        for i in range(depth):
            # Create multiple DMPNN views
            dmpnn_views = []
            for view_idx in range(contrastive_args["num_views"]):
                # Use real DMPNN layer
                h = h0 if i == 0 else h  # Use h0 for first iteration
                m_vw = DMPNNPPoolingEdgesDirected()([n, h, ed, input_tensors[3]])  # input_tensors[3] is edge_indices_reverse
                h = edge_dense_all(m_vw)
                h = LazyAdd()([h, h0])
                h = Activation(**kwargs.get("edge_activation", {"activation": "relu"}))(h)
                
                if kwargs.get("dropout") is not None:
                    h = Dropout(**kwargs["dropout"])(h)
                
                # Aggregate to nodes
                mv = AggregateLocalEdges(**kwargs.get("pooling_args", {"pooling_method": "sum"}))([n, h, ed])
                mv = LazyConcatenate(axis=-1)([mv, n])
                hv = Dense(**kwargs.get("node_dense", {"units": units, "use_bias": True, "activation": "relu"}))(mv)
                
                dmpnn_views.append(hv)
            
            # Store the DMPNN views for contrastive loss computation
            gnn_layers.append(dmpnn_views)
            
            # Combine view outputs
            n = tf.add_n(dmpnn_views) / len(dmpnn_views)
    
    elif gnn_type.lower() == "pna":
        # Contrastive PNA implementation
        gnn_layers = []
        for i in range(depth):
            layer = ContrastivePNAConv(
                units=units,
                depth=2,  # Internal PNA depth
                aggregators=contrastive_args.get("aggregators", ["mean", "max", "sum"]),
                scalers=contrastive_args.get("scalers", ["identity", "amplification", "attenuation"]),
                delta=contrastive_args.get("delta", 1.0),
                dropout_rate=contrastive_args.get("dropout_rate", 0.1),
                num_views=contrastive_args["num_views"],
                edge_drop_rate=contrastive_args.get("edge_drop_rate", 0.1),
                node_mask_rate=contrastive_args.get("node_mask_rate", 0.1),
                feature_noise_std=contrastive_args.get("feature_noise_std", 0.01),
                use_contrastive_loss=contrastive_args["use_contrastive_loss"],
                contrastive_loss_type=contrastive_args["contrastive_loss_type"],
                temperature=contrastive_args["temperature"]
            )
            gnn_layers.append(layer)
        
        # Apply PNA layers
        for layer in gnn_layers:
            n, _ = layer([n, ed])
    
    elif gnn_type.lower() == "moe":
        # Contrastive MoE implementation
        gnn_layers = []
        for i in range(depth):
            # Create multiple MoE views
            moe_views = []
            for view_idx in range(contrastive_args["num_views"]):
                # Simple MoE-like layer with multiple experts
                expert1 = MLP(units=[units, units], use_bias=True, activation="relu")
                expert2 = MLP(units=[units, units], use_bias=True, activation="relu")
                expert3 = MLP(units=[units, units], use_bias=True, activation="relu")
                
                # Gather and aggregate
                gather = GatherNodes()
                aggregate = AggregateLocalEdges(pooling_method="sum")
                
                # Process nodes
                neighbor_features = gather([n, ed])
                aggregated = aggregate([n, neighbor_features, ed])
                
                # Apply experts
                expert1_out = expert1(aggregated)
                expert2_out = expert2(aggregated)
                expert3_out = expert3(aggregated)
                
                # Combine experts (simple averaging for now)
                moe_output = (expert1_out + expert2_out + expert3_out) / 3.0
                
                moe_views.append(moe_output)
            
            # Store the MoE views for contrastive loss computation
            gnn_layers.append(moe_views)
            
            # Combine view outputs
            n = tf.add_n(moe_views) / len(moe_views)
    
    else:
        raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    # Set2Set pooling if enabled
    if use_set2set:
        n = PoolingSet2SetEncoder(**set2set_args)(n)
    
    # Graph state fusion if provided
    if use_graph_state and graph_embedding is not None:
        # Pool node features to match graph embedding shape
        pooled_nodes = PoolingNodes(pooling_method="sum")(n)
        # Concatenate graph embedding with pooled node features
        n = ks.layers.Concatenate()([pooled_nodes, graph_embedding])
    
    # Output embedding choice
    if output_embedding == "graph":
        if use_graph_state and graph_embedding is not None:
            # Already pooled above
            out = n
        else:
            out = PoolingNodes(pooling_method="sum")(n)
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError(f"Unsupported output embedding for mode {output_embedding}")
    
    # Output MLP
    if output_mlp is not None:
        out = MLP(**output_mlp)(out)
    
    # Convert to tensor if needed
    if output_to_tensor:
        out = ks.layers.Lambda(
            lambda x: tf.RaggedTensor.to_tensor(x) if hasattr(x, 'to_tensor') else x
        )(out)
    
    # Create model
    model = ks.models.Model(inputs=input_tensors, outputs=out)
    
    # Store contrastive layers for loss computation
    model.contrastive_layers = gnn_layers
    model.contrastive_args = contrastive_args
    
    return model


def make_contrastive_gin_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive GIN model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="gin",
        **kwargs
    )


def make_contrastive_gat_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive GAT model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="gat",
        **kwargs
    )


def make_contrastive_gatv2_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive GATv2 model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="gatv2",
        **kwargs
    )


def make_contrastive_dmpnn_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive DMPNN model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="dmpnn",
        **kwargs
    )


def make_contrastive_attfp_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive AttentiveFP model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="attfp",
        **kwargs
    )


def make_contrastive_addgnn_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive AddGNN model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="addgnn",
        **kwargs
    )


def make_contrastive_dgin_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive DGIN model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="dgin",
        **kwargs
    )


def make_contrastive_pna_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive PNA model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="pna",
        **kwargs
    )


def make_contrastive_moe_model(inputs, input_embedding=None, **kwargs):
    """Create a contrastive MoE model."""
    return make_contrastive_gnn_model(
        inputs=inputs,
        input_embedding=input_embedding,
        gnn_type="moe",
        **kwargs
    )


def compile_contrastive_gnn_model(
    model,
    optimizer="adam",
    learning_rate=0.001,
    loss="mse",
    metrics=None,
    contrastive_weight=0.1,
    diversity_weight=0.01,
    alignment_weight=0.01,
    use_regression_aware=True,
    target_similarity_threshold=0.1,
    similarity_metric='euclidean',
    **kwargs
):
    """
    Compile a contrastive GNN model with appropriate losses and metrics.
    
    Args:
        model: The contrastive GNN model
        optimizer: Optimizer name or instance
        learning_rate: Learning rate
        loss: Main task loss (e.g., 'mse' for regression)
        metrics: Additional metrics
        contrastive_weight: Weight for contrastive loss
        diversity_weight: Weight for diversity loss
        alignment_weight: Weight for alignment loss
        use_regression_aware: Whether to use regression-aware contrastive losses
        target_similarity_threshold: Threshold for target similarity in regression
        similarity_metric: Similarity metric for regression ('euclidean', 'cosine', 'absolute')
        **kwargs: Additional arguments
        
    Returns:
        Compiled model
    """
    
    # Set up optimizer
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.get(optimizer)
    
    # Set up main task loss
    if isinstance(loss, str):
        if loss.lower() == "mse":
            main_loss = tf.keras.losses.MeanSquaredError()
        elif loss.lower() == "mae":
            main_loss = tf.keras.losses.MeanAbsoluteError()
        elif loss.lower() == "huber":
            main_loss = tf.keras.losses.Huber()
        else:
            main_loss = tf.keras.losses.get(loss)
    else:
        main_loss = loss
    
    # Set up contrastive losses
    if use_regression_aware:
        # Use regression-aware contrastive losses
        contrastive_loss = RegressionContrastiveGNNLoss(
            target_similarity_threshold=target_similarity_threshold,
            similarity_metric=similarity_metric,
            use_adaptive_threshold=True
        )
        triplet_loss = RegressionContrastiveGNNTripletLoss(
            target_distance_threshold=target_similarity_threshold,
            similarity_metric=similarity_metric
        )
    else:
        # Use standard contrastive losses
        contrastive_loss = ContrastiveGNNLoss()
        triplet_loss = ContrastiveGNNTripletLoss()
    
    diversity_loss = ContrastiveGNNDiversityLoss()
    alignment_loss = ContrastiveGNNAlignmentLoss()
    
    # Create custom loss function that combines main task and contrastive losses
    def combined_loss(y_true, y_pred):
        # Main task loss
        main_task_loss = main_loss(y_true, y_pred)
        
        # Get embeddings from the model (assuming model has embeddings attribute)
        if hasattr(model, 'embeddings') and model.embeddings is not None:
            embeddings = model.embeddings
            
            # Contrastive loss
            contrastive_loss_value = contrastive_loss(y_true, embeddings)
            
            # Triplet loss
            triplet_loss_value = triplet_loss(y_true, embeddings)
            
            # Diversity loss
            diversity_loss_value = diversity_loss(y_true, embeddings)
            
            # Alignment loss (if we have multiple views)
            if hasattr(model, 'view_embeddings') and model.view_embeddings is not None:
                alignment_loss_value = alignment_loss(y_true, model.view_embeddings)
            else:
                alignment_loss_value = 0.0
            
            # Combine all losses
            total_loss = (main_task_loss + 
                         contrastive_weight * contrastive_loss_value +
                         contrastive_weight * triplet_loss_value +
                         diversity_weight * diversity_loss_value +
                         alignment_weight * alignment_loss_value)
        else:
            total_loss = main_task_loss
        
        return total_loss
    
    # Set up metrics
    if metrics is None:
        metrics = []
    
    # Add contrastive metrics
    contrastive_metrics = create_contrastive_gnn_metrics()
    all_metrics = metrics + contrastive_metrics
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=all_metrics
    )
    
    return model


# Model factory dictionary
contrastive_gnn_models = {
    "ContrastiveGIN": make_contrastive_gin_model,
    "ContrastiveGAT": make_contrastive_gat_model,
    "ContrastiveDMPNN": make_contrastive_dmpnn_model,
    "ContrastiveGNN": make_contrastive_gnn_model
} 