import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._graphgps_conv import GraphGPSConv
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing, GatherState
from kgcnn.layers.modules import Dense, LazyConcatenate, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.aggr import AggregateLocalEdges
from ...layers.pooling import PoolingNodes
from kgcnn.layers.set2set import PoolingSet2SetEncoder
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer, ShiftPeriodicLattice


def make_graphgps_model(
    inputs,
    input_embedding=None,
    input_node_embedding=None,
    input_edge_embedding=None,
    input_graph_embedding=None,
    cast_disjoint_kwargs=None,
    input_block_cfg=None,
    use_graph_state=False,
    graphgps_args=None,
    depth=None,
    node_dim=None,
    use_set2set=False,
    set2set_args=None,
    node_mlp_args=None,
    edge_mlp_args=None,
    graph_mlp_args=None,
    use_node_embedding=True,
    use_edge_embedding=True,
    use_graph_embedding=False,
    output_embedding="graph",
    output_to_tensor=True,
    output_mlp=None,
    output_scaling=None,
    output_tensor_type="padded",
    output_tensor_kwargs=None,
    **kwargs
):
    """
    Make GraphGPS model.
    
    Args:
        inputs (list): List of input tensors
        input_embedding (dict): Input embedding configuration
        input_node_embedding (dict): Node embedding configuration
        input_edge_embedding (dict): Edge embedding configuration
        input_graph_embedding (dict): Graph embedding configuration
        cast_disjoint_kwargs (dict): Casting arguments
        input_block_cfg (dict): Input block configuration
        use_graph_state (bool): Whether to use graph state
        graphgps_args (dict): GraphGPS layer arguments
        depth (int): Number of GraphGPS layers
        node_dim (int): Node feature dimension
        use_set2set (bool): Whether to use Set2Set pooling
        set2set_args (dict): Set2Set arguments
        node_mlp_args (dict): Node MLP arguments
        edge_mlp_args (dict): Edge MLP arguments
        graph_mlp_args (dict): Graph MLP arguments
        use_node_embedding (bool): Whether to use node embedding
        use_edge_embedding (bool): Whether to use edge embedding
        use_graph_embedding (bool): Whether to use graph embedding
        output_embedding (str): Output embedding type
        output_to_tensor (bool): Whether to convert output to tensor
        output_mlp (dict): Output MLP configuration
        output_scaling (dict): Output scaling configuration
        output_tensor_type (str): Output tensor type
        output_tensor_kwargs (dict): Output tensor arguments
        **kwargs: Additional arguments
        
    Returns:
        tf.keras.Model: GraphGPS model
    """
    
    # Process inputs directly
    if inputs is None:
        raise ValueError("Inputs must be provided for GraphGPS model")
    
    # Extract input configurations
    input_node = inputs[0] if len(inputs) > 0 else {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True}
    input_edge = inputs[1] if len(inputs) > 1 else {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True}
    input_edge_index = inputs[2] if len(inputs) > 2 else {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
    input_graph = inputs[3] if len(inputs) > 3 and use_graph_state else None
    
    # Process embeddings
    input_node_embedding = input_node_embedding or input_embedding.get("node") if input_embedding else None
    input_edge_embedding = input_edge_embedding or input_embedding.get("edge") if input_embedding else None
    input_graph_embedding = input_graph_embedding or input_embedding.get("graph") if input_embedding else None
    
    # Default GraphGPS arguments
    if graphgps_args is None:
        graphgps_args = {
            "units": 200,
            "heads": 8,
            "dropout": 0.1,
            "use_bias": True,
            "activation": "relu",
            "mp_type": "gcn",
            "attn_type": "multihead",
            "use_skip_connection": True,
            "use_layer_norm": True,
            "use_batch_norm": False
        }
    
    # Default depth
    if depth is None:
        depth = 3
    
    # Default node dimension
    if node_dim is None:
        node_dim = graphgps_args.get("units", 200)
    
    # Input layers
    node_input = tf.keras.layers.Input(**input_node)
    edge_input = tf.keras.layers.Input(**input_edge)
    edge_index_input = tf.keras.layers.Input(**input_edge_index)
    
    # Graph descriptor input (if using graph state)
    if use_graph_state and input_graph is not None:
        graph_descriptors_input = tf.keras.layers.Input(**input_graph)
    else:
        graph_descriptors_input = None
    
    # Cast to disjoint representation
    if cast_disjoint_kwargs is not None:
        cast_layer = ChangeTensorType(**cast_disjoint_kwargs)
        node_input, edge_input, edge_index_input = cast_layer([node_input, edge_input, edge_index_input])
    
    # Input embeddings
    if use_node_embedding and input_node_embedding is not None:
        node_embedding = OptionalInputEmbedding(**input_node_embedding)
        node_input = node_embedding(node_input)
    
    if use_edge_embedding and input_edge_embedding is not None:
        edge_embedding = OptionalInputEmbedding(**input_edge_embedding)
        edge_input = edge_embedding(edge_input)
    
    if use_graph_embedding and input_graph_embedding is not None:
        graph_embedding = OptionalInputEmbedding(**input_graph_embedding)
        graph_descriptors_input = graph_embedding(graph_descriptors_input)
    
    # GraphGPS layers
    node_features = node_input
    edge_features = edge_input
    
    for i in range(depth):
        # GraphGPS convolution
        graphgps_layer = GraphGPSConv(**graphgps_args)
        
        if use_graph_state and graph_descriptors_input is not None:
            # Add graph descriptors to node features using a Lambda layer
            def add_graph_descriptors(inputs):
                node_feats, graph_desc = inputs
                # Get the row lengths to know how many nodes per graph
                row_lengths = node_feats.row_lengths()
                # Repeat graph descriptors for each node in each graph
                graph_desc_expanded = tf.repeat(graph_desc, row_lengths, axis=0)
                # Concatenate along feature dimension
                combined_values = tf.concat([node_feats.values, graph_desc_expanded], axis=-1)
                # Convert back to ragged tensor
                return tf.RaggedTensor.from_row_splits(combined_values, node_feats.row_splits)
            
            node_features = tf.keras.layers.Lambda(add_graph_descriptors)([node_features, graph_descriptors_input])
        
        # Apply GraphGPS layer
        node_features = graphgps_layer([node_features, edge_features, edge_index_input])
        
        # Update edge features if needed
        if edge_mlp_args is not None:
            edge_mlp = MLP(**edge_mlp_args)
            edge_features = edge_mlp(edge_features)
    
    # Output processing
    if output_embedding == "graph":
        # Graph-level pooling
        if use_set2set:
            if set2set_args is None:
                set2set_args = {"channels": node_dim, "T": 3, "pooling_method": "sum"}
            # Ensure channels match the node feature dimension
            feature_dim = node_features.shape[-1]
            if feature_dim is None:
                feature_dim = node_dim  # fallback
            set2set_args["channels"] = feature_dim
            pooling = PoolingSet2SetEncoder(**set2set_args)
            out = pooling(node_features)
        else:
            pooling = PoolingNodes(pooling_method="sum")
            out = pooling([node_features, edge_index_input])
    elif output_embedding == "node":
        out = node_features
    else:
        raise ValueError(f"Unsupported output_embedding: {output_embedding}")
    
    # Output MLP
    if output_mlp is not None:
        output_mlp_layer = MLP(**output_mlp)
        out = output_mlp_layer(out)
    
    # Output scaling
    if output_scaling is not None:
        scaling_layer = tf.keras.layers.experimental.preprocessing.Rescaling(**output_scaling)
        out = scaling_layer(out)
    
    # Convert to tensor if needed
    if output_to_tensor:
        out = tf.keras.layers.Lambda(lambda x: tf.RaggedTensor.to_tensor(x) if hasattr(x, 'to_tensor') else x)(out)
    
    # Create model
    if use_graph_state:
        model = tf.keras.Model(inputs=[node_input, edge_input, edge_index_input, graph_descriptors_input], outputs=out)
    else:
        model = tf.keras.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)
    
    return model 