"""Graph Transformer model factory.

This module provides factory functions to create Graph Transformer models
for both molecular and crystal applications.
"""

import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.modules import Dense, OptionalInputEmbedding, LazyConcatenate
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.set2set import PoolingSet2SetEncoder
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer, ShiftPeriodicLattice
from ._graph_transformer_conv import GraphTransformerLayer
from kgcnn.model.utils import update_model_kwargs

ks = tf.keras

__model_version__ = "2025.01.27"

model_default = {
    "name": "GraphTransformer",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None,), "name": "graph_desc", "dtype": "float32", "ragged": False}
    ],
    "input_embedding": {
        "node": {"input_dim": 95, "output_dim": 200},
        "edge": {"input_dim": 5, "output_dim": 200},
        "graph": {"input_dim": 100, "output_dim": 64}
    },
    "use_graph_state": True,  # Default to True for GraphTransformer to use descriptors
    "transformer_args": {
        "units": 200,
        "num_heads": 8,
        "ff_units": 800,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "use_edge_features": True,
        "use_positional_encoding": True,
        "positional_encoding_dim": 64,
        "activation": "relu",
        "layer_norm_epsilon": 1e-6
    },
    "depth": 3,
    "use_set2set": True,
    "set2set_args": {
        "channels": 32,
        "T": 3,
        "pooling_method": "sum",
        "init_qstar": "0"
    },
    "pooling_args": {"pooling_method": "segment_sum"},
    "output_embedding": 'graph',
    "output_to_tensor": True,
    "output_mlp": {
        "use_bias": [True, True, False],
        "units": [200, 100, 1],
        "activation": ["relu", "relu", "linear"]
    },
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               transformer_args: dict = None,
               depth: int = None,
               use_graph_state: bool = True,
               use_set2set: bool = True,
               set2set_args: dict = None,
               pooling_args: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               name: str = None,
               **kwargs):
    """Make Graph Transformer model.
    
    Args:
        inputs: List of input dictionaries for tf.keras.layers.Input
        input_embedding: Dictionary of embedding arguments
        transformer_args: Arguments for GraphTransformerLayer
        depth: Number of transformer layers
        use_graph_state: Whether to use graph descriptors
        use_set2set: Whether to use Set2Set pooling
        set2set_args: Arguments for Set2Set pooling
        pooling_args: Arguments for pooling
        output_embedding: Output embedding type ('graph', 'node', 'edge')
        output_to_tensor: Whether to cast output to tensor
        output_mlp: Arguments for output MLP
        name: Model name
        **kwargs: Additional arguments
        
    Returns:
        tf.keras.models.Model: Graph Transformer model
    """
    # Make inputs
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    
    # Handle graph_desc input if provided
    if len(inputs) > 3:
        graph_desc_input = ks.layers.Input(**inputs[3])
    else:
        graph_desc_input = None

    # Embedding layers
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    e = OptionalInputEmbedding(**input_embedding['edge'],
                               use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    
    # Embed graph_desc if provided
    if graph_desc_input is not None and "graph" in input_embedding:
        graph_desc = OptionalInputEmbedding(
            **input_embedding["graph"],
            use_embedding=len(inputs[3]["shape"]) < 1)(graph_desc_input)
    else:
        graph_desc = None

    # Generate positional encodings if requested
    positional_encoding = None
    if transformer_args.get("use_positional_encoding", True):
        # Create learnable positional encodings based on node indices
        pos_dim = transformer_args.get("positional_encoding_dim", 64)
        positional_encoding = Dense(pos_dim, activation="linear")(n)
    
    # Graph Transformer layers
    for i in range(depth):
        transformer_inputs = [n, e, edge_index_input]
        if positional_encoding is not None:
            transformer_inputs.append(positional_encoding)
        if graph_desc is not None:
            transformer_inputs.append(graph_desc)
        
        n = GraphTransformerLayer(**transformer_args)(transformer_inputs)
    
    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            out = PoolingSet2SetEncoder(**set2set_args)(n)
        else:
            out = PoolingNodes(**pooling_args)(n)
        
        # Add graph descriptor if provided
        if graph_desc is not None:
            # Flatten graph_desc to match the rank of out
            graph_desc_flat = tf.keras.layers.Flatten()(graph_desc)
            out = LazyConcatenate(axis=-1)([out, graph_desc_flat])
        
        out = MLP(**output_mlp)(out)
        
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `GraphTransformer`")

    # Create model
    model_inputs = [node_input, edge_input, edge_index_input]
    if graph_desc_input is not None:
        model_inputs.append(graph_desc_input)
    
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    
    return model


@update_model_kwargs(model_default)
def make_crystal_model(inputs: list = None,
                      input_embedding: dict = None,
                      transformer_args: dict = None,
                      depth: int = None,
                      use_graph_state: bool = True,
                      use_set2set: bool = True,
                      set2set_args: dict = None,
                      pooling_args: dict = None,
                      output_embedding: str = None,
                      output_to_tensor: bool = None,
                      output_mlp: dict = None,
                      name: str = None,
                      **kwargs):
    """Make Graph Transformer model for crystal structures.
    
    This version includes geometric features like distances and positions.
    
    Args:
        inputs: List of input dictionaries for tf.keras.layers.Input
        input_embedding: Dictionary of embedding arguments
        transformer_args: Arguments for GraphTransformerLayer
        depth: Number of transformer layers
        use_graph_state: Whether to use graph descriptors
        use_set2set: Whether to use Set2Set pooling
        set2set_args: Arguments for Set2Set pooling
        pooling_args: Arguments for pooling
        output_embedding: Output embedding type ('graph', 'node', 'edge')
        output_to_tensor: Whether to cast output to tensor
        output_mlp: Arguments for output MLP
        name: Model name
        **kwargs: Additional arguments
        
    Returns:
        tf.keras.models.Model: Graph Transformer model for crystals
    """
    # Make inputs
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    
    # Handle graph_desc input if provided
    if len(inputs) > 3:
        graph_desc_input = ks.layers.Input(**inputs[3])
    else:
        graph_desc_input = None

    # Embedding layers
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    e = OptionalInputEmbedding(**input_embedding['edge'],
                               use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    
    # Embed graph_desc if provided
    if graph_desc_input is not None and "graph" in input_embedding:
        graph_desc = OptionalInputEmbedding(
            **input_embedding["graph"],
            use_embedding=len(inputs[3]["shape"]) < 1)(graph_desc_input)
    else:
        graph_desc = None

    # Add geometric features for crystals
    # Node positions (if available in inputs)
    if len(inputs) > 4:
        pos_input = ks.layers.Input(**inputs[4])
        pos = NodePosition()(pos_input)
        # Add position information to node features
        n = LazyConcatenate(axis=-1)([n, pos])
    
    # Edge distances (if available in inputs)
    if len(inputs) > 5:
        dist_input = ks.layers.Input(**inputs[5])
        dist = NodeDistanceEuclidean()(dist_input)
        # Add distance information to edge features
        e = LazyConcatenate(axis=-1)([e, dist])

    # Generate positional encodings if requested
    positional_encoding = None
    if transformer_args.get("use_positional_encoding", True):
        # Create learnable positional encodings based on node indices
        pos_dim = transformer_args.get("positional_encoding_dim", 64)
        positional_encoding = Dense(pos_dim, activation="linear")(n)
    
    # Graph Transformer layers
    for i in range(depth):
        transformer_inputs = [n, e, edge_index_input]
        if positional_encoding is not None:
            transformer_inputs.append(positional_encoding)
        if graph_desc is not None:
            transformer_inputs.append(graph_desc)
        
        n = GraphTransformerLayer(**transformer_args)(transformer_inputs)
    
    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            out = PoolingSet2SetEncoder(**set2set_args)(n)
        else:
            out = PoolingNodes(**pooling_args)(n)
        
        # Add graph descriptor if provided
        if graph_desc is not None:
            # Flatten graph_desc to match the rank of out
            graph_desc_flat = tf.keras.layers.Flatten()(graph_desc)
            out = LazyConcatenate(axis=-1)([out, graph_desc_flat])
        
        out = MLP(**output_mlp)(out)
        
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `GraphTransformer`")

    # Create model
    model_inputs = [node_input, edge_input, edge_index_input]
    if graph_desc_input is not None:
        model_inputs.append(graph_desc_input)
    if len(inputs) > 4:
        model_inputs.append(pos_input)
    if len(inputs) > 5:
        model_inputs.append(dist_input)
    
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    
    return model 