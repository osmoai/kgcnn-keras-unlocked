"""DHTNN Model Factory.

This module provides the model factory for building DHTNN (Double-Head Transformer Neural Network)
models with double-head attention blocks to enhance DMPNN performance.
"""

import tensorflow as tf
from kgcnn.layers.modules import Dense, Activation, Dropout
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from kgcnn.layers_core.modules import OptionalInputEmbedding
from kgcnn.model.utils import update_model_kwargs
from ._dhtnn_conv import DHTNNConv, DoubleHeadAttention

ks = tf.keras

# Keep track of model version from commit date in literature.
__kgcnn_model_version__ = "2024.01.15"

model_default = {
    "name": "DHTNN",
    "inputs": [
        {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {
        "node": {"input_dim": 95, "output_dim": 128},
        "edge": {"input_dim": 5, "output_dim": 128}
    },
    "dhtnn_args": {
        "units": 128,
        "local_heads": 4,
        "global_heads": 4,
        "local_attention_units": 64,
        "global_attention_units": 64,
        "use_edge_features": True,
        "use_final_activation": True,
        "has_self_loops": True,
        "dropout_rate": 0.1,
        "activation": "relu",
        "use_bias": True
    },
    "depth": 4,
    "verbose": 10,
    "pooling_nodes_args": {"pooling_method": "sum"},
    "use_graph_state": False,
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {
        "use_bias": [True, True, False],
        "units": [128, 64, 1],
        "activation": ["relu", "relu", "linear"]
    }
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               dhtnn_args: dict = None,
               pooling_nodes_args: dict = None,
               use_graph_state: bool = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None):
    """Make DHTNN model.
    
    Args:
        inputs (list): List of input tensors
        input_embedding (dict): Input embedding configuration
        depth (int): Number of DHTNN layers
        dhtnn_args (dict): DHTNN layer arguments
        pooling_nodes_args (dict): Node pooling arguments
        use_graph_state (bool): Whether to use graph state
        name (str): Model name
        verbose (int): Verbosity level
        output_embedding (str): Output embedding type
        output_to_tensor (bool): Whether to convert output to tensor
        output_mlp (dict): Output MLP configuration
        
    Returns:
        tf.keras.Model: DHTNN model
    """
    
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    
    # Handle graph_descriptors input if provided (for descriptors)
    if len(inputs) > 3:
        graph_descriptors_input = ks.layers.Input(**inputs[3])
    else:
        graph_descriptors_input = None
    
    # Embedding
    n = OptionalInputEmbedding(**input_embedding["node"])(node_input)
    e = OptionalInputEmbedding(**input_embedding["edge"])(edge_input)
    
    # Graph state embedding if provided
    if use_graph_state and graph_descriptors_input is not None:
        graph_embedding = OptionalInputEmbedding(
            **input_embedding.get("graph", {"input_dim": 100, "output_dim": 64})
        )(graph_descriptors_input)
    else:
        graph_embedding = None
    
    # DHTNN layers
    for i in range(depth):
        if verbose:
            print("DHTNN layer %i" % i)
        
        # Apply DHTNN convolution with double-head attention
        n = DHTNNConv(**dhtnn_args)([n, e, edge_index_input])
        
        # Apply dropout between layers (except last layer)
        if i < depth - 1 and dhtnn_args.get("dropout_rate", 0) > 0:
            n = Dropout(dhtnn_args["dropout_rate"])(n)
    
    # Graph state fusion if provided
    if use_graph_state and graph_embedding is not None:
        # Concatenate graph embedding with node features
        n = ks.layers.Concatenate()([n, graph_embedding])
    
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_nodes_args)(n)
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError("Unsupported output embedding for mode %s" % output_embedding)
    
    # Output MLP
    out = MLP(**output_mlp)(out)
    
    # Model
    model = ks.Model(inputs=[node_input, edge_input, edge_index_input] + 
                    ([graph_descriptors_input] if graph_descriptors_input is not None else []),
                    outputs=out, name=name)
    
    return model 