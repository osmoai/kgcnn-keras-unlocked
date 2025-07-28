"""KA-GAT Model Factory.

This module provides the model factory for building KA-GAT (Kolmogorov-Arnold Graph Attention Network)
models with Fourier-KAN layers for enhanced attention-based GNN performance.
"""

import tensorflow as tf
from kgcnn.layers.modules import Dense, Activation, Dropout
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers_core.modules import OptionalInputEmbedding
from kgcnn.model.utils import update_model_kwargs
from ._kagat_conv import KAGATConv, FourierKANLayer

ks = tf.keras

# Keep track of model version from commit date in literature.
__kgcnn_model_version__ = "2024.01.15"

model_default = {
    "name": "KAGAT",
    "inputs": [
        {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {
        "node": {"input_dim": 95, "output_dim": 128},
        "edge": {"input_dim": 5, "output_dim": 128}
    },
    "kagat_args": {
        "units": 128,
        "attention_heads": 8,
        "attention_units": 64,
        "use_edge_features": True,
        "use_final_activation": True,
        "has_self_loops": True,
        "dropout_rate": 0.1,
        "fourier_dim": 32,
        "fourier_freq_min": 1.0,
        "fourier_freq_max": 100.0,
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
               kagat_args: dict = None,
               pooling_nodes_args: dict = None,
               use_graph_state: bool = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None):
    """Make KA-GAT model.
    
    Args:
        inputs (list): List of input tensors
        input_embedding (dict): Input embedding configuration
        depth (int): Number of KA-GAT layers
        kagat_args (dict): KA-GAT layer arguments
        pooling_nodes_args (dict): Node pooling arguments
        use_graph_state (bool): Whether to use graph state
        name (str): Model name
        verbose (int): Verbosity level
        output_embedding (str): Output embedding type
        output_to_tensor (bool): Whether to convert output to tensor
        output_mlp (dict): Output MLP configuration
        
    Returns:
        tf.keras.Model: KA-GAT model
    """
    
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    
    # Handle graph_desc input if provided (for descriptors)
    if len(inputs) > 3:
        graph_desc_input = ks.layers.Input(**inputs[3])
    else:
        graph_desc_input = None
    
    # Embedding
    n = OptionalInputEmbedding(**input_embedding["node"])(node_input)
    e = OptionalInputEmbedding(**input_embedding["edge"])(edge_input)
    
    # Graph state embedding if provided
    if use_graph_state and graph_desc_input is not None:
        graph_embedding = OptionalInputEmbedding(
            **input_embedding.get("graph", {"input_dim": 100, "output_dim": 64})
        )(graph_desc_input)
    else:
        graph_embedding = None
    
    # KA-GAT layers
    for i in range(depth):
        if verbose:
            print("KA-GAT layer %i" % i)
        
        # Apply KA-GAT convolution
        n = KAGATConv(**kagat_args)([n, e, edge_index_input])
        
        # Apply dropout between layers (except last layer)
        if i < depth - 1 and kagat_args.get("dropout_rate", 0) > 0:
            n = Dropout(kagat_args["dropout_rate"])(n)
    
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
                    ([graph_desc_input] if graph_desc_input is not None else []),
                    outputs=out, name=name)
    
    return model 