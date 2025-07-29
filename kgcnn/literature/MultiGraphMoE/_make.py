"""Multi-Graph MoE Model Factory.

This module provides the model factory for building Multi-Graph MoE models
that create multiple graph representations and use different GNN experts
for ensemble improvement and variance reduction.
"""

import tensorflow as tf
from kgcnn.layers.modules import Dense, Activation, Dropout
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from kgcnn.layers.modules import OptionalInputEmbedding
from kgcnn.model.utils import update_model_kwargs
from ._multigraph_moe_conv import MultiGraphMoEConv, GraphRepresentationLayer, ExpertRoutingLayer

ks = tf.keras

# Keep track of model version from commit date in literature.
__kgcnn_model_version__ = "2024.01.15"

model_default = {
    "name": "MultiGraphMoE",
    "inputs": [
        {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {
        "node": {"input_dim": 95, "output_dim": 128},
        "edge": {"input_dim": 5, "output_dim": 128}
    },
    "multigraph_moe_args": {
        "num_representations": 4,
        "num_experts": 3,
        "expert_types": ["gin", "gat", "gcn"],
        "representation_types": ["original", "weighted", "augmented", "attention"],
        "use_edge_weights": True,
        "use_node_features": True,
        "use_attention": True,
        "dropout_rate": 0.1,
        "temperature": 1.0,
        "use_noise": True,
        "noise_epsilon": 1e-2,
        "units": 128
    },
    "depth": 3,
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
               multigraph_moe_args: dict = None,
               pooling_nodes_args: dict = None,
               use_graph_state: bool = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None):
    """Make Multi-Graph MoE model.
    
    Args:
        inputs (list): List of input tensors
        input_embedding (dict): Input embedding configuration
        depth (int): Number of Multi-Graph MoE layers
        multigraph_moe_args (dict): Multi-Graph MoE layer arguments
        pooling_nodes_args (dict): Node pooling arguments
        use_graph_state (bool): Whether to use graph state
        name (str): Model name
        verbose (int): Verbosity level
        output_embedding (str): Output embedding type
        output_to_tensor (bool): Whether to convert output to tensor
        output_mlp (dict): Output MLP configuration
        
    Returns:
        tf.keras.Model: Multi-Graph MoE model
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
    
    # Multi-Graph MoE layers
    for i in range(depth):
        if verbose:
            print(f"Multi-Graph MoE layer {i}")
        
        # Apply Multi-Graph MoE convolution
        n = MultiGraphMoEConv(**multigraph_moe_args)([n, e, edge_index_input])
        
        # Apply dropout between layers (except last layer)
        if i < depth - 1 and multigraph_moe_args.get("dropout_rate", 0) > 0:
            n = Dropout(multigraph_moe_args["dropout_rate"])(n)
    
    # Graph state fusion if provided
    if use_graph_state and graph_embedding is not None:
        # Pool node features to match graph embedding shape
        pooled_nodes = PoolingNodes(**pooling_nodes_args)(n)
        # Concatenate graph embedding with pooled node features
        n = ks.layers.Concatenate()([pooled_nodes, graph_embedding])
    
    # Output embedding choice
    if output_embedding == "graph":
        if use_graph_state and graph_embedding is not None:
            # Already pooled above
            out = n
        else:
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