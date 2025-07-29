import tensorflow as tf
from tensorflow import keras as ks
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from kgcnn.layers.modules import DenseEmbedding as Embedding
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.literature.AWARE._aware_conv import AWAREWalkAggregation
from kgcnn.model.utils import update_model_kwargs
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.layers.modules import LazyConcatenate


model_default = {
    "name": "AWARE",
    "inputs": [
        {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {
        "node": {"input_dim": 95, "output_dim": 128},
        "edge": {"input_dim": 5, "output_dim": 128}
    },
    "aware_args": {
        "units": 128,
        "walk_length": 3,
        "num_walks": 10,
        "attention_heads": 4,
        "dropout_rate": 0.1,
        "activation": "relu",
        "use_bias": True
    },
    "depth": 4,
    "dropout": 0.1,
    "verbose": 10,
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {
        "use_bias": [True, True, False],
        "units": [200, 100, 1],
        "activation": ["kgcnn>leaky_relu", "selu", "linear"]
    }
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               aware_args: dict = None,
               depth: int = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               name: str = None,
               dropout: float = None
               ):
    """Make AWARE model.
    
    Args:
        inputs: List of input tensors
        input_embedding: Input embedding configuration
        aware_args: AWARE layer arguments
        depth: Number of AWARE layers
        verbose: Verbosity level
        use_graph_state: Whether to use graph descriptors
        output_embedding: Output embedding type
        output_to_tensor: Whether to convert output to tensor
        output_mlp: Output MLP configuration
        name: Model name
        
    Returns:
        AWARE model
    """
    # Inputs
    node_input = ks.layers.Input(**inputs[0])
    edge_attr_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    
    # Handle graph_descriptors input if provided (for descriptors)
    if len(inputs) > 3:
        graph_descriptors_input = ks.layers.Input(**inputs[3])
    else:
        graph_descriptors_input = None
    
    # Convert output_dim to units for DenseEmbedding
    node_embedding_cfg = dict(input_embedding['node'])
    edge_embedding_cfg = dict(input_embedding['edge'])
    if 'output_dim' in node_embedding_cfg:
        node_embedding_cfg['units'] = node_embedding_cfg.pop('output_dim')
    if 'output_dim' in edge_embedding_cfg:
        edge_embedding_cfg['units'] = edge_embedding_cfg.pop('output_dim')

    # Embedding layers
    n = Embedding(**node_embedding_cfg)(node_input)
    ed = Embedding(**edge_embedding_cfg)(edge_attr_input)
    
    # Process graph_descriptors if provided (use Dense layer for continuous values)
    if use_graph_state and graph_descriptors_input is not None:
        graph_embedding = Dense(input_embedding.get("graph", {"output_dim": 64})["output_dim"],
                               activation='relu',
                               use_bias=True)(graph_descriptors_input)
    else:
        graph_embedding = None
    
    # AWARE layers
    for i in range(depth):
        if verbose:
            print(f"Building AWARE layer {i+1}/{depth}")
        
        # Apply AWARE walk aggregation
        n = AWAREWalkAggregation(**aware_args)([n, ed, edge_index_input])
        
        # Add dropout
        if aware_args.get("dropout_rate", 0) > 0:
            n = Dropout(aware_args["dropout_rate"])(n)
    
    # Output embedding
    if output_embedding == "graph":
        out = PoolingNodes(pooling_method="mean")(n)
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError(f"Unknown output embedding: {output_embedding}")
    
    # Concatenate with graph descriptors if provided
    if graph_embedding is not None:
        # Flatten graph_embedding to match the rank of out
        graph_embedding_flat = tf.keras.layers.Flatten()(graph_embedding)
        out = LazyConcatenate()([graph_embedding_flat, out])
    
    # Output MLP
    out = MLP(**output_mlp)(out)
    
    # Output casting
    if output_to_tensor and hasattr(out, 'ragged_rank') and out.ragged_rank > 0:
        out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    
    # Model
    if graph_descriptors_input is not None:
        model = ks.models.Model(inputs=[node_input, edge_attr_input, edge_index_input, graph_descriptors_input],
                               outputs=out, name=name)
    else:
        model = ks.models.Model(inputs=[node_input, edge_attr_input, edge_index_input],
                               outputs=out, name=name)
    
    model.compile()
    return model 