import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from ._dgat_conv import DGATLayer, PoolingNodesDGAT
from kgcnn.model.utils import update_model_kwargs

# Import the generalized input handling utilities
from kgcnn.utils.input_utils import (
    get_input_names, find_input_by_name, create_input_layer,
    check_descriptor_input, create_descriptor_processing_layer,
    fuse_descriptors_with_output, build_model_inputs
)

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2025.01.15"

# Implementation of DGAT in `tf.keras` from paper:
# "Directed Graph Attention Networks" (ICLR 2019)
# Shikhar Vashishth, et al.
# https://arxiv.org/abs/1901.01396
#
# Key innovation: Direction-aware attention with separate forward/backward attention mechanisms
# - Forward attention (source → target)
# - Backward attention (target → source) 
# - Direction-specific message passing
# - Bidirectional aggregation

model_default = {
    "name": "DGAT",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                        "edge": {"input_dim": 5, "output_dim": 128},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "dgat_args": {"units": 128, "use_bias": True, "activation": "relu",
                  "attention_units": 64, "use_edge_features": True},
    "depth": 4,
    "verbose": 10,
    "use_graph_state": False,
    "use_rms_norm": False,  # Enable RMS normalization
    "rms_norm_args": {"epsilon": 1e-6, "scale": True, "center": False},
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               dgat_args: dict = None,
               depth: int = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               use_rms_norm: bool = None,
               rms_norm_args: dict = None,
               name: str = None
               ):
    r"""Make `DGAT` graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DGAT.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_indices_reverse]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_indices_reverse (tf.RaggedTensor): Index list for reverse edges of shape `(batch, None, 1)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        dgat_args (dict): Dictionary of layer arguments unpacked in :obj:`DGATLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of print output.
        use_graph_state (bool): Whether to use graph state information. Default is False.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        use_rms_norm (bool): Whether to use RMS normalization.
        rms_norm_args (dict): RMS normalization parameters.

    Returns:
        :obj:`tf.keras.models.Model`
    """

    # ROBUST: Use generalized input handling
    input_names = get_input_names(inputs)
    print(f"🔍 Input names: {input_names}")
    
    # Create input layers using name-based lookup
    input_layers = {}
    for i, input_config in enumerate(inputs):
        name = input_config['name']
        input_layers[name] = create_input_layer(input_config)
        print(f"✅ Created input layer: {name} at position {i}")
    
    # Extract required inputs
    node_input = input_layers['node_attributes']
    edge_input = input_layers['edge_attributes'] 
    edge_index_input = input_layers['edge_indices']
    edge_index_reverse_input = input_layers['edge_indices_reverse']
    
    # Check for optional descriptor input
    descriptor_result = check_descriptor_input(inputs)
    graph_descriptors_input = None
    if descriptor_result:
        idx, config = descriptor_result
        graph_descriptors_input = input_layers['graph_descriptors']

    # Embedding
    n = OptionalInputEmbedding(**input_embedding["node"])(node_input)
    e = OptionalInputEmbedding(**input_embedding["edge"])(edge_input)
    
    # ROBUST: Use generalized descriptor processing
    graph_embedding = create_descriptor_processing_layer(
        graph_descriptors_input, 
        input_embedding, 
        layer_name="graph_descriptor_processing"
    )

    # DGAT layers with direction-aware attention
    print(f"🏗️  Building DGAT with depth={depth}, direction-aware attention")
    for i in range(depth):
        if verbose > 5:
            print(f"  Layer {i+1}: Applying DGAT with forward + backward attention")
        
        # Apply DGAT layer with both forward and reverse edges
        n = DGATLayer(**dgat_args)(
            [n, e, edge_index_input, edge_index_reverse_input]
        )

    # Output embedding choice
    if output_embedding == "graph":
        # Use DGAT-specific pooling
        out = PoolingNodesDGAT(pooling_method="mean")(n)
        
        # ROBUST: Use generalized descriptor fusion
        out = fuse_descriptors_with_output(out, graph_embedding, fusion_method="concatenate")
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError("Unsupported output embedding for mode %s" % output_embedding)

    # Pre-output MLP RMS normalization (only place we apply it)
    if use_rms_norm:
        from kgcnn.layers import RMSNormalization
        out = RMSNormalization(**rms_norm_args)(out)
        print(f"🔧 Applied pre-output MLP RMS normalization to DGAT")

    # Output MLP
    out = MLP(**output_mlp)(out)

    # Output casting - only if output is still ragged
    if output_to_tensor and hasattr(out, 'ragged_rank') and out.ragged_rank > 0:
        out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

    # ROBUST: Use generalized model input building
    model_inputs = build_model_inputs(inputs, input_layers)
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.compile()
    return model
