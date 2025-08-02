import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState
from kgcnn.layers.modules import Dense, LazyConcatenate, Activation, LazyAdd, Dropout, \
    OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.aggr import AggregateLocalEdges
from ...layers.pooling import PoolingNodes
from ._cmpnnplus_conv import CMPNNPlusPoolingEdges
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
__model_version__ = "2024.01.15"

# Implementation of CMPNN+ in `tf.keras` from paper:
# Communicative Message Passing Neural Network for Molecular Representation (2023, J. Chem. Inf. Model.)
# Enhanced CMPNN that explicitly integrates multi-level communicative message passing,
# achieving stronger performance on bioactivity and ADMET tasks.

model_default = {
    "name": "CMPNNPlus",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                        "edge": {"input_dim": 5, "output_dim": 128},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "pooling_args": {"pooling_method": "sum"},
    "use_graph_state": False,
    "edge_initialize": {"units": 256, "use_bias": True, "activation": "relu"},
    "edge_dense": {"units": 256, "use_bias": True, "activation": "relu"},
    "edge_activation": {"activation": "relu"},
    "node_dense": {"units": 256, "use_bias": True, "activation": "relu"},
    "verbose": 10, "depth": 6, "dropout": {"rate": 0.15},
    "use_communicative": True, "communication_levels": 3,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, True], "units": [256, 128, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               pooling_args: dict = None,
               edge_initialize: dict = None,
               edge_dense: dict = None,
               edge_activation: dict = None,
               node_dense: dict = None,
               dropout: dict = None,
               depth: int = None,
               verbose: int = None,
               use_graph_state: bool = False,
               use_communicative: bool = True,
               communication_levels: int = 3,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `CMPNNPlus` graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.CMPNNPlus.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_pairs]` or
        `[node_attributes, edge_attributes, edge_indices, edge_pairs, state_attributes]` if `use_graph_state=True`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_pairs (tf.RaggedTensor): Pair mappings for reverse edge for each edge `(batch, None, 1)`.
            - state_attributes (tf.Tensor): Environment or graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "CMPNNPlus".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            :obj:`AggregateLocalEdges` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first edge embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for edge embedding.
        edge_activation (dict): Edge Activation after skip connection.
        node_dense (dict): Dense kwargs for node embedding layer.
        depth (int): Number of graph embedding units or depth of the network.
        dropout (dict): Dictionary of layer arguments unpacked in :obj:`Dropout`.
        verbose (int): Level for print information.
        use_graph_state (bool): Whether to use graph state information. Default is False.
        use_communicative (bool): Whether to use communicative message passing. Default is True.
        communication_levels (int): Number of communication levels. Default is 3.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

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

    # CMPNN+ message passing with communicative layers
    n = CMPNNPlusPoolingEdges(
        edge_initialize=edge_initialize,
        edge_dense=edge_dense,
        edge_activation=edge_activation,
        node_dense=node_dense,
        dropout=dropout,
        depth=depth,
        use_communicative=use_communicative,
        communication_levels=communication_levels
    )([n, e, edge_index_input, edge_index_reverse_input])

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_args)(n)
        # ROBUST: Use generalized descriptor fusion at graph level
        out = fuse_descriptors_with_output(out, graph_embedding, fusion_method="concatenate")
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError("Unsupported output embedding for mode %s" % output_embedding)

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