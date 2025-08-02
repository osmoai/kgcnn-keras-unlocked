import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from ._expc_conv import ExpCLayer, PoolingNodesExpC
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

# Implementation of ExpC in `tf.keras` from paper:
# Going beyond Weisfeiler-Lehman: A Novel Expressive Graph Model (2022, ICML)
# Adds subgraph counting (e.g., triangles) as auxiliary tasks during training,
# significantly enhancing expressivity compared to GIN.

model_default = {
    "name": "ExpC",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "expc_args": {"units": 128, "use_bias": True, "activation": "relu",
                  "use_subgraph_counting": True, "subgraph_types": ["triangle", "square", "pentagon"],
                  "dropout_rate": 0.1},
    "depth": 4,
    "verbose": 10,
    "use_graph_state": False,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               expc_args: dict = None,
               depth: int = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `ExpC` graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.ExpC.model_default`.

    Inputs:
        list: `[node_attributes, edge_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        expc_args (dict): Dictionary of layer arguments unpacked in :obj:`ExpCLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of print output.
        use_graph_state (bool): Whether to use graph state information. Default is False.
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
    edge_index_input = input_layers['edge_indices']
    
    # Check for optional descriptor input
    descriptor_result = check_descriptor_input(inputs)
    graph_descriptors_input = None
    if descriptor_result:
        idx, config = descriptor_result
        graph_descriptors_input = input_layers['graph_descriptors']

    # Embedding
    n = OptionalInputEmbedding(**input_embedding["node"])(node_input)
    
    # ROBUST: Use generalized descriptor processing
    graph_embedding = create_descriptor_processing_layer(
        graph_descriptors_input, 
        input_embedding, 
        layer_name="graph_descriptor_processing"
    )

    # ExpC layers
    for i in range(depth):
        n = ExpCLayer(**expc_args)([n, edge_index_input])

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodesExpC(pooling_method="sum")(n)
        
        # ROBUST: Use generalized descriptor fusion
        out = fuse_descriptors_with_output(out, graph_embedding, fusion_method="concatenate")
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError("Unsupported output embedding for mode %s" % output_embedding)

    # Output MLP
    out = MLP(**output_mlp)(out)

    # Output casting
    if output_to_tensor:
        out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

    # ROBUST: Use generalized model input building
    model_inputs = build_model_inputs(inputs, input_layers)
    model = ks.models.Model(inputs=model_inputs, outputs=out)
    model.compile()
    return model 