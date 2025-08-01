import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._coattentivefp_conv import CoAttentiveHeadFP, PoolingNodesCoAttentive
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.model.utils import update_model_kwargs

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras

# Implementation of CoAttentiveFP in `tf.keras` from paper:
# Collaborative Graph Attention Networks (2022, JCIM)
# Enhanced AttentiveFP by introducing collaborative attention mechanisms
# between atom and bond embeddings, improving ADMET property predictions.


model_default = {
    "name": "CoAttentiveFP",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                        "edge": {"input_dim": 5, "output_dim": 128}},
    "attention_args": {"units": 256, "use_collaborative": True, "collaboration_heads": 8},
    "depthmol": 3,
    "depthato": 3,
    "dropout": 0.15,
    "verbose": 10,
    "use_graph_state": False,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, True], "units": [256, 128, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depthmol: int = None,
               depthato: int = None,
               dropout: float = None,
               attention_args: dict = None,
               name: str = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `CoAttentiveFP` graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.CoAttentiveFP.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depthato (int): Number of graph embedding units or depth of the network.
        depthmol (int): Number of graph embedding units or depth of the graph embedding.
        dropout (float): Dropout to use.
        attention_args (dict): Dictionary of layer arguments unpacked in :obj:`CoAttentiveHeadFP` layer.
        name (str): Name of the model.
        verbose (int): Level of print output.
        use_graph_state (bool): Whether to use graph state information. Default is False.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """

    # Import the generalized input handling utilities
    from kgcnn.utils.input_utils import (
        get_input_names, find_input_by_name, create_input_layer,
        check_descriptor_input, create_descriptor_processing_layer,
        fuse_descriptors_with_output, build_model_inputs
    )

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

    # Model body
    # Atom embedding
    for i in range(depthato):
        n = CoAttentiveHeadFP(**attention_args)([n, e, edge_index_input])
        if dropout > 0:
            n = Dropout(dropout)(n)

    # Molecule embedding
    if depthmol > 0:
        # Pooling
        n = PoolingNodesCoAttentive(pooling_method="sum")(n)
        
        # After pooling, n is a regular tensor, not ragged
        # Use Dense layers instead of GRUUpdate for molecule embedding
        for i in range(depthmol):
            n = Dense(attention_args["units"], activation="relu")(n)
            if dropout > 0:
                n = Dropout(dropout)(n)
    
    # ROBUST: Use generalized descriptor fusion
    n = fuse_descriptors_with_output(n, graph_embedding, fusion_method="concatenate")

    # Output embedding choice
    if output_embedding == "graph":
        out = n
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError("Unsupported output embedding for mode %s" % output_embedding)

    # Output MLP
    out = MLP(**output_mlp)(out)

    # Output casting
    if output_to_tensor:
        # Check if the output is already a regular tensor (after pooling)
        # If it's already a tensor, we don't need to convert it
        if hasattr(out, 'to_tensor'):
            # It's a ragged tensor, convert to regular tensor
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
        # If it's already a regular tensor, do nothing

    # ROBUST: Use generalized model input building
    model_inputs = build_model_inputs(inputs, input_layers)
    model = ks.models.Model(inputs=model_inputs, outputs=out)
    model.compile()
    return model 