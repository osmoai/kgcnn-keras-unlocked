import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.attention import AttentionHeadGATV2
from kgcnn.layers.modules import LazyConcatenate, Dense, LazyAverage, Activation, \
    OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.gather import GatherState

# Import the generalized input handling utilities
from kgcnn.utils.input_utils import (
    get_input_names, find_input_by_name, create_input_layer,
    check_descriptor_input, create_descriptor_processing_layer,
    fuse_descriptors_with_output, build_model_inputs
)

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of GATv2 in `tf.keras` from paper:
# Graph Attention Networks
# by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (2018)
# https://arxiv.org/abs/1710.10903
# Improved by
# How Attentive are Graph Attention Networks?
# by Shaked Brody, Uri Alon, Eran Yahav (2021)
# https://arxiv.org/abs/2105.14491

model_default = {'name': "GATv2",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                     "edge": {"input_dim": 5, "output_dim": 64}},
                 'attention_args': {"units": 32, "use_final_activation": False, "use_edge_features": True,
                                    "has_self_loops": True, "activation": "kgcnn>leaky_relu", "use_bias": True},
                 'pooling_nodes_args': {'pooling_method': 'mean'},
                 'depth': 3, 'attention_heads_num': 5,
                 'use_graph_state': False,
                 'attention_heads_concat': False, 'verbose': 10,
                 'output_embedding': 'graph', "output_to_tensor": True,
                 'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                "activation": ['relu', 'relu', 'sigmoid']}
                 }


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               attention_args: dict = None,
               pooling_nodes_args: dict = None,
               depth: int = None,
               attention_heads_num: int = None,
               attention_heads_concat: bool = None,
               name: str = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `GATv2 <https://arxiv.org/abs/2105.14491>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GATv2.model_default`.

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
        attention_args (dict): Dictionary of layer arguments unpacked in :obj:`AttentionHeadGATV2` layer.
        pooling_nodes_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        depth (int): Number of graph embedding units or depth of the network.
        attention_heads_num (int): Number of attention heads to use.
        attention_heads_concat (bool): Whether to concat attention heads, or simply average heads.
        name (str): Name of the model.
        verbose (int): Level of print output.
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
    
    # Check for optional descriptor input
    descriptor_result = check_descriptor_input(inputs)
    graph_descriptors_input = None
    if descriptor_result:
        idx, config = descriptor_result
        graph_descriptors_input = input_layers['graph_descriptors']

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    # ROBUST: Use generalized descriptor processing
    graph_descriptors = create_descriptor_processing_layer(
        graph_descriptors_input, 
        input_embedding, 
        layer_name="graph_descriptor_processing"
    )

    edi = edge_index_input

    # Model
    nk = Dense(units=attention_args["units"], activation="linear")(n)
    for i in range(0, depth):
        heads = [AttentionHeadGATV2(**attention_args)([nk, ed, edi]) for _ in range(attention_heads_num)]
        if attention_heads_concat:
            nk = LazyConcatenate(axis=-1)(heads)
        else:
            nk = LazyAverage()(heads)
            nk = Activation(activation=attention_args["activation"])(nk)
    n = nk

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_nodes_args)(n)
        # ROBUST: Use generalized descriptor fusion
        out = fuse_descriptors_with_output(out, graph_descriptors, fusion_method="concatenate")
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if graph_descriptors is not None:
            graph_state_node = GatherState()([graph_descriptors, n])
            n = LazyConcatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `GATv2`")

    # ROBUST: Use generalized model input building
    model_inputs = build_model_inputs(inputs, input_layers)
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    return model
