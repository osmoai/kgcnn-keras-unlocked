import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._gin_conv import GIN, GINE
from kgcnn.layers.modules import Dense, OptionalInputEmbedding, LazyConcatenate
from kgcnn.layers.mlp import GraphMLP, MLP
from ...layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.gather import GatherState

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of GIN in `tf.keras` from paper:
# How Powerful are Graph Neural Networks?
# Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
# https://arxiv.org/abs/1810.00826

model_default = {
    "name": "GIN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                "use_normalization": True, "normalization_technique": "graph_batch"},
    "gin_args": {},
    "use_graph_state": False,
    "depth": 3, "dropout": 0.0, "verbose": 10,
    "last_mlp": {"use_bias": [True, True, True], "units": [64, 64, 64],
                 "activation": ["relu", "relu", "linear"]},
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "softmax"}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               gin_args: dict = None,
               gin_mlp: dict = None,
               last_mlp: dict = None,
               dropout: float = None,
               name: str = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `GIN <https://arxiv.org/abs/1810.00826>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GIN.model_default`.

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
        depth (int): Number of graph embedding units or depth of the network.
        gin_args (dict): Dictionary of layer arguments unpacked in :obj:`GIN` convolutional layer.
        gin_mlp (dict): Dictionary of layer arguments unpacked in :obj:`MLP` for convolutional layer.
        last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
        dropout (float): Dropout to use.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # GIN: [node_attributes, edge_indices, (optional) graph_descriptors]
    node_input = ks.layers.Input(**inputs[0])
    edge_index_input = ks.layers.Input(**inputs[1])
    graph_descriptors_input = ks.layers.Input(**inputs[2]) if len(inputs) > 2 else None

    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    graph_descriptors = None
    if graph_descriptors_input is not None and "graph" in input_embedding:
        graph_descriptors = OptionalInputEmbedding(
            **input_embedding["graph"],
            use_embedding=len(inputs[2]["shape"]) < 1)(graph_descriptors_input)
    edi = edge_index_input
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='linear')(n)
    list_embeddings = [n]
    for i in range(0, depth):
        n = GIN(**gin_args)([n, edi])
        n = GraphMLP(**gin_mlp)(n)
        list_embeddings.append(n)
    if output_embedding == "graph":
        out = [PoolingNodes()(x) for x in list_embeddings]
        out = [MLP(**last_mlp)(x) for x in out]
        out = [ks.layers.Dropout(dropout)(x) for x in out]
        out = ks.layers.Add()(out)
        if graph_descriptors is not None:
            out = ks.layers.Concatenate()([graph_descriptors, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = n
        out = GraphMLP(**last_mlp)(out)
        if graph_descriptors is not None:
            graph_state_node = GatherState()([graph_descriptors, out])
            out = LazyConcatenate()([out, graph_state_node])
        out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `GIN`")
    if graph_descriptors_input is not None:
        model = ks.models.Model(inputs=[node_input, edge_index_input, graph_descriptors_input], outputs=out, name=name)
    else:
        model = ks.models.Model(inputs=[node_input, edge_index_input], outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    return model


model_default_edge = {
    "name": "GIN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                "use_normalization": True, "normalization_technique": "graph_batch"},
    "gin_args": {"epsilon_learnable": False},
    "depth": 3, "dropout": 0.0, "verbose": 10,
    "use_graph_state": False,
    "last_mlp": {"use_bias": [True, True, True], "units": [64, 64, 64],
                 "activation": ["relu", "relu", "linear"]},
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "softmax"}
}


@update_model_kwargs(model_default_edge)
def make_model_edge(inputs: list = None,
                    input_embedding: dict = None,
                    depth: int = None,
                    gin_args: dict = None,
                    gin_mlp: dict = None,
                    last_mlp: dict = None,
                    dropout: float = None,
                    name: str = None,
                    verbose: int = None,
                    use_graph_state: bool = False,
                    output_embedding: str = None,
                    output_to_tensor: bool = None,
                    output_mlp: dict = None
                    ):
    r"""Make `GINE <https://arxiv.org/abs/1905.12265>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GIN.model_default_edge`.

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
        depth (int): Number of graph embedding units or depth of the network.
        gin_args (dict): Dictionary of layer arguments unpacked in :obj:`GIN` convolutional layer.
        gin_mlp (dict): Dictionary of layer arguments unpacked in :obj:`MLP` for convolutional layer.
        last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
        dropout (float): Dropout to use.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # GINE: [node_attributes, edge_attributes, edge_indices, (optional) graph_descriptors]
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    graph_descriptors_input = ks.layers.Input(**inputs[3]) if len(inputs) > 3 else None

    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    graph_descriptors = None
    if graph_descriptors_input is not None and "graph" in input_embedding:
        graph_descriptors = OptionalInputEmbedding(
            **input_embedding["graph"],
            use_embedding=len(inputs[3]["shape"]) < 1)(graph_descriptors_input)
    edi = edge_index_input
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='linear')(n)
    ed = Dense(n_units, use_bias=True, activation='linear')(ed)
    list_embeddings = [n]
    for i in range(0, depth):
        n = GINE(**gin_args)([n, edi, ed])
        n = GraphMLP(**gin_mlp)(n)
        list_embeddings.append(n)
    if output_embedding == "graph":
        out = [PoolingNodes()(x) for x in list_embeddings]
        out = [MLP(**last_mlp)(x) for x in out]
        out = [ks.layers.Dropout(dropout)(x) for x in out]
        out = ks.layers.Add()(out)
        if graph_descriptors is not None:
            out = ks.layers.Concatenate()([graph_descriptors, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = n
        out = GraphMLP(**last_mlp)(out)
        if graph_descriptors is not None:
            graph_state_node = GatherState()([graph_descriptors, out])
            out = LazyConcatenate()([out, graph_state_node])
        out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `GIN`")
    if graph_descriptors_input is not None:
        model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, graph_descriptors_input], outputs=out, name=name)
    else:
        model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    return model
