import tensorflow as tf
from typing import Union
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing, GatherEdgesPairs, GatherState
from kgcnn.layers.modules import Dense, LazyConcatenate, Activation, LazyAdd, Dropout, \
    OptionalInputEmbedding, LazySubtract, LazyMultiply
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

# Implementation of CMPNN in `tf.keras` from paper:
# Communicative Representation Learning on Attributed Molecular Graphs
# Ying Song, Shuangjia Zheng, Zhangming Niu, Zhang-Hua Fu, Yutong Lu and Yuedong Yang
# https://www.ijcai.org/proceedings/2020/0392.pdf

model_default = {
    "name": "CMPNN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}],
    'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64},
                        "graph": {"input_dim": 10, "output_dim": 64}},
    "node_initialize": {"units": 300, "activation": "relu"},
    "edge_initialize":  {"units": 300, "activation": "relu"},
    "edge_dense": {"units": 300, "activation": "linear"},
    "node_dense": {"units": 300, "activation": "linear"},
    "edge_activation": {"activation": "relu"},
    "verbose": 10,
    "depth": 5,
    "dropout": {"rate": 0.1},
    "use_final_gru": True,
    "pooling_gru": {"units": 300},
    "pooling_kwargs": {"pooling_method": "sum"},
    "use_graph_state": False,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [300, 100, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               edge_initialize: dict = None,
               node_initialize: dict = None,
               edge_dense: dict = None,
               node_dense: dict = None,
               edge_activation: dict = None,
               depth: int = None,
               dropout: Union[dict, None] = None,
               verbose: int = None,
               use_final_gru: bool = True,
               pooling_gru: dict = None,
               pooling_kwargs: dict = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `CMPNN <https://www.ijcai.org/proceedings/2020/0392.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.CMPNN.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_pairs]` or 
              `[node_attributes, edge_attributes, edge_indices, edge_pairs, state_attributes]` if `use_graph_state=True`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_pairs (tf.RaggedTensor): Pair mappings for reverse edge for each edge `(batch, None, 1)`.
            - state_attributes (tf.RaggedTensor): Graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer if `use_graph_state=True`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "CMPNN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first edge embedding.
        node_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first node embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for edge communicate.
        node_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for node communicate.
        edge_activation (dict): Dictionary of layer arguments unpacked in :obj:`Activation` layer for edge communicate.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level for print information.
        dropout (dict): Dictionary of layer arguments unpacked in :obj:`Dropout`.
        pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            :obj:`AggregateLocalEdges` layers.
        use_final_gru (bool): Whether to use GRU for final readout.
        pooling_gru (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodesGRU`.
        use_graph_state (bool): Whether to use graph state information. Default is False.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_attr_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    bond_index_input = ks.layers.Input(**inputs[3])
    
    # Handle graph_desc input if provided (for descriptors)
    if len(inputs) > 4:
        graph_desc_input = ks.layers.Input(**inputs[4])
    else:
        graph_desc_input = None

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    # Embed graph_desc if provided
    if graph_desc_input is not None and "graph" in input_embedding:
        graph_desc = OptionalInputEmbedding(
            **input_embedding["graph"],
            use_embedding=len(inputs[4]["shape"]) < 1)(graph_desc_input)
    else:
        graph_desc = None

    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_attr_input)
    edi = edge_index_input

    h0 = Dense(**node_initialize)(n)
    he0 = Dense(**edge_initialize)(ed)

    # Model Loop
    h = h0
    he = he0
    for i in range(depth - 1):
        # Node message/update
        m_pool = AggregateLocalEdges(**pooling_kwargs)([h, he, edi])
        m_max = AggregateLocalEdges(pooling_method="segment_max")([h, he, edi])
        m = LazyMultiply()([m_pool, m_max])
        # In paper there is a potential COMMUNICATE() here but in reference code just add() operation.
        h = LazyAdd()([h, m])

        # Edge message/update
        h_out = GatherNodesOutgoing()([h, edi])
        e_rev = GatherEdgesPairs()([he, bond_index_input])
        he = LazySubtract()([h_out, e_rev])
        he = Dense(**edge_dense)(he)
        he = LazyAdd()([he, he0])
        he = Activation(**edge_activation)(he)
        if dropout:
            he = Dropout(**dropout)(he)

    # Last step
    m_pool = AggregateLocalEdges(**pooling_kwargs)([h, he, edi])
    m_max = AggregateLocalEdges(pooling_method="segment_max")([h, he, edi])
    m = LazyMultiply()([m_pool, m_max])
    h_final = LazyConcatenate()([m, h, h0])
    h_final = Dense(**node_dense)(h_final)

    n = h_final
    if output_embedding == 'graph':
        if use_final_gru:
            out = ks.layers.GRU(**pooling_gru)(n)
        else:
            out = PoolingNodes(**pooling_kwargs)(n)
        if graph_desc is not None:
            out = ks.layers.Concatenate()([graph_desc, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if graph_desc is not None:
            graph_state_node = GatherState()([graph_desc, n])
            n = LazyConcatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `CMPNN`")

    if graph_desc_input is not None:
        model = ks.models.Model(
            inputs=[node_input, edge_attr_input, edge_index_input, bond_index_input, graph_desc_input],
            outputs=out,
            name=name
        )
    else:
        model = ks.models.Model(
            inputs=[node_input, edge_attr_input, edge_index_input, bond_index_input],
            outputs=out,
            name=name
        )
    model.__kgcnn_model_version__ = __model_version__
    return model
