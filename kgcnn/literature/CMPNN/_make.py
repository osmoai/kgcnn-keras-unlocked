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
from kgcnn.literature.PNA import GeneralizedPNALayer

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

# Implementation of CMPNN in `tf.keras` from paper:
# Communicative Representation Learning on Attributed Molecular Graphs
# Ying Song, Shuangjia Zheng, Zhangming Niu, Zhang-Hua Fu, Yutong Lu and Yuedong Yang
# https://www.ijcai.org/proceedings/2020/0392.pdf

model_default = {
    "name": "CMPNN",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
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
    edge_attr_input = input_layers['edge_attributes']
    edge_index_input = input_layers['edge_indices']
    bond_index_input = input_layers['edge_indices_reverse']
    
    # Check for optional descriptor input
    descriptor_result = check_descriptor_input(inputs)
    graph_descriptors_input = None
    if descriptor_result:
        idx, config = descriptor_result
        graph_descriptors_input = input_layers['graph_descriptors']

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    
    # ROBUST: Use generalized descriptor processing
    graph_embedding = create_descriptor_processing_layer(
        graph_descriptors_input, 
        input_embedding, 
        layer_name="graph_descriptor_processing"
    )

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
        
        # ROBUST: Use generalized descriptor fusion
        out = fuse_descriptors_with_output(out, graph_embedding, fusion_method="concatenate")
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if graph_embedding is not None:
            graph_state_node = GatherState()([graph_embedding, n])
            n = LazyConcatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `CMPNN`")

    # ROBUST: Use generalized model input building
    model_inputs = build_model_inputs(inputs, input_layers)
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    return model


def make_cmpnn_pna_model(inputs, use_edge_features=True, use_graph_state=False,
                        cmpnn_pna_args=None, depth=3, node_dim=200,
                        output_embedding='graph', output_mlp={"use_bias": [True, True, False], "units": [200, 100, 1],
                                                             "activation": ['relu', 'relu', 'linear']},
                        output_scaling=None, use_set2set=True, set2set_args=None,
                        pooling_args=None, **kwargs):
    """Make CMPNN-PNA model using proper generalized PNA.
    
    This combines CMPNN's message passing with full PNA functionality.
    """
    from kgcnn.layers.modules import LazyConcatenate
    from kgcnn.layers.pooling import PoolingNodes
    from kgcnn.layers.mlp import MLP
    from kgcnn.layers.set2set import PoolingSet2SetEncoder
    import keras as ks
    
    # Handle inputs properly - convert to Input layers if needed
    if isinstance(inputs[0], dict):
        # Inputs are configuration dictionaries, create Input layers
        node_input = ks.layers.Input(**inputs[0])
        edge_input = ks.layers.Input(**inputs[1]) if len(inputs) > 1 else None
        edge_index_input = ks.layers.Input(**inputs[2]) if len(inputs) > 2 else None
        graph_descriptors_input = ks.layers.Input(**inputs[3]) if len(inputs) > 3 else None
    else:
        # Inputs are already tensors
        if len(inputs) == 4:
            node_input, edge_input, edge_index_input, graph_descriptors_input = inputs
        elif len(inputs) == 3:
            node_input, edge_input, edge_index_input = inputs
            graph_descriptors_input = None
        else:
            raise ValueError(f"Expected 3 or 4 inputs, got {len(inputs)}")
    
    if not use_edge_features:
        edge_input = None
    
    # Store initial node features for skip connection
    n0 = node_input
    
    # CMPNN-PNA convolution layers using generalized PNA
    n = node_input
    for i in range(depth):
        if use_edge_features:
            n = GeneralizedPNALayer(
                units=node_dim, 
                use_bias=True, 
                activation="relu",
                aggregators=["mean", "max", "min", "std"],
                scalers=["identity", "amplification", "attenuation"],
                delta=1.0,
                dropout_rate=0.1,
                use_edge_features=True,
                use_skip_connection=True
            )([n, edge_input, edge_index_input])
        else:
            n = GeneralizedPNALayer(
                units=node_dim, 
                use_bias=True, 
                activation="relu",
                aggregators=["mean", "max", "min", "std"],
                scalers=["identity", "amplification", "attenuation"],
                delta=1.0,
                dropout_rate=0.1,
                use_edge_features=False,
                use_skip_connection=True
            )([n, edge_index_input])
    
    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            if set2set_args is None:
                set2set_args = {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"}
            out = PoolingSet2SetEncoder(**set2set_args)(n)
        else:
            if pooling_args is None:
                pooling_args = {"pooling_method": "segment_sum"}
            out = PoolingNodes(**pooling_args)(n)
    elif output_embedding == 'node':
        out = n
    else:
        raise ValueError(f"Unsupported output embedding: {output_embedding}")
    
    # ROBUST: Use generalized descriptor fusion
    graph_embedding = create_descriptor_processing_layer(
        graph_descriptors_input, 
        input_embedding, 
        layer_name="graph_descriptor_processing"
    )
    out = fuse_descriptors_with_output(out, graph_embedding, fusion_method="concatenate")
    
    # Output MLP
    out = MLP(**output_mlp)(out)
    
    # Create Keras model
    if graph_descriptors_input is not None:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, graph_descriptors_input],
            outputs=out, name="CMPNN-PNA")
    else:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input], 
            outputs=out, name="CMPNN-PNA")
    return model


model_cmpnn_pna_default = {
    "name": "CMPNN-PNA",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 200}},
    "cmpnn_pna_args": {"units": 200, "use_bias": True, "activation": "relu",
                       "aggregators": ["mean", "max", "min", "std"],
                       "scalers": ["identity", "amplification", "attenuation"],
                       "delta": 1.0, "dropout_rate": 0.1},
    "use_set2set": True, "depth": 3, "node_dim": 200,
    "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
    "pooling_args": {"pooling_method": "segment_sum"},
    "output_embedding": "graph",
    "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, 1],
                   "activation": ['relu', 'relu', 'linear']},
    "output_scaling": None
}
