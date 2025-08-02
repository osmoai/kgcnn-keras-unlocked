import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._mogat_conv import AttentiveHeadFP_, PoolingNodesAttentive_
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding, LazyConcatenate
from kgcnn.layers.attention import AttentionHeadGAT
from ...layers.pooling import PoolingNodes, PoolingNodesAttention
from kgcnn.layers.gather import GatherState

from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.model.utils import update_model_kwargs
from kgcnn.utils.input_utils import get_input_names, create_input_layer, check_descriptor_input, create_descriptor_processing_layer, fuse_descriptors_with_output, build_model_inputs

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023.03.25"

# import tensorflow.keras as ks
ks = tf.keras

# Implementation of MoGAT in `tf.keras` from paper:
# Multi‑order graph attention network for water solubility prediction and interpretation
# Sangho Lee, Hyunwoo Park, Chihyeon Choi, Wonjoon Kim, Ki Kang Kim, Young‑Kyu Han,
# Joohoon Kang, Chang‑Jong Kang & Youngdoo Son
# published March 2nd 2023
# https://www.nature.com/articles/s41598-022-25701-5
# https://doi.org/10.1038/s41598-022-25701-5

model_default = {
    "name": "MoGAT",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 2), "name": "graph_descriptors", "dtype": "float32", "ragged": False}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "attention_args": {"units": 32},
    "pooling_gat_nodes_args": {'pooling_method': 'mean'},
    "depthmol": 2,
    "depthato": 2,
    "dropout": 0.2,
    "verbose": 10,
    "use_graph_state": False,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True], "units": [1],
                   "activation": ["linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depthmol: int = None,
               depthato: int = None,
               dropout: float = None,
               attention_args: dict = None,
               pooling_gat_nodes_args: dict = None,
               name: str = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `AttentiveFP <https://doi.org/10.1021/acs.jmedchem.9b00959>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.AttentiveFP.model_default`.

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
        attention_args (dict): Dictionary of layer arguments unpacked in :obj:`AttentiveHeadFP` layer. Units parameter
            is also used in GRU-update and :obj:`PoolingNodesAttentive`.
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
    input_layers = {}
    
    for i, input_config in enumerate(inputs):
        input_layers[input_config['name']] = create_input_layer(input_config)
    
    # Get descriptor input if present
    graph_descriptors_input = None
    if check_descriptor_input(inputs):
        graph_descriptors_input = input_layers['graph_descriptors']
    
    # ROBUST: Use generalized descriptor processing
    graph_embedding = create_descriptor_processing_layer(
        graph_descriptors_input,
        input_embedding,
        layer_name="graph_descriptor_processing"
    )
    
    # Get main inputs
    node_input = input_layers['node_attributes']
    edge_attr_input = input_layers['edge_attributes']
    edge_index_input = input_layers['edge_indices']
    
    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_attr_input)

    # Model
    nk = Dense(units=attention_args['units'])(n)
    ck = AttentiveHeadFP_(use_edge_features=True, **attention_args)([nk, ed, edge_index_input])
    nk = GRUUpdate(units=attention_args['units'])([nk, ck])
    nk = Dropout(rate=dropout)(nk) # adding dropout to the first code not in the original AttFP code ?
    list_emb=[nk] # "aka r1"
    for i in range(1, depthato):
        ck = AttentiveHeadFP_(**attention_args)([nk, ed, edge_index_input])
        nk = GRUUpdate(units=attention_args['units'])([nk, ck])
        nk = Dropout(rate=dropout)(nk)
        list_emb.append(nk)
    
    # we store representation of each atomic nodes (at r1,r2,...)

    if output_embedding == 'graph':
        # we apply a super node to each atomic node representation and concate them 
        out = LazyConcatenate()([PoolingNodesAttentive_(units=attention_args['units'], depth=depthmol)(ni) for ni in list_emb]) # Tensor output.        
        # we compute the weigthed scaled self-attention of the super nodes
        at = ks.layers.Attention(dropout=dropout,use_scale=True, score_mode="dot")([out, out])
        # we apply the dot product
        out = at*out
        
        # ROBUST: Use generalized descriptor fusion
        out = fuse_descriptors_with_output(out, graph_embedding, fusion_method="concatenate")
        
        # in the paper this is only one dense layer to the target ... very simple
        out = MLP(**output_mlp)(out)
        

    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `MoGAT`")

    # ROBUST: Use generalized model input building
    model_inputs = build_model_inputs(inputs, input_layers)
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__
    return model
