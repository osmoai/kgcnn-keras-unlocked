import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._attentivefpplus_conv import AttentiveHeadFPPlus, PoolingNodesAttentivePlus
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.model.utils import update_model_kwargs

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

ks = tf.keras

# Implementation of AttentiveFP+ in `tf.keras` from paper:
# AttentiveFP+ (Multi-scale Attention Fusion) (2023)
# Extended AttFP by multi-scale attention fusion across message passing layers,
# capturing both local and global contexts more effectively.


model_default = {
    "name": "AttentiveFPPlus",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                        "edge": {"input_dim": 5, "output_dim": 128}},
    "attention_args": {"units": 256, "use_multiscale": True, "scale_fusion": "weighted_sum", "attention_scales": [1, 2, 4]},
    "depthmol": 4,
    "depthato": 4,
    "dropout": 0.2,
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
    r"""Make `AttentiveFPPlus` graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.AttentiveFPPlus.model_default`.

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
        attention_args (dict): Dictionary of layer arguments unpacked in :obj:`AttentiveHeadFPPlus` layer.
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

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    
    # Handle graph_descriptors input if provided (for descriptors)
    if len(inputs) > 3:
        graph_descriptors_input = ks.layers.Input(**inputs[3])
    else:
        graph_descriptors_input = None

    # Embedding
    n = OptionalInputEmbedding(**input_embedding["node"])(node_input)
    e = OptionalInputEmbedding(**input_embedding["edge"])(edge_input)
    
    # Graph state embedding if provided
    if use_graph_state and graph_descriptors_input is not None:
        graph_embedding = OptionalInputEmbedding(**input_embedding.get("graph", {"input_dim": 100, "output_dim": 64}))(graph_descriptors_input)
    else:
        graph_embedding = None

    # Model body
    # Atom embedding with multi-scale attention
    for i in range(depthato):
        n = AttentiveHeadFPPlus(**attention_args)([n, e, edge_index_input])
        if dropout > 0:
            n = Dropout(dropout)(n)

    # Molecule embedding
    if depthmol > 0:
        # Pooling
        n = PoolingNodesAttentivePlus(pooling_method="sum")(n)
        
        # GRU update for molecule embedding
        for i in range(depthmol):
            n = GRUUpdate(attention_args["units"])(n)
            if dropout > 0:
                n = Dropout(dropout)(n)
    
    # Graph state fusion if provided
    if use_graph_state and graph_embedding is not None:
        # Concatenate or add graph embedding
        n = ks.layers.Concatenate()([n, graph_embedding])
        # Or use attention to fuse
        # n = GraphAttention(units=attention_args["units"])([n, graph_embedding])

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
        out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input] + ([graph_descriptors_input] if graph_descriptors_input is not None else []),
                           outputs=out)
    model.compile()
    return model 