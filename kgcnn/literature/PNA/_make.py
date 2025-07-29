import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from ._pna_conv import PNALayer, PoolingNodesPNA
from kgcnn.model.utils import update_model_kwargs

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2024.01.15"

# Implementation of PNA in `tf.keras` from paper:
# Principal Neighbourhood Aggregation for Graph Nets (Corso et al., NeurIPS 2020)
# Aggregates neighbor information using multiple aggregators (mean, max, min, std)
# with degree-scaling, capturing richer local structural information than GIN.

model_default = {
    "name": "PNA",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "pna_args": {"units": 128, "use_bias": True, "activation": "relu",
                 "aggregators": ["mean", "max", "min", "std"],
                 "scalers": ["identity", "amplification", "attenuation"],
                 "delta": 1.0, "dropout_rate": 0.1},
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
               pna_args: dict = None,
               depth: int = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `PNA` graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.PNA.model_default`.

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
        pna_args (dict): Dictionary of layer arguments unpacked in :obj:`PNALayer` layer.
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

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_index_input = ks.layers.Input(**inputs[1])
    
    # Handle graph_descriptors input if provided (for descriptors)
    if len(inputs) > 2:
        graph_descriptors_input = ks.layers.Input(**inputs[2])
    else:
        graph_descriptors_input = None

    # Embedding
    n = OptionalInputEmbedding(**input_embedding["node"])(node_input)
    
    # Graph state embedding if provided
    if use_graph_state and graph_descriptors_input is not None:
        graph_embedding = OptionalInputEmbedding(**input_embedding.get("graph", {"input_dim": 100, "output_dim": 64}))(graph_descriptors_input)
    else:
        graph_embedding = None

    # PNA layers
    for i in range(depth):
        n = PNALayer(**pna_args)([n, edge_index_input])

    # Graph state fusion if provided
    if use_graph_state and graph_embedding is not None:
        # Concatenate or add graph embedding
        n = ks.layers.Concatenate()([n, graph_embedding])

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodesPNA(pooling_method="sum")(n)
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError("Unsupported output embedding for mode %s" % output_embedding)

    # Output MLP
    out = MLP(**output_mlp)(out)

    # Output casting
    if output_to_tensor:
        out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

    model = ks.models.Model(inputs=[node_input, edge_index_input] + ([graph_descriptors_input] if graph_descriptors_input is not None else []),
                           outputs=out)
    model.compile()
    return model 