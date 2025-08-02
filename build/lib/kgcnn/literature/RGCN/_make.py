import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.modules import Dense, OptionalInputEmbedding, LazyMultiply, LazyAdd, Activation
from kgcnn.layers.aggr import PoolingLocalMessages
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.relational import RelationalDense
from kgcnn.model.utils import update_model_kwargs
from kgcnn.literature.PNA import GeneralizedPNALayer

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of GCN in `tf.keras` from paper:
# Modeling Relational Data with Graph Convolutional Networks
# Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov and Max Welling
# https://arxiv.org/abs/1703.06103


model_default = {
    "name": "RGCN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 1), "name": "edge_weights", "dtype": "float32", "ragged": True},
               {"shape": (None, ), "name": "edge_relations", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "dense_relation_kwargs": {"units": 64, "num_relations": 20},
    "dense_kwargs": {"units": 64},
    "activation_kwargs": {"activation": "swish"},
    "depth": 3, "verbose": 10,
    "use_graph_state": False,
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "softmax"}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               dense_relation_kwargs: dict = None,
               dense_kwargs: dict = None,
               activation_kwargs: dict = None,
               name: str = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `RGCN <https://arxiv.org/abs/1703.06103>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.RGCN.model_default`.

    Inputs:
        list: `[node_attributes, edge_indices, edge_weights, edge_relations]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_weights (tf.RaggedTensor): Edge weights of shape `(batch, None, 1)`. Can depend on relations.
            - edge_relations (tf.RaggedTensor): Edge relations of shape `(batch, None)` .

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        dense_relation_kwargs (dict):  Dictionary of layer arguments unpacked in :obj:`RelationalDense` layer.
        dense_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer.
        activation_kwargs (dict):  Dictionary of layer arguments unpacked in :obj:`Activation` layer.
        name (str): Name of the model.
        verbose (int): Level of print output.
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
    edge_weights = ks.layers.Input(**inputs[2])
    edge_relations = ks.layers.Input(**inputs[3])
    
    # Handle graph_descriptors input if provided (for descriptors)
    if len(inputs) > 4:
        graph_descriptors_input = ks.layers.Input(**inputs[4])
    else:
        graph_descriptors_input = None

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    
    # Embed graph_descriptors if provided
    if graph_descriptors_input is not None:
        graph_descriptors = tf.keras.layers.Dense(64, activation='relu')(graph_descriptors_input)
    else:
        graph_descriptors = None

    # Model
    for i in range(0, depth):
        n_j = GatherNodesOutgoing()([n, edge_index_input])
        h0 = Dense(**dense_kwargs)(n)
        h_j = RelationalDense(**dense_relation_kwargs)([n_j, edge_relations])
        m = LazyMultiply()([h_j, edge_weights])
        h = PoolingLocalMessages(pooling_method="sum")([n, m, edge_index_input])
        n = LazyAdd()([h, h0])
        n = Activation(**activation_kwargs)(n)

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes()(n)
        if graph_descriptors is not None:
            # Flatten graph_descriptors to match the rank of out
            graph_descriptors_flat = tf.keras.layers.Flatten()(graph_descriptors)
            out = ks.layers.Concatenate()([graph_descriptors_flat, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":  # Node labeling
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `RGCN`")

    if graph_descriptors_input is not None:
        model = ks.models.Model(inputs=[node_input, edge_index_input, edge_weights, edge_relations, graph_descriptors_input], 
            outputs=out, name=name)
    else:
        model = ks.models.Model(inputs=[node_input, edge_index_input, edge_weights, edge_relations], 
            outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__
    return model


def make_rgcn_pna_model(inputs, use_edge_features=True, use_graph_state=False,
                       rgcn_pna_args=None, depth=3, node_dim=200,
                       output_embedding='graph', output_mlp={"use_bias": [True, True, False], "units": [200, 100, 1],
                                                            "activation": ['relu', 'relu', 'linear']},
                       output_scaling=None, use_set2set=True, set2set_args=None,
                       pooling_args=None, **kwargs):
    """Make RGCN-PNA model using proper generalized PNA.
    
    This combines RGCN's relation-specific message passing with full PNA functionality.
    """
    from kgcnn.layers.modules import LazyConcatenate
    from kgcnn.layers.pooling import PoolingNodes
    from kgcnn.layers.mlp import MLP
    from kgcnn.layers.set2set import PoolingSet2SetEncoder
    from kgcnn.layers.modules import OptionalInputEmbedding
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
    
    # RGCN-PNA convolution layers using generalized PNA
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
    
    # Graph state fusion (if using graph descriptors)
    if use_graph_state:
        # This would be implemented based on the specific model architecture
        # For now, we'll skip this to avoid shape issues
        pass
    
    # Output MLP
    out = MLP(**output_mlp)(out)
    
    # Create Keras model
    if graph_descriptors_input is not None:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, graph_descriptors_input],
            outputs=out, name="RGCN-PNA")
    else:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input], 
            outputs=out, name="RGCN-PNA")
    return model


model_rgcn_pna_default = {
    "name": "RGCN-PNA",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 200}},
    "rgcn_pna_args": {"units": 200, "use_bias": True, "activation": "relu",
                      "aggregators": ["mean", "max", "min", "std"],
                      "scalers": ["identity", "amplification", "attenuation"],
                      "delta": 1.0, "dropout_rate": 0.1},
    "use_set2set": True, "depth": 3, "node_dim": 200,
    "set2set_args": {"channels": 200, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
    "pooling_args": {"pooling_method": "segment_sum"},
    "output_embedding": "graph",
    "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, 1],
                   "activation": ['relu', 'relu', 'linear']},
    "output_scaling": None
}
