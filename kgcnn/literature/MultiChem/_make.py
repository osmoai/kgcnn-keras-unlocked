import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.modules import Dense, LazyConcatenate, Activation, Dropout, \
    OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from ._multichem_conv import MultiChemLayer, PoolingNodesMultiChem
from kgcnn.model.utils import update_model_kwargs

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2025.01.15"

# Implementation of MultiChem GNN in `tf.keras` from paper:
# MultiChem: Multi-modal Chemical Modeling Framework
# Supports both directed and undirected graphs with dual node/edge features

model_default = {
    "name": "MultiChem",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                        "edge": {"input_dim": 5, "output_dim": 128},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "use_directed": True,
    "use_dual_features": True,
    "units": 128,
    "num_heads": 8,
    "depth": 4,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "use_residual": True,
    "pooling_args": {"pooling_method": "sum", "use_dual_features": True},
    "use_graph_state": False,
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, True], "units": [256, 128, 1],
                   "activation": ["relu", "relu", "linear"]}
}

# Undirected version configuration
model_undirected_default = {
    "name": "MultiChemUndirected",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                        "edge": {"input_dim": 5, "output_dim": 128},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "use_directed": False,
    "use_dual_features": True,
    "units": 128,
    "num_heads": 8,
    "depth": 4,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "use_residual": True,
    "pooling_args": {"pooling_method": "sum", "use_dual_features": True},
    "use_graph_state": False,
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, True], "units": [256, 128, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               use_directed: bool = True,
               use_dual_features: bool = True,
               units: int = 128,
               num_heads: int = 8,
               depth: int = 4,
               dropout: float = 0.1,
               attention_dropout: float = 0.1,
               use_residual: bool = True,
               pooling_args: dict = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `MultiChem` graph network via functional API.
    
    Default parameters can be found in :obj:`kgcnn.literature.MultiChem.model_default`.
    
    MultiChem supports both directed and undirected graphs with dual node/edge features.
    The model uses multi-scale attention mechanisms and chemical-specific message passing.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_indices_reverse]` for directed
        or `[node_attributes, edge_attributes, edge_indices]` for undirected
        or `[node_attributes, edge_attributes, edge_indices, edge_indices_reverse, state_attributes]` 
        if `use_graph_state=True`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_indices_reverse (tf.RaggedTensor): Reverse edge indices for directed graphs `(batch, None, 1)`.
            - state_attributes (tf.Tensor): Graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model.
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc.
        use_directed (bool): Whether to use directed graph processing.
        use_dual_features (bool): Whether to use dual node/edge features.
        units (int): Number of hidden units.
        num_heads (int): Number of attention heads.
        depth (int): Number of MultiChem layers.
        dropout (float): Dropout rate.
        attention_dropout (float): Attention dropout rate.
        use_residual (bool): Whether to use residual connections.
        pooling_args (dict): Arguments for pooling layer.
        use_graph_state (bool): Whether to use graph state information.
        output_embedding (str): Main embedding task for graph network.
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Arguments for the final classification :obj:`MLP` layer block.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    
    # Handle directed vs undirected inputs
    if use_directed and len(inputs) > 3:
        edge_index_reverse_input = ks.layers.Input(**inputs[3])
        graph_descriptors_input = ks.layers.Input(**inputs[4]) if len(inputs) > 4 else None
    else:
        edge_index_reverse_input = None
        graph_descriptors_input = ks.layers.Input(**inputs[3]) if len(inputs) > 3 else None

    # Embedding
    n = OptionalInputEmbedding(**input_embedding["node"])(node_input)
    e = OptionalInputEmbedding(**input_embedding["edge"])(edge_input)
    
    # Graph state embedding if provided
    if use_graph_state and graph_descriptors_input is not None:
        graph_embedding = OptionalInputEmbedding(**input_embedding.get("graph", {"input_dim": 100, "output_dim": 64}))(graph_descriptors_input)
    else:
        graph_embedding = None

    # MultiChem layers
    for i in range(depth):
        # Prepare inputs for MultiChem layer
        layer_inputs = [n, e, edge_index_input]
        if use_directed and edge_index_reverse_input is not None:
            layer_inputs.append(edge_index_reverse_input)
        
        # Apply MultiChem layer
        n, e = MultiChemLayer(
            units=units,
            num_heads=num_heads,
            use_directed=use_directed,
            use_dual_features=use_dual_features,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_residual=use_residual
        )(layer_inputs)

    # Output embedding choice
    if output_embedding == "graph":
        # Use MultiChem-specific pooling
        if use_dual_features:
            out = PoolingNodesMultiChem(**pooling_args)([n, e])
        else:
            out = PoolingNodesMultiChem(**pooling_args)(n)
        
        # Graph state fusion if provided
        if use_graph_state and graph_embedding is not None:
            out = ks.layers.Concatenate()([out, graph_embedding])
            
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError("Unsupported output embedding for mode %s" % output_embedding)

    # Output MLP
    out = MLP(**output_mlp)(out)

    # Output casting
    if output_to_tensor and hasattr(out, 'to_tensor'):
        out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

    # Create model inputs
    model_inputs = [node_input, edge_input, edge_index_input]
    if use_directed and edge_index_reverse_input is not None:
        model_inputs.append(edge_index_reverse_input)
    if graph_descriptors_input is not None:
        model_inputs.append(graph_descriptors_input)
    
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    
    return model


@update_model_kwargs(model_undirected_default)
def make_undirected_model(name: str = None,
                         inputs: list = None,
                         input_embedding: dict = None,
                         use_dual_features: bool = True,
                         units: int = 128,
                         num_heads: int = 8,
                         depth: int = 4,
                         dropout: float = 0.1,
                         attention_dropout: float = 0.1,
                         use_residual: bool = True,
                         pooling_args: dict = None,
                         use_graph_state: bool = False,
                         output_embedding: str = None,
                         output_to_tensor: bool = None,
                         output_mlp: dict = None,
                         **kwargs  # Add kwargs to handle any extra parameters
                         ):
    r"""Make undirected `MultiChem` graph network via functional API.
    
    This is a convenience function for creating undirected MultiChem models.
    It sets `use_directed=False` and calls the main `make_model` function.
    
    Args:
        Same as `make_model` but without `use_directed` parameter.
        
    Returns:
        :obj:`tf.keras.models.Model`
    """
    return make_model(
        name=name,
        inputs=inputs,
        input_embedding=input_embedding,
        use_directed=False,
        use_dual_features=use_dual_features,
        units=units,
        num_heads=num_heads,
        depth=depth,
        dropout=dropout,
        attention_dropout=attention_dropout,
        use_residual=use_residual,
        pooling_args=pooling_args,
        use_graph_state=use_graph_state,
        output_embedding=output_embedding,
        output_to_tensor=output_to_tensor,
        output_mlp=output_mlp
    ) 