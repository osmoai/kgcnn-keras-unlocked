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
                   "activation": "linear"}
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
                   "activation": "linear"}
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


def make_contrastive_gin_model(inputs: list = None,
                              input_embedding: dict = None,
                              depth: int = None,
                              gin_args: dict = None,
                              gin_mlp: dict = None,
                              contrastive_args: dict = None,
                              name: str = None,
                              verbose: int = None,
                              use_graph_state: bool = False,
                              output_embedding: str = None,
                              output_to_tensor: bool = None,
                              output_mlp: dict = None
                              ):
    r"""Make Contrastive GIN model that uses regular GIN as base and adds contrastive learning losses.
    
    This is a simple implementation that:
    1. Uses regular GIN as the base model
    2. Adds contrastive learning losses on top
    3. Properly handles graph_descriptors input
    
    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`.
        input_embedding (dict): Dictionary of embedding arguments.
        depth (int): Number of GIN layers.
        gin_args (dict): Dictionary of GIN layer arguments (only pooling_method, epsilon_learnable).
        gin_mlp (dict): Dictionary of MLP arguments for GIN layers.
        contrastive_args (dict): Dictionary of contrastive learning arguments.
        name (str): Name of the model.
        verbose (int): Level of print output.
        use_graph_state (bool): Whether to use graph descriptors.
        output_embedding (str): Output embedding type.
        output_to_tensor (bool): Whether to convert output to tensor.
        output_mlp (dict): Output MLP configuration.
        
    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Default contrastive arguments
    if contrastive_args is None:
        contrastive_args = {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1
        }
    
    # Default GIN arguments (only the ones GIN layer accepts)
    if gin_args is None:
        gin_args = {
            "pooling_method": "sum",
            "epsilon_learnable": False
        }
    else:
        # Filter out parameters that GIN layer doesn't accept
        valid_gin_args = {}
        if "pooling_method" in gin_args:
            valid_gin_args["pooling_method"] = gin_args["pooling_method"]
        if "epsilon_learnable" in gin_args:
            valid_gin_args["epsilon_learnable"] = gin_args["epsilon_learnable"]
        gin_args = valid_gin_args
    
    # Default GIN MLP arguments
    if gin_mlp is None:
        gin_mlp = {
            "units": [64, 64],
            "use_bias": True,
            "activation": ["relu", "linear"],
            "use_normalization": True,
            "normalization_technique": "graph_batch"
        }
    
    # Create the base GIN model
    base_model = make_model(
        inputs=inputs,
        input_embedding=input_embedding,
        depth=depth,
        gin_args=gin_args,
        gin_mlp=gin_mlp,
        name=name,
        verbose=verbose,
        use_graph_state=use_graph_state,
        output_embedding=output_embedding,
        output_to_tensor=output_to_tensor,
        output_mlp=output_mlp
    )
    
    # Add contrastive learning capabilities
    if contrastive_args.get("use_contrastive_loss", False):
        # Create a custom loss class that combines main task loss with contrastive loss
        class ContrastiveLoss(tf.keras.losses.Loss):
            def __init__(self, contrastive_args, **kwargs):
                super(ContrastiveLoss, self).__init__(**kwargs)
                self.contrastive_args = contrastive_args
                
            def call(self, y_true, y_pred, sample_weight=None):
                # Main task loss (binary crossentropy for classification)
                main_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                
                # Apply sample weights if provided
                if sample_weight is not None:
                    main_loss = main_loss * sample_weight
                
                # Simple contrastive loss based on predictions
                # For similar inputs (same class), predictions should be similar
                # For different inputs (different class), predictions should be different
                batch_size = tf.shape(y_pred)[0]
                
                # Create similarity matrix based on predictions
                pred_norm = tf.nn.l2_normalize(y_pred, axis=1)
                similarity_matrix = tf.matmul(pred_norm, tf.transpose(pred_norm))
                
                # Create target similarity matrix based on true labels
                y_true_expanded = tf.expand_dims(y_true, 1)
                target_similarity = tf.cast(tf.equal(y_true_expanded, tf.transpose(y_true_expanded)), tf.float32)
                
                # Contrastive loss: maximize similarity for same class, minimize for different class
                temperature = self.contrastive_args.get("temperature", 0.1)
                contrastive_loss = tf.reduce_mean(
                    -target_similarity * tf.math.log(tf.nn.sigmoid(similarity_matrix / temperature) + 1e-8)
                )
                
                # Combine losses
                contrastive_weight = self.contrastive_args.get("contrastive_weight", 0.1)
                total_loss = main_loss + contrastive_weight * contrastive_loss
                
                return total_loss
        
        # Compile the model with the contrastive loss
        base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=ContrastiveLoss(contrastive_args),
            metrics=['accuracy']
        )
    
    return base_model
