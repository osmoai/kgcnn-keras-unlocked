import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.attention import AttentionHeadGAT
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

# Implementation of GAT in `tf.keras` from paper:
# Graph Attention Networks
# by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (2018)
# https://arxiv.org/abs/1710.10903

model_default = {
    "name": "GAT",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "attention_args": {"units": 32, "use_final_activation": False, "use_edge_features": True,
                       "has_self_loops": True, "activation": "kgcnn>leaky_relu", "use_bias": True},
    "pooling_nodes_args": {"pooling_method": "mean"},
    "depth": 3, "attention_heads_num": 5,
    "attention_heads_concat": False, "verbose": 10,
    "use_graph_state": False,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["relu", "relu", "sigmoid"]}
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
    r"""Make `GAT <https://arxiv.org/abs/1710.10903>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GAT.model_default`.

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
        attention_args (dict): Dictionary of layer arguments unpacked in :obj:`AttentionHeadGAT` layer.
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
        heads = [AttentionHeadGAT(**attention_args)([nk, ed, edi]) for _ in range(attention_heads_num)]
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
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for `GAT`")

    # ROBUST: Use generalized model input building
    model_inputs = build_model_inputs(inputs, input_layers)
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    
    model.__kgcnn_model_version__ = __model_version__
    return model


def make_contrastive_gat_model(inputs: list = None,
                              input_embedding: dict = None,
                              attention_args: dict = None,
                              pooling_nodes_args: dict = None,
                              depth: int = None,
                              attention_heads_num: int = None,
                              attention_heads_concat: bool = None,
                              contrastive_args: dict = None,
                              name: str = None,
                              verbose: int = None,
                              use_graph_state: bool = False,
                              output_embedding: str = None,
                              output_to_tensor: bool = None,
                              output_mlp: dict = None
                              ):
    r"""Make Contrastive GAT model that uses regular GAT as base and adds contrastive learning losses.
    
    This is a simple implementation that:
    1. Uses regular GAT as the base model
    2. Adds contrastive learning losses on top
    3. Properly handles graph_descriptors input
    
    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`.
        input_embedding (dict): Dictionary of embedding arguments.
        attention_args (dict): Dictionary of attention layer arguments.
        pooling_nodes_args (dict): Dictionary of pooling layer arguments.
        depth (int): Number of GAT layers.
        attention_heads_num (int): Number of attention heads.
        attention_heads_concat (bool): Whether to concatenate attention heads.
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
    
    # Create the base GAT model
    base_model = make_model(
        inputs=inputs,
        input_embedding=input_embedding,
        attention_args=attention_args,
        pooling_nodes_args=pooling_nodes_args,
        depth=depth,
        attention_heads_num=attention_heads_num,
        attention_heads_concat=attention_heads_concat,
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
    else:
        # If not using contrastive loss, still compile with standard loss
        # This ensures the model is always compiled
        base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return base_model
