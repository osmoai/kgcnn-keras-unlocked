import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._attentivefp_conv import AttentiveHeadFP, PoolingNodesAttentive
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.model.utils import update_model_kwargs

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# import tensorflow.keras as ks
ks = tf.keras

# Implementation of AttentiveFP in `tf.keras` from paper:
# Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism
# Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li,
# Xiaomin Luo, Kaixian Chen, Hualiang Jiang*, and Mingyue Zheng*
# Cite this: J. Med. Chem. 2020, 63, 16, 8749–8760
# Publication Date:August 13, 2019
# https://doi.org/10.1021/acs.jmedchem.9b00959


model_default = {
    "name": "AttentiveFP",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "attention_args": {"units": 32},
    "depthmol": 2,
    "depthato": 2,
    "dropout": 0.1,
    "verbose": 10,
    "use_graph_state": False,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["relu", "relu", "sigmoid"]}
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

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    # Embed graph_descriptors if provided
    if graph_descriptors_input is not None and "graph" in input_embedding:
        graph_descriptors = OptionalInputEmbedding(
            **input_embedding["graph"],
            use_embedding=len(inputs[3]["shape"]) < 1)(graph_descriptors_input)
    else:
        graph_descriptors = None

    edi = edge_index_input

    # Model
    nk = Dense(units=attention_args['units'])(n)
    ck = AttentiveHeadFP(use_edge_features=True, **attention_args)([nk, ed, edi])
    nk = GRUUpdate(units=attention_args['units'])([nk, ck])

    for i in range(1, depthato):
        ck = AttentiveHeadFP(**attention_args)([nk, ed, edi])
        nk = GRUUpdate(units=attention_args['units'])([nk, ck])
        nk = Dropout(rate=dropout)(nk)
    n = nk

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodesAttentive(units=attention_args['units'], depth=depthmol)(n)  # Tensor output.
        if graph_descriptors is not None:
            out = ks.layers.Concatenate()([graph_descriptors, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if graph_descriptors is not None:
            graph_state_node = GatherState()([graph_descriptors, n])
            n = LazyConcatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `AttentiveFP`")

    if graph_descriptors_input is not None:
        model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, graph_descriptors_input],
            outputs=out, name=name)
    else:
        model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], 
            outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__
    return model


def make_contrastive_attentivefp_model(inputs: list = None,
                                      input_embedding: dict = None,
                                      depthmol: int = None,
                                      depthato: int = None,
                                      dropout: float = None,
                                      attention_args: dict = None,
                                      contrastive_args: dict = None,
                                      name: str = None,
                                      verbose: int = None,
                                      use_graph_state: bool = False,
                                      output_embedding: str = None,
                                      output_to_tensor: bool = None,
                                      output_mlp: dict = None
                                      ):
    r"""Make Contrastive AttentiveFP model that uses regular AttentiveFP as base and adds contrastive learning losses.
    
    This is a simple implementation that:
    1. Uses regular AttentiveFP as the base model
    2. Adds contrastive learning losses on top
    3. Properly handles graph_descriptors input
    
    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`.
        input_embedding (dict): Dictionary of embedding arguments.
        depthmol (int): Number of graph embedding units for molecular embedding.
        depthato (int): Number of graph embedding units for atomic embedding.
        dropout (float): Dropout rate.
        attention_args (dict): Dictionary of attention layer arguments.
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
    
    # Create the base AttentiveFP model
    base_model = make_model(
        inputs=inputs,
        input_embedding=input_embedding,
        depthmol=depthmol,
        depthato=depthato,
        dropout=dropout,
        attention_args=attention_args,
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
