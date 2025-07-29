import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._addgnn_conv import AddGNNConv
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing, GatherState
from kgcnn.layers.modules import Dense, LazyConcatenate, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.aggr import AggregateLocalEdges
from ...layers.pooling import PoolingNodes
from kgcnn.layers.set2set import PoolingSet2SetEncoder
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer, ShiftPeriodicLattice

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2025.01.27"

# Implementation of Add-GNN in `tf.keras` from paper:
# Add-GNN: A Dual-Representation Fusion Molecular Property Prediction Based on Graph Neural Networks with Additive Attention
# by Zhou, R., Zhang, Y., He, K., & Liu, H. (2025). Symmetry, 17(6), 873.

model_default = {
    "name": "AddGNN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None,), "name": "graph_descriptors", "dtype": "float32", "ragged": False}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 200},
                        "edge": {"input_dim": 5, "output_dim": 200},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "geometric_edge": False, "make_distance": False, "expand_distance": False,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum",
                     "init_qstar": "0"},
    "pooling_args": {"pooling_method": "segment_sum"},
    "addgnn_args": {"units": 200, "heads": 4, "activation": "relu", "use_bias": True},
    "use_set2set": True, "depth": 3, "node_dim": 200,
    "use_graph_state": True,  # Default to True for AddGNN to use descriptors
    "verbose": 10,
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["selu", "selu", "sigmoid"]},
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               geometric_edge: bool = None,
               make_distance: bool = None,
               expand_distance: bool = None,
               gauss_args: dict = None,
               set2set_args: dict = None,
               pooling_args: dict = None,
               addgnn_args: dict = None,
               use_set2set: bool = None,
               node_dim: int = None,
               depth: int = None,
               verbose: int = None,
               name: str = None,
               use_graph_state: bool = True,  # Default to True for AddGNN
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `AddGNN <https://www.mdpi.com/2073-8994/17/6/873>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.AddGNN.model_default`.

    Add-GNN is a dual-representation fusion model that uses additive attention mechanism
    to combine node and edge features for molecular property prediction.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, graph_descriptors]`
        or `[node_attributes, edge_distance, edge_indices, graph_descriptors]` if :obj:`geometric_edge=True`
        or `[node_attributes, node_coordinates, edge_indices, graph_descriptors]` if :obj:`make_distance=True` and
        :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge attributes or distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - graph_descriptors (tf.Tensor): Graph-level descriptors of shape `(batch, D)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        geometric_edge (bool): Whether the edges are geometric, like distance or coordinates.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingSet2SetEncoder` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`, `AggregateLocalEdges` layers.
        addgnn_args (dict): Dictionary of layer arguments unpacked in :obj:`AddGNNConv` layer.
        use_set2set (bool): Whether to use :obj:`PoolingSet2SetEncoder` layer.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        use_graph_state (bool): Whether to use graph-level descriptors. Default True for AddGNN.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])  # Or coordinates
    edge_index_input = ks.layers.Input(**inputs[2])
    graph_descriptors_input = ks.layers.Input(**inputs[3]) if len(inputs) > 3 else None

    edi = edge_index_input

    # embedding, if no feature dimension
    n0 = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    if not geometric_edge:
        ed = OptionalInputEmbedding(**input_embedding['edge'], use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    else:
        ed = edge_input

    # Embed graph_descriptors if provided
    if graph_descriptors_input is not None and "graph" in input_embedding:
        graph_descriptors = OptionalInputEmbedding(
            **input_embedding["graph"],
            use_embedding=len(inputs[3]["shape"]) < 1)(graph_descriptors_input)
    else:
        graph_descriptors = None

    # If coordinates are in place of edges
    if make_distance:
        pos1, pos2 = NodePosition()([ed, edi])
        ed = NodeDistanceEuclidean()([pos1, pos2])

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Make hidden dimension
    n = Dense(node_dim, activation="linear")(n0)

    # Add-GNN convolution layers with additive attention
    for i in range(0, depth):
        n = AddGNNConv(**addgnn_args)([n, ed, edi])

    n = LazyConcatenate(axis=-1)([n0, n])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            out = Dense(set2set_args['channels'], activation="linear")(n)
            out = PoolingSet2SetEncoder(**set2set_args)(out)
        else:
            out = PoolingNodes(**pooling_args)(n)
        out = ks.layers.Flatten()(out)  # Flatten() required for to Set2Set output.
        if graph_descriptors is not None:
            out = ks.layers.Concatenate()([graph_descriptors, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if graph_descriptors is not None:
            graph_state_node = GatherState()([graph_descriptors, n])
            n = LazyConcatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `AddGNN`")

    if graph_descriptors_input is not None:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, graph_descriptors_input],
            outputs=out, name=name)
    else:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input], 
            outputs=out, name=name)
    return model


model_crystal_default = {
    "name": "AddGNN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 1), "name": "edge_image", "dtype": "int64", "ragged": True},
               {"shape": (3, 3), "name": "lattice", "dtype": "float32", "ragged": False},
               {"shape": (None,), "name": "graph_descriptors", "dtype": "float32", "ragged": False}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 200},
                        "edge": {"input_dim": 5, "output_dim": 200},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "geometric_edge": False, "make_distance": False, "expand_distance": False,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum",
                     "init_qstar": "0"},
    "pooling_args": {"pooling_method": "segment_sum"},
    "addgnn_args": {"units": 200, "heads": 4, "activation": "relu", "use_bias": True},
    "use_set2set": True, "depth": 3, "node_dim": 200,
    "use_graph_state": True,  # Default to True for AddGNN to use descriptors
    "verbose": 10,
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["selu", "selu", "sigmoid"]},
}


@update_model_kwargs(model_crystal_default)
def make_crystal_model(inputs: list = None,
                       input_embedding: dict = None,
                       geometric_edge: bool = None,
                       make_distance: bool = None,
                       expand_distance: bool = None,
                       gauss_args: dict = None,
                       set2set_args: dict = None,
                       pooling_args: dict = None,
                       addgnn_args: dict = None,
                       use_set2set: bool = None,
                       node_dim: int = None,
                       depth: int = None,
                       verbose: int = None,
                       name: str = None,
                       use_graph_state: bool = True,  # Default to True for AddGNN
                       output_embedding: str = None,
                       output_to_tensor: bool = None,
                       output_mlp: dict = None
                       ):
    r"""Make `AddGNN <https://www.mdpi.com/2073-8994/17/6/873>`_ graph network via functional API for crystal structures.
    Default parameters can be found in :obj:`kgcnn.literature.AddGNN.model_crystal_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_image, lattice, graph_descriptors]`
        or `[node_attributes, edge_distance, edge_indices, edge_image, lattice, graph_descriptors]` if :obj:`geometric_edge=True`
        or `[node_attributes, node_coordinates, edge_indices, edge_image, lattice, graph_descriptors]` if :obj:`make_distance=True` and
        :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge attributes or distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - lattice (tf.Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
            - edge_image (tf.RaggedTensor): Indices of the periodic image the sending node is located. The indices
                of and edge are :math:`(i, j)` with :math:`j` being the sending node.
            - graph_descriptors (tf.Tensor): Graph-level descriptors of shape `(batch, D)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        geometric_edge (bool): Whether the edges are geometric, like distance or coordinates.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingSet2SetEncoder` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`, `AggregateLocalEdges` layers.
        addgnn_args (dict): Dictionary of layer arguments unpacked in :obj:`AddGNNConv` layer.
        use_set2set (bool): Whether to use :obj:`PoolingSet2SetEncoder` layer.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        use_graph_state (bool): Whether to use graph-level descriptors. Default True for AddGNN.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])  # Or coordinates
    edge_index_input = ks.layers.Input(**inputs[2])
    edge_image = ks.layers.Input(**inputs[3])
    lattice = ks.layers.Input(**inputs[4])
    graph_descriptors_input = ks.layers.Input(**inputs[5]) if len(inputs) > 5 else None

    edi = edge_index_input

    # embedding, if no feature dimension
    n0 = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    if not geometric_edge:
        ed = OptionalInputEmbedding(**input_embedding['edge'], use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    else:
        ed = edge_input

    # Embed graph_descriptors if provided
    if graph_descriptors_input is not None and "graph" in input_embedding:
        graph_descriptors = OptionalInputEmbedding(
            **input_embedding["graph"],
            use_embedding=len(inputs[5]["shape"]) < 1)(graph_descriptors_input)
    else:
        graph_descriptors = None

    # If coordinates are in place of edges
    if make_distance:
        x = ed
        pos1, pos2 = NodePosition()([x, edi])
        pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice])
        ed = NodeDistanceEuclidean()([pos1, pos2])

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Make hidden dimension
    n = Dense(node_dim, activation="linear")(n0)

    # Add-GNN convolution layers with additive attention
    for i in range(0, depth):
        n = AddGNNConv(**addgnn_args)([n, ed, edi])

    n = LazyConcatenate(axis=-1)([n0, n])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            out = Dense(set2set_args['channels'], activation="linear")(n)
            out = PoolingSet2SetEncoder(**set2set_args)(out)
        else:
            out = PoolingNodes(**pooling_args)(n)
        out = ks.layers.Flatten()(out)  # Flatten() required for to Set2Set output.
        if graph_descriptors is not None:
            out = ks.layers.Concatenate()([graph_descriptors, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if graph_descriptors is not None:
            graph_state_node = GatherState()([graph_descriptors, n])
            n = LazyConcatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `AddGNN`")

    if graph_descriptors_input is not None:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, edge_image, lattice, graph_descriptors_input],
            outputs=out, name=name)
    else:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, edge_image, lattice], 
            outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__
    return model 


def make_contrastive_addgnn_model(inputs: list = None,
                                 input_embedding: dict = None,
                                 geometric_edge: bool = None,
                                 make_distance: bool = None,
                                 expand_distance: bool = None,
                                 gauss_args: dict = None,
                                 set2set_args: dict = None,
                                 pooling_args: dict = None,
                                 addgnn_args: dict = None,
                                 use_set2set: bool = None,
                                 node_dim: int = None,
                                 depth: int = None,
                                 contrastive_args: dict = None,
                                 verbose: int = None,
                                 name: str = None,
                                 use_graph_state: bool = True,
                                 output_embedding: str = None,
                                 output_to_tensor: bool = None,
                                 output_mlp: dict = None
                                 ):
    r"""Make Contrastive AddGNN model that uses regular AddGNN as base and adds contrastive learning losses.
    
    This is a simple implementation that:
    1. Uses regular AddGNN as the base model
    2. Adds contrastive learning losses on top
    3. Properly handles graph_descriptors input
    
    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`.
        input_embedding (dict): Dictionary of embedding arguments.
        geometric_edge (bool): Whether the edges are geometric.
        make_distance (bool): Whether input is distance or coordinates.
        expand_distance (bool): Whether to expand coordinates to edges.
        gauss_args (dict): Dictionary of gauss basis arguments.
        set2set_args (dict): Dictionary of set2set arguments.
        pooling_args (dict): Dictionary of pooling arguments.
        addgnn_args (dict): Dictionary of AddGNN layer arguments.
        use_set2set (bool): Whether to use set2set pooling.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of AddGNN layers.
        contrastive_args (dict): Dictionary of contrastive learning arguments.
        verbose (int): Level of print output.
        name (str): Name of the model.
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
    
    # Create the base AddGNN model
    base_model = make_model(
        inputs=inputs,
        input_embedding=input_embedding,
        geometric_edge=geometric_edge,
        make_distance=make_distance,
        expand_distance=expand_distance,
        gauss_args=gauss_args,
        set2set_args=set2set_args,
        pooling_args=pooling_args,
        addgnn_args=addgnn_args,
        use_set2set=use_set2set,
        node_dim=node_dim,
        depth=depth,
        verbose=verbose,
        name=name,
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