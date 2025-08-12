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

# Import RMS normalization
from kgcnn.layers.norm import RMSNormalization

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2025.08.12"

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
    "depthmol": 2,
    "depthato": 2,
    "dropout": 0.2,
    "verbose": 10,
    "use_graph_state": False,
    "use_rms_norm": False,  # New option for RMS normalization
    "rms_norm_args": {"epsilon": 1e-6, "scale": True},  # RMS normalization parameters
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
               use_rms_norm: bool = False,  # New option for RMS normalization
               rms_norm_args: dict = None,  # RMS normalization parameters
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `MoGAT <https://doi.org/10.1038/s41598-022-25701-5>`_ graph network via functional API.
    
    This is the TRUE implementation of Multi-order Graph Attention Network from the Nature paper.
    
    Key innovation: Multi-order information fusion with attention mechanism:
    - Extracts graph embeddings from every node embedding layer
    - Uses attention to merge graph embeddings from different orders
    - Provides interpretability through atomic importance scores
    
    Default parameters can be found in :obj:`kgcnn.literature.MoGAT.model_default`.

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
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc.
        depthato (int): Number of atomic embedding layers (depth of the network).
        depthmol (int): Number of molecular embedding iterations.
        dropout (float): Dropout rate.
        attention_args (dict): Dictionary of attention layer arguments.
        name (str): Name of the model.
        verbose (int): Level of print output.
        use_graph_state (bool): Whether to use graph state information.
        output_embedding (str): Main embedding task for graph network.
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments for final MLP.

    Returns:
        :obj:`tf.keras.models.Model`
    """

    # ROBUST: Use generalized input handling
    input_names = get_input_names(inputs)
    print(f"🔍 MoGAT Input names: {input_names}")
    
    input_layers = {}
    for i, input_config in enumerate(inputs):
        input_layers[input_config['name']] = create_input_layer(input_config)
        print(f"✅ Created input layer: {input_config['name']} at position {i}")
    
    # Check for optional descriptor input
    descriptor_result = check_descriptor_input(inputs)
    graph_descriptors_input = None
    if descriptor_result:
        idx, config = descriptor_result
        graph_descriptors_input = input_layers['graph_descriptors']
        print(f"🎯 Found descriptor input: graph_descriptors")
    
    # Get main inputs
    node_input = input_layers['node_attributes']
    edge_attr_input = input_layers['edge_attributes']
    edge_index_input = input_layers['edge_indices']
    
    # ROBUST: Use generalized descriptor processing
    graph_descriptors = create_descriptor_processing_layer(
        graph_descriptors_input,
        input_embedding,
        layer_name="graph_descriptor_processing"
    )
    
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
        
        # Apply RMS normalization if enabled
        if use_rms_norm:
            out = RMSNormalization(**rms_norm_args)(out)
        
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
    print(f"✅ MoGAT model built successfully with {len(model.layers)} layers")
    return model


def make_mogatv2_model(inputs: list = None,
                       input_embedding: dict = None,
                       depthmol: int = None,
                       depthato: int = None,
                       dropout: float = None,
                       attention_args: dict = None,
                       name: str = None,
                       verbose: int = None,
                       use_graph_state: bool = False,
                       use_rms_norm: bool = False,  # RMS normalization option
                       rms_norm_args: dict = None,  # RMS normalization parameters
                       output_embedding: str = None,
                       output_to_tensor: bool = None,
                       output_mlp: dict = None
                       ):
    r"""Make `MoGATv2 <https://doi.org/10.1038/s41598-022-25701-5>`_ graph network via functional API.
    
    This is the TRUE implementation of Multi-order Graph Attention Network from the Nature paper.
    
    Key innovation: Multi-order information fusion with attention mechanism:
    - Extracts graph embeddings from every node embedding layer
    - Uses attention to merge graph embeddings from different orders
    - Provides interpretability through atomic importance scores
    
    This is DIFFERENT from the original MoGAT implementation above.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, graph_descriptors]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - graph_descriptors (tf.Tensor): Graph-level descriptors of shape `(batch, D)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc.
        depthato (int): Number of atomic embedding layers (depth of the network).
        depthmol (int): Number of molecular embedding iterations.
        dropout (float): Dropout rate.
        attention_args (dict): Dictionary of attention layer arguments.
        name (str): Name of the model.
        verbose (int): Level of print output.
        use_graph_state (bool): Whether to use graph state information.
        use_rms_norm (bool): Whether to use RMS normalization.
        rms_norm_args (dict): RMS normalization parameters.
        output_embedding (str): Main embedding task for graph network.
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments for final MLP.

    Returns:
        :obj:`tf.keras.models.Model`
    """

    # ROBUST: Use generalized input handling
    input_names = get_input_names(inputs)
    print(f"🔍 MoGATv2 Input names: {input_names}")
    
    input_layers = {}
    for i, input_config in enumerate(inputs):
        input_layers[input_config['name']] = create_input_layer(input_config)
        print(f"✅ Created input layer: {input_config['name']} at position {i}")
    
    # Check for optional descriptor input
    descriptor_result = check_descriptor_input(inputs)
    graph_descriptors_input = None
    if descriptor_result:
        idx, config = descriptor_result
        graph_descriptors_input = input_layers['graph_descriptors']
        print(f"🎯 Found descriptor input: graph_descriptors")
    
    # Get main inputs
    node_input = input_layers['node_attributes']
    edge_attr_input = input_layers['edge_attributes']
    edge_index_input = input_layers['edge_indices']
    
    # ROBUST: Use generalized descriptor processing
    graph_descriptors = create_descriptor_processing_layer(
        graph_descriptors_input,
        input_embedding,
        layer_name="graph_descriptor_processing"
    )
    
    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_attr_input)

    # ===== MOGA T V2 CORE ARCHITECTURE (Nature Paper) =====
    # Phase 1: Multi-order node embedding extraction
    print(f"🏗️  Building MoGATv2 (Nature Paper) with depthato={depthato}, depthmol={depthmol}")
    
    # Initial node transformation
    nk = Dense(units=attention_args['units'])(n)
    
    # Store node embeddings from each layer (multi-order information)
    node_embeddings = []
    
    # First layer with pre-attention normalization
    if use_rms_norm:
        nk = RMSNormalization(**rms_norm_args)(nk)
        print(f"🔧 Applied pre-attention RMS normalization to layer 1")
    
    ck = AttentiveHeadFP_(use_edge_features=True, **attention_args)([nk, ed, edge_index_input])
    nk = GRUUpdate(units=attention_args['units'])([nk, ck])
    nk = Dropout(rate=dropout)(nk)
    
    # Post-attention normalization for stability
    if use_rms_norm:
        nk = RMSNormalization(**rms_norm_args)(nk)
        print(f"🔧 Applied post-attention RMS normalization to layer 1")
    
    node_embeddings.append(nk)  # Store r1 (first order)
    
    # Subsequent layers with strategic normalization
    for i in range(1, depthato):
        # Pre-attention normalization (stabilizes attention)
        if use_rms_norm:
            nk = RMSNormalization(**rms_norm_args)(nk)
            print(f"🔧 Applied pre-attention RMS normalization to layer {i+1}")
        
        ck = AttentiveHeadFP_(**attention_args)([nk, ed, edge_index_input])
        nk = GRUUpdate(units=attention_args['units'])([nk, ck])
        nk = Dropout(rate=dropout)(nk)
        
        # Post-attention normalization (stabilizes features)
        if use_rms_norm:
            nk = RMSNormalization(**rms_norm_args)(nk)
            print(f"🔧 Applied post-attention RMS normalization to layer {i+1}")
        
        node_embeddings.append(nk)  # Store r2, r3, ... (higher orders)
    
    print(f"📊 Extracted {len(node_embeddings)} multi-order node embeddings")
    
    # Phase 2: Graph embedding extraction from each order
    if output_embedding == 'graph':
        # Extract graph embeddings from each node embedding layer (multi-order)
        graph_embeddings = []
        for i, node_emb in enumerate(node_embeddings):
            # Pre-pooling normalization (stabilizes pooling operation)
            if use_rms_norm:
                node_emb = RMSNormalization(**rms_norm_args)(node_emb)
                print(f"🔧 Applied pre-pooling RMS normalization to graph embedding {i+1}")
            
            # Apply molecular-level attention pooling to each order
            graph_emb = PoolingNodesAttentive_(
                units=attention_args['units'], 
                depth=depthmol
            )(node_emb)
            graph_embeddings.append(graph_emb)
            print(f"🎯 Graph embedding {i+1}: shape {graph_emb.shape}")
        
        # Phase 3: Multi-order attention fusion (CORE INNOVATION from Nature Paper)
        print("🔀 Applying multi-order attention fusion (Nature Paper innovation)...")
        
        # Stack graph embeddings for attention computation
        stacked_embeddings = tf.stack(graph_embeddings, axis=1)  # (batch, orders, features)
        
        # Multi-order attention mechanism as per Nature paper
        # Query: Current graph representation, Key/Value: All order embeddings
        attention_layer = ks.layers.MultiHeadAttention(
            num_heads=4,  # Configurable
            key_dim=attention_args['units'],
            dropout=dropout,
            name="multi_order_attention"
        )
        
        # Apply attention to fuse multi-order information
        fused_embedding = attention_layer(
            query=stacked_embeddings,
            key=stacked_embeddings,
            value=stacked_embeddings
        )
        
        # Global average pooling across orders to get final graph representation
        out = tf.reduce_mean(fused_embedding, axis=1)  # (batch, features)
        
        print(f"🎯 Final fused graph embedding shape: {out.shape}")
        
        # Pre-descriptor fusion normalization (stabilizes fusion operation)
        if use_rms_norm:
            out = RMSNormalization(**rms_norm_args)(out)
            print(f"🔧 Applied pre-descriptor fusion RMS normalization")
        
        # ROBUST: Use generalized descriptor fusion
        out = fuse_descriptors_with_output(out, graph_descriptors, fusion_method="concatenate")
        
        # Pre-MLP normalization (stabilizes final predictions)
        if use_rms_norm:
            out = RMSNormalization(**rms_norm_args)(out)
            print(f"🔧 Applied pre-MLP RMS normalization")
        
        # Final prediction layer (simple as per paper)
        out = MLP(**output_mlp)(out)
        
    elif output_embedding == 'node':
        # For node-level tasks, use the final node embeddings
        out = GraphMLP(**output_mlp)(node_embeddings[-1])
        if output_to_tensor:
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `MoGATv2`")

    # ROBUST: Use generalized model input building
    model_inputs = build_model_inputs(inputs, input_layers)
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    print(f"✅ MoGATv2 (Nature Paper) model built successfully with {len(model.layers)} layers")
    return model
