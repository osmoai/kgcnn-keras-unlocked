"""Multi-Order Directed Message Passing Neural Network (MoDMPNN) Model Factory.

This module provides the functional API for creating MoDMPNN models.
"""

import tensorflow as tf
from kgcnn.literature.MoDMPNN._modmpnn_conv import MoDMPNNLayer, PoolingNodesMoDMPNN
from kgcnn.layers.modules import Dense, OptionalInputEmbedding
from kgcnn.layers.norm import RMSNormalization
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.utils.input_utils import create_mixed_input_embedding, get_molecular_feature_types


def model_default(
    name="MoDMPNN",
    inputs=None,
    input_embedding=None,
    units=128,
    depth=4,
    dropout_rate=0.1,
    verbose=10,
    use_rms_norm=True,
    rms_norm_args=None,
    use_graph_state=True,
    output_embedding="graph",
    output_to_tensor=True,
    output_mlp=None,
    **kwargs
):
    """Default model configuration for MoDMPNN.
    
    Args:
        name: Model name
        inputs: Input layer configuration
        input_embedding: Input embedding configuration
        units: Number of hidden units
        depth: Number of DMPNN layers
        dropout_rate: Dropout rate
        verbose: Verbosity level
        use_rms_norm: Whether to use RMS normalization
        rms_norm_args: RMS normalization arguments
        use_graph_state: Whether to use graph state
        output_embedding: Output embedding type
        output_to_tensor: Whether to convert output to tensor
        output_mlp: Output MLP configuration
        **kwargs: Additional arguments
        
    Returns:
        Model configuration dictionary
    """
    if inputs is None:
        inputs = [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True},
            {"shape": [2], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ]
    
    if input_embedding is None:
        input_embedding = {
            "node": {"output_dim": 128},  # Mixed categorical/continuous - handled automatically 
            "edge": {"input_dim": 5, "output_dim": 128},  # Categorical bond types
            "graph": {"output_dim": 64}  # Continuous descriptors
        }
    
    if rms_norm_args is None:
        rms_norm_args = {"epsilon": 1e-6, "scale": True}
    
    if output_mlp is None:
        output_mlp = {
            "use_bias": [True, True, False],
            "units": [200, 100, 1],
            "activation": ["kgcnn>leaky_relu", "selu", "linear"]
        }
    
    return {
        "name": name,
        "inputs": inputs,
        "input_embedding": input_embedding,
        "units": units,
        "depth": depth,
        "dropout_rate": dropout_rate,
        "verbose": verbose,
        "use_rms_norm": use_rms_norm,
        "rms_norm_args": rms_norm_args,
        "use_graph_state": use_graph_state,
        "output_embedding": output_embedding,
        "output_to_tensor": output_to_tensor,
        "output_mlp": output_mlp,
        **kwargs
    }


def make_model(inputs=None,
               input_embedding=None,
               units=128,
               depth=4,
               dropout_rate=0.1,
               verbose=10,
               use_rms_norm=True,
               rms_norm_args=None,
               use_graph_state=True,
               output_embedding="graph",
               output_to_tensor=True,
               output_mlp=None,
               name="MoDMPNN",
               **kwargs):
    """Make MoDMPNN model.
    
    Args:
        inputs: Input layer configuration
        input_embedding: Input embedding configuration
        units: Number of hidden units
        depth: Number of DMPNN layers
        dropout_rate: Dropout rate
        verbose: Verbosity level
        use_rms_norm: Whether to use RMS normalization
        rms_norm_args: RMS normalization arguments
        use_graph_state: Whether to use graph state
        output_embedding: Output embedding type
        output_to_tensor: Whether to convert output to tensor
        output_mlp: Output MLP configuration
        name: Model name
        **kwargs: Additional arguments
        
    Returns:
        Keras model
    """
    # Get default configuration
    config = model_default(
        name=name,
        inputs=inputs,
        input_embedding=input_embedding,
        units=units,
        depth=depth,
        dropout_rate=dropout_rate,
        verbose=verbose,
        use_rms_norm=use_rms_norm,
        rms_norm_args=rms_norm_args,
        use_graph_state=use_graph_state,
        output_embedding=output_embedding,
        output_to_tensor=output_to_tensor,
        output_mlp=output_mlp,
        **kwargs
    )
    
    # Extract configuration
    inputs = config["inputs"]
    input_embedding = config["input_embedding"]
    units = config["units"]
    depth = config["depth"]
    dropout_rate = config["dropout_rate"]
    verbose = config["verbose"]
    use_rms_norm = config["use_rms_norm"]
    rms_norm_args = config["rms_norm_args"]
    use_graph_state = config["use_graph_state"]
    output_embedding = config["output_embedding"]
    output_to_tensor = config["output_to_tensor"]
    output_mlp = config["output_mlp"]
    
    # Create input layers
    input_layers = {}
    for i, input_config in enumerate(inputs):
        input_layers[input_config["name"]] = tf.keras.layers.Input(**input_config)
    
    # Input embeddings - handle mixed categorical/continuous features properly
    node_feature_types = get_molecular_feature_types()
    node_embedding = create_mixed_input_embedding(
        input_layers["node_attributes"], 
        input_embedding["node"], 
        node_feature_types,
        verbose=verbose
    )
    
    # Edge features are typically categorical (bond types)
    edge_embedding = OptionalInputEmbedding(
        **input_embedding["edge"],
        use_embedding=True,
        name="edge_embedding"
    )(input_layers["edge_attributes"])
    
    # Check if we have graph descriptors
    has_descriptors = "graph_descriptors" in input_layers
    
    if has_descriptors:
        # Graph descriptors are always continuous
        graph_embedding = Dense(
            units=input_embedding["graph"]["output_dim"], 
            activation='relu', 
            use_bias=True,
            name="graph_descriptor_projection"
        )(input_layers["graph_descriptors"])
    
    # MoDMPNN core layers
    if verbose > 0:
        print(f"🏗️  Building MoDMPNN (Multi-Order Directed MPNN) with depth={depth}")
    
    # Apply pre-attention RMS normalization
    if use_rms_norm:
        if verbose > 0:
            print(f"🔧 Applied pre-attention RMS normalization to layer 1")
        node_embedding = RMSNormalization(**rms_norm_args)(node_embedding)
    
    # Main MoDMPNN layer
    modmpnn_output = MoDMPNNLayer(
        units=units,
        depth=depth,
        dropout_rate=dropout_rate,
        use_rms_norm=use_rms_norm,
        rms_norm_args=rms_norm_args,
        name="modmpnn_layer"
    )([
        node_embedding, edge_embedding, 
        input_layers["edge_indices"], input_layers["edge_indices_reverse"]
    ])
    
    # Pooling
    if output_embedding == "graph":
        if use_graph_state:
            # Use graph state pooling
            out = PoolingNodesMoDMPNN(pooling_method="mean")(modmpnn_output)
        else:
            # Use node pooling
            out = PoolingNodes(pooling_method="mean")(modmpnn_output)
    else:
        out = modmpnn_output
    
    # Descriptor fusion
    if has_descriptors:
        if verbose > 0:
            print("🔧 Applied pre-descriptor fusion RMS normalization")
        if use_rms_norm:
            out = RMSNormalization(**rms_norm_args)(out)
        
        # Concatenate with graph descriptors
        out = LazyConcatenate(axis=-1)([out, graph_embedding])
        if verbose > 0:
            print("✅ Fused descriptors using concatenate")
    
    # Pre-MLP RMS normalization
    if use_rms_norm:
        if verbose > 0:
            print("🔧 Applied pre-MLP RMS normalization")
        out = RMSNormalization(**rms_norm_args)(out)
    
    # Output MLP
    if output_mlp is not None:
        out = GraphMLP(**output_mlp)(out)
    
    # Convert to tensor if needed
    if output_to_tensor:
        out = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(x))(out)
    
    if verbose > 0:
        print(f"✅ MoDMPNN (Multi-Order Directed MPNN) model built successfully with {len(config)} layers")
    
    # Create model
    model = tf.keras.Model(inputs=list(input_layers.values()), outputs=out, name=name)
    
    return model
