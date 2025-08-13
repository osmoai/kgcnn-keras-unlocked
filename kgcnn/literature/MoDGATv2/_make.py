"""Multi-Order Directed Graph Attention Network v2 (MoDGATv2) Model Factory.

This module provides the functional API for creating MoDGATv2 models.
"""

from kgcnn.literature.MoDGATv2._modgatv2_conv import MoDGATv2Layer, PoolingNodesMoDGATv2
from kgcnn.layers.embedding import Embedding
from kgcnn.layers.norm import RMSNormalization
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.modules import Concatenate
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer


def model_default(
    name="MoDGATv2",
    inputs=None,
    input_embedding=None,
    attention_args=None,
    depth=4,
    dropout=0.1,
    verbose=10,
    use_rms_norm=True,
    rms_norm_args=None,
    use_graph_state=True,
    output_embedding="graph",
    output_to_tensor=True,
    output_mlp=None,
    **kwargs
):
    """Default model configuration for MoDGATv2.
    
    Args:
        name: Model name
        inputs: Input layer configuration
        input_embedding: Input embedding configuration
        attention_args: Attention layer arguments
        depth: Number of DGAT layers
        dropout: Dropout rate
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
            {"shape": [None, 2], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True},
            {"shape": [2], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ]
    
    if input_embedding is None:
        input_embedding = {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        }
    
    if attention_args is None:
        attention_args = {"units": 128}
    
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
        "attention_args": attention_args,
        "depth": depth,
        "dropout": dropout,
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
               attention_args=None,
               depth=4,
               dropout=0.1,
               verbose=10,
               use_rms_norm=True,
               rms_norm_args=None,
               use_graph_state=True,
               output_embedding="graph",
               output_to_tensor=True,
               output_mlp=None,
               name="MoDGATv2",
               **kwargs):
    """Make MoDGATv2 model.
    
    Args:
        inputs: Input layer configuration
        input_embedding: Input embedding configuration
        attention_args: Attention layer arguments
        depth: Number of DGAT layers
        dropout: Dropout rate
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
        attention_args=attention_args,
        depth=depth,
        dropout=dropout,
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
    attention_args = config["attention_args"]
    depth = config["depth"]
    dropout = config["dropout"]
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
    
    # Input embeddings
    node_embedding = Embedding(**input_embedding["node"])(input_layers["node_attributes"])
    edge_embedding = Embedding(**input_embedding["edge"])(input_layers["edge_attributes"])
    
    # Check if we have graph descriptors
    has_descriptors = "graph_descriptors" in input_layers
    
    if has_descriptors:
        graph_embedding = Embedding(**input_embedding["graph"])(input_layers["graph_descriptors"])
    
    # MoDGATv2 core layers
    if verbose > 0:
        print(f"🏗️  Building MoDGATv2 (Multi-Order Directed) with depth={depth}")
    
    # Apply pre-attention RMS normalization
    if use_rms_norm:
        if verbose > 0:
            print(f"🔧 Applied pre-attention RMS normalization to layer 1")
        node_embedding = RMSNormalization(**rms_norm_args)(node_embedding)
    
    # Main MoDGATv2 layer
    modgatv2_output = MoDGATv2Layer(
        units=attention_args["units"],
        depth=depth,
        attention_heads=8,  # Default attention heads
        dropout_rate=dropout,
        use_rms_norm=use_rms_norm,
        rms_norm_args=rms_norm_args,
        name="modgatv2_layer"
    )([
        node_embedding, edge_embedding, 
        input_layers["edge_indices"], input_layers["edge_indices_reverse"]
    ])
    
    # Pooling
    if output_embedding == "graph":
        if use_graph_state:
            # Use graph state pooling
            out = PoolingNodesMoDGATv2(pooling_method="mean")(modgatv2_output)
        else:
            # Use node pooling
            out = PoolingNodes(pooling_method="mean")(modgatv2_output)
    else:
        out = modgatv2_output
    
    # Descriptor fusion
    if has_descriptors:
        if verbose > 0:
            print("🔧 Applied pre-descriptor fusion RMS normalization")
        if use_rms_norm:
            out = RMSNormalization(**rms_norm_args)(out)
        
        # Concatenate with graph descriptors
        out = Concatenate(axis=-1)([out, graph_embedding])
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
        print(f"✅ MoDGATv2 (Multi-Order Directed) model built successfully with {len(config)} layers")
    
    # Create model
    model = tf.keras.Model(inputs=list(input_layers.values()), outputs=out, name=name)
    
    return model
