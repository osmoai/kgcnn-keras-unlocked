"""
Utility functions for robust input handling in kgcnn models.

This module provides functions to handle optional inputs, check input presence by name,
and ensure consistent model architecture regardless of input variations.
"""

import tensorflow.keras as ks
from typing import List, Dict, Optional, Tuple, Any


def get_input_names(inputs: List[Dict[str, Any]]) -> List[str]:
    """
    Extract input names from input configuration list.
    
    Args:
        inputs: List of input configuration dictionaries
        
    Returns:
        List of input names
    """
    return [inp['name'] for inp in inputs]


def find_input_by_name(inputs: List[Dict[str, Any]], name: str) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Find input by name and return its index and configuration.
    
    Args:
        inputs: List of input configuration dictionaries
        name: Name of the input to find
        
    Returns:
        Tuple of (index, input_config) if found, None otherwise
    """
    input_names = get_input_names(inputs)
    if name in input_names:
        idx = input_names.index(name)
        return idx, inputs[idx]
    return None


def create_input_layer(input_config: Dict[str, Any]) -> ks.layers.Input:
    """
    Create an Input layer from input configuration.
    
    Args:
        input_config: Input configuration dictionary
        
    Returns:
        Keras Input layer
    """
    return ks.layers.Input(**input_config)


def get_optional_inputs(inputs: List[Dict[str, Any]], required_names: List[str]) -> Dict[str, Optional[ks.layers.Input]]:
    """
    Create Input layers for all inputs, handling optional ones gracefully.
    
    Args:
        inputs: List of input configuration dictionaries
        required_names: List of required input names (must be present)
        
    Returns:
        Dictionary mapping input names to Input layers (None if not present)
    """
    input_layers = {}
    input_names = get_input_names(inputs)
    
    # Create layers for all inputs
    for i, input_config in enumerate(inputs):
        name = input_config['name']
        input_layers[name] = create_input_layer(input_config)
        print(f"✅ Created input layer: {name}")
    
    # Check for missing required inputs
    for name in required_names:
        if name not in input_names:
            raise ValueError(f"Required input '{name}' not found in inputs: {input_names}")
    
    return input_layers


def check_descriptor_input(inputs: List[Dict[str, Any]]) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Check if graph_descriptors input is present and return its configuration.
    
    Args:
        inputs: List of input configuration dictionaries
        
    Returns:
        Tuple of (index, input_config) if found, None otherwise
    """
    result = find_input_by_name(inputs, 'graph_descriptors')
    if result:
        idx, config = result
        print(f"✅ Found graph_descriptors input at position {idx}")
        return result
    else:
        print("⚠️  No graph_descriptors input found")
        return None


def create_descriptor_processing_layer(graph_descriptors_input: ks.layers.Input, 
                                     input_embedding: Dict[str, Any],
                                     layer_name: str = "graph_descriptor_processing") -> Optional[ks.layers.Layer]:
    """
    Create descriptor processing layer if descriptors are present.
    
    Args:
        graph_descriptors_input: Input layer for descriptors
        input_embedding: Input embedding configuration
        layer_name: Name for the processing layer
        
    Returns:
        Dense layer for descriptor processing, or None if no descriptors
    """
    if graph_descriptors_input is not None:
        # FIX: Use Dense layer for continuous float descriptors instead of OptionalInputEmbedding
        # Descriptors are float values, not categorical indices!
        graph_embedding = ks.layers.Dense(
            input_embedding.get("graph", {"output_dim": 64})["output_dim"], 
            activation='relu', 
            use_bias=True,
            name=layer_name
        )(graph_descriptors_input)
        print(f"✅ Created descriptor processing layer: {layer_name}")
        return graph_embedding
    else:
        print("❌ No descriptor processing layer created (no descriptors)")
        return None


def fuse_descriptors_with_output(output_tensor: ks.layers.Layer, 
                                descriptor_tensor: Optional[ks.layers.Layer],
                                fusion_method: str = "concatenate") -> ks.layers.Layer:
    """
    Fuse descriptors with model output using specified method.
    
    Args:
        output_tensor: Main model output tensor
        descriptor_tensor: Descriptor processing tensor (can be None)
        fusion_method: Method for fusion ("concatenate", "add", "attention")
        
    Returns:
        Fused output tensor
    """
    if descriptor_tensor is not None:
        if fusion_method == "concatenate":
            fused = ks.layers.Concatenate()([output_tensor, descriptor_tensor])
            print(f"✅ Fused descriptors using {fusion_method}")
        elif fusion_method == "add":
            # Ensure dimensions match for addition
            if output_tensor.shape[-1] == descriptor_tensor.shape[-1]:
                fused = ks.layers.Add()([output_tensor, descriptor_tensor])
            else:
                # Project descriptor to match output dimension
                projection = ks.layers.Dense(
                    output_tensor.shape[-1], 
                    activation="linear", 
                    use_bias=False,
                    name="descriptor_projection"
                )(descriptor_tensor)
                fused = ks.layers.Add()([output_tensor, projection])
            print(f"✅ Fused descriptors using {fusion_method}")
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        return fused
    else:
        print("⚠️  No descriptor fusion (no descriptors)")
        return output_tensor


def build_model_inputs(inputs: List[Dict[str, Any]], 
                      input_layers: Dict[str, ks.layers.Input]) -> List[ks.layers.Input]:
    """
    Build model inputs list in the correct order.
    
    Args:
        inputs: Original input configuration list (defines order)
        input_layers: Dictionary of input layers by name
        
    Returns:
        List of input layers in the correct order
    """
    model_inputs = []
    for input_config in inputs:
        name = input_config['name']
        if name in input_layers:
            model_inputs.append(input_layers[name])
        else:
            raise ValueError(f"Input layer '{name}' not found in input_layers")
    
    return model_inputs 