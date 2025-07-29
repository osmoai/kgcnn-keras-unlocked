#!/usr/bin/env python3
"""
Test script for contrastive augmentation layers.
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Add the parent directory to the path to import kgcnn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kgcnn.literature.ContrastiveGNN._contrastive_augmentation import (
    create_augmentation_layer,
    MolCLRAugmentation,
    GraphCLAugmentation,
    MoCLAugmentation,
    DIGMolAugmentation,
    CLAPSAugmentation
)

def create_test_graph():
    """Create a simple test graph."""
    # Batch size 2, max 3 nodes, 2 features
    node_attributes = tf.ragged.constant([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Graph 1: 3 nodes
        [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]]   # Graph 2: 2 nodes (padded)
    ], ragged_rank=1)
    
    # Edge attributes: batch size 2, max 4 edges, 1 feature
    edge_attributes = tf.ragged.constant([
        [[1.0], [2.0], [3.0], [4.0]],  # Graph 1: 4 edges
        [[5.0], [6.0], [0.0], [0.0]]   # Graph 2: 2 edges (padded)
    ], ragged_rank=1)
    
    # Edge indices: batch size 2, max 4 edges, 2 indices
    edge_indices = tf.ragged.constant([
        [[0, 1], [1, 2], [2, 0], [0, 2]],  # Graph 1: 4 edges
        [[0, 1], [1, 0], [0, 0], [0, 0]]   # Graph 2: 2 edges (padded)
    ], ragged_rank=1)
    
    return node_attributes, edge_attributes, edge_indices

def test_augmentation_layer(augmentation_type, **kwargs):
    """Test a specific augmentation layer."""
    print(f"\n{'='*50}")
    print(f"Testing {augmentation_type.upper()} augmentation")
    print(f"{'='*50}")
    
    # Create test data
    node_attrs, edge_attrs, edge_idx = create_test_graph()
    
    print("Original inputs:")
    print(f"Node attributes shape: {node_attrs.shape}")
    print(f"Edge attributes shape: {edge_attrs.shape}")
    print(f"Edge indices shape: {edge_idx.shape}")
    
    # Create augmentation layer
    try:
        aug_layer = create_augmentation_layer(augmentation_type, **kwargs)
        print(f"Created {augmentation_type} augmentation layer successfully")
        
        # Apply augmentation
        aug_inputs = aug_layer([node_attrs, edge_attrs, edge_idx])
        aug_nodes, aug_edges, aug_indices = aug_inputs
        
        print("\nAfter augmentation:")
        print(f"Augmented node attributes shape: {aug_nodes.shape}")
        print(f"Augmented edge attributes shape: {aug_edges.shape}")
        print(f"Augmented edge indices shape: {aug_indices.shape}")
        
        # Check if shapes are preserved
        assert aug_nodes.shape == node_attrs.shape, f"Node shape mismatch: {aug_nodes.shape} vs {node_attrs.shape}"
        assert aug_edges.shape == edge_attrs.shape, f"Edge shape mismatch: {aug_edges.shape} vs {edge_attrs.shape}"
        assert aug_indices.shape == edge_idx.shape, f"Index shape mismatch: {aug_indices.shape} vs {edge_idx.shape}"
        
        print("✅ Shape preservation: PASSED")
        
        # Check if values changed (augmentation should modify values)
        nodes_changed = not tf.reduce_all(tf.equal(aug_nodes, node_attrs))
        edges_changed = not tf.reduce_all(tf.equal(aug_edges, edge_attrs))
        
        print(f"Nodes modified: {nodes_changed}")
        print(f"Edges modified: {edges_changed}")
        
        if nodes_changed or edges_changed:
            print("✅ Augmentation applied: PASSED")
        else:
            print("⚠️  No changes detected (might be due to low augmentation rates)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {augmentation_type}: {str(e)}")
        return False

def test_all_augmentations():
    """Test all augmentation strategies."""
    print("Testing Contrastive Augmentation Layers")
    print("="*60)
    
    # Test configurations for each augmentation type
    test_configs = {
        "molclr": {
            "node_mask_rate": 0.3,
            "edge_drop_rate": 0.3,
            "subgraph_ratio": 0.8,
            "feature_noise_std": 0.1
        },
        "graphcl": {
            "node_drop_rate": 0.3,
            "edge_drop_rate": 0.3,
            "subgraph_ratio": 0.8
        },
        "mocl": {
            "substructure_mask_rate": 0.3,
            "feature_noise_std": 0.1
        },
        "digmol": {
            "bond_deletion_rate": 0.3,
            "node_mask_rate": 0.3,
            "feature_noise_std": 0.1
        },
        "claps": {
            "attention_drop_rate": 0.3,
            "feature_noise_std": 0.1
        }
    }
    
    results = {}
    
    # Test each augmentation type
    for aug_type, config in test_configs.items():
        results[aug_type] = test_augmentation_layer(aug_type, **config)
    
    # Test random augmentation
    results["random"] = test_augmentation_layer("random")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for aug_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{aug_type.upper():<12}: {status}")
    
    print(f"\nOverall: {passed}/{total} augmentation types passed")
    
    if passed == total:
        print("🎉 All augmentation tests passed!")
    else:
        print("⚠️  Some augmentation tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = test_all_augmentations()
    sys.exit(0 if success else 1) 