"""
Examples of different contrastive augmentation strategies for GNNs.

This file shows how to configure various augmentation strategies based on
state-of-the-art contrastive learning methods.
"""

# Example 1: MolCLR-style augmentation (default)
molclr_config = {
    "contrastive_args": {
        "use_contrastive_loss": True,
        "contrastive_loss_type": "infonce",
        "temperature": 0.1,
        "contrastive_weight": 0.1,
        "num_views": 2,
        "use_diversity_loss": False,
        "use_auxiliary_loss": False,
        "augmentation_type": "molclr",
        "augmentation_args": {
            "node_mask_rate": 0.15,      # Mask 15% of node features
            "edge_drop_rate": 0.15,      # Drop 15% of edges
            "subgraph_ratio": 0.8,       # Keep 80% of nodes
            "feature_noise_std": 0.01    # Add small noise to features
        }
    }
}

# Example 2: GraphCL-style augmentation (random strategy selection)
graphcl_config = {
    "contrastive_args": {
        "use_contrastive_loss": True,
        "contrastive_loss_type": "infonce",
        "temperature": 0.1,
        "contrastive_weight": 0.1,
        "num_views": 2,
        "use_diversity_loss": False,
        "use_auxiliary_loss": False,
        "augmentation_type": "graphcl",
        "augmentation_args": {
            "node_drop_rate": 0.2,       # Drop 20% of nodes
            "edge_drop_rate": 0.2,       # Drop 20% of edges
            "subgraph_ratio": 0.8        # Keep 80% of nodes
        }
    }
}

# Example 3: MoCL-style augmentation (domain-knowledge based)
mocl_config = {
    "contrastive_args": {
        "use_contrastive_loss": True,
        "contrastive_loss_type": "infonce",
        "temperature": 0.1,
        "contrastive_weight": 0.1,
        "num_views": 2,
        "use_diversity_loss": False,
        "use_auxiliary_loss": False,
        "augmentation_type": "mocl",
        "augmentation_args": {
            "substructure_mask_rate": 0.3,  # Mask 30% of substructures
            "feature_noise_std": 0.05       # Add moderate noise
        }
    }
}

# Example 4: DIG-Mol-style augmentation (bond deletion focus)
digmol_config = {
    "contrastive_args": {
        "use_contrastive_loss": True,
        "contrastive_loss_type": "infonce",
        "temperature": 0.1,
        "contrastive_weight": 0.1,
        "num_views": 2,
        "use_diversity_loss": False,
        "use_auxiliary_loss": False,
        "augmentation_type": "digmol",
        "augmentation_args": {
            "bond_deletion_rate": 0.2,   # Delete 20% of bonds
            "node_mask_rate": 0.15,      # Mask 15% of nodes
            "feature_noise_std": 0.02    # Add small noise
        }
    }
}

# Example 5: CLAPS-style augmentation (attention-guided)
claps_config = {
    "contrastive_args": {
        "use_contrastive_loss": True,
        "contrastive_loss_type": "infonce",
        "temperature": 0.1,
        "contrastive_weight": 0.1,
        "num_views": 2,
        "use_diversity_loss": False,
        "use_auxiliary_loss": False,
        "augmentation_type": "claps",
        "augmentation_args": {
            "attention_drop_rate": 0.2,  # Drop 20% based on attention
            "feature_noise_std": 0.01    # Add small noise
        }
    }
}

# Example 6: Random augmentation (randomly choose strategy each time)
random_config = {
    "contrastive_args": {
        "use_contrastive_loss": True,
        "contrastive_loss_type": "infonce",
        "temperature": 0.1,
        "contrastive_weight": 0.1,
        "num_views": 2,
        "use_diversity_loss": False,
        "use_auxiliary_loss": False,
        "augmentation_type": "random",
        "augmentation_args": {
            # Will use default args for randomly chosen strategy
        }
    }
}

# Example 7: Custom augmentation with specific parameters
custom_config = {
    "contrastive_args": {
        "use_contrastive_loss": True,
        "contrastive_loss_type": "infonce",
        "temperature": 0.1,
        "contrastive_weight": 0.1,
        "num_views": 3,  # More views for better contrastive learning
        "use_diversity_loss": True,  # Enable diversity loss
        "use_auxiliary_loss": True,  # Enable auxiliary loss
        "augmentation_type": "molclr",
        "augmentation_args": {
            "node_mask_rate": 0.2,       # More aggressive masking
            "edge_drop_rate": 0.25,      # More aggressive edge dropping
            "subgraph_ratio": 0.7,       # Keep fewer nodes
            "feature_noise_std": 0.02    # More noise
        }
    }
}

# Example 8: No augmentation (for comparison)
no_augmentation_config = {
    "contrastive_args": {
        "use_contrastive_loss": True,
        "contrastive_loss_type": "infonce",
        "temperature": 0.1,
        "contrastive_weight": 0.1,
        "num_views": 2,
        "use_diversity_loss": False,
        "use_auxiliary_loss": False,
        "augmentation_type": "molclr",
        "augmentation_args": {
            "node_mask_rate": 0.0,       # No masking
            "edge_drop_rate": 0.0,       # No edge dropping
            "subgraph_ratio": 1.0,       # Keep all nodes
            "feature_noise_std": 0.0     # No noise
        }
    }
}

# Usage examples:
if __name__ == "__main__":
    print("Available augmentation strategies:")
    print("1. MolCLR: Masking, deletion, subgraph")
    print("2. GraphCL: Generic augmentations with random selection")
    print("3. MoCL: Domain-knowledge substructure augmentation")
    print("4. DIG-Mol: Bond deletion and masking")
    print("5. CLAPS: Attention-guided augmentation")
    print("6. Random: Randomly choose between strategies")
    print("7. Custom: User-defined parameters")
    print("8. No augmentation: For baseline comparison")
    
    print("\nTo use in your config file, replace the contrastive_args section with one of the above examples.")
    print("For example, to use MolCLR-style augmentation:")
    print("contrastive_args = molclr_config['contrastive_args']") 