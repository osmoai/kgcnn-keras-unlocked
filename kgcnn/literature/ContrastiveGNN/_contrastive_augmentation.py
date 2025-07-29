"""
Flexible Contrastive Learning Augmentation Strategies for Graph Neural Networks.

This module implements various augmentation strategies used in state-of-the-art
contrastive learning methods like MolCLR, GraphCL, MoCL, etc.

References:
- MolCLR: https://arxiv.org/abs/2102.10789
- GraphCL: https://arxiv.org/abs/2010.13902
- MoCL: https://arxiv.org/abs/2003.12002
"""

import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dropout
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.aggr import AggregateLocalEdges


class GraphAugmentationLayer(GraphBaseLayer):
    """Base class for graph augmentation strategies."""
    
    def __init__(self, augmentation_type="random", **kwargs):
        super(GraphAugmentationLayer, self).__init__(**kwargs)
        self.augmentation_type = augmentation_type
        
    def call(self, inputs, training=None):
        if training:
            return self._augment(inputs)
        else:
            return inputs
    
    def _augment(self, inputs):
        """Override this method in subclasses."""
        raise NotImplementedError


class MolCLRAugmentation(GraphAugmentationLayer):
    """
    MolCLR-style augmentation: Masking, deletion, subgraph.
    
    Reference: https://arxiv.org/abs/2102.10789
    """
    
    def __init__(self, 
                 node_mask_rate=0.15,
                 edge_drop_rate=0.15,
                 subgraph_ratio=0.8,
                 feature_noise_std=0.01,
                 **kwargs):
        super(MolCLRAugmentation, self).__init__(augmentation_type="molclr", **kwargs)
        self.node_mask_rate = node_mask_rate
        self.edge_drop_rate = edge_drop_rate
        self.subgraph_ratio = subgraph_ratio
        self.feature_noise_std = feature_noise_std
        
    def _augment(self, inputs):
        node_attributes, edge_attributes, edge_indices = inputs
        
        # 1. Node masking
        if self.node_mask_rate > 0:
            node_attributes = self._mask_nodes(node_attributes)
        
        # 2. Edge dropping
        if self.edge_drop_rate > 0:
            edge_attributes, edge_indices = self._drop_edges(edge_attributes, edge_indices)
        
        # 3. Subgraph sampling
        if self.subgraph_ratio < 1.0:
            node_attributes, edge_attributes, edge_indices = self._sample_subgraph(
                node_attributes, edge_attributes, edge_indices
            )
        
        # 4. Feature noise
        if self.feature_noise_std > 0:
            node_attributes = self._add_feature_noise(node_attributes)
            edge_attributes = self._add_feature_noise(edge_attributes)
        
        return [node_attributes, edge_attributes, edge_indices]
    
    def _mask_nodes(self, node_attributes):
        """Mask node features with zeros."""
        mask = tf.random.uniform(tf.shape(node_attributes)[:2]) > self.node_mask_rate
        mask = tf.expand_dims(mask, -1)
        return node_attributes * tf.cast(mask, node_attributes.dtype)
    
    def _drop_edges(self, edge_attributes, edge_indices):
        """Randomly drop edges."""
        mask = tf.random.uniform(tf.shape(edge_attributes)[:2]) > self.edge_drop_rate
        mask = tf.expand_dims(mask, -1)
        
        edge_attributes = edge_attributes * tf.cast(mask, edge_attributes.dtype)
        edge_indices = edge_indices * tf.cast(mask, edge_indices.dtype)
        
        return edge_attributes, edge_indices
    
    def _sample_subgraph(self, node_attributes, edge_attributes, edge_indices):
        """Sample a subgraph by keeping a subset of nodes."""
        # Simple implementation: keep top nodes by degree
        # In practice, you might want more sophisticated sampling
        num_nodes = tf.shape(node_attributes)[1]
        keep_nodes = tf.cast(num_nodes * self.subgraph_ratio, tf.int32)
        
        # Keep first keep_nodes nodes
        node_attributes = node_attributes[:, :keep_nodes, :]
        
        # Filter edges that connect to kept nodes
        mask = tf.reduce_all(edge_indices < keep_nodes, axis=-1)
        mask = tf.expand_dims(mask, -1)
        
        edge_attributes = edge_attributes * tf.cast(mask, edge_attributes.dtype)
        edge_indices = edge_indices * tf.cast(mask, edge_indices.dtype)
        
        return node_attributes, edge_attributes, edge_indices
    
    def _add_feature_noise(self, features):
        """Add Gaussian noise to features."""
        noise = tf.random.normal(tf.shape(features), stddev=self.feature_noise_std)
        return features + noise


class GraphCLAugmentation(GraphAugmentationLayer):
    """
    GraphCL-style augmentation: Generic augmentations (drop/mask/subgraph).
    
    Reference: https://arxiv.org/abs/2010.13902
    """
    
    def __init__(self,
                 node_drop_rate=0.2,
                 edge_drop_rate=0.2,
                 subgraph_ratio=0.8,
                 **kwargs):
        super(GraphCLAugmentation, self).__init__(augmentation_type="graphcl", **kwargs)
        self.node_drop_rate = node_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.subgraph_ratio = subgraph_ratio
        
    def _augment(self, inputs):
        node_attributes, edge_attributes, edge_indices = inputs
        
        # Randomly choose one augmentation strategy
        strategy = tf.random.uniform([], 0, 4, dtype=tf.int32)
        
        if strategy == 0:
            # Node dropping
            return self._drop_nodes(inputs)
        elif strategy == 1:
            # Edge dropping
            return self._drop_edges(inputs)
        elif strategy == 2:
            # Subgraph sampling
            return self._sample_subgraph(inputs)
        else:
            # No augmentation
            return inputs
    
    def _drop_nodes(self, inputs):
        node_attributes, edge_attributes, edge_indices = inputs
        mask = tf.random.uniform(tf.shape(node_attributes)[:2]) > self.node_drop_rate
        mask = tf.expand_dims(mask, -1)
        node_attributes = node_attributes * tf.cast(mask, node_attributes.dtype)
        return [node_attributes, edge_attributes, edge_indices]
    
    def _drop_edges(self, inputs):
        node_attributes, edge_attributes, edge_indices = inputs
        mask = tf.random.uniform(tf.shape(edge_attributes)[:2]) > self.edge_drop_rate
        mask = tf.expand_dims(mask, -1)
        edge_attributes = edge_attributes * tf.cast(mask, edge_attributes.dtype)
        edge_indices = edge_indices * tf.cast(mask, edge_indices.dtype)
        return [node_attributes, edge_attributes, edge_indices]
    
    def _sample_subgraph(self, inputs):
        node_attributes, edge_attributes, edge_indices = inputs
        num_nodes = tf.shape(node_attributes)[1]
        keep_nodes = tf.cast(num_nodes * self.subgraph_ratio, tf.int32)
        node_attributes = node_attributes[:, :keep_nodes, :]
        
        mask = tf.reduce_all(edge_indices < keep_nodes, axis=-1)
        mask = tf.expand_dims(mask, -1)
        edge_attributes = edge_attributes * tf.cast(mask, edge_attributes.dtype)
        edge_indices = edge_indices * tf.cast(mask, edge_indices.dtype)
        
        return [node_attributes, edge_attributes, edge_indices]


class MoCLAugmentation(GraphAugmentationLayer):
    """
    MoCL-style augmentation: Domain-knowledge substructure augmentation.
    
    Reference: https://arxiv.org/abs/2003.12002
    """
    
    def __init__(self,
                 substructure_mask_rate=0.3,
                 feature_noise_std=0.05,
                 **kwargs):
        super(MoCLAugmentation, self).__init__(augmentation_type="mocl", **kwargs)
        self.substructure_mask_rate = substructure_mask_rate
        self.feature_noise_std = feature_noise_std
        
    def _augment(self, inputs):
        node_attributes, edge_attributes, edge_indices = inputs
        
        # 1. Substructure masking (simplified - in practice would use domain knowledge)
        node_attributes = self._mask_substructures(node_attributes, edge_indices)
        
        # 2. Feature noise
        if self.feature_noise_std > 0:
            node_attributes = self._add_feature_noise(node_attributes)
            edge_attributes = self._add_feature_noise(edge_attributes)
        
        return [node_attributes, edge_attributes, edge_indices]
    
    def _mask_substructures(self, node_attributes, edge_indices):
        """Mask substructures based on connectivity patterns."""
        # Simplified: mask nodes with high degree
        degrees = tf.reduce_sum(tf.ones_like(edge_indices[:, :, 0]), axis=1)
        mask = degrees < tf.reduce_mean(degrees) + tf.math.reduce_std(degrees)
        mask = tf.expand_dims(mask, -1)
        return node_attributes * tf.cast(mask, node_attributes.dtype)
    
    def _add_feature_noise(self, features):
        """Add Gaussian noise to features."""
        noise = tf.random.normal(tf.shape(features), stddev=self.feature_noise_std)
        return features + noise


class DIGMolAugmentation(GraphAugmentationLayer):
    """
    DIG-Mol-style augmentation: One-way bond deletion, masking + momentum distillation.
    
    Reference: https://arxiv.org/abs/2106.10234
    """
    
    def __init__(self,
                 bond_deletion_rate=0.2,
                 node_mask_rate=0.15,
                 feature_noise_std=0.02,
                 **kwargs):
        super(DIGMolAugmentation, self).__init__(augmentation_type="digmol", **kwargs)
        self.bond_deletion_rate = bond_deletion_rate
        self.node_mask_rate = node_mask_rate
        self.feature_noise_std = feature_noise_std
        
    def _augment(self, inputs):
        node_attributes, edge_attributes, edge_indices = inputs
        
        # 1. One-way bond deletion (edge dropping)
        edge_attributes, edge_indices = self._delete_bonds(edge_attributes, edge_indices)
        
        # 2. Node masking
        node_attributes = self._mask_nodes(node_attributes)
        
        # 3. Feature noise
        if self.feature_noise_std > 0:
            node_attributes = self._add_feature_noise(node_attributes)
            edge_attributes = self._add_feature_noise(edge_attributes)
        
        return [node_attributes, edge_attributes, edge_indices]
    
    def _delete_bonds(self, edge_attributes, edge_indices):
        """Delete bonds (edges) randomly."""
        mask = tf.random.uniform(tf.shape(edge_attributes)[:2]) > self.bond_deletion_rate
        mask = tf.expand_dims(mask, -1)
        edge_attributes = edge_attributes * tf.cast(mask, edge_attributes.dtype)
        edge_indices = edge_indices * tf.cast(mask, edge_indices.dtype)
        return edge_attributes, edge_indices
    
    def _mask_nodes(self, node_attributes):
        """Mask node features."""
        mask = tf.random.uniform(tf.shape(node_attributes)[:2]) > self.node_mask_rate
        mask = tf.expand_dims(mask, -1)
        return node_attributes * tf.cast(mask, node_attributes.dtype)
    
    def _add_feature_noise(self, features):
        """Add Gaussian noise to features."""
        noise = tf.random.normal(tf.shape(features), stddev=self.feature_noise_std)
        return features + noise


class CLAPSAugmentation(GraphAugmentationLayer):
    """
    CLAPS-style augmentation: Attention-guided positive sample from SMILES.
    
    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8438750/
    """
    
    def __init__(self,
                 attention_drop_rate=0.2,
                 feature_noise_std=0.01,
                 **kwargs):
        super(CLAPSAugmentation, self).__init__(augmentation_type="claps", **kwargs)
        self.attention_drop_rate = attention_drop_rate
        self.feature_noise_std = feature_noise_std
        
    def _augment(self, inputs):
        node_attributes, edge_attributes, edge_indices = inputs
        
        # 1. Attention-guided dropping (simplified)
        node_attributes = self._attention_guided_drop(node_attributes, edge_indices)
        
        # 2. Feature noise
        if self.feature_noise_std > 0:
            node_attributes = self._add_feature_noise(node_attributes)
            edge_attributes = self._add_feature_noise(edge_attributes)
        
        return [node_attributes, edge_attributes, edge_indices]
    
    def _attention_guided_drop(self, node_attributes, edge_indices):
        """Drop nodes based on attention weights (simplified)."""
        # Simplified: drop nodes with low degree (less important)
        degrees = tf.reduce_sum(tf.ones_like(edge_indices[:, :, 0]), axis=1)
        attention_weights = degrees / tf.reduce_max(degrees)
        mask = attention_weights > self.attention_drop_rate
        mask = tf.expand_dims(mask, -1)
        return node_attributes * tf.cast(mask, node_attributes.dtype)
    
    def _add_feature_noise(self, features):
        """Add Gaussian noise to features."""
        noise = tf.random.normal(tf.shape(features), stddev=self.feature_noise_std)
        return features + noise


# Factory function for creating augmentation layers
def create_augmentation_layer(augmentation_type="molclr", **kwargs):
    """
    Factory function to create augmentation layers.
    
    Args:
        augmentation_type: Type of augmentation strategy
            - "molclr": MolCLR-style (masking, deletion, subgraph)
            - "graphcl": GraphCL-style (generic augmentations)
            - "mocl": MoCL-style (domain-knowledge substructure)
            - "digmol": DIG-Mol-style (bond deletion, masking)
            - "claps": CLAPS-style (attention-guided)
            - "random": Random choice between strategies
        **kwargs: Additional arguments for the augmentation layer
    
    Returns:
        GraphAugmentationLayer instance
    """
    
    augmentation_map = {
        "molclr": MolCLRAugmentation,
        "graphcl": GraphCLAugmentation,
        "mocl": MoCLAugmentation,
        "digmol": DIGMolAugmentation,
        "claps": CLAPSAugmentation,
    }
    
    if augmentation_type == "random":
        # Randomly choose an augmentation strategy
        strategies = list(augmentation_map.keys())
        strategy = np.random.choice(strategies)
        return augmentation_map[strategy](**kwargs)
    elif augmentation_type in augmentation_map:
        return augmentation_map[augmentation_type](**kwargs)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}. "
                        f"Available types: {list(augmentation_map.keys())} + 'random'") 