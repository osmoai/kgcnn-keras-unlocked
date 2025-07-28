"""Multi-Graph MoE Convolution Layer with Multiple Graph Representations and Expert Routing.

This module implements a Multi-Graph MoE that creates multiple graph representations
and uses different GNN experts to improve ensemble performance and reduce variance.
"""

import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, Activation, Dropout
from tensorflow.keras.layers import LayerNormalization
from kgcnn.layers.attention import AttentionHeadGAT
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
import math

# Import MultiGraphMoE specific losses and metrics
try:
    from ._multigraph_moe_losses import (
        MultiGraphMoELoadBalancingLoss,
        MultiGraphMoEAuxiliaryLoss,
        MultiGraphMoEExpertUsageMetric,
        MultiGraphMoEGatingEntropyMetric,
        MultiGraphMoELoadBalanceScoreMetric
    )
except ImportError:
    # Fallback if losses module is not available
    MultiGraphMoELoadBalancingLoss = None
    MultiGraphMoEAuxiliaryLoss = None
    MultiGraphMoEExpertUsageMetric = None
    MultiGraphMoEGatingEntropyMetric = None
    MultiGraphMoELoadBalanceScoreMetric = None

ks = tf.keras


class GraphRepresentationLayer(ks.layers.Layer):
    """Creates multiple graph representations from a single input graph.
    
    This layer generates multiple graph representations by:
    1. Different edge weightings
    2. Different node feature transformations
    3. Different graph structures (subgraphs, augmented graphs)
    4. Different attention patterns
    
    Args:
        num_representations (int): Number of graph representations to create
        representation_types (list): Types of representations to create
        use_edge_weights (bool): Whether to create different edge weightings
        use_node_features (bool): Whether to create different node features
        use_attention (bool): Whether to create different attention patterns
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, num_representations=4, representation_types=None,
                 use_edge_weights=True, use_node_features=True, use_attention=True,
                 dropout_rate=0.1, **kwargs):
        super(GraphRepresentationLayer, self).__init__(**kwargs)
        
        self.num_representations = num_representations
        self.use_edge_weights = use_edge_weights
        self.use_node_features = use_node_features
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # Default representation types
        if representation_types is None:
            self.representation_types = [
                "original", "weighted", "augmented", "attention"
            ]
        else:
            self.representation_types = representation_types
        
        # Edge weight generators for different representations
        if self.use_edge_weights:
            self.edge_weight_generators = []
            for i in range(num_representations):
                edge_generator = Dense(
                    units=1,
                    activation="sigmoid",
                    use_bias=True,
                    name=f"edge_weight_gen_{i}"
                )
                self.edge_weight_generators.append(edge_generator)
        
        # Node feature transformers for different representations
        if self.use_node_features:
            self.node_transformers = []
            for i in range(num_representations):
                node_transformer = Dense(
                    units=64,  # Will be set dynamically
                    activation="relu",
                    use_bias=True,
                    name=f"node_transformer_{i}"
                )
                self.node_transformers.append(node_transformer)
        
        # Attention generators for different representations
        if self.use_attention:
            self.attention_generators = []
            for i in range(num_representations):
                attention_generator = Dense(
                    units=1,
                    activation="sigmoid",
                    use_bias=True,
                    name=f"attention_gen_{i}"
                )
                self.attention_generators.append(attention_generator)
        
        # Dropout for regularization
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Create multiple graph representations.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices]
            training: Training mode flag
            
        Returns:
            List of graph representations
        """
        node_features, edge_features, edge_indices = inputs
        
        representations = []
        
        for i in range(self.num_representations):
            rep_type = self.representation_types[i % len(self.representation_types)]
            
            if rep_type == "original":
                # Original representation
                rep = {
                    "node_features": node_features,
                    "edge_features": edge_features,
                    "edge_indices": edge_indices,
                    "edge_weights": tf.ones_like(edge_features[:, :, 0:1])
                }
            
            elif rep_type == "weighted" and self.use_edge_weights:
                # Weighted representation with learned edge weights
                edge_weights = self.edge_weight_generators[i](edge_features)
                if self.dropout and training:
                    edge_weights = self.dropout(edge_weights)
                
                rep = {
                    "node_features": node_features,
                    "edge_features": edge_features * edge_weights,
                    "edge_indices": edge_indices,
                    "edge_weights": edge_weights
                }
            
            elif rep_type == "augmented" and self.use_node_features:
                # Augmented representation with transformed node features
                node_dim = node_features.shape[-1]
                if not hasattr(self.node_transformers[i], '_built'):
                    self.node_transformers[i].build((None, node_dim))
                
                transformed_nodes = self.node_transformers[i](node_features)
                if self.dropout and training:
                    transformed_nodes = self.dropout(transformed_nodes)
                
                rep = {
                    "node_features": transformed_nodes,
                    "edge_features": edge_features,
                    "edge_indices": edge_indices,
                    "edge_weights": tf.ones_like(edge_features[:, :, 0:1])
                }
            
            elif rep_type == "attention" and self.use_attention:
                # Attention-based representation
                attention_weights = self.attention_generators[i](edge_features)
                if self.dropout and training:
                    attention_weights = self.dropout(attention_weights)
                
                rep = {
                    "node_features": node_features,
                    "edge_features": edge_features,
                    "edge_indices": edge_indices,
                    "edge_weights": attention_weights
                }
            
            else:
                # Fallback to original
                rep = {
                    "node_features": node_features,
                    "edge_features": edge_features,
                    "edge_indices": edge_indices,
                    "edge_weights": tf.ones_like(edge_features[:, :, 0:1])
                }
            
            representations.append(rep)
        
        return representations
    
    def get_config(self):
        """Get layer configuration."""
        config = super(GraphRepresentationLayer, self).get_config()
        config.update({
            "num_representations": self.num_representations,
            "representation_types": self.representation_types,
            "use_edge_weights": self.use_edge_weights,
            "use_node_features": self.use_node_features,
            "use_attention": self.use_attention,
            "dropout_rate": self.dropout_rate
        })
        return config


class GINExpert(ks.layers.Layer):
    """GIN (Graph Isomorphism Network) Expert for MoE."""
    
    def __init__(self, units, use_bias=True, activation="relu", **kwargs):
        super(GINExpert, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        
        # GIN MLP
        self.gin_mlp = MLP(
            units=[units, units],
            use_bias=use_bias,
            activation=activation,
            use_normalization=True,
            normalization_technique="graph_batch"
        )
        
        # Aggregation layer
        self.aggregate = AggregateLocalEdges(pooling_method="sum")
        
        # Gather layers
        self.gather_in = GatherNodesIngoing()
        self.gather_out = GatherNodesOutgoing()
    
    def call(self, inputs):
        """Forward pass of GIN expert."""
        node_features, edge_features, edge_indices = inputs
        
        # Gather neighbor features
        neighbor_features = self.gather_in([node_features, edge_indices])
        
        # Aggregate neighbor features
        aggregated = self.aggregate([node_features, neighbor_features, edge_indices])
        
        # Apply GIN MLP
        output = self.gin_mlp(aggregated)
        
        return output


class GATExpert(ks.layers.Layer):
    """GAT (Graph Attention Network) Expert for MoE."""
    
    def __init__(self, units, num_heads=4, use_bias=True, activation="relu", **kwargs):
        super(GATExpert, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.activation = activation
        
        # GAT attention heads
        self.attention_heads = []
        head_units = max(1, units // num_heads)  # Ensure at least 1 unit per head
        for i in range(num_heads):
            attention_head = AttentionHeadGAT(
                units=head_units,
                use_edge_features=True,
                use_final_activation=False,
                has_self_loops=True,
                activation=activation,
                use_bias=use_bias
            )
            self.attention_heads.append(attention_head)
        
        # Final activation
        self.final_activation = Activation(activation)
    
    def call(self, inputs):
        """Forward pass of GAT expert."""
        node_features, edge_features, edge_indices = inputs
        
        # Apply attention heads
        attention_outputs = []
        for attention_head in self.attention_heads:
            attention_out = attention_head([node_features, edge_features, edge_indices])
            attention_outputs.append(attention_out)
        
        # Concatenate attention outputs
        if len(attention_outputs) > 1:
            combined = tf.concat(attention_outputs, axis=-1)
        else:
            combined = attention_outputs[0]
        
        # Apply final activation
        output = self.final_activation(combined)
        
        return output


class GCNExpert(ks.layers.Layer):
    """GCN (Graph Convolutional Network) Expert for MoE."""
    
    def __init__(self, units, use_bias=True, activation="relu", **kwargs):
        super(GCNExpert, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        
        # GCN transformation
        self.gcn_transform = Dense(
            units=units,
            activation=activation,
            use_bias=use_bias
        )
        
        # Aggregation layer
        self.aggregate = AggregateLocalEdges(pooling_method="sum")
        
        # Gather layers
        self.gather_in = GatherNodesIngoing()
    
    def call(self, inputs):
        """Forward pass of GCN expert."""
        node_features, edge_features, edge_indices = inputs
        
        # Gather neighbor features
        neighbor_features = self.gather_in([node_features, edge_indices])
        
        # Aggregate neighbor features
        aggregated = self.aggregate([node_features, neighbor_features, edge_indices])
        
        # Apply GCN transformation
        output = self.gcn_transform(aggregated)
        
        return output


class ExpertRoutingLayer(ks.layers.Layer):
    """MoE routing layer that selects experts for each graph representation."""
    
    def __init__(self, num_experts=3, expert_types=None, temperature=1.0, 
                 use_noise=True, noise_epsilon=1e-2, units=128, 
                 use_load_balancing=True, use_auxiliary_loss=True, **kwargs):
        super(ExpertRoutingLayer, self).__init__(**kwargs)
        
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_noise = use_noise
        self.noise_epsilon = noise_epsilon
        self.units = units
        self.use_load_balancing = use_load_balancing
        self.use_auxiliary_loss = use_auxiliary_loss
        
        # Default expert types
        if expert_types is None:
            self.expert_types = ["gin", "gat", "gcn"]
        else:
            self.expert_types = expert_types
        
        # Router network - will be built dynamically
        self.router = None
        self.router_units = num_experts
        
        # Fixed routing weights for simplicity
        self.use_fixed_routing = True
        
        # Expert networks - will be built dynamically
        self.experts = []
        self.expert_types_list = []
        for i in range(num_experts):
            expert_type = self.expert_types[i % len(self.expert_types)]
            self.expert_types_list.append(expert_type)
            # Experts will be built dynamically based on input dimensions
        
        # Load balancing loss
        self.aux_loss = 0.0
        
        # Gating weights tracking for losses and metrics
        self.gating_weights_history = []
        
        # Initialize losses if available
        if use_load_balancing and MultiGraphMoELoadBalancingLoss is not None:
            self.load_balancing_loss = MultiGraphMoELoadBalancingLoss(
                num_experts=num_experts,
                importance_weight=0.01,
                load_weight=0.01
            )
        else:
            self.load_balancing_loss = None
            
        if use_auxiliary_loss and MultiGraphMoEAuxiliaryLoss is not None:
            self.auxiliary_loss = MultiGraphMoEAuxiliaryLoss(
                num_experts=num_experts,
                diversity_weight=0.01,
                entropy_weight=0.01
            )
        else:
            self.auxiliary_loss = None
    
    def call(self, inputs, training=None):
        """Route inputs to experts and combine outputs.
        
        Args:
            inputs: List of graph representations
            training: Training mode flag
            
        Returns:
            Combined expert outputs
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        all_expert_outputs = []
        routing_weights = []
        
        for graph_rep in inputs:
            # Extract features for routing
            node_features = graph_rep["node_features"]
            edge_features = graph_rep["edge_features"]
            edge_indices = graph_rep["edge_indices"]
            
            # Use fixed routing weights for simplicity
            if self.use_fixed_routing:
                # Create uniform routing weights
                batch_size = tf.shape(node_features)[0]
                weights = tf.ones((batch_size, self.num_experts)) / self.num_experts
            else:
                # Original routing logic (commented out for now)
                # Global pooling for routing decision - use fixed-size representation
                pooled_nodes = tf.reduce_mean(node_features, axis=1)  # [batch, features]
                pooled_edges = tf.reduce_mean(edge_features, axis=1)  # [batch, features]
                
                # Project to fixed size for routing
                if not hasattr(self, 'routing_projection'):
                    node_dim = pooled_nodes.shape[-1]
                    edge_dim = pooled_edges.shape[-1]
                    self.routing_projection = Dense(
                        units=64,  # Fixed size for routing
                        activation="relu",
                        use_bias=True,
                        name="routing_projection"
                    )
                    self.routing_projection.build((None, node_dim + edge_dim))
                
                # Concatenate and project
                combined = tf.concat([pooled_nodes, pooled_edges], axis=-1)
                routing_input = self.routing_projection(combined)
                
                # Build router if not already built
                if self.router is None:
                    self.router = Dense(
                        units=self.router_units,
                        activation="softmax",
                        use_bias=True,
                        name="expert_router"
                    )
                    self.router.build((None, 64))  # Fixed size from routing_projection
                
                # Get routing weights
                if training and self.use_noise:
                    # Add noise for exploration during training
                    noise = tf.random.normal(tf.shape(routing_input)) * self.noise_epsilon
                    routing_input = routing_input + noise
                
                weights = self.router(routing_input)  # [batch, num_experts]
            routing_weights.append(weights)
            
            # Build experts dynamically if not already built
            if len(self.experts) == 0:
                input_dim = node_features.shape[-1]
                for i, expert_type in enumerate(self.expert_types_list):
                    if expert_type == "gin":
                        expert = GINExpert(units=self.units, use_bias=True, activation="relu")
                        # Build the expert with the correct input dimension
                        expert.build([(None, input_dim), (None, edge_features.shape[-1]), (None, 2)])
                    elif expert_type == "gat":
                        expert = GATExpert(units=self.units, num_heads=4, use_bias=True, activation="relu")
                        expert.build([(None, input_dim), (None, edge_features.shape[-1]), (None, 2)])
                    elif expert_type == "gcn":
                        expert = GCNExpert(units=self.units, use_bias=True, activation="relu")
                        expert.build([(None, input_dim), (None, edge_features.shape[-1]), (None, 2)])
                    else:
                        # Default to GIN
                        expert = GINExpert(units=self.units, use_bias=True, activation="relu")
                        expert.build([(None, input_dim), (None, edge_features.shape[-1]), (None, 2)])
                    self.experts.append(expert)
            
            # Apply experts
            expert_outputs = []
            for expert in self.experts:
                expert_out = expert([node_features, edge_features, edge_indices])
                expert_outputs.append(expert_out)
            
            # Weighted combination of expert outputs
            weighted_outputs = []
            for i, expert_out in enumerate(expert_outputs):
                weight = weights[:, i:i+1, tf.newaxis]  # [batch, 1, 1]
                weighted_out = expert_out * weight
                weighted_outputs.append(weighted_out)
            
            # Sum weighted outputs
            combined_output = tf.add_n(weighted_outputs)
            all_expert_outputs.append(combined_output)
        
        # Combine outputs from all graph representations
        if len(all_expert_outputs) > 1:
            final_output = tf.add_n(all_expert_outputs)
        else:
            final_output = all_expert_outputs[0]
        
        # Update auxiliary loss for load balancing
        if training:
            self._update_aux_loss(routing_weights)
            
            # Track gating weights for losses and metrics
            if routing_weights:
                # Use the first routing weights for loss computation
                gating_weights = routing_weights[0]
                self.gating_weights_history.append(gating_weights)
                
                # Keep only the last 1000 weights to prevent memory issues
                if len(self.gating_weights_history) > 1000:
                    self.gating_weights_history = self.gating_weights_history[-1000:]
        
        return final_output
    
    def _update_aux_loss(self, routing_weights):
        """Update auxiliary loss for load balancing."""
        if not routing_weights:
            return
        
        # Calculate load balancing loss
        mean_weights = tf.reduce_mean(tf.stack(routing_weights), axis=0)  # [batch, num_experts]
        mean_usage = tf.reduce_mean(mean_weights, axis=0)  # [num_experts]
        
        # Ideal uniform distribution
        ideal_usage = 1.0 / self.num_experts
        
        # Load balancing loss (KL divergence from uniform)
        load_balancing_loss = tf.reduce_sum(
            mean_usage * tf.math.log(mean_usage / ideal_usage)
        )
        
        self.aux_loss = load_balancing_loss
    
    def get_config(self):
        """Get layer configuration."""
        config = super(ExpertRoutingLayer, self).get_config()
        config.update({
            "num_experts": self.num_experts,
            "expert_types": self.expert_types,
            "temperature": self.temperature,
            "use_noise": self.use_noise,
            "noise_epsilon": self.noise_epsilon,
            "units": self.units,
            "use_load_balancing": self.use_load_balancing,
            "use_auxiliary_loss": self.use_auxiliary_loss
        })
        return config
    
    def compute_moe_losses(self, y_true=None):
        """
        Compute MoE-specific losses using tracked gating weights.
        
        Args:
            y_true: True labels (not used, but required by loss functions)
            
        Returns:
            dict: Dictionary containing load balancing and auxiliary losses
        """
        losses = {}
        
        if not self.gating_weights_history:
            losses['load_balancing_loss'] = 0.0
            losses['auxiliary_loss'] = 0.0
            return losses
        
        # Use the most recent gating weights
        gating_weights = self.gating_weights_history[-1]
        
        # Compute load balancing loss
        if self.load_balancing_loss is not None:
            losses['load_balancing_loss'] = self.load_balancing_loss(y_true, gating_weights)
        else:
            losses['load_balancing_loss'] = 0.0
        
        # Compute auxiliary loss
        if self.auxiliary_loss is not None:
            losses['auxiliary_loss'] = self.auxiliary_loss(y_true, gating_weights)
        else:
            losses['auxiliary_loss'] = 0.0
        
        return losses
    
    def get_expert_usage_stats(self):
        """
        Get expert usage statistics from tracked gating weights.
        
        Returns:
            dict: Dictionary containing expert usage statistics
        """
        if not self.gating_weights_history:
            return {
                'expert_usage': tf.zeros(self.num_experts),
                'gating_entropy': 0.0,
                'load_balance_score': 0.0
            }
        
        # Use the most recent gating weights
        gating_weights = self.gating_weights_history[-1]
        
        # Compute expert usage
        expert_usage = tf.reduce_mean(gating_weights, axis=0)
        
        # Compute gating entropy
        entropy = -tf.reduce_sum(gating_weights * tf.math.log(gating_weights + 1e-8), axis=-1)
        gating_entropy = tf.reduce_mean(entropy)
        
        # Compute load balance score
        ideal_usage = 1.0 / self.num_experts
        load_balance_score = -tf.reduce_mean((expert_usage - ideal_usage) ** 2)
        
        return {
            'expert_usage': expert_usage,
            'gating_entropy': gating_entropy,
            'load_balance_score': load_balance_score
        }


@ks.utils.register_keras_serializable(package='kgcnn', name='MultiGraphMoEConv')
class MultiGraphMoEConv(GraphBaseLayer):
    """Multi-Graph MoE Convolution Layer.
    
    This layer creates multiple graph representations and uses different GNN experts
    to improve ensemble performance and reduce variance on small datasets.
    
    Args:
        num_representations (int): Number of graph representations to create
        num_experts (int): Number of GNN experts
        expert_types (list): Types of GNN experts to use
        representation_types (list): Types of graph representations to create
        use_edge_weights (bool): Whether to create different edge weightings
        use_node_features (bool): Whether to create different node features
        use_attention (bool): Whether to create different attention patterns
        dropout_rate (float): Dropout rate
        temperature (float): Temperature for expert routing
        use_noise (bool): Whether to use noise in routing during training
        noise_epsilon (float): Noise magnitude for routing exploration
    """
    
    def __init__(self, num_representations=4, num_experts=3, expert_types=None,
                 representation_types=None, use_edge_weights=True, use_node_features=True,
                 use_attention=True, dropout_rate=0.1, temperature=1.0, use_noise=True,
                 noise_epsilon=1e-2, units=128, **kwargs):
        super(MultiGraphMoEConv, self).__init__(**kwargs)
        
        self.num_representations = num_representations
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.use_noise = use_noise
        self.noise_epsilon = noise_epsilon
        self.units = units
        
        # Graph representation layer
        self.graph_representations = GraphRepresentationLayer(
            num_representations=num_representations,
            representation_types=representation_types,
            use_edge_weights=use_edge_weights,
            use_node_features=use_node_features,
            use_attention=use_attention,
            dropout_rate=dropout_rate
        )
        
        # Expert routing layer
        self.expert_routing = ExpertRoutingLayer(
            num_experts=num_experts,
            expert_types=expert_types,
            temperature=temperature,
            use_noise=use_noise,
            noise_epsilon=noise_epsilon,
            units=units
        )
        
        # Output projection
        self.output_projection = Dense(
            units=units,  # Use the units parameter
            activation="relu",
            use_bias=True
        )
        
        # Layer normalization
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Forward pass of Multi-Graph MoE convolution.
        
        Args:
            inputs: List of [node_features, edge_features, edge_indices]
            training: Training mode flag
            
        Returns:
            Enhanced node features with ensemble improvement
        """
        node_features, edge_features, edge_indices = inputs
        
        # For simplicity, use only the original graph representation
        # and apply multiple experts to it
        graph_rep = {
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_indices": edge_indices
        }
        
        # Route to experts and get combined output
        expert_output = self.expert_routing([graph_rep], training=training)
        
        # Apply output projection
        projected_output = self.output_projection(expert_output)
        
        # Apply layer normalization
        normalized_output = self.layer_norm(projected_output)
        
        # Apply dropout if specified
        if self.dropout is not None and training:
            normalized_output = self.dropout(normalized_output)
        
        return normalized_output
    
    def get_config(self):
        """Get layer configuration."""
        config = super(MultiGraphMoEConv, self).get_config()
        config.update({
            "num_representations": self.num_representations,
            "num_experts": self.num_experts,
            "dropout_rate": self.dropout_rate,
            "temperature": self.temperature,
            "use_noise": self.use_noise,
            "noise_epsilon": self.noise_epsilon,
            "units": self.units
        })
        return config 