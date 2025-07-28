import tensorflow as tf
import numpy as np


class MultiGraphMoELoadBalancingLoss(tf.keras.losses.Loss):
    """
    Load Balancing Loss for MultiGraphMoE
    
    Ensures that all experts are used equally during training.
    This prevents expert collapse where only a few experts are used.
    
    Args:
        num_experts (int): Number of experts
        importance_weight (float): Weight for the importance loss
        load_weight (float): Weight for the load loss
    """
    
    def __init__(self, num_experts=3, importance_weight=0.01, load_weight=0.01, name="multigraph_moe_load_balancing", **kwargs):
        super(MultiGraphMoELoadBalancingLoss, self).__init__(name=name, **kwargs)
        self.num_experts = num_experts
        self.importance_weight = importance_weight
        self.load_weight = load_weight
    
    def call(self, y_true, gating_weights):
        """
        Compute load balancing loss
        
        Args:
            y_true: True labels (not used, but required by Keras)
            gating_weights: Expert gating weights of shape (batch_size, num_experts)
        """
        # Importance loss: encourage equal importance across experts
        importance = tf.reduce_mean(gating_weights, axis=0)  # (num_experts,)
        target_importance = tf.ones_like(importance) / self.num_experts
        importance_loss = tf.reduce_mean((importance - target_importance) ** 2)
        
        # Load loss: encourage equal load across experts
        load = tf.reduce_mean(gating_weights, axis=0)  # (num_experts,)
        target_load = tf.ones_like(load) / self.num_experts
        load_loss = tf.reduce_mean((load - target_load) ** 2)
        
        total_loss = self.importance_weight * importance_loss + self.load_weight * load_loss
        return total_loss
    
    def get_config(self):
        config = super(MultiGraphMoELoadBalancingLoss, self).get_config()
        config.update({
            "num_experts": self.num_experts,
            "importance_weight": self.importance_weight,
            "load_weight": self.load_weight
        })
        return config


class MultiGraphMoEAuxiliaryLoss(tf.keras.losses.Loss):
    """
    Auxiliary Loss for MultiGraphMoE
    
    Provides additional supervision to prevent expert collapse
    and encourage diverse expert usage.
    
    Args:
        num_experts (int): Number of experts
        diversity_weight (float): Weight for diversity loss
        entropy_weight (float): Weight for entropy regularization
    """
    
    def __init__(self, num_experts=3, diversity_weight=0.01, entropy_weight=0.01, name="multigraph_moe_auxiliary", **kwargs):
        super(MultiGraphMoEAuxiliaryLoss, self).__init__(name=name, **kwargs)
        self.num_experts = num_experts
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
    
    def call(self, y_true, gating_weights):
        """
        Compute auxiliary loss
        
        Args:
            y_true: True labels (not used, but required by Keras)
            gating_weights: Expert gating weights of shape (batch_size, num_experts)
        """
        # Diversity loss: encourage different experts for different samples
        batch_size = tf.shape(gating_weights)[0]
        
        # Compute pairwise diversity
        gating_expanded = tf.expand_dims(gating_weights, axis=1)  # (batch_size, 1, num_experts)
        gating_expanded_2 = tf.expand_dims(gating_weights, axis=0)  # (1, batch_size, num_experts)
        
        # Cosine similarity between expert usage patterns
        similarity = tf.reduce_sum(gating_expanded * gating_expanded_2, axis=-1)  # (batch_size, batch_size)
        similarity = similarity / (tf.norm(gating_expanded, axis=-1) * tf.norm(gating_expanded_2, axis=-1) + 1e-8)
        
        # Remove diagonal (self-similarity)
        mask = 1.0 - tf.eye(batch_size, dtype=tf.float32)
        diversity_loss = tf.reduce_mean(similarity * mask) / (tf.cast(batch_size, tf.float32) - 1.0)
        
        # Entropy regularization: encourage more uniform expert usage
        entropy = -tf.reduce_sum(gating_weights * tf.math.log(gating_weights + 1e-8), axis=-1)
        entropy_loss = -tf.reduce_mean(entropy)  # Negative because we want to maximize entropy
        
        total_loss = self.diversity_weight * diversity_loss + self.entropy_weight * entropy_loss
        return total_loss
    
    def get_config(self):
        config = super(MultiGraphMoEAuxiliaryLoss, self).get_config()
        config.update({
            "num_experts": self.num_experts,
            "diversity_weight": self.diversity_weight,
            "entropy_weight": self.entropy_weight
        })
        return config


class MultiGraphMoEExpertUsageMetric(tf.keras.metrics.Metric):
    """
    Metric to track expert usage across layers
    
    Args:
        layer_idx (int): Layer index
        num_experts (int): Number of experts
        name (str): Metric name
    """
    
    def __init__(self, layer_idx=0, num_experts=3, name="expert_usage", **kwargs):
        super(MultiGraphMoEExpertUsageMetric, self).__init__(name=f"{name}_layer_{layer_idx}", **kwargs)
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.expert_usage = self.add_weight(
            name=f"expert_usage_layer_{layer_idx}",
            shape=(num_experts,),
            initializer="zeros",
            dtype=tf.float32
        )
        self.count = self.add_weight(
            name=f"count_layer_{layer_idx}",
            initializer="zeros",
            dtype=tf.float32
        )
    
    def update_state(self, gating_weights):
        """
        Update expert usage statistics
        
        Args:
            gating_weights: Expert gating weights of shape (batch_size, num_experts)
        """
        # Compute mean usage across batch
        batch_usage = tf.reduce_mean(gating_weights, axis=0)  # (num_experts,)
        
        # Update running average
        self.expert_usage.assign_add(batch_usage)
        self.count.assign_add(1.0)
    
    def result(self):
        """Return current expert usage statistics"""
        return self.expert_usage / tf.maximum(self.count, 1.0)
    
    def reset_state(self):
        """Reset metric state"""
        self.expert_usage.assign(tf.zeros_like(self.expert_usage))
        self.count.assign(0.0)


class MultiGraphMoEGatingEntropyMetric(tf.keras.metrics.Metric):
    """
    Metric to track gating entropy across layers
    
    Args:
        layer_idx (int): Layer index
        name (str): Metric name
    """
    
    def __init__(self, layer_idx=0, name="gating_entropy", **kwargs):
        super(MultiGraphMoEGatingEntropyMetric, self).__init__(name=f"{name}_layer_{layer_idx}", **kwargs)
        self.layer_idx = layer_idx
        self.entropy_sum = self.add_weight(
            name=f"entropy_sum_layer_{layer_idx}",
            initializer="zeros",
            dtype=tf.float32
        )
        self.count = self.add_weight(
            name=f"count_layer_{layer_idx}",
            initializer="zeros",
            dtype=tf.float32
        )
    
    def update_state(self, gating_weights):
        """
        Update entropy statistics
        
        Args:
            gating_weights: Expert gating weights of shape (batch_size, num_experts)
        """
        # Compute entropy for each sample
        entropy = -tf.reduce_sum(gating_weights * tf.math.log(gating_weights + 1e-8), axis=-1)
        batch_entropy = tf.reduce_mean(entropy)
        
        # Update running average
        self.entropy_sum.assign_add(batch_entropy)
        self.count.assign_add(1.0)
    
    def result(self):
        """Return current average entropy"""
        return self.entropy_sum / tf.maximum(self.count, 1.0)
    
    def reset_state(self):
        """Reset metric state"""
        self.entropy_sum.assign(0.0)
        self.count.assign(0.0)


class MultiGraphMoELoadBalanceScoreMetric(tf.keras.metrics.Metric):
    """
    Metric to compute load balance score
    
    Higher score indicates more balanced expert usage
    """
    
    def __init__(self, num_experts=3, name="load_balance_score", **kwargs):
        super(MultiGraphMoELoadBalanceScoreMetric, self).__init__(name=name, **kwargs)
        self.num_experts = num_experts
        self.usage_sum = self.add_weight(
            name="usage_sum",
            shape=(num_experts,),
            initializer="zeros",
            dtype=tf.float32
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros",
            dtype=tf.float32
        )
    
    def update_state(self, gating_weights):
        """
        Update load balance statistics
        
        Args:
            gating_weights: Expert gating weights of shape (batch_size, num_experts)
        """
        # Compute mean usage across batch
        batch_usage = tf.reduce_mean(gating_weights, axis=0)  # (num_experts,)
        
        # Update running average
        self.usage_sum.assign_add(batch_usage)
        self.count.assign_add(1.0)
    
    def result(self):
        """Return load balance score"""
        mean_usage = self.usage_sum / tf.maximum(self.count, 1.0)
        
        # Ideal balanced usage
        ideal_usage = 1.0 / self.num_experts
        
        # Compute balance score as negative MSE from ideal (higher is better)
        balance_score = -tf.reduce_mean((mean_usage - ideal_usage) ** 2)
        
        return balance_score
    
    def reset_state(self):
        """Reset metric state"""
        self.usage_sum.assign(tf.zeros_like(self.usage_sum))
        self.count.assign(0.0)


def create_multigraph_moe_metrics(num_layers=3, num_experts=3):
    """
    Create a comprehensive set of MultiGraphMoE metrics
    
    Args:
        num_layers (int): Number of MultiGraphMoE layers
        num_experts (int): Number of experts per layer
        
    Returns:
        list: List of metrics
    """
    metrics = []
    
    # Add expert usage metrics for each layer
    for i in range(num_layers):
        metrics.append(MultiGraphMoEExpertUsageMetric(layer_idx=i, num_experts=num_experts))
        metrics.append(MultiGraphMoEGatingEntropyMetric(layer_idx=i))
    
    # Add overall load balance score
    metrics.append(MultiGraphMoELoadBalanceScoreMetric(num_experts=num_experts))
    
    return metrics


def compute_expert_balance_score(expert_usage):
    """
    Compute how balanced the expert usage is
    
    Args:
        expert_usage: Expert usage array of shape (num_batches, num_layers, num_experts)
        
    Returns:
        Balance score (higher is more balanced)
    """
    # Compute mean usage across batches for each layer
    mean_usage = np.mean(expert_usage, axis=0)  # (num_layers, num_experts)
    
    # Ideal balanced usage
    ideal_usage = 1.0 / expert_usage.shape[-1]
    
    # Compute balance score as negative MSE from ideal
    balance_scores = -np.mean((mean_usage - ideal_usage) ** 2, axis=1)
    
    return balance_scores.tolist() 