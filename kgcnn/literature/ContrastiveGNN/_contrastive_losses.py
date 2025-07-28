import tensorflow as tf
import numpy as np


class RegressionContrastiveGNNLoss(tf.keras.losses.Loss):
    """
    Regression-Aware Contrastive Loss for GNN architectures
    
    Implements contrastive learning that considers target similarity
    for regression tasks. Similar targets should have similar embeddings.
    
    Args:
        temperature (float): Temperature parameter for softmax
        target_similarity_threshold (float): Threshold for considering targets similar
        similarity_metric (str): How to compute target similarity ('euclidean', 'cosine', 'absolute')
        use_adaptive_threshold (bool): Whether to use adaptive thresholding
        margin (float): Margin for hard negative mining
    """
    
    def __init__(self, temperature=0.1, target_similarity_threshold=0.1, 
                 similarity_metric='euclidean', use_adaptive_threshold=True,
                 margin=1.0, name="regression_contrastive_gnn", **kwargs):
        super(RegressionContrastiveGNNLoss, self).__init__(name=name, **kwargs)
        self.temperature = temperature
        self.target_similarity_threshold = target_similarity_threshold
        self.similarity_metric = similarity_metric
        self.use_adaptive_threshold = use_adaptive_threshold
        self.margin = margin
    
    def compute_target_similarity(self, targets):
        """Compute similarity matrix between targets"""
        if self.similarity_metric == 'euclidean':
            # Convert to distance, then to similarity
            distances = tf.sqrt(tf.reduce_sum(tf.square(
                tf.expand_dims(targets, 1) - tf.expand_dims(targets, 0)
            ), axis=-1))
            # Convert distance to similarity (inverse relationship)
            similarity = 1.0 / (1.0 + distances)
        elif self.similarity_metric == 'cosine':
            # Normalize targets for cosine similarity
            normalized_targets = tf.nn.l2_normalize(targets, axis=-1)
            similarity = tf.matmul(normalized_targets, normalized_targets, transpose_b=True)
        elif self.similarity_metric == 'absolute':
            # Absolute difference converted to similarity
            abs_diff = tf.abs(tf.expand_dims(targets, 1) - tf.expand_dims(targets, 0))
            similarity = 1.0 / (1.0 + abs_diff)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity
    
    def call(self, y_true, embeddings):
        """
        Compute regression-aware contrastive loss
        
        Args:
            y_true: True regression targets of shape (batch_size, target_dim)
            embeddings: Graph embeddings of shape (batch_size, embedding_dim)
        """
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        # Compute embedding similarity
        embedding_similarity = tf.matmul(embeddings, embeddings, transpose_b=True)
        
        # Compute target similarity
        target_similarity = self.compute_target_similarity(y_true)
        
        # Determine positive/negative pairs based on target similarity
        if self.use_adaptive_threshold:
            # Use adaptive threshold based on target distribution
            target_std = tf.math.reduce_std(y_true)
            threshold = tf.maximum(self.target_similarity_threshold, target_std * 0.1)
        else:
            threshold = self.target_similarity_threshold
        
        # Create positive mask (similar targets)
        positive_mask = target_similarity > threshold
        
        # Create negative mask (dissimilar targets)
        negative_mask = target_similarity <= threshold
        
        # Remove self-comparisons from positive mask
        batch_size = tf.shape(embeddings)[0]
        self_mask = tf.eye(batch_size, dtype=tf.bool)
        positive_mask = tf.logical_and(positive_mask, ~self_mask)
        
        # Extract positive and negative similarities
        positive_similarities = tf.boolean_mask(embedding_similarity, positive_mask)
        negative_similarities = tf.boolean_mask(embedding_similarity, negative_mask)
        
        # Weight by target similarity
        positive_weights = tf.boolean_mask(target_similarity, positive_mask)
        negative_weights = 1.0 - tf.boolean_mask(target_similarity, negative_mask)
        
        # Compute weighted loss
        if tf.shape(positive_similarities)[0] > 0:
            # Positive loss: encourage similar embeddings for similar targets
            positive_loss = -tf.reduce_mean(positive_weights * positive_similarities)
        else:
            positive_loss = 0.0
        
        if tf.shape(negative_similarities)[0] > 0:
            # Negative loss: discourage similar embeddings for dissimilar targets
            negative_loss = tf.reduce_mean(negative_weights * tf.nn.relu(
                negative_similarities + self.margin
            ))
        else:
            negative_loss = 0.0
        
        total_loss = positive_loss + negative_loss
        
        return total_loss
    
    def get_config(self):
        config = super(RegressionContrastiveGNNLoss, self).get_config()
        config.update({
            "temperature": self.temperature,
            "target_similarity_threshold": self.target_similarity_threshold,
            "similarity_metric": self.similarity_metric,
            "use_adaptive_threshold": self.use_adaptive_threshold,
            "margin": self.margin
        })
        return config


class RegressionContrastiveGNNTripletLoss(tf.keras.losses.Loss):
    """
    Regression-Aware Triplet Loss for Contrastive GNN
    
    Implements triplet loss that considers target similarity for regression.
    Creates triplets based on target distances.
    
    Args:
        margin (float): Margin for triplet loss
        target_distance_threshold (float): Threshold for anchor-positive pairs
        use_hard_negative (bool): Whether to use hard negative mining
        similarity_metric (str): How to compute target similarity
    """
    
    def __init__(self, margin=1.0, target_distance_threshold=0.1, 
                 use_hard_negative=True, similarity_metric='euclidean',
                 name="regression_contrastive_gnn_triplet", **kwargs):
        super(RegressionContrastiveGNNTripletLoss, self).__init__(name=name, **kwargs)
        self.margin = margin
        self.target_distance_threshold = target_distance_threshold
        self.use_hard_negative = use_hard_negative
        self.similarity_metric = similarity_metric
    
    def compute_target_distances(self, targets):
        """Compute distance matrix between targets"""
        if self.similarity_metric == 'euclidean':
            distances = tf.sqrt(tf.reduce_sum(tf.square(
                tf.expand_dims(targets, 1) - tf.expand_dims(targets, 0)
            ), axis=-1))
        elif self.similarity_metric == 'absolute':
            distances = tf.reduce_sum(tf.abs(
                tf.expand_dims(targets, 1) - tf.expand_dims(targets, 0)
            ), axis=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return distances
    
    def call(self, y_true, embeddings):
        """
        Compute regression-aware triplet loss
        
        Args:
            y_true: True regression targets of shape (batch_size, target_dim)
            embeddings: Graph embeddings of shape (batch_size, embedding_dim)
        """
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        # Compute target distances
        target_distances = self.compute_target_distances(y_true)
        
        # Compute embedding distances
        embedding_distances = tf.sqrt(2.0 - 2.0 * tf.matmul(embeddings, embeddings, transpose_b=True))
        
        batch_size = tf.shape(embeddings)[0]
        total_loss = 0.0
        valid_triplets = 0
        
        for i in range(batch_size):
            # Find positive samples (similar targets)
            positive_mask = target_distances[i] <= self.target_distance_threshold
            positive_mask = tf.logical_and(positive_mask, tf.range(batch_size) != i)
            positive_indices = tf.where(positive_mask)[:, 0]
            
            if tf.shape(positive_indices)[0] == 0:
                continue
            
            # Find negative samples (dissimilar targets)
            negative_mask = target_distances[i] > self.target_distance_threshold
            negative_indices = tf.where(negative_mask)[:, 0]
            
            if tf.shape(negative_indices)[0] == 0:
                continue
            
            # For each positive sample, find hardest negative
            for pos_idx in positive_indices:
                anchor_embedding = embeddings[i]
                positive_embedding = embeddings[pos_idx]
                
                # Compute distances
                anchor_positive_dist = tf.norm(anchor_embedding - positive_embedding)
                
                # Find hardest negative
                negative_embeddings = tf.gather(embeddings, negative_indices)
                anchor_negative_dists = tf.norm(anchor_embedding - negative_embeddings, axis=1)
                
                if self.use_hard_negative:
                    hardest_negative_dist = tf.reduce_min(anchor_negative_dists)
                else:
                    hardest_negative_dist = tf.reduce_mean(anchor_negative_dists)
                
                # Compute triplet loss
                triplet_loss = tf.nn.relu(anchor_positive_dist - hardest_negative_dist + self.margin)
                total_loss += triplet_loss
                valid_triplets += 1
        
        if valid_triplets > 0:
            return total_loss / tf.cast(valid_triplets, tf.float32)
        else:
            return 0.0
    
    def get_config(self):
        config = super(RegressionContrastiveGNNTripletLoss, self).get_config()
        config.update({
            "margin": self.margin,
            "target_distance_threshold": self.target_distance_threshold,
            "use_hard_negative": self.use_hard_negative,
            "similarity_metric": self.similarity_metric
        })
        return config


class ContrastiveGNNLoss(tf.keras.losses.Loss):
    """
    Contrastive Loss for GNN architectures
    
    Implements InfoNCE (Info Noise Contrastive Estimation) loss
    for contrastive learning with graph neural networks.
    
    Args:
        temperature (float): Temperature parameter for softmax
        negative_samples (int): Number of negative samples per positive
        use_hard_negatives (bool): Whether to use hard negative mining
        margin (float): Margin for triplet loss variant
    """
    
    def __init__(self, temperature=0.1, negative_samples=16, 
                 use_hard_negatives=True, margin=1.0, name="contrastive_gnn", **kwargs):
        super(ContrastiveGNNLoss, self).__init__(name=name, **kwargs)
        self.temperature = temperature
        self.negative_samples = negative_samples
        self.use_hard_negatives = use_hard_negatives
        self.margin = margin
    
    def call(self, y_true, embeddings):
        """
        Compute contrastive loss
        
        Args:
            y_true: True labels (not used, but required by Keras)
            embeddings: Graph embeddings of shape (batch_size, embedding_dim)
        """
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)
        
        # Create positive pairs (diagonal)
        positive_mask = tf.eye(tf.shape(embeddings)[0], dtype=tf.bool)
        
        # Create negative pairs (off-diagonal)
        negative_mask = ~positive_mask
        
        # Extract positive and negative similarities
        positive_similarities = tf.boolean_mask(similarity_matrix, positive_mask)
        negative_similarities = tf.boolean_mask(similarity_matrix, negative_mask)
        
        # Reshape negative similarities
        batch_size = tf.shape(embeddings)[0]
        negative_similarities = tf.reshape(negative_similarities, [batch_size, -1])
        
        # Select top-k hardest negatives if enabled
        if self.use_hard_negatives:
            k = min(self.negative_samples, tf.shape(negative_similarities)[1])
            negative_similarities = tf.nn.top_k(negative_similarities, k=k).values
        
        # Compute InfoNCE loss
        logits = tf.concat([
            tf.expand_dims(positive_similarities, axis=1),
            negative_similarities
        ], axis=1) / self.temperature
        
        # Labels are zeros (first column is positive)
        labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int32)
        
        # Cross-entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super(ContrastiveGNNLoss, self).get_config()
        config.update({
            "temperature": self.temperature,
            "negative_samples": self.negative_samples,
            "use_hard_negatives": self.use_hard_negatives,
            "margin": self.margin
        })
        return config


class ContrastiveGNNTripletLoss(tf.keras.losses.Loss):
    """
    Triplet Loss for Contrastive GNN
    
    Implements triplet loss with hard negative mining
    for contrastive learning with graph neural networks.
    
    Args:
        margin (float): Margin for triplet loss
        use_semi_hard (bool): Whether to use semi-hard negative mining
        use_hard_negative (bool): Whether to use hard negative mining
    """
    
    def __init__(self, margin=1.0, use_semi_hard=True, use_hard_negative=True, 
                 name="contrastive_gnn_triplet", **kwargs):
        super(ContrastiveGNNTripletLoss, self).__init__(name=name, **kwargs)
        self.margin = margin
        self.use_semi_hard = use_semi_hard
        self.use_hard_negative = use_hard_negative
    
    def call(self, y_true, embeddings):
        """
        Compute triplet loss
        
        Args:
            y_true: True labels (not used, but required by Keras)
            embeddings: Graph embeddings of shape (batch_size, embedding_dim)
        """
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        # Compute pairwise distances
        distances = tf.reduce_sum(tf.square(embeddings[:, None, :] - embeddings[None, :, :]), axis=2)
        
        # Create positive pairs (diagonal)
        positive_mask = tf.eye(tf.shape(embeddings)[0], dtype=tf.bool)
        
        # Create negative pairs (off-diagonal)
        negative_mask = ~positive_mask
        
        # Extract positive and negative distances
        positive_distances = tf.boolean_mask(distances, positive_mask)
        negative_distances = tf.boolean_mask(distances, negative_mask)
        
        # Reshape negative distances
        batch_size = tf.shape(embeddings)[0]
        negative_distances = tf.reshape(negative_distances, [batch_size, -1])
        
        # Select hardest negatives
        if self.use_hard_negative:
            hardest_negatives = tf.reduce_min(negative_distances, axis=1)
        else:
            # Random negative selection
            indices = tf.random.uniform([batch_size], 0, tf.shape(negative_distances)[1], dtype=tf.int32)
            hardest_negatives = tf.gather(negative_distances, indices, batch_dims=1)
        
        # Compute triplet loss
        triplet_loss = tf.maximum(0.0, positive_distances - hardest_negatives + self.margin)
        
        # Apply semi-hard mining if enabled
        if self.use_semi_hard:
            # Only keep triplets where negative is harder than positive but within margin
            semi_hard_mask = (hardest_negatives > positive_distances) & (hardest_negatives < positive_distances + self.margin)
            triplet_loss = tf.where(semi_hard_mask, triplet_loss, tf.zeros_like(triplet_loss))
        
        return tf.reduce_mean(triplet_loss)
    
    def get_config(self):
        config = super(ContrastiveGNNTripletLoss, self).get_config()
        config.update({
            "margin": self.margin,
            "use_semi_hard": self.use_semi_hard,
            "use_hard_negative": self.use_hard_negative
        })
        return config


class ContrastiveGNNDiversityLoss(tf.keras.losses.Loss):
    """
    Diversity Loss for Contrastive GNN
    
    Encourages diverse representations across different views
    and prevents representation collapse.
    
    Args:
        diversity_weight (float): Weight for diversity loss
        entropy_weight (float): Weight for entropy regularization
        use_cosine_diversity (bool): Whether to use cosine-based diversity
    """
    
    def __init__(self, diversity_weight=0.01, entropy_weight=0.01, 
                 use_cosine_diversity=True, name="contrastive_gnn_diversity", **kwargs):
        super(ContrastiveGNNDiversityLoss, self).__init__(name=name, **kwargs)
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.use_cosine_diversity = use_cosine_diversity
    
    def call(self, y_true, embeddings):
        """
        Compute diversity loss
        
        Args:
            y_true: True labels (not used, but required by Keras)
            embeddings: Graph embeddings of shape (batch_size, embedding_dim)
        """
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        # Compute pairwise similarities
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)
        
        # Remove diagonal (self-similarity)
        mask = 1.0 - tf.eye(tf.shape(embeddings)[0], dtype=tf.float32)
        similarities = similarity_matrix * mask
        
        if self.use_cosine_diversity:
            # Cosine-based diversity: minimize average cosine similarity
            diversity_loss = tf.reduce_mean(similarities)
        else:
            # L2-based diversity: maximize average L2 distance
            distances = 2.0 - 2.0 * similarities  # Convert similarity to distance
            diversity_loss = -tf.reduce_mean(distances)
        
        # Entropy regularization: encourage more uniform distribution
        # Compute entropy of embedding distribution
        embedding_mean = tf.reduce_mean(embeddings, axis=0)
        embedding_std = tf.math.reduce_std(embeddings, axis=0)
        
        # Normal distribution entropy
        entropy = 0.5 * tf.reduce_sum(tf.math.log(2.0 * np.pi * np.e * tf.square(embedding_std + 1e-8)))
        entropy_loss = -entropy  # Negative because we want to maximize entropy
        
        total_loss = self.diversity_weight * diversity_loss + self.entropy_weight * entropy_loss
        return total_loss
    
    def get_config(self):
        config = super(ContrastiveGNNDiversityLoss, self).get_config()
        config.update({
            "diversity_weight": self.diversity_weight,
            "entropy_weight": self.entropy_weight,
            "use_cosine_diversity": self.use_cosine_diversity
        })
        return config


class ContrastiveGNNAlignmentLoss(tf.keras.losses.Loss):
    """
    Alignment Loss for Contrastive GNN
    
    Ensures that different views of the same graph are aligned
    while different graphs are separated.
    
    Args:
        alignment_weight (float): Weight for alignment loss
        separation_weight (float): Weight for separation loss
        temperature (float): Temperature parameter
    """
    
    def __init__(self, alignment_weight=1.0, separation_weight=1.0, 
                 temperature=0.1, name="contrastive_gnn_alignment", **kwargs):
        super(ContrastiveGNNAlignmentLoss, self).__init__(name=name, **kwargs)
        self.alignment_weight = alignment_weight
        self.separation_weight = separation_weight
        self.temperature = temperature
    
    def call(self, y_true, view_embeddings):
        """
        Compute alignment loss
        
        Args:
            y_true: True labels (not used, but required by Keras)
            view_embeddings: List of embeddings from different views
        """
        if len(view_embeddings) < 2:
            return 0.0
        
        # Normalize embeddings
        normalized_embeddings = [tf.nn.l2_normalize(emb, axis=1) for emb in view_embeddings]
        
        # Compute alignment loss between views
        alignment_loss = 0.0
        for i in range(len(normalized_embeddings)):
            for j in range(i + 1, len(normalized_embeddings)):
                # Compute similarity between views
                similarity = tf.reduce_sum(normalized_embeddings[i] * normalized_embeddings[j], axis=1)
                # Alignment loss: maximize similarity between views of same graph
                alignment_loss += tf.reduce_mean(1.0 - similarity)
        
        # Compute separation loss between different graphs
        separation_loss = 0.0
        for i, emb1 in enumerate(normalized_embeddings):
            for j, emb2 in enumerate(normalized_embeddings):
                if i != j:
                    # Compute similarity between different graphs
                    similarity_matrix = tf.matmul(emb1, emb2, transpose_b=True)
                    # Separation loss: minimize similarity between different graphs
                    separation_loss += tf.reduce_mean(tf.nn.relu(similarity_matrix - 0.5))
        
        total_loss = self.alignment_weight * alignment_loss + self.separation_weight * separation_loss
        return total_loss
    
    def get_config(self):
        config = super(ContrastiveGNNAlignmentLoss, self).get_config()
        config.update({
            "alignment_weight": self.alignment_weight,
            "separation_weight": self.separation_weight,
            "temperature": self.temperature
        })
        return config


class ContrastiveGNNMetric(tf.keras.metrics.Metric):
    """
    Metric to track contrastive learning performance
    
    Args:
        name (str): Metric name
        metric_type (str): Type of metric ('similarity', 'diversity', 'alignment')
    """
    
    def __init__(self, name="contrastive_metric", metric_type="similarity", **kwargs):
        super(ContrastiveGNNMetric, self).__init__(name=name, **kwargs)
        self.metric_type = metric_type
        self.similarity_sum = self.add_weight(name="similarity_sum", initializer="zeros", dtype=tf.float32)
        self.diversity_sum = self.add_weight(name="diversity_sum", initializer="zeros", dtype=tf.float32)
        self.alignment_sum = self.add_weight(name="alignment_sum", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)
    
    def update_state(self, embeddings, view_embeddings=None):
        """
        Update metric statistics
        
        Args:
            embeddings: Graph embeddings
            view_embeddings: List of embeddings from different views (optional)
        """
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        # Compute similarity metric
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)
        mask = 1.0 - tf.eye(tf.shape(embeddings)[0], dtype=tf.float32)
        similarities = tf.boolean_mask(similarity_matrix, mask)
        avg_similarity = tf.reduce_mean(similarities)
        
        # Compute diversity metric
        diversity = -tf.reduce_mean(similarities)  # Lower similarity = higher diversity
        
        # Compute alignment metric (if view embeddings provided)
        alignment = 0.0
        if view_embeddings is not None and len(view_embeddings) >= 2:
            normalized_views = [tf.nn.l2_normalize(emb, axis=1) for emb in view_embeddings]
            view_similarities = []
            for i in range(len(normalized_views)):
                for j in range(i + 1, len(normalized_views)):
                    similarity = tf.reduce_sum(normalized_views[i] * normalized_views[j], axis=1)
                    view_similarities.append(tf.reduce_mean(similarity))
            alignment = tf.reduce_mean(view_similarities)
        
        # Update running averages
        self.similarity_sum.assign_add(avg_similarity)
        self.diversity_sum.assign_add(diversity)
        self.alignment_sum.assign_add(alignment)
        self.count.assign_add(1.0)
    
    def result(self):
        """Return current metric value"""
        if self.metric_type == "similarity":
            return self.similarity_sum / tf.maximum(self.count, 1.0)
        elif self.metric_type == "diversity":
            return self.diversity_sum / tf.maximum(self.count, 1.0)
        elif self.metric_type == "alignment":
            return self.alignment_sum / tf.maximum(self.count, 1.0)
        else:
            return 0.0
    
    def reset_state(self):
        """Reset metric state"""
        self.similarity_sum.assign(0.0)
        self.diversity_sum.assign(0.0)
        self.alignment_sum.assign(0.0)
        self.count.assign(0.0)


def create_contrastive_gnn_metrics():
    """
    Create a comprehensive set of contrastive GNN metrics
    
    Returns:
        list: List of metrics
    """
    metrics = [
        ContrastiveGNNMetric(name="contrastive_similarity", metric_type="similarity"),
        ContrastiveGNNMetric(name="contrastive_diversity", metric_type="diversity"),
        ContrastiveGNNMetric(name="contrastive_alignment", metric_type="alignment")
    ]
    return metrics


def compute_contrastive_quality_score(embeddings, view_embeddings=None):
    """
    Compute overall contrastive learning quality score
    
    Args:
        embeddings: Graph embeddings
        view_embeddings: List of embeddings from different views (optional)
        
    Returns:
        dict: Dictionary containing quality scores
    """
    # Normalize embeddings
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    
    # Compute similarity matrix
    similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)
    mask = 1.0 - tf.eye(tf.shape(embeddings)[0], dtype=tf.float32)
    similarities = tf.boolean_mask(similarity_matrix, mask)
    
    # Quality metrics
    avg_similarity = tf.reduce_mean(similarities)
    diversity_score = -avg_similarity  # Higher is better
    separation_score = tf.reduce_mean(tf.nn.relu(0.5 - similarities))  # Higher is better
    
    # Alignment score (if view embeddings provided)
    alignment_score = 0.0
    if view_embeddings is not None and len(view_embeddings) >= 2:
        normalized_views = [tf.nn.l2_normalize(emb, axis=1) for emb in view_embeddings]
        view_similarities = []
        for i in range(len(normalized_views)):
            for j in range(i + 1, len(normalized_views)):
                similarity = tf.reduce_sum(normalized_views[i] * normalized_views[j], axis=1)
                view_similarities.append(tf.reduce_mean(similarity))
        alignment_score = tf.reduce_mean(view_similarities)
    
    # Overall quality score
    quality_score = (diversity_score + separation_score + alignment_score) / 3.0
    
    return {
        'avg_similarity': float(avg_similarity),
        'diversity_score': float(diversity_score),
        'separation_score': float(separation_score),
        'alignment_score': float(alignment_score),
        'quality_score': float(quality_score)
    } 