#!/usr/bin/env python3
"""
Adam-atan2 Optimizer for TensorFlow/Keras
Based on the PyTorch implementation from lucidrains/adam-atan2-pytorch
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K


class AdamAtan2(Optimizer):
    """
    Adam-atan2 optimizer for TensorFlow/Keras.
    
    This optimizer uses atan2 for adaptive learning rate adjustment,
    which can provide better convergence in some cases.
    
    Reference: https://github.com/lucidrains/adam-atan2-pytorch
    """
    
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 name='AdamAtan2',
                 **kwargs):
        """
        Initialize Adam-atan2 optimizer.
        
        Args:
            learning_rate: Initial learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            amsgrad: Whether to apply AMSGrad variant
            name: Optimizer name
            **kwargs: Additional arguments
        """
        super(AdamAtan2, self).__init__(name=name, **kwargs)
        
        # Set learning rate using the proper method
        self._learning_rate = learning_rate
        
        # Store other hyperparameters directly
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon or K.epsilon()
        self.amsgrad = amsgrad
    
    @property
    def learning_rate(self):
        """Get the learning rate."""
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        """Set the learning rate."""
        self._learning_rate = value
        
    def _create_slots(self, var_list):
        """Create optimizer slots for variables."""
        for var in var_list:
            self.add_slot(var, 'm')  # First moment
            self.add_slot(var, 'v')  # Second moment
            if self.amsgrad:
                self.add_slot(var, 'vhat')  # AMSGrad variant
    
    def _resource_apply_dense(self, grad, var):
        """Apply gradients to dense variables."""
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self.learning_rate, var_dtype)
        beta_1_t = tf.cast(self.beta_1, var_dtype)
        beta_2_t = tf.cast(self.beta_2, var_dtype)
        
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        
        # Update biased first moment estimate
        m_t = beta_1_t * m + (1 - beta_1_t) * grad
        m_t = tf.cast(m_t, var_dtype)
        
        # Update biased second moment estimate
        v_t = beta_2_t * v + (1 - beta_2_t) * tf.square(grad)
        v_t = tf.cast(v_t, var_dtype)
        
        # Compute bias-corrected first moment estimate
        m_hat = m_t / (1 - beta_1_t)
        
        # Compute bias-corrected second moment estimate
        v_hat = v_t / (1 - beta_2_t)
        
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = tf.maximum(vhat, v_hat)
            v_hat = vhat_t
        
        # Apply atan2-based adaptive learning rate
        # The key insight: use atan2(m_hat, sqrt(v_hat)) for adaptive scaling
        adaptive_factor = tf.atan2(m_hat, tf.sqrt(v_hat + self.epsilon))
        
        # Scale the gradient by the adaptive factor
        scaled_grad = adaptive_factor * grad
        
        # Update variable
        var_update = var - lr_t * scaled_grad
        
        # Update slots
        m.assign(m_t)
        v.assign(v_t)
        if self.amsgrad:
            vhat.assign(v_hat)
        
        return var.assign(var_update)
    
    def _resource_apply_sparse(self, grad, var, indices):
        """Apply gradients to sparse variables."""
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self.learning_rate, var_dtype)
        beta_1_t = tf.cast(self.beta_1, var_dtype)
        beta_2_t = tf.cast(self.beta_2, var_dtype)
        
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        
        # Sparse update for first moment
        m_scaled_grad = grad * (1 - beta_1_t)
        m_t = tf.compat.v1.scatter_update(m, indices, m_scaled_grad, use_locking=self._use_locking)
        m_t = beta_1_t * m + m_scaled_grad
        
        # Sparse update for second moment
        v_scaled_grad = tf.square(grad) * (1 - beta_2_t)
        v_t = tf.compat.v1.scatter_update(v, indices, v_scaled_grad, use_locking=self._use_locking)
        v_t = beta_2_t * v + v_scaled_grad
        
        # Compute bias-corrected estimates
        m_hat = m_t / (1 - beta_1_t)
        v_hat = v_t / (1 - beta_2_t)
        
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = tf.maximum(vhat, v_hat)
            v_hat = vhat_t
        
        # Apply atan2-based adaptive learning rate
        adaptive_factor = tf.atan2(m_hat, tf.sqrt(v_hat + self.epsilon))
        scaled_grad = adaptive_factor * grad
        
        # Update variable
        var_update = var - lr_t * scaled_grad
        
        # Update slots
        m.assign(m_t)
        v.assign(v_t)
        if self.amsgrad:
            vhat.assign(v_hat)
        
        return var.assign(var_update)
    
    def get_config(self):
        """Get optimizer configuration."""
        config = super(AdamAtan2, self).get_config()
        config.update({
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config


# Alias for backward compatibility
AdamAtan2Optimizer = AdamAtan2
