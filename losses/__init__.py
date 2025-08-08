from typing import Dict, Any, Callable, Union, Optional

import torch
from torch import Tensor

from .adversarial import (
    critic_loss, 
    generator_adversarial_loss, 
    estimate_wasserstein_distance, 
    compute_gradient_penalty
)
from .reconstruction import reconstruction_loss
from .regularizers import (
    displacement_norm_loss, 
    smoothness_loss, 
    edge_length_consistency_loss
)
from .spectral import laplacian_spectrum_loss
from .stats_loss import (
    distribution_1d_wasserstein,
    kl_divergence_histograms
)


class LossFactory:
    """
    Factory class for managing multiple loss functions with weights.
    Simplifies loss calculation in trainers.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize loss factory with configuration.
        
        Args:
            config: Dictionary with loss types and weights
                   Format: {'loss_name': {'weight': float, 'params': {...}}}
        """
        self.config = config or {}
        self.loss_fns = {}
        self._setup_loss_functions()
        
    def _setup_loss_functions(self):
        """Configure loss functions based on config."""
        # Register all available loss functions
        available_losses = {
            # Adversarial losses
            'critic': critic_loss,
            'generator_adv': generator_adversarial_loss,
            'gradient_penalty': compute_gradient_penalty,
            
            # Reconstruction losses
            'reconstruction': reconstruction_loss,
            
            # Regularizers
            'displacement_norm': displacement_norm_loss,
            'smoothness': smoothness_loss,
            'edge_length': edge_length_consistency_loss,
            
            # Spectral losses
            'spectral': laplacian_spectrum_loss,
            
            # Distribution losses
            'wasserstein_1d': distribution_1d_wasserstein,
            'kl_divergence': kl_divergence_histograms,
        }
        
        # Initialize requested losses with their configs
        for loss_name, loss_config in self.config.items():
            if loss_name in available_losses:
                weight = loss_config.get('weight', 1.0)
                params = loss_config.get('params', {})
                
                # Store function and its parameters
                self.loss_fns[loss_name] = {
                    'fn': available_losses[loss_name],
                    'weight': weight,
                    'params': params
                }
    
    def compute_loss(self, loss_name: str, *args, **kwargs) -> torch.Tensor:
        """
        Compute a specific loss with its weight applied.
        
        Args:
            loss_name: Name of the loss function to compute
            *args, **kwargs: Arguments to pass to the loss function
            
        Returns:
            Weighted loss tensor
        """
        if loss_name not in self.loss_fns:
            raise ValueError(f"Loss {loss_name} not configured")
            
        loss_config = self.loss_fns[loss_name]
        fn = loss_config['fn']
        weight = loss_config['weight']
        
        # Override default params with provided kwargs
        params = {**loss_config['params'], **kwargs}
        
        # Compute loss
        loss = fn(*args, **params)
        
        # Ensure loss is finite
        if not torch.isfinite(loss):
            raise ValueError(f"Loss {loss_name} returned non-finite value: {loss}")
            
        return weight * loss
    
    def compute_total_loss(self, losses_dict: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute multiple losses and their weighted sum.
        
        Args:
            losses_dict: Dictionary mapping loss names to {args, kwargs} dicts
            
        Returns:
            Dictionary with individual losses and total loss
        """
        result = {}
        total_loss = 0.0
        
        for loss_name, loss_inputs in losses_dict.items():
            args = loss_inputs.get('args', [])
            kwargs = loss_inputs.get('kwargs', {})
            
            loss = self.compute_loss(loss_name, *args, **kwargs)
            result[loss_name] = loss
            total_loss += loss
            
        result['total'] = total_loss
        return result


__all__ = [
    'LossFactory',
    'critic_loss',
    'generator_adversarial_loss',
    'estimate_wasserstein_distance',
    'compute_gradient_penalty',
    'reconstruction_loss',
    'displacement_norm_loss',
    'smoothness_loss',
    'edge_length_consistency_loss',
    'laplacian_spectrum_loss',
    'distribution_1d_wasserstein',
    'kl_divergence_histograms'
]