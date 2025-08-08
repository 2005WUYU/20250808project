import torch
from torch import Tensor
from typing import Dict, Callable, Union, Any


def critic_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """
    WGAN critic loss: maximize E[real] - E[fake]
    
    Args:
        real_scores: Discriminator outputs for real samples [B]
        fake_scores: Discriminator outputs for fake samples [B]
        
    Returns:
        Scalar tensor with critic loss (minimize this for D)
    """
    # Shape checks
    if real_scores.shape != fake_scores.shape:
        raise ValueError(f"Shape mismatch: real_scores {real_scores.shape} != fake_scores {fake_scores.shape}")
        
    # WGAN critic loss: E[fake] - E[real] (minimize this for D)
    loss_basic = torch.mean(fake_scores) - torch.mean(real_scores)
    return loss_basic


def generator_adversarial_loss(fake_scores: Tensor) -> Tensor:
    """
    WGAN generator loss: maximize E[D(G(z))]
    
    Args:
        fake_scores: Discriminator outputs for fake samples [B]
        
    Returns:
        Scalar tensor with generator adversarial loss
    """
    # G tries to maximize D(G) -> minimize -mean(D(G))
    return -torch.mean(fake_scores)


def estimate_wasserstein_distance(real_scores: Tensor, fake_scores: Tensor) -> float:
    """
    Estimate Wasserstein distance between real and generated distributions.
    
    Args:
        real_scores: Discriminator outputs for real samples [B]
        fake_scores: Discriminator outputs for fake samples [B]
        
    Returns:
        Float estimate of Wasserstein distance
    """
    # Return float for logger
    return float(torch.mean(real_scores).item() - torch.mean(fake_scores).item())


def compute_gradient_penalty(
    D: Callable,
    real_batch: Dict[str, Any],
    fake_batch: Dict[str, Any],
    device: torch.device,
    gp_lambda: float = 10.0
) -> Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        D: Discriminator network
        real_batch: Batch of real data
        fake_batch: Batch of generated data
        device: Device to compute on
        gp_lambda: Gradient penalty coefficient
        
    Returns:
        Gradient penalty loss term
    """
    # Precondition checks
    assert real_batch['X'].shape == fake_batch['X'].shape, f"Node-feature shape mismatch: {real_batch['X'].shape} vs {fake_batch['X'].shape}"
    B, N, F = real_batch['X'].shape

    # Sample alpha [B,1,1] uniform(0,1) on device
    alpha = torch.rand(B, 1, 1, device=device)
    
    # Interpolate between real and fake data
    interpolated_X = alpha * real_batch['X'] + (1 - alpha) * fake_batch['X']
    
    # Build interpolated batch replicating other fields from real_batch
    interpolated_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in real_batch.items()}
    interpolated_batch['X'] = interpolated_X
    
    # Set requires_grad for interpolated X
    interpolated_X.requires_grad_(True)

    # Forward pass through discriminator
    d_interpolates = D(interpolated_batch)  # expect shape [B] (scalar per sample)
    
    # If output is [B,1], reshape to [B]
    if d_interpolates.dim() > 1:
        d_interpolates = d_interpolates.view(-1)
        
    # Compute gradients with respect to inputs
    grad_outputs = torch.ones_like(d_interpolates, device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolated_X,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient norms
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=(1, 2)) + 1e-12)
    
    # Compute gradient penalty: (||grad|| - 1)^2
    gradient_penalty = gp_lambda * torch.mean((gradients_norm - 1.0) ** 2)
    
    return gradient_penalty