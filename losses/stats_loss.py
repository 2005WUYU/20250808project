import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Optional, Union


def distribution_1d_wasserstein(
    p_samples: Tensor,
    q_samples: Tensor
) -> Tensor:
    """
    Compute 1D Wasserstein distance between two empirical distributions.
    
    For 1D distributions, this is equivalent to computing the area between the CDFs.
    
    Args:
        p_samples: Samples from first distribution [n_p]
        q_samples: Samples from second distribution [n_q]
        
    Returns:
        Scalar tensor with Wasserstein distance
    """
    # Sort both arrays in ascending order
    p_sorted, _ = torch.sort(p_samples)
    q_sorted, _ = torch.sort(q_samples)
    
    # Handle case of different sample counts
    n_p = p_sorted.size(0)
    n_q = q_sorted.size(0)
    
    if n_p == n_q:
        # Simple case: same number of samples
        return torch.mean(torch.abs(p_sorted - q_sorted))
    else:
        # Different number of samples: interpolate larger to match smaller
        if n_p > n_q:
            # Interpolate p to match q size
            indices = torch.linspace(0, n_p - 1, n_q, device=p_sorted.device)
            p_interp = torch.zeros_like(q_sorted)
            for i, idx in enumerate(indices):
                # Linear interpolation
                idx_low = int(idx)
                idx_high = min(idx_low + 1, n_p - 1)
                weight_high = idx - idx_low
                p_interp[i] = (1 - weight_high) * p_sorted[idx_low] + weight_high * p_sorted[idx_high]
            return torch.mean(torch.abs(p_interp - q_sorted))
        else:
            # Interpolate q to match p size
            indices = torch.linspace(0, n_q - 1, n_p, device=q_sorted.device)
            q_interp = torch.zeros_like(p_sorted)
            for i, idx in enumerate(indices):
                # Linear interpolation
                idx_low = int(idx)
                idx_high = min(idx_low + 1, n_q - 1)
                weight_high = idx - idx_low
                q_interp[i] = (1 - weight_high) * q_sorted[idx_low] + weight_high * q_sorted[idx_high]
            return torch.mean(torch.abs(p_sorted - q_interp))


def kl_divergence_histograms(
    p_samples: Tensor,
    q_samples: Tensor,
    bins: int = 50,
    eps: float = 1e-8,
    density: bool = True
) -> Tensor:
    """
    Compute KL divergence between two empirical distributions using histograms.
    
    Args:
        p_samples: Samples from first distribution
        q_samples: Samples from second distribution
        bins: Number of histogram bins
        eps: Small epsilon for numerical stability
        density: Whether to normalize the histograms
        
    Returns:
        Scalar tensor with KL divergence
    """
    device = p_samples.device
    
    # Determine bin edges based on combined range
    min_val = min(p_samples.min().item(), q_samples.min().item())
    max_val = max(p_samples.max().item(), q_samples.max().item())
    
    # Add small margin to avoid edge effects
    margin = (max_val - min_val) * 0.01
    edges = torch.linspace(min_val - margin, max_val + margin, bins + 1, device=device)
    
    # Compute histograms
    p_hist = torch.histc(p_samples, bins=bins, min=edges[0], max=edges[-1])
    q_hist = torch.histc(q_samples, bins=bins, min=edges[0], max=edges[-1])
    
    # Normalize if density is True
    if density:
        p_hist = p_hist / (p_hist.sum() + eps)
        q_hist = q_hist / (q_hist.sum() + eps)
    
    # Add small epsilon for stability
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    
    # Compute KL divergence: sum(p * log(p/q))
    kl = torch.sum(p_hist * torch.log(p_hist / q_hist))
    
    return kl