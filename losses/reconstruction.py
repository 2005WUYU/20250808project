import torch
from torch import Tensor
from typing import Dict, Optional, Literal


def reconstruction_loss(
    gen_batch: Dict[str, Tensor],
    ref_batch: Dict[str, Tensor],
    mode: Literal['l1', 'l2', 'huber'] = 'l1',
    coord_weight: float = 1.0,
    feat_weight: float = 1.0,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Compute reconstruction loss between generated and reference batches.
    
    Args:
        gen_batch: Generated batch dictionary with 'X' key
        ref_batch: Reference batch dictionary with 'X' key
        mode: Loss type ('l1', 'l2', or 'huber')
        coord_weight: Weight for coordinate loss component
        feat_weight: Weight for feature loss component
        mask: Optional mask tensor [B,N] to apply to the loss
        
    Returns:
        Scalar reconstruction loss
    """
    # Shape checks
    assert gen_batch['X'].shape == ref_batch['X'].shape, \
        f"Shape mismatch in reconstruction: {gen_batch['X'].shape} vs {ref_batch['X'].shape}"
    
    B, N, F = gen_batch['X'].shape

    # Split coordinates and features
    coords_gen = gen_batch['X'][..., :3]   # [B,N,3]
    coords_ref = ref_batch['X'][..., :3]
    
    # Extract features if they exist
    has_features = F > 3
    if has_features:
        feats_gen = gen_batch['X'][..., 3:]   # [B,N,F-3]
        feats_ref = ref_batch['X'][..., 3:]

    # Apply mask if provided
    if mask is not None:
        mask_expand = mask.unsqueeze(-1)  # [B,N,1]
        coords_diff = (coords_gen - coords_ref) * mask_expand
        if has_features:
            feats_diff = (feats_gen - feats_ref) * mask_expand
        denom = max(1.0, mask.sum().item())
    else:
        coords_diff = coords_gen - coords_ref
        if has_features:
            feats_diff = feats_gen - feats_ref
        denom = B * N

    # Compute per-element loss based on mode
    if mode == 'l1':
        L_coord = torch.abs(coords_diff).mean()
        L_feat = torch.abs(feats_diff).mean() if has_features else 0.0
    
    elif mode == 'l2':
        L_coord = (coords_diff ** 2).mean()
        L_feat = (feats_diff ** 2).mean() if has_features else 0.0
    
    elif mode == 'huber':
        L_coord = torch.nn.functional.smooth_l1_loss(coords_gen, coords_ref, reduction='mean')
        L_feat = torch.nn.functional.smooth_l1_loss(feats_gen, feats_ref, reduction='mean') if has_features else 0.0
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Combine coordinate and feature losses with weights
    total_loss = coord_weight * L_coord
    if has_features and feat_weight > 0:
        total_loss = total_loss + feat_weight * L_feat
    
    return total_loss