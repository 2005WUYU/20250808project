import torch
from torch import Tensor
from typing import Optional, Literal, Tuple


def displacement_norm_loss(
    disp: Tensor,
    p: int = 2,
    reduction: Literal['mean', 'sum'] = 'mean'
) -> Tensor:
    """
    Computes the norm of displacement vectors.
    
    Args:
        disp: Displacement vectors [B,N,3]
        p: Order of norm (1=L1, 2=L2)
        reduction: Reduction method ('mean' or 'sum')
        
    Returns:
        Scalar tensor with displacement norm loss
    """
    if disp.ndim != 3:
        raise ValueError(f"Expected 3D tensor [B,N,3], got shape {disp.shape}")
        
    # Compute p-norm of each displacement vector
    norms = torch.norm(disp, p=p, dim=-1)  # [B,N]
    
    # Apply reduction
    if reduction == 'mean':
        return norms.mean()
    elif reduction == 'sum':
        return norms.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def smoothness_loss(
    disp: Tensor,
    edge_index: Tensor,
    reduction: Literal['mean', 'sum'] = 'mean'
) -> Tensor:
    """
    Computes smoothness loss by penalizing displacement differences between connected nodes.
    
    Args:
        disp: Displacement vectors [B,N,3]
        edge_index: Graph connectivity [2,M]
        reduction: Reduction method ('mean' or 'sum')
        
    Returns:
        Scalar tensor with smoothness loss
    """
    B, N, _ = disp.shape
    
    # Extract source and target node indices
    i, j = edge_index[0], edge_index[1]
    
    # Gather displacements for both ends of each edge
    disp_i = disp[:, i, :]  # [B,M,3]
    disp_j = disp[:, j, :]  # [B,M,3]
    
    # Compute squared differences
    edge_diff = disp_i - disp_j  # [B,M,3]
    per_edge_sq = torch.sum(edge_diff ** 2, dim=-1)  # [B,M]
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(per_edge_sq)
    elif reduction == 'sum':
        return torch.sum(per_edge_sq)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def edge_length_consistency_loss(
    coords_gen: Tensor,
    coords_ref: Tensor,
    edge_index: Tensor,
    mode: Literal['relative', 'absolute', 'emd'] = 'relative'
) -> Tensor:
    """
    Computes edge length consistency loss between generated and reference coordinates.
    
    Args:
        coords_gen: Generated coordinates [B,N,3]
        coords_ref: Reference coordinates [B,N,3]
        edge_index: Graph connectivity [2,M]
        mode: Loss mode ('relative', 'absolute', or 'emd')
        
    Returns:
        Scalar tensor with edge length consistency loss
    """
    from .stats_loss import distribution_1d_wasserstein
    
    # Extract source and target node indices
    i, j = edge_index[0], edge_index[1]
    
    # Gather coordinates for both ends of each edge
    coords_gen_i = coords_gen[:, i, :]  # [B,M,3]
    coords_gen_j = coords_gen[:, j, :]  # [B,M,3]
    coords_ref_i = coords_ref[:, i, :]  # [B,M,3]
    coords_ref_j = coords_ref[:, j, :]  # [B,M,3]
    
    # Compute edge lengths
    l_gen = torch.norm(coords_gen_i - coords_gen_j, dim=-1)  # [B,M]
    l_ref = torch.norm(coords_ref_i - coords_ref_j, dim=-1)  # [B,M]
    
    eps = 1e-8  # Small epsilon for numerical stability
    
    if mode == 'relative':
        return torch.mean(torch.abs(l_gen - l_ref) / (l_ref + eps))
    
    elif mode == 'absolute':
        return torch.mean(torch.abs(l_gen - l_ref))
    
    elif mode == 'emd':
        # Flatten edge lengths from batch dimension for EMD calculation
        return distribution_1d_wasserstein(l_gen.flatten(), l_ref.flatten())
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def jacobian_determinant_penalty(
    coords_gen: Tensor,
    coords_ref: Tensor,
    edge_index: Tensor,
    k_neighbors: int = 6
) -> Tensor:
    """
    Penalizes negative Jacobian determinants of local transformations.
    
    Note: This is a computationally expensive loss function and should be used selectively.
    
    Args:
        coords_gen: Generated coordinates [B,N,3]
        coords_ref: Reference coordinates [B,N,3]
        edge_index: Graph connectivity [2,M]
        k_neighbors: Number of neighbors to use for affine mapping estimation
        
    Returns:
        Scalar tensor with Jacobian determinant penalty
    """
    B, N, _ = coords_gen.shape
    device = coords_gen.device
    
    # Warning: this is a computationally intensive implementation
    # This could be optimized using sparse operations or custom CUDA kernels
    
    # Build adjacency lists for efficient neighborhood queries
    adj_lists = [[] for _ in range(N)]
    for e in range(edge_index.shape[1]):
        i, j = edge_index[0, e].item(), edge_index[1, e].item()
        adj_lists[i].append(j)
    
    # For each node, compute local Jacobian
    all_dets = []
    
    for b in range(B):
        batch_dets = []
        
        for node_idx in range(N):
            # Get k nearest neighbors (or all if less than k)
            neighbors = adj_lists[node_idx]
            if len(neighbors) < 3:  # Need at least 3 points to fit affine transformation
                batch_dets.append(1.0)  # Default to identity transformation
                continue
                
            # Select up to k neighbors
            if len(neighbors) > k_neighbors:
                # Sort by distance and take closest k
                neighbor_coords = coords_ref[b, neighbors, :]
                center_coord = coords_ref[b, node_idx, :].unsqueeze(0)
                dists = torch.norm(neighbor_coords - center_coord, dim=1)
                _, indices = torch.topk(dists, k_neighbors, largest=False)
                neighbors = [neighbors[idx.item()] for idx in indices]
            
            # Gather neighbor coordinates
            src_coords = coords_ref[b, [node_idx] + neighbors, :]  # Template coords
            tgt_coords = coords_gen[b, [node_idx] + neighbors, :]  # Generated coords
            
            # Center coordinates
            src_center = src_coords[0]
            tgt_center = tgt_coords[0]
            src_centered = src_coords[1:] - src_center
            tgt_centered = tgt_coords[1:] - tgt_center
            
            # Fit affine transformation: X @ A = Y => A = X^+ @ Y
            # Use pseudoinverse for least squares solution
            try:
                X = torch.cat([src_centered, torch.ones_like(src_centered[:, :1])], dim=1)
                Y = tgt_centered
                
                # Compute pseudoinverse manually or use torch.pinverse
                X_pinv = torch.pinverse(X)
                A = X_pinv @ Y  # Transformation matrix
                
                # Extract 3x3 Jacobian matrix
                J = A[:3, :3]
                
                # Compute determinant
                det = torch.det(J)
                batch_dets.append(det.item())
                
            except Exception as e:
                # Fallback for numerical issues
                batch_dets.append(1.0)
        
        all_dets.append(torch.tensor(batch_dets, device=device))
    
    # Stack results and compute penalty
    det_tensor = torch.stack(all_dets)  # [B,N]
    
    # Penalize negative determinants (folding in the transformation)
    penalty = torch.nn.functional.relu(-det_tensor).mean()
    
    return penalty