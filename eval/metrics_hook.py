import torch
from torch import Tensor
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy import stats


def compute_per_node_error(gen_batch: Dict[str, Any], ref_batch: Dict[str, Any]) -> Tensor:
    """
    Compute per-node error between generated and reference point clouds.
    
    Args:
        gen_batch: Generated batch with 'X' tensor [B,N,F]
        ref_batch: Reference batch with 'X' tensor [B,N,F]
        
    Returns:
        Per-node error tensor [B,N]
    """
    # Ensure batches have same shape
    assert gen_batch['X'].shape == ref_batch['X'].shape, \
        f"Shape mismatch: {gen_batch['X'].shape} vs {ref_batch['X'].shape}"
    
    # Extract coordinates (first 3 dims)
    gen_coords = gen_batch['X'][..., :3]  # [B,N,3]
    ref_coords = ref_batch['X'][..., :3]
    
    # Compute Euclidean distance per node
    error = torch.sqrt(((gen_coords - ref_coords) ** 2).sum(dim=-1) + 1e-8)  # [B,N]
    
    # Apply mask if available
    if 'mask' in ref_batch:
        error = error * ref_batch['mask']
    
    return error


def compute_graph_stats_distance(gen_batch: Dict[str, Any], ref_batch: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute distance between graph statistics of generated and reference graphs.
    
    Args:
        gen_batch: Generated batch with graph data
        ref_batch: Reference batch with graph data
        
    Returns:
        Dictionary of graph statistics distances
    """
    metrics = {}
    
    # Node count difference (if masks are available)
    if 'mask' in gen_batch and 'mask' in ref_batch:
        gen_node_count = gen_batch['mask'].sum(dim=1)
        ref_node_count = ref_batch['mask'].sum(dim=1)
        node_count_diff = torch.abs(gen_node_count - ref_node_count).mean().item()
        metrics['node_count_diff'] = node_count_diff
    
    # Degree distribution comparison (if edges available)
    if 'edge_index' in gen_batch and 'edge_index' in ref_batch:
        # This would require processing each graph individually
        # Implementation depends on how edge_index is structured
        pass
    
    # Density comparison
    # Additional graph statistics...
    
    return metrics


def compute_spectral_metrics(gen_batch: Dict[str, Any], ref_batch: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute spectral metrics between generated and reference graphs.
    
    Args:
        gen_batch: Generated batch with graph data
        ref_batch: Reference batch with graph data
        
    Returns:
        Dictionary of spectral metrics
    """
    metrics = {}
    
    try:
        # This would compute Laplacian eigenvalues and compare them
        # Requires constructing Laplacians from edge_index
        # Implementation depends on specific needs
        
        # Placeholder for actual implementation
        metrics['spectral_dist'] = 0.0
    except Exception as e:
        # Fallback if computation fails
        metrics['spectral_dist'] = float('nan')
    
    return metrics


def compute_edge_length_metrics(gen_batch: Dict[str, Any], ref_batch: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute edge length distribution metrics between generated and reference graphs.
    
    Args:
        gen_batch: Generated batch with graph data
        ref_batch: Reference batch with graph data
        
    Returns:
        Dictionary of edge length metrics
    """
    metrics = {}
    
    # Check if we have edge information
    if 'edge_index' not in gen_batch or 'edge_index' not in ref_batch:
        metrics['edge_length_jsd'] = float('nan')
        return metrics
    
    try:
        # Compute edge lengths for generated graph
        gen_edge_index = gen_batch['edge_index']
        gen_coords = gen_batch['X'][..., :3]
        gen_lengths = compute_edge_lengths(gen_edge_index, gen_coords)
        
        # Compute edge lengths for reference graph
        ref_edge_index = ref_batch['edge_index']
        ref_coords = ref_batch['X'][..., :3]
        ref_lengths = compute_edge_lengths(ref_edge_index, ref_coords)
        
        # Compute histogram distance
        jsd = jensen_shannon_distance(gen_lengths, ref_lengths)
        metrics['edge_length_jsd'] = float(jsd)
        
        # Additional edge statistics
        metrics['edge_length_mean_diff'] = float(abs(gen_lengths.mean().item() - ref_lengths.mean().item()))
        
    except Exception as e:
        metrics['edge_length_jsd'] = float('nan')
        metrics['edge_length_mean_diff'] = float('nan')
    
    return metrics


def compute_edge_lengths(edge_index: Tensor, coords: Tensor) -> Tensor:
    """
    Compute lengths of edges in a graph.
    
    Args:
        edge_index: Edge index tensor [2, E] or similar
        coords: Node coordinates [B, N, 3] or [N, 3]
        
    Returns:
        Tensor of edge lengths
    """
    # Handle batch dimension if present
    if coords.dim() == 3:
        # For simplicity, assume first batch item
        coords = coords[0]
    
    # Get source and target node indices
    src, dst = edge_index[0], edge_index[1]
    
    # Compute Euclidean distances
    lengths = torch.sqrt(((coords[src] - coords[dst]) ** 2).sum(dim=-1) + 1e-8)
    
    return lengths


def jensen_shannon_distance(sample1: Tensor, sample2: Tensor, bins: int = 50) -> float:
    """
    Compute Jensen-Shannon distance between two samples.
    
    Args:
        sample1: First sample
        sample2: Second sample
        bins: Number of histogram bins
        
    Returns:
        Jensen-Shannon distance [0,1]
    """
    # Convert to numpy
    s1 = sample1.cpu().numpy()
    s2 = sample2.cpu().numpy()
    
    # Determine common binning
    min_val = min(np.min(s1), np.min(s2))
    max_val = max(np.max(s1), np.max(s2))
    
    # Create histograms
    hist1, bin_edges = np.histogram(s1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(s2, bins=bins, range=(min_val, max_val), density=True)
    
    # Add small epsilon to avoid log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Compute JS divergence
    m = 0.5 * (hist1 + hist2)
    js_div = 0.5 * (stats.entropy(hist1, m) + stats.entropy(hist2, m))
    
    # Convert to distance
    js_dist = np.sqrt(js_div)
    
    return float(js_dist)