import torch
import random
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import copy


def adapt_batch_to_schema(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Adapt a batch to the expected schema and move it to the specified device.
    
    Args:
        batch: Input batch dictionary
        device: Device to move tensors to
        
    Returns:
        Adapted batch on the specified device
    """
    result = {}
    
    # Process each key in the batch
    for key, value in batch.items():
        # Skip None values
        if value is None:
            continue
            
        # Move tensors to device
        if isinstance(value, torch.Tensor):
            if key == 'edge_index':
                # Ensure edge_index is of type torch.long
                result[key] = value.long().to(device)
            else:
                result[key] = value.to(device)
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in value.items()}
        else:
            # Keep other types as is
            result[key] = value
    
    # Ensure required keys are present
    required_keys = ['X', 'edge_index']
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Required key '{key}' missing from batch")
    
    # Ensure coords is available (duplicate from X if needed)
    if 'coords' not in result and 'X' in result:
        result['coords'] = result['X'][..., :3]
    
    # Ensure proper batch schema
    if result['X'].ndim != 3:
        raise ValueError(f"Expected X to have 3 dimensions [B,N,F], got {result['X'].shape}")
    
    # Check edge_index format and validate indices
    if result['edge_index'].ndim == 2:
        # Single graph edge_index [2,M]
        N = result['X'].shape[1]
        if torch.max(result['edge_index']) >= N:
            raise ValueError(f"Edge index contains out-of-bounds indices: max={torch.max(result['edge_index'])}, N={N}")
            
    elif result['edge_index'].ndim == 3:
        # Batched edge_index [B,2,M]
        # This requires special handling in loss functions
        pass
    else:
        raise ValueError(f"Unexpected edge_index shape: {result['edge_index'].shape}")
    
    return result


class DataLoopAdapter:
    """
    Adapts data batches to a standardized format for training and evaluation.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def adapt_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt a raw batch to the standardized format and move to device."""
        return adapt_batch_to_schema(batch, self.device)
    
    def adapt_batches(self, batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adapt multiple batches to the standardized format."""
        return [self.adapt_batch(batch) for batch in batches]


def select_template_for_batch(
    batch: Dict[str, Any], 
    device: torch.device,
    template_loader: Optional[torch.utils.data.DataLoader] = None,
    template_cache: Optional[Dict[Any, Dict[str, Any]]] = None,
    template_mode: str = 'identity'
) -> Dict[str, Any]:
    """
    Select an appropriate template for the given batch.
    
    Args:
        batch: Input batch
        device: Device to place tensors on
        template_loader: Optional loader for template data
        template_cache: Optional cache for templates
        template_mode: Mode for template selection ('identity', 'fixed', 'batch_specific', 'random')
        
    Returns:
        Template batch matching input batch size
    """
    template_cache = template_cache or {}
    B = batch['X'].shape[0]
    
    if template_mode == 'identity':
        # Use the batch itself as template (identity mapping)
        return batch
        
    elif template_mode == 'fixed':
        # Use a single fixed template for all samples
        template_id = 0  # Default template ID
        
        # Check if template is already cached
        if template_id not in template_cache:
            # Load template if template_loader is available
            if template_loader is not None:
                for temp_batch in template_loader:
                    temp_batch = adapt_batch_to_schema(temp_batch, torch.device('cpu'))
                    # Store on CPU to save GPU memory
                    template_cache[template_id] = temp_batch
                    break
            else:
                # Fallback to using the first batch as template
                # Store on CPU to save GPU memory
                template_cache[template_id] = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
        
        # Replicate template to match batch size
        template = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in template_cache[template_id].items()
        }
        return replicate_template(template, B)
        
    elif template_mode == 'batch_specific':
        # Use template specified in batch metadata
        if 'meta' not in batch or 'template_id' not in batch['meta']:
            raise ValueError("template_mode='batch_specific' requires batch['meta']['template_id']")
            
        template_ids = batch['meta']['template_id']
        
        # Load and cache templates if needed
        templates = []
        for tid in template_ids:
            if tid not in template_cache and template_loader is not None:
                # Need to load this template
                # This is inefficient for online loading - in practice, preload all templates
                for temp_batch in template_loader:
                    if temp_batch['meta']['id'] == tid:
                        temp_batch = adapt_batch_to_schema(temp_batch, torch.device('cpu'))
                        # Store on CPU to save GPU memory
                        template_cache[tid] = temp_batch
                        break
            
            if tid in template_cache:
                # Move to device before use
                templates.append({
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in template_cache[tid].items()
                })
            else:
                # Fallback to using the batch itself
                templates.append(batch)
        
        # Combine templates (advanced implementation would create a proper batch)
        # For simplicity, this implementation uses the first template replicated
        if templates:
            return replicate_template(templates[0], B)
        else:
            return batch
            
    elif template_mode == 'random':
        # Use a randomly selected template from template_loader
        template_samples = []
        
        if template_loader is not None:
            for temp_batch in template_loader:
                adapted_batch = adapt_batch_to_schema(temp_batch, torch.device('cpu'))
                template_samples.append(adapted_batch)
                if len(template_samples) >= 10:  # Cache up to 10 templates
                    break
        
        if not template_samples:
            # Fallback to using the batch itself
            template_samples = [{
                k: v.cpu() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }]
        
        # Use random.choice instead of np.random.choice for list of dictionaries
        template_dict = random.choice(template_samples)
        
        # Move template to device and replicate
        template = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in template_dict.items()
        }
        return replicate_template(template, B)
        
    else:
        raise ValueError(f"Unknown template_mode: {template_mode}")


def select_n_templates(
    n: int,
    device: torch.device,
    template_loader: Optional[torch.utils.data.DataLoader] = None,
    template_cache: Optional[Dict[Any, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Select n templates for visualization or evaluation.
    
    Args:
        n: Number of templates to select
        device: Device to place tensors on
        template_loader: Optional loader for template data
        template_cache: Optional cache for templates
        
    Returns:
        Batch containing n templates
    """
    template_cache = template_cache or {}
    template_samples = []
    
    # Try to use existing templates from cache
    if template_cache:
        for template_dict in template_cache.values():
            template_samples.append(template_dict)
            if len(template_samples) >= n:
                break
    
    # If we need more templates, load from template_loader
    if len(template_samples) < n and template_loader is not None:
        for temp_batch in template_loader:
            adapted_batch = adapt_batch_to_schema(temp_batch, torch.device('cpu'))
            template_samples.append(adapted_batch)
            if len(template_samples) >= n:
                break
    
    # If still no templates, create dummy templates
    if not template_samples:
        raise ValueError("No templates available. Please provide template_loader or template_cache.")
    
    # Select n templates (with replacement if needed)
    indices = np.random.choice(len(template_samples), size=n, replace=len(template_samples) < n)
    selected_templates = []
    
    for idx in indices:
        # Move template to device
        template = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in template_samples[idx].items()
        }
        selected_templates.append(template)
    
    # Combine templates into a single batch
    return combine_templates(selected_templates)


def replicate_template(template: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    """
    Replicate a single template to match the batch size.
    
    Args:
        template: Template batch (with batch size 1)
        batch_size: Target batch size
        
    Returns:
        Replicated template batch
    """
    result = {}
    
    for key, value in template.items():
        if isinstance(value, torch.Tensor):
            # Check if tensor has batch dimension
            if value.ndim > 0 and value.shape[0] == 1:
                # Replicate along batch dimension
                result[key] = value.repeat([batch_size] + [1] * (value.ndim - 1))
            elif value.ndim == 0 or key == 'edge_index':
                # Scalar tensor or special case for edge_index
                result[key] = value
            else:
                # Tensor without proper batch dim
                result[key] = value.unsqueeze(0).repeat([batch_size] + [1] * value.ndim)
        elif isinstance(value, dict):
            # Handle nested dictionaries (e.g., 'meta')
            result[key] = value
        else:
            # Non-tensor values
            result[key] = value
            
    return result


def combine_templates(templates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple template batches into a single batch.
    
    Args:
        templates: List of template batches
        
    Returns:
        Combined batch
    """
    if not templates:
        return {}
        
    result = {}
    n = len(templates)
    
    # First, determine total number of nodes to calculate offsets
    node_counts = [t['X'].shape[1] for t in templates]
    cumulative_nodes = [0] + list(np.cumsum(node_counts[:-1]))
    
    # Process each key
    for key in templates[0].keys():
        if key == 'edge_index':
            # Special handling for edge_index - apply node offsets and concatenate
            combined_edges = []
            
            for i, template in enumerate(templates):
                edge_index = template[key]
                # Add node offset to this template's edge indices
                offset = cumulative_nodes[i]
                
                if edge_index.ndim == 2:
                    # Format [2, M]
                    offset_edge_index = edge_index.clone()
                    # Ensure indices are within bounds and convert to long
                    valid_edges = (offset_edge_index < node_counts[i]).all(dim=0)
                    if not valid_edges.all():
                        print(f"Warning: Found out-of-bounds indices in edge_index for template {i}")
                        offset_edge_index = offset_edge_index[:, valid_edges]
                    
                    offset_edge_index[0] += offset
                    offset_edge_index[1] += offset
                    combined_edges.append(offset_edge_index)
                else:
                    # For more complex formats, just use the first template
                    combined_edges = [templates[0][key]]
                    break
                    
            # Concatenate all edges along the second dimension
            if combined_edges and all(e.ndim == 2 for e in combined_edges):
                result[key] = torch.cat(combined_edges, dim=1).long()
            else:
                # Fallback
                result[key] = templates[0][key].long()
                
        elif isinstance(templates[0][key], torch.Tensor):
            # Try to concatenate tensors along batch dimension
            try:
                tensors = [t[key] for t in templates]
                result[key] = torch.cat(tensors, dim=0)
            except:
                # Fallback to first template's tensor
                result[key] = templates[0][key]
        elif key == 'meta':
            # Combine metadata if present
            result[key] = {k: [t[key].get(k, None) for t in templates] 
                          for k in templates[0][key].keys()}
        else:
            # For other types, just use the first template's value
            result[key] = templates[0][key]
            
    return result