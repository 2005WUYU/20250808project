import torch
from torch import nn, Tensor
from typing import Dict, Any, Optional, Union, List, Tuple
from torch_geometric.data import Batch, Data

from utils.data_adapter import DataAdapter


class ModelIntegrator:
    """
    Utility class to facilitate integration between components with different data formats.
    """
    
    def __init__(self, 
                 data_adapter: Optional[DataAdapter] = None,
                 compute_edge_attr: bool = True):
        """
        Initialize model integrator.
        
        Args:
            data_adapter: Data adapter for format conversion
            compute_edge_attr: Whether to compute edge attributes if missing
        """
        self.data_adapter = data_adapter or DataAdapter()
        self.compute_edge_attr = compute_edge_attr
    
    def wrap_generator(self, G: nn.Module) -> nn.Module:
        """
        Wrap generator to ensure consistent output format.
        
        Args:
            G: Generator model
            
        Returns:
            Wrapped generator model
        """
        original_forward = G.forward
        
        def wrapped_forward(z, *args, **kwargs):
            # Call original generator
            output = original_forward(z, *args, **kwargs)
            
            # Convert to standard format if not already
            if not isinstance(output, dict):
                output = self.data_adapter.adapt_batch(output)
                
            return output
        
        G.forward = wrapped_forward
        return G
    
    def wrap_discriminator(self, D: nn.Module) -> nn.Module:
        """
        Wrap discriminator to ensure consistent input format.
        
        Args:
            D: Discriminator model
            
        Returns:
            Wrapped discriminator model
        """
        original_forward = D.forward
        
        def wrapped_forward(x, *args, **kwargs):
            # Convert input to format expected by discriminator
            if isinstance(x, dict):
                # Ensure we have edge attributes if needed and requested
                if self.compute_edge_attr and 'edge_index' in x and 'edge_attr' not in x:
                    x['edge_attr'] = self._compute_edge_features(
                        x['edge_index'], 
                        x['X'][..., :3] if x['X'].dim() == 3 else x['X'][:, :3]
                    )
                
                # If discriminator expects PyG format
                if isinstance(getattr(D, 'expected_format', None), str) and D.expected_format == 'pyg':
                    x = self.data_adapter.dict_to_pyg(x)
            
            # Call original discriminator
            return original_forward(x, *args, **kwargs)
        
        D.forward = wrapped_forward
        return D
    
    def _compute_edge_features(self, edge_index: Tensor, coords: Tensor) -> Tensor:
        """
        Compute basic edge features from coordinates.
        
        Args:
            edge_index: Edge indices [2, E]
            coords: Node coordinates
            
        Returns:
            Edge features
        """
        # Handle batch dimension
        if coords.dim() == 3:
            # For simplicity, assume single batch for now
            # Full implementation would handle per-graph edge indices
            coords = coords[0]
            
        src, dst = edge_index
        
        # Compute displacement vectors
        disp = coords[dst] - coords[src]  # [E, 3]
        
        # Compute lengths
        lengths = torch.norm(disp, dim=1, keepdim=True)  # [E, 1]
        
        # Normalize displacements
        normed_disp = disp / (lengths + 1e-8)  # [E, 3]
        
        # Create edge features [E, 4]: [length, normalized_direction]
        edge_attr = torch.cat([lengths, normed_disp], dim=1)
        
        return edge_attr