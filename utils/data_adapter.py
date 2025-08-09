import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np


class DataAdapter:
    """
    Utility class for standardizing data formats between PyG and Dict representations
    and ensuring consistent field naming and attribute availability.
    """
    
    def __init__(self, 
                 node_feature_dim: int = 3,
                 edge_feature_dim: int = 0, 
                 compute_edges: bool = True,
                 coordinate_field: str = 'X',
                 edge_field: str = 'edge_index'):
        """
        Initialize the data adapter.
        
        Args:
            node_feature_dim: Dimension of node features (default: 3 for xyz)
            edge_feature_dim: Dimension of edge features (default: 0)
            compute_edges: Whether to compute edges if missing
            coordinate_field: Name of the field to use for coordinates in dict format
            edge_field: Name of the field to use for edge indices
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.compute_edges = compute_edges
        self.coordinate_field = coordinate_field
        self.edge_field = edge_field
        
    def pyg_to_dict(self, batch: Union[Batch, Data]) -> Dict[str, Any]:
        """
        Convert PyG Batch/Data to standardized dictionary format.
        
        Args:
            batch: PyTorch Geometric Batch or Data object
            
        Returns:
            Standardized dictionary format
        """
        result = {}
        
        # Handle batch dimension
        if isinstance(batch, Batch):
            batch_size = batch.num_graphs
            ptr = batch.ptr
        else:
            batch_size = 1
            ptr = torch.tensor([0, batch.num_nodes])
        
        # Extract coordinates and features
        coords = None
        features = None
        
        # Check for position data
        if hasattr(batch, 'pos'):
            coords = batch.pos
        elif hasattr(batch, 'x') and batch.x.size(1) >= 3:
            # Assume first 3 dims of x are coordinates
            coords = batch.x[:, :3]
            if batch.x.size(1) > 3:
                features = batch.x[:, 3:]
        
        # Check for separate feature data
        if features is None and hasattr(batch, 'x') and coords is not None and batch.x is not coords:
            features = batch.x
            
        # Create X tensor combining coordinates and features
        if coords is not None:
            if features is not None:
                X = torch.cat([coords, features], dim=1)
            else:
                X = coords
        else:
            raise ValueError("No coordinate data found in batch")
        
        # Reshape based on batch size
        X_batched = []
        for i in range(batch_size):
            start, end = ptr[i], ptr[i+1]
            X_batched.append(X[start:end])
        
        # Pad if necessary to make same size
        if batch_size > 1:
            max_nodes = max(x.size(0) for x in X_batched)
            X_padded = []
            mask = []
            
            for x in X_batched:
                n_nodes = x.size(0)
                if n_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - n_nodes, x.size(1), 
                                          device=x.device, dtype=x.dtype)
                    x_pad = torch.cat([x, padding], dim=0)
                    node_mask = torch.cat([
                        torch.ones(n_nodes, device=x.device),
                        torch.zeros(max_nodes - n_nodes, device=x.device)
                    ])
                else:
                    x_pad = x
                    node_mask = torch.ones(max_nodes, device=x.device)
                
                X_padded.append(x_pad)
                mask.append(node_mask)
            
            result[self.coordinate_field] = torch.stack(X_padded)
            result['mask'] = torch.stack(mask)
        else:
            # Single graph case
            result[self.coordinate_field] = X.unsqueeze(0)
            result['mask'] = torch.ones(1, X.size(0), device=X.device)
        
        # Handle edge information
        if hasattr(batch, 'edge_index'):
            result[self.edge_field] = batch.edge_index
            
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                result['edge_attr'] = batch.edge_attr
            elif self.compute_edges and self.edge_feature_dim > 0:
                # Compute edge features if needed
                result['edge_attr'] = self._compute_edge_features(
                    batch.edge_index, coords, self.edge_feature_dim
                )
        
        # Copy any additional attributes
        for key, value in batch:
            if key not in ['x', 'pos', 'edge_index', 'edge_attr', 'batch', 'ptr'] and key not in result:
                result[key] = value
                
        return result
    
    def dict_to_pyg(self, batch_dict: Dict[str, Any]) -> Union[Batch, List[Data]]:
        """
        Convert dictionary format to PyG Batch/Data.
        
        Args:
            batch_dict: Dictionary with standardized format
            
        Returns:
            PyTorch Geometric Batch or list of Data objects
        """
        if self.coordinate_field not in batch_dict:
            raise ValueError(f"Missing required field {self.coordinate_field}")
        
        # Extract data
        X = batch_dict[self.coordinate_field]  # [B, N, F]
        mask = batch_dict.get('mask', None)
        
        # Handle batch dimension
        batch_size = X.size(0)
        data_list = []
        
        for i in range(batch_size):
            x_i = X[i]
            
            # Handle masking
            if mask is not None:
                valid_nodes = int(mask[i].sum().item())
                x_i = x_i[:valid_nodes]
            
            # Split coordinates and features
            coords = x_i[:, :3]
            features = x_i[:, 3:] if x_i.size(1) > 3 else None
            
            data = Data(pos=coords)
            
            if features is not None:
                data.x = features
            
            # Handle edges
            if self.edge_field in batch_dict:
                # Need to adjust edge_index for this particular graph
                # This depends on how edge_index is stored in the dict
                # (typically needs to be handled specially for each batch)
                pass
            
            # Copy additional attributes
            for key, value in batch_dict.items():
                if key not in [self.coordinate_field, 'mask', self.edge_field] and key not in data:
                    # Handle tensors with batch dimension
                    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == batch_size:
                        data[key] = value[i]
                    else:
                        data[key] = value
                        
            data_list.append(data)
        
        # Return as batch if requested
        return Batch.from_data_list(data_list)
    
    def adapt_batch(self, batch: Union[Dict[str, Any], Batch, Data]) -> Dict[str, Any]:
        """
        Convert any batch format to the standardized dictionary format.
        
        Args:
            batch: Input batch in any supported format
            
        Returns:
            Standardized dictionary format
        """
        if isinstance(batch, (Batch, Data)):
            return self.pyg_to_dict(batch)
        elif isinstance(batch, dict):
            # Check if the batch needs standardization
            if self.coordinate_field not in batch and ('x' in batch or 'pos' in batch):
                # Convert from dict with non-standard fields to standard dict
                standardized = {}
                
                # Handle coordinates
                if 'pos' in batch:
                    coords = batch['pos']
                elif 'x' in batch and batch['x'].size(-1) >= 3:
                    coords = batch['x'][..., :3]
                else:
                    raise ValueError("No coordinate data found in batch dict")
                
                # Handle features
                if 'x' in batch and batch['x'].size(-1) > 3:
                    features = batch['x'][..., 3:]
                    X = torch.cat([coords, features], dim=-1)
                else:
                    X = coords
                
                standardized[self.coordinate_field] = X
                
                # Copy other fields
                for key, value in batch.items():
                    if key not in ['x', 'pos'] and key not in standardized:
                        standardized[key] = value
                
                return standardized
            else:
                # Already in standard format
                return batch
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
    
    def _compute_edge_features(self, edge_index: Tensor, pos: Tensor, edge_feat_dim: int) -> Tensor:
        """
        Compute edge features based on node positions.
        
        Args:
            edge_index: Edge index tensor [2, E]
            pos: Node position tensor [N, 3]
            edge_feat_dim: Dimension of edge features to compute
            
        Returns:
            Edge feature tensor [E, edge_feat_dim]
        """
        # Default implementation: compute distances between connected nodes
        src, dst = edge_index
        dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
        
        if edge_feat_dim == 1:
            return dist
        elif edge_feat_dim > 1:
            # Expand with zeros or compute additional features
            return torch.cat([dist, torch.zeros(dist.size(0), edge_feat_dim - 1, device=dist.device)], dim=-1)
        else:
            return torch.zeros(edge_index.size(1), 0, device=edge_index.device)


def compute_edges_from_points(points: Tensor, method: str = 'knn', k: int = 8, 
                             radius: float = None, loop: bool = False) -> Tensor:
    """
    Compute edges between points using different methods.
    
    Args:
        points: Point cloud tensor [B, N, 3] or [N, 3]
        method: Edge computation method ('knn', 'radius', or 'delaunay')
        k: Number of neighbors for knn
        radius: Radius for radius-based graphs
        loop: Whether to include self-loops
        
    Returns:
        edge_index tensor [2, E]
    """
    # Handle batched input
    if points.dim() == 3:
        # Currently only implementing for single point cloud
        # For batched version, would need to handle offsets
        points = points[0]
    
    try:
        # Use PyTorch Geometric functions if available
        import torch_geometric.nn as pyg_nn
        
        if method == 'knn':
            edge_index = pyg_nn.knn_graph(points, k=k, loop=loop)
        elif method == 'radius':
            if radius is None:
                raise ValueError("Radius must be specified for radius graph")
            edge_index = pyg_nn.radius_graph(points, r=radius, loop=loop)
        else:
            raise ValueError(f"Unsupported edge computation method: {method}")
            
    except ImportError:
        # Fallback to manual KNN computation
        # This is a simplified version - full implementation would be more complex
        N = points.size(0)
        dists = torch.cdist(points, points)
        
        if not loop:
            # Fill diagonal with large value to exclude self connections
            dists.fill_diagonal_(float('inf'))
        
        # Get K nearest neighbors
        _, nn_idx = dists.topk(k, dim=1, largest=False)
        
        rows = torch.arange(N, device=points.device).repeat_interleave(k)
        cols = nn_idx.view(-1)
        
        edge_index = torch.stack([rows, cols], dim=0)
        
    return edge_index