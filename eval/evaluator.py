import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import time

from trainer.data_loop import adapt_batch_to_schema, select_template_for_batch
from eval.metrics_hooks import (
    compute_per_node_error,
    compute_graph_stats_distance,
    compute_spectral_metrics,
    compute_edge_length_metrics
)


class Evaluator:
    """
    Evaluator for graph generative models.
    
    Handles validation and test-time evaluation including metrics
    for anomaly detection and generation quality.
    """
    
    def __init__(self, cfg: Dict[str, Any], device: str = 'cuda:0'):
        """
        Initialize the evaluator.
        
        Args:
            cfg: Configuration dictionary
            device: Device to run evaluation on
        """
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.z_dim = cfg.get('z_dim', 128)
    
    def evaluate(
        self,
        G: torch.nn.Module,
        D: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        data_adapter,
        template_loader: Optional[torch.utils.data.DataLoader] = None,
        template_cache: Optional[Dict] = None,
        mode: str = 'validation'
    ) -> Dict[str, float]:
        """
        Evaluate models on the provided data loader.
        
        Args:
            G: Generator model
            D: Discriminator model
            data_loader: Data loader for evaluation
            data_adapter: Adapter for processing data batches
            template_loader: Optional loader for templates
            template_cache: Optional cache for templates
            mode: 'validation' or 'test'
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nRunning evaluation ({mode})...")
        start_time = time.time()
        
        # Set models to eval mode
        G.eval()
        D.eval()
        
        # Collect metrics
        all_metrics = {}
        batch_metrics = []
        
        # For anomaly detection
        all_scores = []
        all_labels = []
        
        # For detailed per-node errors
        all_per_node_errors = []
        all_meta_info = []
        
        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(data_loader):
                # Process batch
                batch = data_adapter.adapt_batch(raw_batch)
                
                # Get batch size
                B = batch['X'].shape[0]
                
                # Sample noise
                z = self.sample_noise([B, self.z_dim], device=self.device)
                
                # Get templates
                template_batch = select_template_for_batch(
                    batch, 
                    self.device, 
                    template_loader, 
                    template_cache,
                    self.cfg.get('template_mode', 'identity')
                )
                
                # Generate fake samples
                fake_batch = G(z, template=template_batch)
                
                # Compute metrics
                metrics = self.compute_metrics(fake_batch, batch, D)
                batch_metrics.append(metrics)
                
                # Store per-node errors for detailed analysis
                per_node_err = compute_per_node_error(fake_batch, batch)
                all_per_node_errors.append(per_node_err)
                
                # Store metadata if available
                if 'meta' in batch:
                    all_meta_info.append(batch['meta'])
                
                # For anomaly detection, collect scores and labels
                if 'labels' in batch:
                    # Use per-node max error as anomaly score
                    scores = per_node_err.max(dim=1)[0].cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    
                    all_scores.append(scores)
                    all_labels.append(labels)
        
        # Aggregate metrics across batches
        for key in batch_metrics[0].keys():
            values = [m[key] for m in batch_metrics]
            all_metrics[key] = np.mean(values)
        
        # Calculate ROC/AUC if labels are available
        if all_labels and all_scores:
            all_metrics.update(self.compute_anomaly_detection_metrics(all_scores, all_labels))
        
        # Finalize metrics
        all_metrics['eval_time'] = time.time() - start_time
        
        return all_metrics
    
    def compute_metrics(self, fake_batch: Dict[str, Any], real_batch: Dict[str, Any], D: torch.nn.Module) -> Dict[str, float]:
        """
        Compute evaluation metrics between generated and real batches.
        
        Args:
            fake_batch: Generated batch
            real_batch: Real batch
            D: Discriminator model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Reconstruction loss
        from losses.reconstruction import reconstruction_loss
        recon_loss = reconstruction_loss(
            fake_batch, 
            real_batch, 
            mode=self.cfg.get('recon_mode', '