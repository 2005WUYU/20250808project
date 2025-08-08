import os
import torch
import numpy as np
import random  # Added for proper random choice
from torch import Tensor
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import logging
from tqdm import tqdm
import copy

from losses import (
    adversarial,
    reconstruction,
    regularizers,
    spectral,
    stats_loss,
    utils_losses
)


class MetricsAccumulator:
    """Helper class to accumulate and average metrics during training."""
    
    def __init__(self, window_size=100):
        self.metrics = {}
        self.counts = {}
        self.recent_values = {}
        self.window_size = window_size
    
    def add(self, metric_dict):
        """Add a dictionary of metrics to the accumulator."""
        for key, value in metric_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
                self.recent_values[key] = []
            
            # Add to total
            self.metrics[key] += value
            self.counts[key] += 1
            
            # Add to recent window
            self.recent_values[key].append(value)
            if len(self.recent_values[key]) > self.window_size:
                self.recent_values[key].pop(0)
    
    def average(self):
        """Get the average of all accumulated metrics."""
        return {k: self.metrics[k] / max(1, self.counts[k]) 
                for k in self.metrics}
    
    def recent(self):
        """Get the average of recent metrics within window size."""
        return {k: sum(v) / max(1, len(v)) 
                for k, v in self.recent_values.items()}
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.metrics = {}
        self.counts = {}
        self.recent_values = {}


class Logger:
    """Simple logging interface that can be extended to support different backends."""
    
    def __init__(self, log_dir=None, use_tensorboard=False):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
                self.use_tensorboard = False
    
    def log_scalar(self, name, value, step):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(name, value, step)
        
        # Also print to console
        print(f"[Step {step}] {name}: {value:.5f}")
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars under the same main tag."""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Print summary to console
        print(f"[Step {step}] {main_tag}:")
        for tag, value in tag_scalar_dict.items():
            print(f"  {tag}: {value:.5f}")
    
    def log_histogram(self, name, values, step):
        """Log a histogram of values."""
        if self.writer:
            self.writer.add_histogram(name, values, step)
    
    def log_image(self, name, image_tensor, step):
        """Log an image tensor."""
        if self.writer:
            self.writer.add_image(name, image_tensor, step)
    
    def log_graph(self, model, input_data):
        """Log model graph."""
        if self.writer:
            self.writer.add_graph(model, input_data)
    
    def close(self):
        """Close the logger and release resources."""
        if self.writer:
            self.writer.close()


class Trainer:
    """
    Trainer for graph generative adversarial networks.
    
    Implements WGAN-GP training for graph generation with various loss functions
    and regularization terms.
    """
    
    def __init__(self, 
                 cfg: Dict[str, Any],
                 G: torch.nn.Module,
                 D: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 template_loader: Optional[torch.utils.data.DataLoader] = None,
                 device: str = 'cuda:0',
                 logger: Optional[Logger] = None):
        """
        Initialize the trainer with models, data loaders, and configuration.
        
        Args:
            cfg: Configuration dictionary
            G: Generator network
            D: Discriminator/critic network
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            template_loader: Optional DataLoader for templates
            device: Device to run training on ('cuda:0', 'cpu', etc.)
            logger: Optional logger for metrics
        """
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.G = G.to(self.device)
        self.D = D.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.template_loader = template_loader
        
        # Set default logger if none provided
        self.logger = logger if logger is not None else Logger()
        
        # Setup training components
        self.setup_optimizers()
        self.setup_loss_functions()
        
        # Setup AMP if configured
        self.use_amp = cfg.get('use_amp', False)
        self.amp_scaler = GradScaler() if self.use_amp else None
        
        # Initialize training state
        self.step = 0
        self.epoch = 0
        self.best_val_metric = float('inf')  # Can be changed based on metric direction
        
        # Get z dimension from config
        self.z_dim = cfg.get('z_dim', 128)
        
        # Template cache for efficiency (stored on CPU to save GPU memory)
        self.template_cache = {}
        
        # Eigenvalue cache for spectral computations
        self.eigenvalue_cache = {}
        
    def setup_optimizers(self):
        """Configure optimizers for generator and discriminator."""
        # Get hyperparameters from config
        lr_G = self.cfg.get('lr_G', 1e-4)
        lr_D = self.cfg.get('lr_D', 1e-4)
        betas = self.cfg.get('betas', (0.5, 0.9))
        
        # Create optimizers
        self.optim_G = Adam(self.G.parameters(), lr=lr_G, betas=betas)
        self.optim_D = Adam(self.D.parameters(), lr=lr_D, betas=betas)
        
        # Setup learning rate schedulers if configured
        if 'lr_schedule' in self.cfg:
            sched_cfg = self.cfg['lr_schedule']
            if sched_cfg['type'] == 'step':
                from torch.optim.lr_scheduler import StepLR
                self.sched_G = StepLR(
                    self.optim_G, 
                    step_size=sched_cfg['step_size'], 
                    gamma=sched_cfg['gamma']
                )
                self.sched_D = StepLR(
                    self.optim_D, 
                    step_size=sched_cfg['step_size'], 
                    gamma=sched_cfg['gamma']
                )
            elif sched_cfg['type'] == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.sched_G = CosineAnnealingLR(
                    self.optim_G, 
                    T_max=sched_cfg['T_max']
                )
                self.sched_D = CosineAnnealingLR(
                    self.optim_D, 
                    T_max=sched_cfg['T_max']
                )
            else:
                self.sched_G = None
                self.sched_D = None
        else:
            self.sched_G = None
            self.sched_D = None
    
    def setup_loss_functions(self):
        """Bind loss functions for easy access during training."""
        self.loss_fns = {
            'critic_loss': adversarial.critic_loss,
            'gen_adv': adversarial.generator_adversarial_loss,
            'gp': utils_losses.compute_gradient_penalty,  # Correct module binding
            'recon': reconstruction.reconstruction_loss,
            'smooth': regularizers.smoothness_loss,
            'disp_norm': regularizers.displacement_norm_loss,
            'spec': spectral.laplacian_spectrum_loss,
            'edge_len': regularizers.edge_length_consistency_loss,
            'stats': stats_loss.distribution_1d_wasserstein,
            'wasserstein_dist': adversarial.estimate_wasserstein_distance,
        }
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        print(f"Starting training for {num_epochs} epochs")
        start_epoch = self.epoch  # Store starting epoch for correct display
        
        # Precompute eigenvalues for templates if spectral loss is used
        if self.cfg.get('spec_weight', 0.0) > 0 and self.template_loader is not None:
            self._precompute_template_eigenvalues()
        
        for epoch in range(self.epoch, self.epoch + num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch+1}/{start_epoch + num_epochs}")
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Update learning rate schedulers if used
            if self.sched_G is not None:
                self.sched_G.step()
                self.sched_D.step()
            
            # Log epoch-level metrics
            self.logger.log_scalars('train', train_metrics, epoch)
            
            # Validate if a validation loader was provided
            if self.val_loader is not None:
                val_metrics = self.validate(self.val_loader)
                self.logger.log_scalars('val', val_metrics, epoch)
                
                # Save best model based on validation metric
                selected_metric = val_metrics.get(
                    self.cfg.get('selected_metric', 'loss_recon'), 
                    float('inf')
                )
                
                better_metric = self._is_better_metric(selected_metric, self.best_val_metric)
                if better_metric:
                    print(f"Validation metric improved from {self.best_val_metric:.6f} to {selected_metric:.6f}")
                    self.save_checkpoint('best.pth')
                    self.best_val_metric = selected_metric
            
            # Save periodic checkpoints
            save_every = self.cfg.get('save_every_epochs', 10)
            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
            
            # Visualize samples periodically
            viz_every = self.cfg.get('visualize_every_epochs', 5)
            if epoch % viz_every == 0:
                self.sample_and_visualize(
                    n_samples=4, 
                    save_path=os.path.join(self.cfg.get('output_dir', 'output'), f'epoch_{epoch}')
                )
        
        # Save final model
        self.save_checkpoint('final.pth')
        print("Training completed!")

    def _precompute_template_eigenvalues(self):
        """Precompute eigenvalues for all templates to avoid computing them during training."""
        print("Precomputing template eigenvalues for spectral loss...")
        k = self.cfg.get('spec_k', 30)
        
        for i, template_batch in enumerate(self.template_loader):
            # Process batch
            template_batch = self.adapt_batch_to_schema(template_batch)
            
            # Get template ID if available, otherwise use index
            template_id = template_batch.get('meta', {}).get('id', i)
            
            # Compute eigenvalues efficiently
            eigenvalues = self._compute_laplacian_eigenvalues(template_batch, k)
            
            # Cache eigenvalues on CPU
            self.eigenvalue_cache[template_id] = eigenvalues.cpu()
            
            if i % 10 == 0:
                print(f"Processed {i+1} templates")
    
    def _is_better_metric(self, current: float, best: float) -> bool:
        """
        Determine if the current metric is better than the best so far.
        
        Args:
            current: Current metric value
            best: Best metric value so far
            
        Returns:
            True if current is better than best
        """
        # Default to lower is better, but can be configured
        metric_mode = self.cfg.get('metric_mode', 'min')
        
        if metric_mode == 'min':
            return current < best
        elif metric_mode == 'max':
            return current > best
        else:
            raise ValueError(f"Unknown metric mode: {metric_mode}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.G.train()
        self.D.train()
        
        # Initialize metric accumulator
        accum_metrics = MetricsAccumulator()
        
        # Get training parameters
        d_steps = self.cfg.get('d_steps', 5)  # WGAN-GP typically uses 5 D steps per G step
        g_steps = self.cfg.get('g_steps', 1)  # Usually 1 G step
        log_interval = self.cfg.get('log_interval', 50)  # How often to log metrics
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}")
        except ImportError:
            pbar = None
        
        # Main training loop
        for batch_idx, real_batch in enumerate(self.train_loader):
            # Move batch to device and adapt to expected schema
            real_batch = self.adapt_batch_to_schema(real_batch)
            
            # Train discriminator for d_steps
            for d_iter in range(d_steps):
                metrics_D = self.train_step_D(real_batch)
                accum_metrics.add(metrics_D)
                self.step += 1
            
            # Train generator for g_steps
            for g_iter in range(g_steps):
                metrics_G = self.train_step_G(real_batch)
                accum_metrics.add(metrics_G)
            
            # Log periodically
            if self.step % log_interval == 0:
                recent_metrics = accum_metrics.recent()
                self.logger.log_scalars('training', recent_metrics, self.step)
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                if self.step % 10 == 0:  # Update stats every 10 steps
                    pbar.set_postfix(accum_metrics.recent())
        
        # Clean up progress bar
        if pbar is not None:
            pbar.close()
        
        # Return average metrics for the epoch
        return accum_metrics.average()
    
    def train_step_D(self, real_batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Training step for discriminator.
        
        Args:
            real_batch: Batch of real data
            
        Returns:
            Dictionary of metrics from this step
        """
        # Get batch size
        B = real_batch['X'].shape[0]
        
        # 1) Sample noise vector
        z = self.sample_noise([B, self.z_dim], device=self.device)
        
        # 2) Sample template or get from batch
        template_batch = self.select_template_for_batch(real_batch)
        
        # 3) Generate fake batch with no gradient tracking for G
        with torch.no_grad():
            fake_batch = self.G(z=z, template=template_batch)
        
        # 4) Forward pass through discriminator for both real and fake
        with autocast(enabled=self.use_amp):
            real_scores = self.D(real_batch)
            fake_scores = self.D(fake_batch)
            
            # Ensure scores have shape [B]
            if real_scores.ndim > 1:
                real_scores = real_scores.view(-1)
            if fake_scores.ndim > 1:
                fake_scores = fake_scores.view(-1)
            
            # 5) Compute critic loss
            loss_basic = self.loss_fns['critic_loss'](real_scores, fake_scores)
        
        # 6) Compute gradient penalty outside of autocast context to avoid numerical issues
        with autocast(enabled=False):
            gp = self.loss_fns['gp'](
                self.D, 
                real_batch, 
                fake_batch, 
                device=self.device, 
                gp_lambda=self.cfg.get('gp_lambda', 10.0)
            )
            
            # Total D loss
            loss_D = loss_basic + gp
        
        # 7) Backward and optimization step
        self.optim_D.zero_grad()
        
        if self.use_amp:
            # AMP: scaled backward pass
            self.amp_scaler.scale(loss_D).backward()
            self.amp_scaler.unscale_(self.optim_D)
            
            # Optional gradient clipping
            if self.cfg.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(
                    self.D.parameters(), 
                    self.cfg.get('max_grad_norm', 1.0)
                )
                
            self.amp_scaler.step(self.optim_D)
            self.amp_scaler.update()
        else:
            # Normal backward pass
            loss_D.backward()
            
            # Optional gradient clipping
            if self.cfg.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(
                    self.D.parameters(), 
                    self.cfg.get('max_grad_norm', 1.0)
                )
                
            self.optim_D.step()
        
        # 8) Compute metrics for logging
        with torch.no_grad():
            w_dist = self.loss_fns['wasserstein_dist'](real_scores, fake_scores)
            
        # Convert all metrics to Python floats for consistent logging
        metrics = {
            'loss_D': float(loss_D.item()),
            'loss_D_basic': float(loss_basic.item()),
            'gp': float(gp.item()),
            'mean_real_score': float(torch.mean(real_scores).item()),
            'mean_fake_score': float(torch.mean(fake_scores).item()),
            'wasserstein_distance': float(w_dist)
        }
        
        return metrics
    
    def train_step_G(self, real_batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Training step for generator.
        
        Args:
            real_batch: Batch of real data
            
        Returns:
            Dictionary of metrics from this step
        """
        # Get batch size
        B = real_batch['X'].shape[0]
        
        # 1) Sample noise for generator
        z = self.sample_noise([B, self.z_dim], device=self.device)
        
        # 2) Sample/choose template
        template_batch = self.select_template_for_batch(real_batch)
        
        # 3) Generate fake samples
        with autocast(enabled=self.use_amp):
            fake_batch = self.G(z=z, template=template_batch)
            
            # 4) Forward through discriminator
            fake_scores_for_G = self.D(fake_batch)
            
            # Ensure scores have shape [B]
            if fake_scores_for_G.ndim > 1:
                fake_scores_for_G = fake_scores_for_G.view(-1)
            
            # 5) Compute adversarial loss
            adv_loss = self.loss_fns['gen_adv'](fake_scores_for_G)
            
            # 6) Compute reconstruction loss
            recon_loss = self.loss_fns['recon'](
                fake_batch, 
                real_batch, 
                mode=self.cfg.get('recon_mode', 'l1'),
                coord_weight=self.cfg.get('coord_weight', 1.0),
                feat_weight=self.cfg.get('feat_weight', 1.0),
                mask=real_batch.get('mask', None)
            )
            
            # 7) Compute regularization losses
            # Displacement from template
            disp = fake_batch['X'][..., :3] - template_batch['X'][..., :3]
            
            # Displacement norm regularization
            disp_norm_loss = self.loss_fns['disp_norm'](
                disp, 
                p=self.cfg.get('disp_norm_p', 2),
                reduction='mean'
            )
            
            # Smoothness loss
            smooth_loss = self.loss_fns['smooth'](
                disp, 
                real_batch['edge_index']
            )
            
            # Spectral loss
            spec_loss = 0.0
            if self.cfg.get('spec_weight', 0.0) > 0:
                try:
                    # Get template ID for cached eigenvalues
                    template_id = template_batch.get('meta', {}).get('id', None)
                    
                    # Get cached eigenvalues if available
                    eigs_ref = None
                    if template_id is not None and template_id in self.eigenvalue_cache:
                        eigs_ref = self.eigenvalue_cache[template_id].to(self.device)
                    
                    # Compute eigenvalues for fake batch
                    k = self.cfg.get('spec_k', 30)
                    eigs_gen = self._compute_laplacian_eigenvalues(fake_batch, k)
                    
                    # Compute spectral loss
                    if eigs_ref is None:
                        # Compute eigenvalues for reference if not cached
                        eigs_ref = self._compute_laplacian_eigenvalues(template_batch, k)
                    
                    spec_loss = self.loss_fns['spec'](
                        eigs_gen=eigs_gen,
                        eigs_ref=eigs_ref,
                        mode=self.cfg.get('spec_mode', 'l2')
                    )
                    
                except Exception as e:
                    print(f"Warning: Error in spectral loss calculation: {e}")
                    spec_loss = torch.tensor(0.0, device=self.device)
            
            # Edge length consistency loss
            edge_len_loss = 0.0
            if self.cfg.get('edge_len_weight', 0.0) > 0:
                edge_len_loss = self.loss_fns['edge_len'](
                    coords_gen=fake_batch['X'][..., :3],
                    coords_ref=real_batch['X'][..., :3],
                    edge_index=real_batch['edge_index'],
                    mode=self.cfg.get('edge_len_mode', 'relative')
                )
            
            # Stats distributional loss
            stats_loss = 0.0
            if self.cfg.get('stats_weight', 0.0) > 0:
                # Example: compare edge length distributions
                l_fake = self.compute_edge_lengths(fake_batch)
                l_real = self.compute_edge_lengths(real_batch)
                
                stats_loss = self.loss_fns['stats'](
                    l_fake.flatten(), 
                    l_real.flatten()
                )
            
            # 8) Total G loss with weights from config
            loss_G = self.cfg.get('adv_weight', 1.0) * adv_loss + \
                     self.cfg.get('recon_weight', 10.0) * recon_loss + \
                     self.cfg.get('smooth_weight', 1.0) * smooth_loss + \
                     self.cfg.get('disp_weight', 1.0) * disp_norm_loss + \
                     self.cfg.get('spec_weight', 0.0) * spec_loss + \
                     self.cfg.get('stats_weight', 0.0) * stats_loss + \
                     self.cfg.get('edge_len_weight', 0.0) * edge_len_loss
        
        # 9) Backward and optimization step
        self.optim_G.zero_grad()
        
        if self.use_amp:
            # AMP: scaled backward pass
            self.amp_scaler.scale(loss_G).backward()
            self.amp_scaler.unscale_(self.optim_G)
            
            # Optional gradient clipping
            if self.cfg.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(
                    self.G.parameters(), 
                    self.cfg.get('max_grad_norm', 1.0)
                )
                
            self.amp_scaler.step(self.optim_G)
            self.amp_scaler.update()
        else:
            # Normal backward pass
            loss_G.backward()
            
            # Optional gradient clipping
            if self.cfg.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(
                    self.G.parameters(), 
                    self.cfg.get('max_grad_norm', 1.0)
                )
                
            self.optim_G.step()
        
        # 10) Convert all metrics to Python floats for consistent logging
        metrics = {
            'loss_G': float(loss_G.item()),
            'adv_loss': float(adv_loss.item()),
            'recon_loss': float(recon_loss.item()),
            'smooth_loss': float(smooth_loss.item()),
            'disp_norm': float(disp_norm_loss.item()),
            'spec_loss': float(spec_loss.item() if isinstance(spec_loss, torch.Tensor) else spec_loss),
            'edge_len_loss': float(edge_len_loss.item() if isinstance(edge_len_loss, torch.Tensor) else edge_len_loss),
            'stats_loss': float(stats_loss.item() if isinstance(stats_loss, torch.Tensor) else stats_loss)
        }
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate current models using a validation data loader.
        
        Args:
            val_loader: Data loader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.G.eval()
        self.D.eval()
        
        # Initialize metric accumulator
        metrics_accum = MetricsAccumulator()
        
        # Store all scores and labels for AUC calculation
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device and adapt to schema
                batch = self.adapt_batch_to_schema(batch)
                B = batch['X'].shape[0]
                
                # Generate fake samples
                z = self.sample_noise([B, self.z_dim], device=self.device)
                template_batch = self.select_template_for_batch(batch)
                fake_batch = self.G(z, template_batch)
                
                # Compute per-node reconstruction error
                per_node_err = self.compute_per_node_error(fake_batch, batch)
                
                # Compute other metrics
                recon_loss = self.loss_fns['recon'](
                    fake_batch, 
                    batch, 
                    mode=self.cfg.get('recon_mode', 'l1')
                )
                
                # Compute graph stats distance
                stats_diff = self.compute_graph_stats_distance(fake_batch, batch)
                
                # Discriminator scores for real and fake
                real_scores = self.D(batch).cpu()
                fake_scores = self.D(fake_batch).cpu()
                
                # Compute batch metrics
                batch_metrics = {
                    'loss_recon': float(recon_loss.item()),
                    'per_node_err_mean': float(per_node_err.mean().item()),
                    'per_node_err_max': float(per_node_err.max().item()),
                    'stats_diff': float(stats_diff.item()),
                    'mean_real_score': float(real_scores.mean().item()),
                    'mean_fake_score': float(fake_scores.mean().item()),
                }
                
                # If ground truth anomaly labels are available
                if 'labels' in batch:
                    labels = batch['labels'].cpu().numpy()
                    # Use max error per sample as anomaly score
                    scores = per_node_err.max(dim=1)[0].cpu().numpy()
                    
                    all_labels.append(labels)
                    all_scores.append(scores)
                
                metrics_accum.add(batch_metrics)
        
        # Compute average metrics
        avg_metrics = metrics_accum.average()
        
        # Compute ROC/AUC if ground truth labels are available
        if all_labels and all_scores:
            try:
                from sklearn.metrics import roc_auc_score
                labels = np.concatenate(all_labels)
                scores = np.concatenate(all_scores)
                auc = roc_auc_score(labels, scores)
                avg_metrics['auc'] = float(auc)
            except ImportError:
                print("Warning: sklearn not available for ROC/AUC calculation")
            except Exception as e:
                print(f"Error calculating ROC/AUC: {e}")
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str):
        """
        Save a checkpoint of the current training state.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint_dir = self.cfg.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Prepare checkpoint state
        state = {
            'G_state': self.G.state_dict(),
            'D_state': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val_metric': self.best_val_metric,
            'cfg': self.cfg
        }
        
        # Save schedulers if they exist
        if self.sched_G is not None:
            state['sched_G'] = self.sched_G.state_dict()
            state['sched_D'] = self.sched_D.state_dict()
        
        # Save checkpoint
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """
        Load a checkpoint to resume training.
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model states
        self.G.load_state_dict(checkpoint['G_state'])
        self.D.load_state_dict(checkpoint['D_state'])
        
        # Load optimizer states if requested
        if load_optimizer:
            self.optim_G.load_state_dict(checkpoint['optim_G'])
            self.optim_D.load_state_dict(checkpoint['optim_D'])
            
            # Load scheduler states if they exist
            if 'sched_G' in checkpoint and self.sched_G is not None:
                self.sched_G.load_state_dict(checkpoint['sched_G'])
                self.sched_D.load_state_dict(checkpoint['sched_D'])
        
        # Load training state
        self.step = checkpoint['step']
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
        
        print(f"Resumed from step {self.step}, epoch {self.epoch}")
    
    def sample_and_visualize(self, n_samples: int = 4, save_path: Optional[str] = None):
        """
        Generate samples and visualize them.
        
        Args:
            n_samples: Number of samples to generate
            save_path: Directory to save visualizations
        """
        self.G.eval()
        
        # Create save directory if needed
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        
        # Select templates
        templates = self.select_n_templates(n_samples)
        
        # Sample latent codes
        z = self.sample_noise([n_samples, self.z_dim], device=self.device)
        
        # Generate samples
        with torch.no_grad():
            fake_samples = self.G(z, template=templates)
        
        # Visualize each sample
        for i in range(n_samples):
            template = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.ndim > 1 else v 
                       for k, v in templates.items()}
            
            generated = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.ndim > 1 else v 
                        for k, v in fake_samples.items()}
            
            # Call visualization function
            self.visualize_overlay(
                template=template,
                generated=generated,
                real=None,  # No corresponding real sample for pure generation
                out_path=f"{save_path}/sample_{i}.png" if save_path else None
            )
    
    def adapt_batch_to_schema(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a batch to the expected schema, moving it to the correct device.
        
        Args:
            batch: Input batch
            
        Returns:
            Adapted batch on the correct device
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
                    # FIX: Ensure edge_index is of type torch.long
                    result[key] = value.long().to(self.device)
                else:
                    result[key] = value.to(self.device)
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                result[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
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
        
        # Check edge_index format (could be either [2,M] or [B,2,M])
        if result['edge_index'].ndim == 2:
            # Single graph edge_index [2,M]
            # FIX: Validate indices are within bounds
            N = result['X'].shape[1]
            if torch.max(result['edge_index']) >= N:
                raise ValueError(f"Edge index contains out-of-bounds indices: max={torch.max(result['edge_index'])}, N={N}")
                
        elif result['edge_index'].ndim == 3:
            # Batched edge_index [B,2,M]
            # Note: This requires special handling in loss functions
            pass
        else:
            raise ValueError(f"Unexpected edge_index shape: {result['edge_index'].shape}")
        
        return result
    
    def select_template_for_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an appropriate template for the given batch.
        
        This could be the batch itself (identity mapping), a fixed template,
        or a randomly selected template from a template dataset.
        
        Args:
            batch: Input batch
            
        Returns:
            Template batch matching input batch size
        """
        template_mode = self.cfg.get('template_mode', 'identity')
        B = batch['X'].shape[0]
        
        if template_mode == 'identity':
            # Use the batch itself as template (identity mapping)
            return batch
            
        elif template_mode == 'fixed':
            # Use a single fixed template for all samples
            template_id = self.cfg.get('template_id', 0)
            
            # Check if template is already cached
            if template_id not in self.template_cache:
                # Load template if template_loader is available
                if self.template_loader is not None:
                    for temp_batch in self.template_loader:
                        temp_batch = self.adapt_batch_to_schema(temp_batch)
                        # FIX: Store on CPU to save GPU memory
                        self.template_cache[template_id] = {
                            k: v.cpu() if isinstance(v, torch.Tensor) else v 
                            for k, v in temp_batch.items()
                        }
                        break
                else:
                    # Fallback to using the first batch as template
                    # FIX: Store on CPU to save GPU memory
                    self.template_cache[template_id] = {
                        k: v.cpu() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()
                    }
            
            # Replicate template to match batch size
            template = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.template_cache[template_id].items()
            }
            return self.replicate_template(template, B)
            
        elif template_mode == 'batch_specific':
            # Use template specified in batch metadata
            if 'meta' not in batch or 'template_id' not in batch['meta']:
                raise ValueError("template_mode='batch_specific' requires batch['meta']['template_id']")
                
            template_ids = batch['meta']['template_id']
            
            # Load and cache templates if needed
            templates = []
            for tid in template_ids:
                if tid not in self.template_cache and self.template_loader is not None:
                    # Need to load this template
                    # This is inefficient for online loading - in practice, preload all templates
                    for temp_batch in self.template_loader:
                        if temp_batch['meta']['id'] == tid:
                            temp_batch = self.adapt_batch_to_schema(temp_batch)
                            # FIX: Store on CPU to save GPU memory
                            self.template_cache[tid] = {
                                k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                for k, v in temp_batch.items()
                            }
                            break
                
                if tid in self.template_cache:
                    # FIX: Move to device before use
                    templates.append({
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in self.template_cache[tid].items()
                    })
                else:
                    # Fallback to using the batch itself
                    templates.append(batch)
            
            # Combine templates (advanced implementation would create a proper batch)
            # For simplicity, this implementation uses the first template replicated
            if templates:
                return self.replicate_template(templates[0], B)
            else:
                return batch
                
        elif template_mode == 'random':
            # Use a randomly selected template from template_loader
            if not hasattr(self, 'template_samples') or not self.template_samples:
                # Cache some template samples if we haven't already
                self.template_samples = []
                if self.template_loader is not None:
                    for temp_batch in self.template_loader:
                        # FIX: Store templates on CPU to save GPU memory
                        adapted_batch = self.adapt_batch_to_schema(temp_batch)
                        self.template_samples.append({
                            k: v.cpu() if isinstance(v, torch.Tensor) else v 
                            for k, v in adapted_batch.items()
                        })
                        if len(self.template_samples) >= 10:  # Cache up to 10 templates
                            break
                
                if not self.template_samples:
                    # Fallback to using the batch itself
                    self.template_samples = [{
                        k: v.cpu() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()
                    }]
            
            # FIX: Use random.choice instead of np.random.choice for list of dictionaries
            template_dict = random.choice(self.template_samples)
            
            # Move template to device and replicate
            template = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in template_dict.items()
            }
            return self.replicate_template(template, B)
            
        else:
            raise ValueError(f"Unknown template_mode: {template_mode}")
    
    def select_n_templates(self, n: int) -> Dict[str, Any]:
        """
        Select n templates for visualization or evaluation.
        
        Args:
            n: Number of templates to select
            
        Returns:
            Batch containing n templates
        """
        if not hasattr(self, 'template_samples') or not self.template_samples:
            # Initialize template cache if needed
            if self.template_loader is not None:
                self.template_samples = []
                for temp_batch in self.template_loader:
                    adapted_batch = self.adapt_batch_to_schema(temp_batch)
                    # FIX: Store on CPU to save GPU memory
                    self.template_samples.append({
                        k: v.cpu() if isinstance(v, torch.Tensor) else v 
                        for k, v in adapted_batch.items()
                    })
                    if len(self.template_samples) >= max(10, n):
                        break
            
            # Fallback if no templates available
            if not self.template_samples and self.train_loader:
                for batch in self.train_loader:
                    adapted_batch = self.adapt_batch_to_schema(batch)
                    # FIX: Store on CPU to save GPU memory
                    self.template_samples = [{
                        k: v.cpu() if isinstance(v, torch.Tensor) else v 
                        for k, v in adapted_batch.items()
                    }]
                    break
        
        # Select n templates (with replacement if needed)
        if not self.template_samples:
            raise ValueError("No templates available")
            
        templates = []
        # FIX: Use numpy for index selection but not for direct selection of dictionaries
        indices = np.random.choice(len(self.template_samples), size=n, replace=len(self.template_samples) < n)
        for idx in indices:
            # Move template to device
            template = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.template_samples[idx].items()
            }
            templates.append(template)
        
        # Combine templates into a single batch
        return self.combine_templates(templates)
    
    def replicate_template(self, template: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
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
    
    def combine_templates(self, templates: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        
        # FIX: Properly handle edge_index when combining templates
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
                        # FIX: Ensure indices are within bounds and convert to long
                        valid_edges = (offset_edge_index < node_counts[i]).all(dim=0)
                        if not valid_edges.all():
                            print(f"Warning: Found out-of-bounds indices in edge_index for template {i}")
                            offset_edge_index = offset_edge_index[:, valid_edges]
                        
                        offset_edge_index[0] += offset
                        offset_edge_index[1] += offset
                        combined_edges.append(offset_edge_index)
                    else:
                        # For more complex formats, just use the first template
                        # Real implementation would need to handle this properly
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

    def _compute_laplacian_eigenvalues(self, batch: Dict[str, Any], k: int) -> torch.Tensor:
        """
        Compute the k smallest eigenvalues of the graph Laplacian without materializing the full Laplacian matrix.
        
        Args:
            batch: Batch containing graph data
            k: Number of eigenvalues to compute
            
        Returns:
            Tensor of shape [B,k] containing smallest k eigenvalues
        """
        B = batch['X'].shape[0]
        device = batch['X'].device
        
        # Check if we should use CPU for computation (more memory efficient)
        use_cpu = self.cfg.get('use_cpu_for_eigenvalues', True)
        compute_device = torch.device('cpu') if use_cpu else device
        
        # Result tensor
        eigenvalues = torch.zeros(B, k, device=device)
        
        # Compute eigenvalues for each batch element
        for b in range(B):
            try:
                # Extract this batch's edge_index
                if batch['edge_index'].ndim == 3:
                    edge_index = batch['edge_index'][b]
                else:
                    edge_index = batch['edge_index']
                
                # Move to computation device
                if use_cpu:
                    edge_index = edge_index.cpu()
                
                # Convert edge_index to scipy sparse matrix format
                import scipy.sparse as sp
                import numpy as np
                
                # Convert to numpy arrays
                edge_index_np = edge_index.detach().numpy()
                N = batch['X'].shape[1]
                
                # Create sparse adjacency matrix
                rows, cols = edge_index_np[0], edge_index_np[1]
                
                # Ensure indices are valid
                valid_mask = (rows >= 0) & (rows < N) & (cols >= 0) & (cols < N)
                rows, cols = rows[valid_mask], cols[valid_mask]
                
                # Create adjacency matrix
                adj = sp.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(N, N))
                
                # Make it symmetric (undirected graph)
                adj = adj + adj.T
                adj.data = np.ones_like(adj.data)  # Binary adjacency
                
                # Convert to CSR format for efficient operations
                adj_csr = adj.tocsr()
                
                # Compute degree matrix
                degrees = np.array(adj_csr.sum(axis=1)).flatten()
                
                # Create Laplacian matrix L = D - A
                # For eigenvalue calculation, we don't need to materialize the full matrix
                # Instead, we can define a function that computes L @ x
                def laplacian_matvec(x):
                    # L @ x = D @ x - A @ x
                    return degrees * x - adj_csr @ x
                
                # Create a linear operator for the Laplacian
                from scipy.sparse.linalg import LinearOperator
                L_op = LinearOperator((N, N), matvec=laplacian_matvec)
                
                # Compute k smallest eigenvalues
                from scipy.sparse.linalg import eigsh
                evals, _ = eigsh(L_op, k=min(k, N-1), which='SM', tol=1e-3, maxiter=1000)
                
                # Sort eigenvalues
                evals = np.sort(evals)
                
                # Pad if needed
                if len(evals) < k:
                    evals = np.pad(evals, (0, k - len(evals)), mode='constant', constant_values=0)
                
                # Move back to PyTorch and the original device
                eigenvalues[b, :] = torch.tensor(evals[:k], device=device)
            
            except Exception as e:
                print(f"Error computing eigenvalues for batch element {b}: {e}")
                # Use placeholder eigenvalues
                eigenvalues[b, :] = torch.linspace(0, 1, k, device=device)
        
        return eigenvalues
    
    def compute_laplacian(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute k smallest eigenvalues of the graph Laplacian.
        
        Args:
            batch: Batch containing graph data
            
        Returns:
            Tensor of shape [B,k] containing k smallest eigenvalues
        """
        k = self.cfg.get('spec_k', 30)
        return self._compute_laplacian_eigenvalues(batch, k)
    
    def sample_noise(self, shape: List[int], device: torch.device) -> torch.Tensor:
        """
        Sample noise vector for generator input.
        
        Args:
            shape: Shape of noise tensor
            device: Device to create tensor on
            
        Returns:
            Noise tensor
        """
        noise_type = self.cfg.get('noise_type', 'normal')
        
        if noise_type == 'normal':
            return torch.randn(shape, device=device)
        elif noise_type == 'uniform':
            return torch.rand(shape, device=device) * 2 - 1
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def compute_per_node_error(self, generated: Dict[str, Any], reference: Dict[str, Any]) -> torch.Tensor:
        """
        Compute per-node reconstruction error between generated and reference batches.
        
        Args:
            generated: Generated batch
            reference: Reference batch
            
        Returns:
            Tensor of shape [B,N] with per-node errors
        """
        # Extract coordinates
        coords_gen = generated['X'][..., :3]  # [B,N,3]
        coords_ref = reference['X'][..., :3]
        
        # Compute L2 distance for each node
        per_node_err = torch.norm(coords_gen - coords_ref, dim=-1)  # [B,N]
        
        return per_node_err
    
    def compute_graph_stats_distance(self, generated: Dict[str, Any], reference: Dict[str, Any]) -> torch.Tensor:
        """
        Compute distance between graph statistics of generated and reference batches.
        
        Args:
            generated: Generated batch
            reference: Reference batch
            
        Returns:
            Scalar tensor with graph statistics distance
        """
        # Compute edge lengths
        l_gen = self.compute_edge_lengths(generated)  # [B,M]
        l_ref = self.compute_edge_lengths(reference)  # [B,M]
        
        # Compute distances between distributions
        dist = self.loss_fns['stats'](l_gen.flatten(), l_ref.flatten())
        
        return dist
    
    def compute_edge_lengths(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute lengths of all edges in the graph.
        
        Args:
            batch: Batch containing 'X' and 'edge_index'
            
        Returns:
            Tensor of shape [B,M] with edge lengths
        """
        coords = batch['X'][..., :3]  # [B,N,3]
        edge_index = batch['edge_index']  # [2,M] or [B,2,M]
        device = coords.device
        B, N = coords.shape[:2]
        
        if edge_index.ndim == 2:
            # Non-batched edge_index [2,M]
            # FIX: Convert to long and ensure indices are within bounds
            src, dst = edge_index[0].long(), edge_index[1].long()
            
            # Validate indices
            valid_edges = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
            if not valid_edges.all():
                # Filter out invalid edges
                src = src[valid_edges]
                dst = dst[valid_edges]
            
            # Extract coordinates for both endpoints [B,M,3]
            src_coords = coords[:, src, :]
            dst_coords = coords[:, dst, :]
            
            # Compute edge lengths [B,M]
            lengths = torch.norm(src_coords - dst_coords, dim=-1)
            
        elif edge_index.ndim == 3:
            # Batched edge_index [B,2,M]
            lengths = []
            
            for b in range(B):
                # FIX: Convert to long and ensure indices are within bounds
                src = edge_index[b, 0].long()
                dst = edge_index[b, 1].long()
                
                # Validate indices
                valid_edges = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
                if not valid_edges.all():
                    # Filter out invalid edges
                    src = src[valid_edges]
                    dst = dst[valid_edges]
                
                src_coords = coords[b, src]  # [M,3]
                dst_coords = coords[b, dst]  # [M,3]
                batch_lengths = torch.norm(src_coords - dst_coords, dim=-1)  # [M]
                lengths.append(batch_lengths)
                
            # FIX: Handle the case where some batches might have no valid edges
            if not lengths:
                return torch.zeros(B, 1, device=device)
                
            # Pad lengths to same size if needed
            max_len = max(len(l) for l in lengths)
            if not all(len(l) == max_len for l in lengths):
                lengths = [torch.cat([l, torch.zeros(max_len - len(l), device=device)]) for l in lengths]
                
            lengths = torch.stack(lengths)  # [B,M]
            
        else:
            raise ValueError(f"Unexpected edge_index shape: {edge_index.shape}")
            
        return lengths
    
    def visualize_overlay(self, template, generated, real=None, out_path=None):
        """
        Visualize overlay of template, generated, and real samples.
        
        Args:
            template: Template batch
            generated: Generated batch
            real: Optional real batch for comparison
            out_path: Path to save visualization
        """
        # This is a placeholder - you would implement actual visualization
        # based on your specific needs and visualization libraries
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            
            # Extract coordinates
            template_coords = template['X'][0, :, :3].cpu().numpy()
            generated_coords = generated['X'][0, :, :3].cpu().numpy()
            
            # Create 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot template points
            ax.scatter(
                template_coords[:, 0], 
                template_coords[:, 1], 
                template_coords[:, 2],
                color='blue', 
                alpha=0.5, 
                label='Template'
            )
            
            # Plot generated points
            ax.scatter(
                generated_coords[:, 0], 
                generated_coords[:, 1], 
                generated_coords[:,