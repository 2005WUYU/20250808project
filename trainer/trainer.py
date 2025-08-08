import os
import torch
from torch import Tensor
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import copy
import time

from trainer.optim import create_optimizers_and_schedulers
from trainer.data_loop import DataLoopAdapter
from trainer.callbacks import (
    CallbackRunner,
    CheckpointCallback,
    EarlyStoppingCallback,
    VisualizationCallback
)
from eval.evaluator import Evaluator
from logs.logger import Logger


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
                 logger: Optional[Logger] = None,
                 callbacks: Optional[List] = None):
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
            callbacks: List of callbacks for training events
        """
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.G = G.to(self.device)
        self.D = D.to(self.device)
        
        # Create data adapters
        self.data_adapter = DataLoopAdapter(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.template_loader = template_loader
        
        # Set default logger if none provided
        self.logger = logger if logger is not None else Logger()
        
        # Create evaluator
        self.evaluator = Evaluator(cfg, device=device)
        
        # Setup optimizers and loss functions
        self._setup_training_components()
        
        # Initialize training state
        self.step = 0
        self.epoch = 0
        self.best_val_metric = float('inf')  # Can be changed based on metric direction
        
        # Get z dimension from config
        self.z_dim = cfg.get('z_dim', 128)
        
        # Template cache for efficiency (stored on CPU to save GPU memory)
        self.template_cache = {}
        
        # Setup callbacks
        self.callbacks = CallbackRunner(callbacks if callbacks else self._default_callbacks())
        
    def _default_callbacks(self) -> List:
        """Create default callbacks if none provided."""
        return [
            CheckpointCallback(self.cfg),
            EarlyStoppingCallback(self.cfg),
            VisualizationCallback(self.cfg)
        ]
        
    def _setup_training_components(self):
        """Setup optimizers, schedulers, and loss functions."""
        # Create optimizers and schedulers
        optim_results = create_optimizers_and_schedulers(
            self.G, self.D, self.cfg
        )
        self.optim_G = optim_results['optim_G']
        self.optim_D = optim_results['optim_D']
        self.sched_G = optim_results.get('sched_G')
        self.sched_D = optim_results.get('sched_D')
        
        # Setup AMP if configured
        self.use_amp = self.cfg.get('use_amp', False)
        self.amp_scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Setup loss functions - these will be handled by the training steps
        # Since losses were implemented in separate module
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        print(f"Starting training for {num_epochs} epochs")
        start_epoch = self.epoch  # Store starting epoch for correct display
        total_epochs = start_epoch + num_epochs
        
        # Initialize callbacks
        self.callbacks.on_training_begin(self)
        
        for epoch in range(self.epoch, total_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch+1}/{total_epochs}")
            
            # Train for one epoch
            epoch_start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_duration = time.time() - epoch_start_time
            
            # Log training metrics
            train_metrics['epoch_time'] = epoch_duration
            self.logger.log_metrics('train', train_metrics, epoch)
            
            # Update learning rate schedulers if used
            if self.sched_G is not None:
                self.sched_G.step()
                self.sched_D.step()
                
                # Log learning rates
                self.logger.log_scalar('lr_G', self.optim_G.param_groups[0]['lr'], epoch)
                self.logger.log_scalar('lr_D', self.optim_D.param_groups[0]['lr'], epoch)
            
            # Run validation if a validation loader was provided
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.evaluator.evaluate(
                    self.G, self.D, self.val_loader, self.data_adapter,
                    template_loader=self.template_loader,
                    template_cache=self.template_cache
                )
                self.logger.log_metrics('val', val_metrics, epoch)
            
            # Run callbacks for epoch end
            self.callbacks.on_epoch_end(self, train_metrics, val_metrics)
        
        # Run callbacks for training end
        self.callbacks.on_training_end(self)
        print("Training completed!")
    
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
        
        # Use MetricsAccumulator from callback
        metrics_accumulator = self.callbacks.get_metrics_accumulator()
        
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
        
        # Notify callbacks about epoch start
        self.callbacks.on_epoch_begin(self, epoch)
        
        # Main training loop
        for batch_idx, raw_batch in enumerate(self.train_loader):
            # Move batch to device and adapt to expected schema
            real_batch = self.data_adapter.adapt_batch(raw_batch)
            
            # Notify callbacks about batch start
            self.callbacks.on_batch_begin(self, real_batch)
            
            # Train discriminator for d_steps
            for d_iter in range(d_steps):
                metrics_D = self.train_step_D(real_batch)
                metrics_accumulator.add(metrics_D)
                self.step += 1
            
            # Train generator for g_steps
            for g_iter in range(g_steps):
                metrics_G = self.train_step_G(real_batch)
                metrics_accumulator.add(metrics_G)
            
            # Log periodically
            if self.step % log_interval == 0:
                recent_metrics = metrics_accumulator.recent()
                self.logger.log_metrics('training', recent_metrics, self.step)
            
            # Notify callbacks about batch end
            self.callbacks.on_batch_end(self, metrics_accumulator.recent())
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                if self.step % 10 == 0:  # Update stats every 10 steps
                    pbar.set_postfix(metrics_accumulator.recent())
        
        # Clean up progress bar
        if pbar is not None:
            pbar.close()
        
        # Return average metrics for the epoch
        return metrics_accumulator.average()
    
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
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            real_scores = self.D(real_batch)
            fake_scores = self.D(fake_batch)
            
            # Ensure scores have shape [B]
            if real_scores.ndim > 1:
                real_scores = real_scores.view(-1)
            if fake_scores.ndim > 1:
                fake_scores = fake_scores.view(-1)
            
            # 5) Import and compute critic loss
            from losses.adversarial import critic_loss
            loss_basic = critic_loss(real_scores, fake_scores)
        
        # 6) Compute gradient penalty outside of autocast context to avoid numerical issues
        with torch.cuda.amp.autocast(enabled=False):
            from losses.utils_losses import compute_gradient_penalty
            gp = compute_gradient_penalty(
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
            from losses.adversarial import estimate_wasserstein_distance
            w_dist = estimate_wasserstein_distance(real_scores, fake_scores)
            
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
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            fake_batch = self.G(z=z, template=template_batch)
            
            # 4) Forward through discriminator
            fake_scores_for_G = self.D(fake_batch)
            
            # Ensure scores have shape [B]
            if fake_scores_for_G.ndim > 1:
                fake_scores_for_G = fake_scores_for_G.view(-1)
            
            # 5) Import and compute adversarial loss
            from losses.adversarial import generator_adversarial_loss
            adv_loss = generator_adversarial_loss(fake_scores_for_G)
            
            # 6) Compute reconstruction loss
            from losses.reconstruction import reconstruction_loss
            recon_loss = reconstruction_loss(
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
            from losses.regularizers import displacement_norm_loss
            disp_norm_loss = displacement_norm_loss(
                disp, 
                p=self.cfg.get('disp_norm_p', 2),
                reduction='mean'
            )
            
            # Smoothness loss
            from losses.regularizers import smoothness_loss
            smooth_loss = smoothness_loss(
                disp, 
                real_batch['edge_index']
            )
            
            # Spectral loss
            spec_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.get('spec_weight', 0.0) > 0:
                try:
                    from eval.metrics_hooks import compute_spectral_loss
                    spec_loss = compute_spectral_loss(
                        fake_batch,
                        template_batch,
                        k=self.cfg.get('spec_k', 30),
                        mode=self.cfg.get('spec_mode', 'l2')
                    )
                except Exception as e:
                    print(f"Warning: Error in spectral loss calculation: {e}")
            
            # Edge length consistency loss
            edge_len_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.get('edge_len_weight', 0.0) > 0:
                from losses.regularizers import edge_length_consistency_loss
                edge_len_loss = edge_length_consistency_loss(
                    coords_gen=fake_batch['X'][..., :3],
                    coords_ref=real_batch['X'][..., :3],
                    edge_index=real_batch['edge_index'],
                    mode=self.cfg.get('edge_len_mode', 'relative')
                )
            
            # Stats distributional loss
            stats_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.get('stats_weight', 0.0) > 0:
                from eval.metrics_hooks import compute_distribution_loss
                stats_loss = compute_distribution_loss(fake_batch, real_batch)
            
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
    
    def select_template_for_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an appropriate template for the given batch.
        This method delegates to the data adapter.
        """
        from trainer.data_loop import select_template_for_batch
        return select_template_for_batch(
            batch, 
            self.device,
            self.template_loader, 
            self.template_cache,
            self.cfg.get('template_mode', 'identity')
        )

    def save_checkpoint(self, filename: str):
        """Save a checkpoint of current training state."""
        self.callbacks.save_checkpoint(self, filename)
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load a checkpoint to resume training."""
        self.callbacks.load_checkpoint(self, path, load_optimizer)