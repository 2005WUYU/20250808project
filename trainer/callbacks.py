import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
import time

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


class Callback:
    """Base class for all callbacks."""
    
    def __init__(self, cfg: Dict[str, Any] = None):
        self.cfg = cfg or {}
    
    def on_training_begin(self, trainer):
        """Called at the beginning of training."""
        pass
        
    def on_training_end(self, trainer):
        """Called at the end of training."""
        pass
        
    def on_epoch_begin(self, trainer, epoch):
        """Called at the beginning of an epoch."""
        pass
        
    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        """Called at the end of an epoch."""
        pass
        
    def on_batch_begin(self, trainer, batch):
        """Called at the beginning of a batch."""
        pass
        
    def on_batch_end(self, trainer, batch_metrics):
        """Called at the end of a batch."""
        pass


class CallbackRunner:
    """Manages and executes multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []
        self.metrics_accumulator = MetricsAccumulator()
    
    def get_metrics_accumulator(self) -> MetricsAccumulator:
        """Get the metrics accumulator instance."""
        return self.metrics_accumulator
        
    def on_training_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_training_begin(trainer)
            
    def on_training_end(self, trainer):
        for callback in self.callbacks:
            callback.on_training_end(trainer)
            
    def on_epoch_begin(self, trainer, epoch):
        # Reset metrics accumulator at the start of each epoch
        self.metrics_accumulator.reset()
        
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
            
    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, train_metrics, val_metrics)
            
    def on_batch_begin(self, trainer, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch)
            
    def on_batch_end(self, trainer, batch_metrics):
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_metrics)

    def save_checkpoint(self, trainer, filename: str):
        """Save a checkpoint of current training state."""
        checkpoint_dir = trainer.cfg.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Prepare checkpoint state
        state = {
            'G_state': trainer.G.state_dict(),
            'D_state': trainer.D.state_dict(),
            'optim_G': trainer.optim_G.state_dict(),
            'optim_D': trainer.optim_D.state_dict(),
            'step': trainer.step,
            'epoch': trainer.epoch,
            'best_val_metric': trainer.best_val_metric,
            'cfg': trainer.cfg
        }
        
        # Save schedulers if they exist
        if trainer.sched_G is not None:
            state['sched_G'] = trainer.sched_G.state_dict()
            state['sched_D'] = trainer.sched_D.state_dict()
        
        # Save checkpoint
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, trainer, path: str, load_optimizer: bool = True):
        """Load a checkpoint to resume training."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=trainer.device)
        
        # Load model states
        trainer.G.load_state_dict(checkpoint['G_state'])
        trainer.D.load_state_dict(checkpoint['D_state'])
        
        # Load optimizer states if requested
        if load_optimizer:
            trainer.optim_G.load_state_dict(checkpoint['optim_G'])
            trainer.optim_D.load_state_dict(checkpoint['optim_D'])
            
            # Load scheduler states if they exist
            if 'sched_G' in checkpoint and trainer.sched_G is not None:
                trainer.sched_G.load_state_dict(checkpoint['sched_G'])
                trainer.sched_D.load_state_dict(checkpoint['sched_D'])
        
        # Load training state
        trainer.step = checkpoint['step']
        trainer.epoch = checkpoint.get('epoch', 0)
        trainer.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
        
        print(f"Resumed from step {trainer.step}, epoch {trainer.epoch}")


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints during training.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.save_every = cfg.get('save_every_epochs', 10)
        self.checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints')
        self.save_best = cfg.get('save_best', True)
        self.best_metric_name = cfg.get('best_metric_name', 'loss_recon')
        self.metric_mode = cfg.get('metric_mode', 'min')  # 'min' or 'max'
        self.best_metric = float('inf') if self.metric_mode == 'min' else float('-inf')
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        # Save checkpoint at regular intervals
        if trainer.epoch % self.save_every == 0:
            filename = f"epoch_{trainer.epoch}.pth"
            path = os.path.join(self.checkpoint_dir, filename)
            self._save_checkpoint(trainer, path)
        
        # Save best checkpoint if validation metrics improved
        if self.save_best and val_metrics:
            current_metric = val_metrics.get(self.best_metric_name)
            if current_metric is not None:
                is_improved = False
                
                if self.metric_mode == 'min':
                    is_improved = current_metric < self.best_metric
                else:
                    is_improved = current_metric > self.best_metric
                    
                if is_improved:
                    print(f"{self.best_metric_name} improved from {self.best_metric:.6f} to {current_metric:.6f}")
                    self.best_metric = current_metric
                    path = os.path.join(self.checkpoint_dir, "best.pth")
                    self._save_checkpoint(trainer, path)
    
    def on_training_end(self, trainer):
        # Save final checkpoint
        path = os.path.join(self.checkpoint_dir, "final.pth")
        self._save_checkpoint(trainer, path)
        
    def _save_checkpoint(self, trainer, path: str):
        """Save model checkpoint."""
        state = {
            'G_state': trainer.G.state_dict(),
            'D_state': trainer.D.state_dict(),
            'optim_G': trainer.optim_G.state_dict(),
            'optim_D': trainer.optim_D.state_dict(),
            'step': trainer.step,
            'epoch': trainer.epoch,
            'best_val_metric': self.best_metric,
            'cfg': trainer.cfg
        }
        
        # Save schedulers if they exist
        if trainer.sched_G is not None:
            state['sched_G'] = trainer.sched_G.state_dict()
            state['sched_D'] = trainer.sched_D.state_dict()
        
        torch.save(state, path)
        print(f"Checkpoint saved to {path}")


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on validation metrics.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.patience = cfg.get('early_stopping_patience', 10)
        self.metric_name = cfg.get('early_stopping_metric', 'loss_recon')
        self.metric_mode = cfg.get('metric_mode', 'min')  # 'min' or 'max'
        self.best_metric = float('inf') if self.metric_mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
    
    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        # Skip if no validation metrics
        if not val_metrics or self.metric_name not in val_metrics:
            return
            
        current_metric = val_metrics[self.metric_name]
        
        # Check if metric improved
        if self.metric_mode == 'min':
            is_improved = current_metric < self.best_metric
        else:
            is_improved = current_metric > self.best_metric
            
        if is_improved:
            # Reset counter and update best metric
            self.best_metric = current_metric
            self.counter = 0
        else:
            # Increment counter
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            # Check if we should stop
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after {self.counter} epochs without improvement")
                # Terminate training loop
                trainer.early_stop = True


class VisualizationCallback(Callback):
    """
    Callback for generating and saving visualizations during training.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.visualize_every = cfg.get('visualize_every_epochs', 5)
        self.n_samples = cfg.get('n_vis_samples', 4)
        self.output_dir = cfg.get('output_dir', 'output')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        # Generate and save visualizations at regular intervals
        if trainer.epoch % self.visualize_every == 0:
            save_path = os.path.join(self.output_dir, f'epoch_{trainer.epoch}')
            self._generate_and_save_samples(trainer, save_path)
    
    def on_training_end(self, trainer):
        # Generate final visualizations
        save_path = os.path.join(self.output_dir, 'final')
        self._generate_and_save_samples(trainer, save_path)
        
    def _generate_and_save_samples(self, trainer, save_path: str):
        """Generate and save sample visualizations."""
        # Set model to eval mode
        trainer.G.eval()
        
        # Create save directory if needed
        os.makedirs(save_path, exist_ok=True)
        
        # Get templates
        from trainer.data_loop import select_n_templates
        templates = select_n_templates(
            self.n_samples, 
            trainer.device,
            trainer.template_loader, 
            trainer.template_cache
        )
        
        # Sample latent codes
        z = trainer.sample_noise([self.n_samples, trainer.z_dim], device=trainer.device)
        
        # Generate samples
        with torch.no_grad():
            fake_samples = trainer.G(z, template=templates)
        
        # Visualize each sample
        for i in range(self.n_samples):
            template = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.ndim > 1 else v 
                       for k, v in templates.items()}
            
            generated = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.ndim > 1 else v 
                        for k, v in fake_samples.items()}
            
            # Call visualization function
            from eval.metrics_hooks import visualize_overlay
            visualize_overlay(
                template=template,
                generated=generated,
                real=None,
                out_path=f"{save_path}/sample_{i}.png"
            )
            
        # Set model back to train mode
        trainer.G.train()