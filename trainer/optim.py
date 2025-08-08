import torch
from typing import Dict, Any, Optional, Union


def create_optimizers_and_schedulers(
    G: torch.nn.Module, 
    D: torch.nn.Module, 
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create optimizers and learning rate schedulers for generator and discriminator.
    
    Args:
        G: Generator model
        D: Discriminator model
        cfg: Configuration dictionary with optimizer settings
        
    Returns:
        Dictionary containing optimizers and schedulers
    """
    # Extract hyperparameters
    lr_G = cfg.get('lr_G', 1e-4)
    lr_D = cfg.get('lr_D', 1e-4)
    betas = cfg.get('betas', (0.5, 0.9))
    weight_decay = cfg.get('weight_decay', 0)
    
    # Create optimizers
    optim_type = cfg.get('optimizer_type', 'adam').lower()
    
    if optim_type == 'adam':
        optim_G = torch.optim.Adam(
            G.parameters(), 
            lr=lr_G, 
            betas=betas, 
            weight_decay=weight_decay
        )
        optim_D = torch.optim.Adam(
            D.parameters(), 
            lr=lr_D, 
            betas=betas, 
            weight_decay=weight_decay
        )
    elif optim_type == 'adamw':
        optim_G = torch.optim.AdamW(
            G.parameters(), 
            lr=lr_G, 
            betas=betas, 
            weight_decay=weight_decay
        )
        optim_D = torch.optim.AdamW(
            D.parameters(), 
            lr=lr_D, 
            betas=betas, 
            weight_decay=weight_decay
        )
    elif optim_type == 'rmsprop':
        optim_G = torch.optim.RMSprop(
            G.parameters(), 
            lr=lr_G, 
            weight_decay=weight_decay
        )
        optim_D = torch.optim.RMSprop(
            D.parameters(), 
            lr=lr_D, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")
    
    # Create schedulers if configured
    result = {
        'optim_G': optim_G,
        'optim_D': optim_D
    }
    
    if 'lr_schedule' in cfg:
        sched_cfg = cfg['lr_schedule']
        sched_type = sched_cfg.get('type', '').lower()
        
        if sched_type == 'step':
            result['sched_G'] = torch.optim.lr_scheduler.StepLR(
                optim_G, 
                step_size=sched_cfg.get('step_size', 30), 
                gamma=sched_cfg.get('gamma', 0.1)
            )
            result['sched_D'] = torch.optim.lr_scheduler.StepLR(
                optim_D, 
                step_size=sched_cfg.get('step_size', 30), 
                gamma=sched_cfg.get('gamma', 0.1)
            )
            
        elif sched_type == 'cosine':
            result['sched_G'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_G, 
                T_max=sched_cfg.get('T_max', 100),
                eta_min=sched_cfg.get('eta_min', 0)
            )
            result['sched_D'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_D, 
                T_max=sched_cfg.get('T_max', 100),
                eta_min=sched_cfg.get('eta_min', 0)
            )
            
        elif sched_type == 'plateau':
            result['sched_G'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_G,
                mode=sched_cfg.get('mode', 'min'),
                factor=sched_cfg.get('factor', 0.1),
                patience=sched_cfg.get('patience', 10),
                verbose=True
            )
            result['sched_D'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_D,
                mode=sched_cfg.get('mode', 'min'),
                factor=sched_cfg.get('factor', 0.1),
                patience=sched_cfg.get('patience', 10),
                verbose=True
            )
            
        elif sched_type == 'warmup_cosine':
            # Linear warmup followed by cosine annealing
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            # Create custom schedulers with warmup
            class WarmupCosineScheduler:
                def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
                    self.warmup_epochs = warmup_epochs
                    self.max_epochs = max_epochs
                    self.cosine_scheduler = CosineAnnealingLR(
                        optimizer, 
                        T_max=max_epochs - warmup_epochs,
                        eta_min=eta_min
                    )
                    self.optimizer = optimizer
                    self.eta_min = eta_min
                    self.last_epoch = -1
                    self.base_lrs = [group['lr'] for group in optimizer.param_groups]
                    
                def step(self):
                    self.last_epoch += 1
                    
                    if self.last_epoch < self.warmup_epochs:
                        # Linear warmup
                        for i, group in enumerate(self.optimizer.param_groups):
                            group['lr'] = self.base_lrs[i] * (self.last_epoch / self.warmup_epochs)
                    else:
                        # Cosine annealing
                        self.cosine_scheduler.step()
                        
                def state_dict(self):
                    state = {
                        'last_epoch': self.last_epoch,
                        'base_lrs': self.base_lrs,
                    }
                    if self.last_epoch >= self.warmup_epochs:
                        state['cosine_state'] = self.cosine_scheduler.state_dict()
                    return state
                    
                def load_state_dict(self, state_dict):
                    self.last_epoch = state_dict['last_epoch']
                    self.base_lrs = state_dict['base_lrs']
                    if self.last_epoch >= self.warmup_epochs:
                        self.cosine_scheduler.load_state_dict(state_dict['cosine_state'])
            
            # Create warmup cosine schedulers
            warmup_epochs = sched_cfg.get('warmup_epochs', 10)
            max_epochs = sched_cfg.get('max_epochs', 100)
            eta_min = sched_cfg.get('eta_min', 0)
            
            result['sched_G'] = WarmupCosineScheduler(
                optim_G, 
                warmup_epochs=warmup_epochs, 
                max_epochs=max_epochs,
                eta_min=eta_min
            )
            result['sched_D'] = WarmupCosineScheduler(
                optim_D, 
                warmup_epochs=warmup_epochs, 
                max_epochs=max_epochs,
                eta_min=eta_min
            )
    
    return result