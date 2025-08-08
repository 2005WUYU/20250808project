import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from tqdm import tqdm
import logging

class Trainer:
    def __init__(self, generator, discriminator, opt_g, opt_d, dataloader, template_graph, config):
        """
        初始化训练器
        
        Args:
            generator: 生成器模型
            discriminator: 判别器模型
            opt_g: 生成器优化器
            opt_d: 判别器优化器
            dataloader: 数据加载器
            template_graph: 模板图结构
            config: 配置字典，包含训练参数
        """
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.dataloader = dataloader
        self.template_graph = template_graph
        self.config = config
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        计算梯度惩罚项
        """
        batch_size = real_samples.num_graphs
        # 创建随机插值系数
        alpha = torch.rand(batch_size, 1).to(real_samples.x.device)
        
        # 在真实和生成样本之间进行插值
        interpolates = real_samples.clone()
        interpolates.x = alpha * real_samples.x + (1 - alpha) * fake_samples.x
        interpolates.requires_grad_(True)
        
        # 计算判别器对插值样本的输出
        d_interpolates = self.discriminator(interpolates)
        
        # 计算梯度
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates.x,
            grad_outputs=torch.ones_like(d_interpolates).to(real_samples.x.device),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 计算梯度范数
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def compute_reconstruction_loss(self, fake_batch, target_batch):
        """
        计算重建损失
        """
        # 这里使用MSE损失作为重建损失的示例
        # 具体实现可能需要根据实际图结构定制
        return nn.MSELoss()(fake_batch.x, target_batch.x)
    
    def train_epoch(self):
        """
        训练一个epoch
        """
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_d_real = []
        epoch_d_fake = []
        
        for real_batch in self.dataloader:
            batch_size = real_batch.num_graphs
            real_batch = real_batch.to(next(self.generator.parameters()).device)
            
            # ==================== 训练判别器 ====================
            for _ in range(self.config['ncritic']):
                self.opt_d.zero_grad()
                
                # 生成假样本
                z = torch.randn(batch_size, self.config['latent_dim']).to(real_batch.x.device)
                fake_batch = self.generator(z, self.template_graph).detach()
                
                # 计算判别器输出
                d_real = self.discriminator(real_batch)
                d_fake = self.discriminator(fake_batch)
                
                # 计算梯度惩罚
                gp = self.compute_gradient_penalty(real_batch, fake_batch)
                
                # 计算判别器损失
                loss_d = d_fake.mean() - d_real.mean() + self.config['lambda_gp'] * gp
                
                # 反向传播和优化
                loss_d.backward()
                self.opt_d.step()
                
                epoch_d_losses.append(loss_d.item())
                epoch_d_real.append(d_real.mean().item())
                epoch_d_fake.append(d_fake.mean().item())
            
            # ==================== 训练生成器 ====================
            self.opt_g.zero_grad()
            
            # 生成新的假样本
            z = torch.randn(batch_size, self.config['latent_dim']).to(real_batch.x.device)
            fake_batch_for_g = self.generator(z, self.template_graph)
            
            # 计算对抗损失
            loss_adv = -self.discriminator(fake_batch_for_g).mean()
            
            # 计算重建损失
            target_batch = real_batch  # 使用当前batch作为目标
            loss_recon = self.compute_reconstruction_loss(fake_batch_for_g, target_batch)
            
            # 计算总损失
            loss_g = loss_adv + self.config['lambda_recon'] * loss_recon
            
            # 反向传播和优化
            loss_g.backward()
            self.opt_g.step()
            
            epoch_g_losses.append(loss_g.item())
        
        # 返回平均损失
        return {
            'loss_d': np.mean(epoch_d_losses),
            'loss_g': np.mean(epoch_g_losses),
            'd_real': np.mean(epoch_d_real),
            'd_fake': np.mean(epoch_d_fake)
        }
    
    def train(self, num_epochs):
        """
        训练指定数量的epoch
        
        Args:
            num_epochs: 要训练的epoch数量
        """
        for epoch in tqdm(range(num_epochs)):
            self.generator.train()
            self.discriminator.train()
            
            metrics = self.train_epoch()
            
            # 记录训练指标
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"D_loss: {metrics['loss_d']:.4f}, "
                f"G_loss: {metrics['loss_g']:.4f}, "
                f"D(x): {metrics['d_real']:.4f}, "
                f"D(G(z)): {metrics['d_fake']:.4f}"
            )