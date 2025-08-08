# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import itertools

# --- 辅助函数和类的定义 (在实际项目中，这些会是独立的复杂模块) ---

class DummyGraphBatch:
    """一个模拟的图数据批次，用于演示。"""
    def __init__(self, data, num_graphs):
        self.data = data
        self.num_graphs = num_graphs
        # 在 PyTorch Geometric 中，这会是一个 Batch 对象，包含 x, edge_index, batch 等属性
        self.x = data # 假设节点特征就是数据本身
        self.edge_index = None # 简化处理，不包含边信息

    def to(self, device):
        self.data = self.data.to(device)
        return self

class DummyModel(nn.Module):
    """一个模拟的生成器或判别器。"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, template_graph=None):
        # 模板图参数是为了匹配接口，这里简化了，没有使用它
        return self.fc(x)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """计算 WGAN-GP 的梯度惩罚。"""
    # 随机权重
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # 获取真实样本和生成样本之间的插值
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # 计算判别器对插值的输出
    d_interpolated = discriminator(interpolated)
    
    # 计算梯度
    grad_outputs = torch.ones(d_interpolated.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # 计算梯度的 L2 范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_reconstruction_loss(fake_batch, target_batch):
    """
    计算重建损失（或分布匹配损失）的占位符。
    这是一个关键的、需要根据具体任务定制的函数。
    
    例如，我们可以比较节点特征的均值和标准差。
    """
    # 示例: 比较两批图的节点特征均值的 L1 距离
    # 这是一个非常简化的例子！
    mean_fake = fake_batch.mean(dim=0)
    mean_target = target_batch.mean(dim=0)
    loss = torch.abs(mean_fake - mean_target).mean()
    return loss

# --- 核心训练器类 ---

class Trainer:
    """
    封装完整的训练循环逻辑，协调模型、数据和损失函数。
    """
    def __init__(self, generator, discriminator, opt_g, opt_d, dataloader, template_graph, config, device):
        """
        初始化训练器。
        
        Args:
            generator (nn.Module): 生成器模型。
            discriminator (nn.Module): 判别器模型。
            opt_g (optim.Optimizer): 生成器的优化器。
            opt_d (optim.Optimizer): 判别器的优化器。
            dataloader (DataLoader): 训练数据加载器。
            template_graph: 用于生成过程的模板图 (可能为 None)。
            config (dict): 包含超参数的字典 (ncritic, lambda_gp, etc.)。
            device (torch.device): 训练设备 (e.g., 'cuda' or 'cpu')。
        """
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.dataloader = dataloader
        self.template_graph = template_graph
        self.config = config
        self.device = device

        # 将模型移动到指定设备
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def train(self, num_epochs: int):
        """主训练循环，按指定的 epoch 数进行训练。"""
        print("--- 开始训练 ---")
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            epoch_logs = self.train_epoch()
            
            # 打印日志
            log_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_logs.items()])
            print(log_str)
        print("\n--- 训练完成 ---")

    def train_epoch(self):
        """执行一个完整的 epoch 训练。"""
        # 使用 tqdm 创建进度条
        progress_bar = tqdm(self.dataloader, desc="Epoch Progress")
        
        # 用于记录整个 epoch 的损失
        epoch_loss_g = []
        epoch_loss_d = []
        epoch_d_real = []
        epoch_d_fake = []

        # 使用 itertools.cycle 避免数据耗尽问题
        recon_dataloader_iter = itertools.cycle(self.dataloader)

        for real_batch in progress_bar:
            # 实际项目中，real_batch 是一个图批次对象，这里我们用 DummyGraphBatch 模拟
            # 将数据移动到设备
            real_data = real_batch.data.to(self.device)
            batch_size = real_batch.num_graphs

            # --- 步骤 1: 训练判别器 (ncritic 次) ---
            d_loss_items = []
            for _ in range(self.config['ncritic']):
                self.opt_d.zero_grad()

                # a. 生成假样本
                z = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
                # .detach() 是关键，梯度不回传给生成器
                with torch.no_grad():
                    fake_data = self.generator(z, self.template_graph)

                # b. 计算判别器输出
                d_real = self.discriminator(real_data)
                d_fake = self.discriminator(fake_data)

                # c. 计算梯度惩罚
                gradient_penalty = compute_gradient_penalty(
                    self.discriminator, real_data, fake_data, self.device
                )

                # d. 计算判别器损失 (WGAN-GP loss)
                loss_d = (
                    d_fake.mean()
                    - d_real.mean()
                    + self.config['lambda_gp'] * gradient_penalty
                )

                # e. 反向传播和优化
                loss_d.backward()
                self.opt_d.step()
                d_loss_items.append(loss_d.item())
            
            avg_loss_d = sum(d_loss_items) / len(d_loss_items)

            # --- 步骤 2: 训练生成器 (1 次) ---
            self.opt_g.zero_grad()

            # a. 生成新的假样本 (这次需要计算梯度)
            z = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
            fake_data_for_g = self.generator(z, self.template_graph)

            # b. 计算对抗损失
            # 我们希望判别器给假样本打高分，所以取负数
            loss_adv = -self.discriminator(fake_data_for_g).mean()

            # c. 计算重建损失
            target_batch = next(recon_dataloader_iter)
            target_data = target_batch.data.to(self.device)
            loss_recon = compute_reconstruction_loss(fake_data_for_g, target_data)

            # d. 计算生成器总损失
            loss_g = loss_adv + self.config['lambda_recon'] * loss_recon

            # e. 反向传播和优化
            loss_g.backward()
            self.opt_g.step()

            # --- 记录日志 ---
            epoch_loss_d.append(avg_loss_d)
            epoch_loss_g.append(loss_g.item())
            epoch_d_real.append(d_real.mean().item())
            epoch_d_fake.append(d_fake.mean().item())
            
            # 更新进度条显示
            progress_bar.set_postfix({
                "Loss_D": f"{avg_loss_d:.4f}",
                "Loss_G": f"{loss_g.item():.4f}",
                "D(real)": f"{d_real.mean().item():.4f}",
                "D(fake)": f"{d_fake.mean().item():.4f}",
            })

        # 返回整个 epoch 的平均日志
        return {
            "loss_d": sum(epoch_loss_d) / len(epoch_loss_d),
            "loss_g": sum(epoch_loss_g) / len(epoch_loss_g),
            "d_real_mean": sum(epoch_d_real) / len(epoch_d_real),
            "d_fake_mean": sum(epoch_d_fake) / len(epoch_d_fake),
        }

if __name__ == '__main__':
    # --- 这是一个演示如何使用 Trainer 的示例 ---

    # 1. 配置
    config = {
        'latent_dim': 64,    # 噪声向量维度
        'data_dim': 128,     # 模拟数据维度
        'ncritic': 5,        # 判别器更新次数
        'lambda_gp': 10,     # 梯度惩罚系数
        'lambda_recon': 0.1, # 重建损失系数
        'lr_g': 1e-4,        # 生成器学习率
        'lr_d': 1e-4,        # 判别器学习率
        'batch_size': 32
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. 创建模拟组件
    generator = DummyModel(config['latent_dim'], config['data_dim'])
    discriminator = DummyModel(config['data_dim'], 1)
    
    # 使用 Adam 优化器，betas=(0.5, 0.9) 在 GAN 中常用
    opt_g = optim.Adam(generator.parameters(), lr=config['lr_g'], betas=(0.5, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=config['lr_d'], betas=(0.5, 0.9))

    # 创建模拟数据加载器
    dummy_dataset = [
        DummyGraphBatch(
            data=torch.randn(config['data_dim']), 
            num_graphs=config['batch_size']
        ) for _ in range(100) # 100个批次的数据
    ]
    # 在实际项目中，这里会是 torch.utils.data.DataLoader
    dataloader = dummy_dataset 
    
    template_graph = None # 在这个简化例子中不需要模板

    # 3. 初始化并运行训练器
    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        opt_g=opt_g,
        opt_d=opt_d,
        dataloader=dataloader,
        template_graph=template_graph,
        config=config,
        device=DEVICE
    )
    
    trainer.train(num_epochs=5)