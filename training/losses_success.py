# 文件路径: src/training/losses.py

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

# 确保在需要时可以从其他模块导入判别器
# from src.models.discriminator import Discriminator 

# ----------------------------------------------------------------------------
# 核心函数 1: WGAN-GP 梯度惩罚
# ----------------------------------------------------------------------------
def compute_gradient_penalty(discriminator: nn.Module, real_data: Batch, fake_data: Batch, device: torch.device) -> torch.Tensor:
    """
    计算 WGAN-GP 的梯度惩罚。
    惩罚项计算作用于在真实数据和生成数据之间的随机插值点上。
    
    Args:
        discriminator (nn.Module): 判别器模型。
        real_data (Batch): 一批次的真实图数据。
        fake_data (Batch): 一批次的生成图数据。
        device (torch.device): 计算设备 (e.g., 'cuda' or 'cpu')。

    Returns:
        torch.Tensor: 计算出的梯度惩罚项 (一个标量)。
    """
    # 确保真实数据和生成数据的节点总数相同，这是插值的前提
    assert real_data.x.size(0) == fake_data.x.size(0)
    
    num_nodes_total = real_data.x.size(0)
    
    # 1. 生成插值权重 epsilon，在拼接后的大张量上进行逐节点插值
    # epsilon 的形状为 (B*N_total, 1)，以便与 (B*N_total, D) 的节点特征广播
    epsilon = torch.rand(num_nodes_total, 1, device=device)
    epsilon = epsilon.expand_as(real_data.x)

    # 2. 创建插值节点特征
    interpolated_x = epsilon * real_data.x + (1 - epsilon) * fake_data.x
    interpolated_x.requires_grad_(True)
    
    # 3. 创建一个新的 Batch 对象用于判别器计算
    # 我们使用真实数据的图结构 (edge_index, batch) 和插值的节点特征 (x)
    interpolated_data = real_data.clone()
    interpolated_data.x = interpolated_x

    # 4. 计算判别器在插值点上的输出
    d_interpolated = discriminator(interpolated_data)

    # 5. 计算判别器输出相对于插值输入的梯度
    grad_outputs = torch.ones_like(d_interpolated, device=device)
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated_x,
        grad_outputs=grad_outputs,
        create_graph=True,  # 必须为 True，因为惩罚项是总损失的一部分，需要通过它反向传播
        retain_graph=True,  # 必须为 True，以保持计算图用于后续的判别器/生成器更新
        only_inputs=True,
    )[0]
    
    # 6. 计算梯度的L2范数并计算惩罚项
    # gradients 的形状是 (B*N_total, D)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


# ----------------------------------------------------------------------------
# 核心函数 2: 生成器的重建损失
# ----------------------------------------------------------------------------
def compute_reconstruction_loss(generated_graph: Data, target_graph: Data, w_pos: float, w_feat: float) -> torch.Tensor:
    """
    计算生成图和目标图之间的重建损失。
    这通常用于 VAE 或自编码器类型的生成器。
    
    Args:
        generated_graph (Data): 生成的图。
        target_graph (Data): 原始的目标图。
        w_pos (float): 坐标损失的权重。
        w_feat (float): 特征损失的权重。

    Returns:
        torch.Tensor: 计算出的加权重建损失 (一个标量)。
    """
    # 假设节点特征 x 的前3维是坐标，其余是其他特征
    # [..., :3] 用于处理批处理和非批处理情况
    p_gen, f_gen = generated_graph.x[..., :3], generated_graph.x[..., 3:]
    p_target, f_target = target_graph.x[..., :3], target_graph.x[..., 3:]

    # 使用 L1 损失计算坐标和特征的差异
    loss_pos = torch.nn.functional.l1_loss(p_gen, p_target)
    loss_feat = torch.nn.functional.l1_loss(f_gen, f_target)

    # 返回加权和
    return w_pos * loss_pos + w_feat * loss_feat


# ============================================================================
# 使用示例 (Usage Example)
# ============================================================================
if __name__ == '__main__':
    # --- 1. 测试 compute_reconstruction_loss ---
    print("--- Testing compute_reconstruction_loss ---")
    # 创建模拟图数据，节点特征维度为10 (3 for pos, 7 for feat)
    gen_graph = Data(x=torch.randn(5, 10)) # 5个节点，10维特征
    tar_graph = Data(x=torch.randn(5, 10))
    w_p, w_f = 10.0, 1.0
    
    recon_loss = compute_reconstruction_loss(gen_graph, tar_graph, w_pos=w_p, w_feat=w_f)
    print(f"Reconstruction Loss: {recon_loss.item()}")
    assert recon_loss.ndim == 0 # 损失必须是标量

    # --- 2. 测试 compute_gradient_penalty ---
    print("\n--- Testing compute_gradient_penalty ---")
    device = torch.device('cpu')
    
    # a. 创建一个简单的虚拟判别器用于测试
    class DummyDiscriminator(nn.Module):
        def __init__(self, node_dim=10, out_dim=256):
            super().__init__()
            self.mp = nn.Linear(node_dim, out_dim)
            self.pool = global_mean_pool
            self.mlp = nn.Linear(out_dim, 1)
        
        def forward(self, data: Batch) -> torch.Tensor:
            h = self.mp(data.x)
            graph_h = self.pool(h, data.batch)
            return self.mlp(graph_h)

    # b. 创建模拟的真实和生成批次数据
    data_list_real, data_list_fake = [], []
    for _ in range(4): # Batch size = 4
        num_nodes = torch.randint(5, 10, (1,)).item()
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), dtype=torch.long)
        # Real
        data_list_real.append(Data(x=torch.randn(num_nodes, 10), edge_index=edge_index))
        # Fake (有相同的结构，但特征不同)
        data_list_fake.append(Data(x=torch.randn(num_nodes, 10), edge_index=edge_index))

    from torch_geometric.loader import DataLoader
    
    # 修复: 使用正确的方式获取batch数据
    real_loader = DataLoader(data_list_real, batch_size=4, shuffle=False)
    fake_loader = DataLoader(data_list_fake, batch_size=4, shuffle=False)
    
    # 使用 next(iter()) 替代 .next()
    real_batch = next(iter(real_loader))
    fake_batch = next(iter(fake_loader))
    
    # c. 实例化判别器并计算梯度惩罚
    dummy_d = DummyDiscriminator()
    gp = compute_gradient_penalty(dummy_d, real_batch, fake_batch, device)
    
    print(f"Gradient Penalty: {gp.item()}")
    assert gp.ndim == 0 # 惩罚项必须是标量
    print("✅ 所有函数测试通过!")