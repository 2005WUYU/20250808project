# utils.py
import torch
import torch.autograd as autograd
from torch_geometric.data import Data, Batch

def compute_gradient_penalty(D, real_samples_x, fake_samples_x, edge_index, edge_attr, device, lambda_gp):
    """
    计算WGAN-GP的梯度惩罚。
    插值是在节点特征矩阵 X 上进行的。
    
    参数:
        D (torch.nn.Module): 判别器模型。
        real_samples_x (Tensor): 真实样本的节点特征 (batch_size * N, num_features)。
        fake_samples_x (Tensor): 生成样本的节点特征 (batch_size * N, num_features)。
        edge_index (Tensor): 批处理后的边索引。
        edge_attr (Tensor): 批处理后的边特征。
        device (str): 'cuda' 或 'cpu'。
        lambda_gp (float): 梯度惩罚的权重。

    返回:
        Tensor: 计算出的梯度惩罚损失。
    """
    # 获取批次中每个图的节点数，用于正确地重塑 alpha
    num_graphs = real_samples_x.size(0) // fake_samples_x.size(0) # This is a simple trick assuming N is constant
    if isinstance(real_samples_x, torch.Tensor) and hasattr(real_samples_x, 'ptr'): # Check if it's from a Batch object
        num_nodes_per_graph = (real_samples_x.ptr[1:] - real_samples_x.ptr[:-1])[0] # Assuming constant N
        num_graphs = len(real_samples_x.ptr) - 1
        alpha_shape = (num_graphs, 1, 1) # (batch_size, 1, 1)
    else: # Fallback for simple tensors
        # This part requires more robust handling if N varies.
        # For now, we assume a constant N, so batching is simple stacking.
        # A more robust way is to pass the `batch` vector.
        num_graphs = real_samples_x.shape[0] // fake_samples_x.shape[0]
        num_nodes_per_graph = real_samples_x.shape[0] // num_graphs
        alpha_shape = (num_graphs, 1, 1)


    alpha = torch.rand(alpha_shape, device=device)
    # 扩展alpha以匹配节点特征的维度
    alpha = alpha.expand(-1, num_nodes_per_graph, real_samples_x.size(-1)).reshape(real_samples_x.size())

    # 创建插值样本
    interpolated_x = (alpha * real_samples_x.data + (1 - alpha) * fake_samples_x.data).requires_grad_(True)
    
    # 构建一个完整的插值图对象
    # 注意：这里的 A, E, D, L 等对于插值样本来说是复用的，这在实践中是标准做法
    interpolated_graph = Batch.from_data_list([
        Data(x=interpolated_x[i*num_nodes_per_graph:(i+1)*num_nodes_per_graph],
             edge_index=edge_index, # simplified assumption
             edge_attr=edge_attr) # simplified assumption
        for i in range(num_graphs)
    ]).to(device)


    # 计算判别器对插值样本的输出
    d_interpolated = D(interpolated_graph)

    # 计算梯度
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated_graph.x,
        grad_outputs=torch.ones(d_interpolated.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # 计算梯度的L2范数并计算惩罚
    gradients = gradients.view(num_graphs * num_nodes_per_graph, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    
    return gradient_penalty