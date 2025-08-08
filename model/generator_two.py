# src/models/generator.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from typing import List

class Generator(nn.Module):
    """
    它接收一个噪声向量 z 和一个模板图，通过风格调制和 GAT 网络，
    生成一个新的、变形后的图。
    """
    def __init__(self, latent_dim: int, style_dim: int, node_feature_dim: int, 
                 gat_hidden_dims: List[int], num_gat_layers: int = 3, num_heads: int = 4):
        """
        初始化生成器。

        参数:
        - latent_dim (int): 输入噪声向量 z 的维度。
        - style_dim (int): 内部风格向量 S 的维度。
        - node_feature_dim (int): 模板图中节点特征 x 的维度。
        - gat_hidden_dims (list[int]): GAT 各隐藏层的维度列表。
        - num_gat_layers (int): GAT 层的总数。
        - num_heads (int): GAT 中间层的多头注意力头数。
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.node_feature_dim = node_feature_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads

        # 1. 风格MLP: 将噪声z映射到风格向量S
        self.style_mlp = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=style_dim)
        )

        # 2. GAT 网络
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList() 
        self.activation = nn.LeakyReLU(0.2)

        # 输入层
        in_channels = node_feature_dim + style_dim # (e.g., 10 + 64 = 74)
        self.gat_layers.append(
            GATConv(in_channels, gat_hidden_dims[0], heads=num_heads, concat=True)
        )
        
        # 中间层
        for i in range(num_gat_layers - 2):
            # 前一层的输出维度是 gat_hidden_dims[i] * num_heads
            in_channels_hidden = gat_hidden_dims[i] * num_heads
            out_channels_hidden = gat_hidden_dims[i+1]
            self.norms.append(nn.LayerNorm(in_channels_hidden))
            self.gat_layers.append(
                GATConv(in_channels_hidden, out_channels_hidden, heads=num_heads, concat=True)
            )
        
        # 输出层
        in_channels_last = gat_hidden_dims[-1] * num_heads
        # 输出维度为 3 (位置偏移) + node_feature_dim (特征偏移)
        out_channels_final = 3 + node_feature_dim
        self.norms.append(nn.LayerNorm(in_channels_last))
        self.gat_layers.append(
            GATConv(in_channels_last, out_channels_final, heads=1, concat=False)
        )


    def forward(self, z: torch.Tensor, template_data: Data) -> Batch:
        """
        前向传播函数。

        参数:
        - z (torch.Tensor): 噪声张量，形状为 (B, latent_dim)，B是批处理大小。
        - template_data (PyG.Data): 模板图，必须包含 `x` (N, node_feature_dim),
                                   `pos` (N, 3), 和 `edge_index` (2, M)。

        返回:
        - PyG.Batch: 一个包含 B 个生成图的批处理对象。
        """
        # 检查模板数据是否符合要求
        if not all(hasattr(template_data, attr) for attr in ['x', 'pos', 'edge_index']):
            raise ValueError("`template_data` must contain 'x', 'pos', and 'edge_index' attributes.")
        
        batch_size = z.size(0)
        num_nodes = template_data.num_nodes

        # 1. 获取风格向量 S
        style_vector = self.style_mlp(z) # 形状: (B, style_dim)

        # 2. 构建批处理图数据 (PyG Best Practice)
        # 将模板图复制B次，并使用`Batch.from_data_list`来正确处理批处理
        data_list = [template_data.clone() for _ in range(batch_size)]
        graph_batch = Batch.from_data_list(data_list)
        # graph_batch.x 形状: (B*N, node_feature_dim)
        # graph_batch.pos 形状: (B*N, 3)
        # graph_batch.edge_index 形状: (2, B*M)，值已自动偏移
        # graph_batch.batch 形状: (B*N)，记录每个节点属于哪个图

        # 3. 特征拼接准备
        # 将风格向量S扩展，使其能与每个节点特征拼接
        # style_vector 形状: (B, style_dim) -> (B, 1, style_dim) -> (B, N, style_dim) -> (B*N, style_dim)
        style_expanded = style_vector.unsqueeze(1).expand(-1, num_nodes, -1).reshape(-1, self.style_dim)
        
        # 4. 构建GAT的初始节点特征 H0
        # H0 形状: (B*N, node_feature_dim + style_dim)
        H = torch.cat([graph_batch.x, style_expanded], dim=1)

        # 5. 通过GAT网络
        edge_index = graph_batch.edge_index
        for i in range(self.num_gat_layers - 1):
            H = self.gat_layers[i](H, edge_index)
            H = self.norms[i](H)
            H = self.activation(H)
        
        # 最后一层，不经过激活和归一化
        H_out = self.gat_layers[-1](H, edge_index) # 形状: (B*N, 3 + node_feature_dim)

        # 6. 分离位移和特征变化
        delta_p = H_out[:, :3]               # 形状: (B*N, 3)
        delta_f = H_out[:, 3:]               # 形状: (B*N, node_feature_dim)

        # 7. 计算生成图的新属性
        # 直接在批处理数据上进行计算，更高效
        p_new = graph_batch.pos + delta_p    # 新位置
        f_new = graph_batch.x + delta_f      # 新特征
        
        # 8. 构建输出批次对象
        # 创建一个新的Batch对象来存储结果，这能保持数据流的清晰
        # 也可以直接修改 graph_batch.x 和 graph_batch.pos，但创建新的更安全
        output_batch = Batch(
            batch=graph_batch.batch,
            ptr=graph_batch.ptr,
            x=f_new,
            pos=p_new,
            edge_index=graph_batch.edge_index
        )
        
        return output_batch


# --- 单元测试 ---
if __name__ == '__main__':
    # 定义超参数
    BATCH_SIZE = 4
    LATENT_DIM = 128
    STYLE_DIM = 64
    NODE_FEATURE_DIM = 10
    GAT_HIDDEN_DIMS = [128, 128]
    NUM_GAT_LAYERS = 3
    NUM_HEADS = 4
    
    # 模拟一个模板图 (15个节点，20条边)
    NUM_NODES = 15
    NUM_EDGES = 20
    template_graph = Data(
        x=torch.randn(NUM_NODES, NODE_FEATURE_DIM),      # 节点特征
        pos=torch.randn(NUM_NODES, 3),                  # 节点位置
        edge_index=torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long) # 边连接
    )

    print("--- 输入 ---")
    print(f"批处理大小 (B): {BATCH_SIZE}")
    print(f"模板图: {template_graph}")

    # 实例化生成器
    generator = Generator(
        latent_dim=LATENT_DIM,
        style_dim=STYLE_DIM,
        node_feature_dim=NODE_FEATURE_DIM,
        gat_hidden_dims=GAT_HIDDEN_DIMS,
        num_gat_layers=NUM_GAT_LAYERS,
        num_heads=NUM_HEADS
    )
    print("\n--- 生成器结构 ---")
    print(generator)

    # 创建一个批次的噪声向量
    z_noise = torch.randn(BATCH_SIZE, LATENT_DIM)

    # 执行前向传播
    generated_batch = generator(z_noise, template_graph)

    print("\n--- 输出 ---")
    print(f"生成的批处理对象: {generated_batch}")
    print(f"批处理中的图数量: {generated_batch.num_graphs}")
    print(f"批处理中的总节点数: {generated_batch.num_nodes}")
    print(f"新节点特征 `x` 的形状: {generated_batch.x.shape}")
    print(f"新节点位置 `pos` 的形状: {generated_batch.pos.shape}")
    
    # 验证输出
    assert generated_batch.num_graphs == BATCH_SIZE
    assert generated_batch.num_nodes == BATCH_SIZE * NUM_NODES
    assert generated_batch.x.shape == (BATCH_SIZE * NUM_NODES, NODE_FEATURE_DIM)
    assert generated_batch.pos.shape == (BATCH_SIZE * NUM_NODES, 3)
    assert generated_batch.edge_index.shape == (2, BATCH_SIZE * NUM_EDGES)
    
    print("\n✅ 测试通过，输出维度正确！")