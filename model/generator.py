import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.data import Data, Batch

# ==============================================================================
# 辅助函数: 用于重新计算边特征
# ==============================================================================

def recompute_edge_features(pos, edge_index):
    """
    根据更新后的节点坐标重新计算边特征。
    
    这里的边特征被定义为连接两个节点的位置向量 (displacement vector)。
    这是一个非常有用的几何特征，因为它同时包含了边的长度和方向信息。

    Args:
        pos (torch.Tensor): 节点的坐标矩阵，形状为 [N, 3]，N是节点数。
        edge_index (torch.Tensor): 描述图连接的边索引，形状为 [2, M]，M是边数。

    Returns:
        torch.Tensor: 新的边特征矩阵，形状为 [M, 3]。
    """
    # edge_index[0] 是所有边的源节点索引
    # edge_index[1] 是所有边的目标节点索引
    source_nodes_pos = pos[edge_index[0]]
    destination_nodes_pos = pos[edge_index[1]]
    
    # 计算目标节点到源节点的位置向量差
    # 这个向量代表了边的方向和长度
    edge_features = destination_nodes_pos - source_nodes_pos
    
    return edge_features

# ==============================================================================
# 生成器模型 (Generator)
# ==============================================================================

class Generator(nn.Module):
    """
    生成器模型 G(z)。
    
    该模型接收一个随机噪声向量z和固定的拓扑模板，
    输出一个“变形”后的图结构，该图结构在几何和节点特征上都发生了改变。
    """
    def __init__(self, latent_dim, style_dim, node_feature_dim, heads=4, gat_hidden_dim=128):
        """
        初始化生成器。

        Args:
            latent_dim (int): 输入的随机噪声z的维度。
            style_dim (int): 通过MLP从z生成的风格向量S的维度。
            node_feature_dim (int): 模板图中每个节点的特征维度 (这里是10)。
            heads (int, optional): GAT中多头注意力的头数。默认为 4。
            gat_hidden_dim (int, optional): GAT隐藏层的维度。默认为 128。
        """
        super(Generator, self).__init__()

        # --- 1. 风格提取网络 (MLP) ---
        # 将随机噪声z映射为更有意义的风格向量S
        self.style_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, style_dim) # 最后一层无激活函数
        )

        # GAT输入特征的维度 = 原始节点特征维度 + 风格向量维度
        gat_input_dim = node_feature_dim + style_dim
        
        # GAT输出维度 = 3D位移向量(Δp) + 10D特征变化向量(Δf)
        gat_output_dim = 3 + node_feature_dim

        # --- 2. 变形网络 (GAT) ---
        # 使用图注意力网络来学习每个节点应该如何更新
        # 我们使用3层GAT，并加入残差连接和层归一化来稳定训练

        # GAT第一层
        self.gat1 = GATConv(gat_input_dim, gat_hidden_dim, heads=heads, concat=True)
        self.norm1 = LayerNorm(gat_hidden_dim * heads)

        # GAT第二层
        # 注意：输入维度是上一层多头拼接后的维度
        self.gat2 = GATConv(gat_hidden_dim * heads, gat_hidden_dim, heads=heads, concat=True)
        self.norm2 = LayerNorm(gat_hidden_dim * heads)

        # GAT第三层 (输出层)
        # 输出最终的更新向量，不进行多头拼接(concat=False)，而是取平均
        self.gat3 = GATConv(gat_hidden_dim * heads, gat_output_dim, heads=1, concat=False)

    def forward(self, z, template_data):
        """
        生成器的前向传播过程。

        Args:
            z (torch.Tensor): 一个批次的随机噪声向量，形状为 [batch_size, latent_dim]。
            template_data (torch_geometric.data.Data): 包含标准模板图信息的PyG Data对象。
                                                     应包含 template_data.x 和 template_data.edge_index。

        Returns:
            torch_geometric.data.Batch: 一个批次的生成图数据。
        """
        # --- 步骤 1: 生成风格向量 S ---
        batch_size = z.shape[0]
        style_vectors = self.style_mlp(z) # -> [batch_size, style_dim]

        # --- 步骤 2: 为GAT准备批处理数据和初始特征 H0 ---
        # 将单个模板图复制 N=batch_size 次，构建一个批处理图
        data_list = [template_data.clone() for _ in range(batch_size)]
        batched_template = Batch.from_data_list(data_list)
        
        # 将风格向量S扩展，使其能与每个节点的特征拼接
        # [batch_size, style_dim] -> [batch_size, N, style_dim] -> [batch_size * N, style_dim]
        # N 是单个图的节点数
        num_nodes_per_graph = template_data.num_nodes
        style_vectors_expanded = style_vectors.repeat_interleave(num_nodes_per_graph, dim=0)

        # 拼接得到GAT的初始节点特征 H0
        # H0 = [原始节点特征, 风格向量]
        H0 = torch.cat([batched_template.x, style_vectors_expanded], dim=1)

        # --- 步骤 3: 通过GAT网络进行变形信息传播 ---
        edge_index = batched_template.edge_index
        
        # GAT Layer 1 + Residual + Norm
        H1 = F.leaky_relu(self.gat1(H0, edge_index), 0.2)
        # 残差连接需要对H0进行维度匹配
        # 这里为了简化，我们不添加H0到H1的残差，而是从H1到H2开始
        H1_norm = self.norm1(H1)
        
        # GAT Layer 2 + Residual + Norm
        H2 = F.leaky_relu(self.gat2(H1_norm, edge_index), 0.2)
        H2 = H2 + H1_norm # 残差连接
        H2_norm = self.norm2(H2)

        # GAT Layer 3 (输出层)
        # 输出的 H_out 即为每个节点的“更新信息”
        H_out = self.gat3(H2_norm, edge_index) # -> [batch_size * N, 13]

        # --- 步骤 4: 解析输出并生成新图 ---
        # 分离出位移向量 Δp 和特征变化向量 Δf
        delta_p = H_out[:, :3]        # 形状: [B*N, 3]
        delta_f = H_out[:, 3:]        # 形状: [B*N, 10]

        # 获取批处理模板的原始数据
        X_template_batched = batched_template.x
        p_template_batched = X_template_batched[:, :3]

        # 计算新的节点特征和坐标
        # 1. 首先用 Δf 更新整个10维特征向量
        X_generated = X_template_batched + delta_f
        # 2. 然后用更专门的 Δp 更新位置坐标
        p_generated = p_template_batched + delta_p
        # 3. 将更新后的坐标 p_generated 覆盖到特征矩阵 X_generated 的前三列
        X_generated[:, :3] = p_generated

        # --- 步骤 5: 组装并返回最终的生成图批次 ---
        # 重新计算边特征
        EF_generated = recompute_edge_features(p_generated, edge_index)

        # 创建一个新的批处理Data对象用于输出
        output_batch = batched_template.clone()
        output_batch.x = X_generated
        output_batch.pos = p_generated # PyG中通常用 pos 属性存储坐标
        output_batch.edge_attr = EF_generated

        return output_batch

# ==============================================================================
# 示例用法
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 定义超参数 ---
    BATCH_SIZE = 4
    NUM_NODES = 4000  # 假设每个图有4000个节点
    NUM_EDGES = 20000 # 假设每个图有20000条边
    NODE_FEATURE_DIM = 10
    LATENT_DIM = 128
    STYLE_DIM = 64
    
    # 将模型移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 2. 创建一个虚拟的标准拓扑模板 ---
    # 节点特征 (前3维是坐标)
    template_x = torch.randn(NUM_NODES, NODE_FEATURE_DIM, device=device)
    template_x[:, :3] = torch.randn(NUM_NODES, 3, device=device) * 10 # 放大坐标范围
    # 边索引
    template_edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), device=device)
    # 创建PyG Data对象
    template_graph_data = Data(x=template_x, edge_index=template_edge_index)
    print("标准模板图信息:")
    print(template_graph_data)
    
    # --- 3. 实例化生成器模型 ---
    generator = Generator(
        latent_dim=LATENT_DIM,
        style_dim=STYLE_DIM,
        node_feature_dim=NODE_FEATURE_DIM
    ).to(device)
    
    print("\n生成器模型结构:")
    print(generator)
    
    # --- 4. 创建一批随机噪声z ---
    z_noise = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    
    # --- 5. 执行前向传播 ---
    # 将噪声和模板输入生成器
    # 在实际训练中，应将模型设置为训练模式 model.train()
    generator.train()
    generated_graph_batch = generator(z_noise, template_graph_data)
    
    # --- 6. 检查输出 ---
    print("\n生成器输出信息:")
    print(f"输出类型: {type(generated_graph_batch)}")
    print("输出的批处理图信息:")
    print(generated_graph_batch)
    
    # 验证输出的维度是否正确
    expected_nodes = BATCH_SIZE * NUM_NODES
    expected_edges = BATCH_SIZE * NUM_EDGES
    
    print(f"\n期望的总节点数: {expected_nodes}, 实际: {generated_graph_batch.num_nodes}")
    print(f"期望的总边数: {expected_edges}, 实际: {generated_graph_batch.num_edges}")
    
    assert isinstance(generated_graph_batch, Batch)
    assert generated_graph_batch.num_nodes == expected_nodes
    assert generated_graph_batch.x.shape == (expected_nodes, NODE_FEATURE_DIM)
    assert generated_graph_batch.pos.shape == (expected_nodes, 3)
    assert generated_graph_batch.edge_attr.shape == (expected_edges, 3)
    
    print("\n所有断言通过，代码结构和维度正确！")