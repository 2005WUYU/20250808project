import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data, Batch

# ==============================================================================
# 判别器模型 (Discriminator)
# ==============================================================================

class Discriminator(nn.Module):
    """
    判别器模型 D(x)。
    
    该模型接收一个图结构（来自真实数据或生成器），并输出一个标量（实数），
    表示该图的“真实度”得分。在WGAN的设定中，这个分数是未经过激活的，被称为"Critic Score"。
    
    架构:
    1. MPN: 堆叠多层GINEConv，从图中提取和聚合局部与全局信息。
    2. Pooling: 使用全局平均池化将所有节点的特征聚合成一个图级别的特征向量。
    3. MLP: 使用一个MLP将图级别向量映射为最终的单一分数。
    """
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=128):
        """
        初始化判别器。

        Args:
            node_feature_dim (int): 输入图中每个节点的特征维度 (这里是10)。
            edge_feature_dim (int): 输入图中每个边的特征维度 (这里是3)。
            hidden_dim (int, optional): MPN和MLP中隐藏层的维度。默认为 128。
        """
        super(Discriminator, self).__init__()

        # --- 1. 消息传递网络 (MPN) ---
        # 我们使用GINEConv，因为它能有效结合节点和边的特征。
        # GINEConv需要一个内部MLP来转换节点特征。
        
        # GINE Layer 1
        # (节点特征维度) -> hidden_dim
        nn1 = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(nn1, edge_dim=edge_feature_dim)
        self.bn1 = BatchNorm(hidden_dim)

        # GINE Layer 2
        # hidden_dim -> hidden_dim
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn2, edge_dim=edge_feature_dim)
        self.bn2 = BatchNorm(hidden_dim)

        # GINE Layer 3
        # hidden_dim -> hidden_dim
        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv3 = GINEConv(nn3, edge_dim=edge_feature_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # GINE Layer 4 (我们堆叠4层以获得更大的感受野)
        # hidden_dim -> hidden_dim
        nn4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv4 = GINEConv(nn4, edge_dim=edge_feature_dim)
        self.bn4 = BatchNorm(hidden_dim)

        # --- 2. 最终的MLP分类器 ---
        # 在全局池化后，这个MLP将图级别的向量映射为一个分数
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1) # 输出一个单一的、未经激活的分数
        )

    def forward(self, data):
        """
        判别器的前向传播过程。

        Args:
            data (torch_geometric.data.Batch): 一个批次的图数据。
                                             需要包含 x, edge_index, edge_attr, 和 batch 属性。

        Returns:
            torch.Tensor: 对批次中每个图的评分，形状为 [batch_size, 1]。
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # --- 步骤 1: 通过MPN层进行特征提取 ---
        # GINEConv的标准用法是：卷积 -> 批归一化 -> 激活函数
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.relu(x)
        
        # --- 步骤 2: 全局池化 ---
        # 将每个图的所有节点特征聚合成一个单一的图级别向量
        # `global_mean_pool` 会根据 `batch` 向量自动处理
        graph_embedding = global_mean_pool(x, batch) # -> [batch_size, hidden_dim]

        # --- 步骤 3: 通过MLP得到最终分数 ---
        score = self.mlp(graph_embedding) # -> [batch_size, 1]

        return score

# ==============================================================================
# 示例用法
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 定义超参数 ---
    BATCH_SIZE = 4
    NUM_NODES = 4000
    NUM_EDGES = 20000
    NODE_FEATURE_DIM = 10
    EDGE_FEATURE_DIM = 3
    HIDDEN_DIM = 128
    
    # 将模型移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # --- 2. 创建一个虚拟的、代表一批图的 Batch 对象 ---
    # 这模拟了从数据加载器或生成器得到的输入
    data_list = []
    for _ in range(BATCH_SIZE):
        # 为批次中的每个图创建虚拟数据
        nodes = torch.randn(NUM_NODES, NODE_FEATURE_DIM)
        edges = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
        edge_feats = torch.randn(NUM_EDGES, EDGE_FEATURE_DIM)
        data_list.append(Data(x=nodes, edge_index=edges, edge_attr=edge_feats))
    
    # 使用 PyG 的 Batch.from_data_list 来正确地构建批处理对象
    input_batch = Batch.from_data_list(data_list).to(device)
    
    print("输入判别器的批处理图信息:")
    print(input_batch)
    
    # --- 3. 实例化判别器模型 ---
    discriminator = Discriminator(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    print("\n判别器模型结构:")
    print(discriminator)
    
    # --- 4. 执行前向传播 ---
    # 在实际训练中，应将模型设置为训练模式 model.train()
    discriminator.train()
    scores = discriminator(input_batch)
    
    # --- 5. 检查输出 ---
    print("\n判别器输出信息:")
    print(f"输出类型: {type(scores)}")
    print(f"输出形状: {scores.shape}")
    print("输出分数示例:")
    print(scores)
    
    # 验证输出的维度是否正确
    expected_shape = (BATCH_SIZE, 1)
    
    print(f"\n期望的输出形状: {expected_shape}, 实际: {scores.shape}")
    assert scores.shape == expected_shape, "输出形状不正确！"
    
    print("\n所有断言通过，代码结构和维度正确！")