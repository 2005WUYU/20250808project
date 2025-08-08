# 文件路径: src/models/discriminator.py

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch_geometric.data as PyGData
from torch_geometric.loader import DataLoader

# ----------------------------------------------------------------------------
# 内部模块: 自定义消息传递层
# ----------------------------------------------------------------------------
class CustomMessagePassingLayer(MessagePassing):
    """
    一个自定义的消息传递层。
    消息的计算会同时考虑邻居节点特征和边特征。
    节点的更新会同时考虑聚合后的消息和节点自身的旧特征。
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int):
        # "mean" 聚合: 对所有邻居节点传来的消息取平均
        super().__init__(aggr='mean')
        
        # message_mlp 用于根据源节点和边特征生成消息
        # 输入维度是 node_in_dim (来自x_j) + edge_in_dim (来自edge_attr)
        self.message_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # update_mlp 用于结合聚合消息和中心节点特征来更新节点
        # 输入维度是 out_dim (来自aggr_out) + node_in_dim (来自x_i)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_dim + node_in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # self.propagate 会调用 message(), aggregate() 和 update()
        # x=[node_in_dim], edge_attr=[edge_in_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_j 的形状是 [num_edges, node_in_dim]
        # edge_attr 的形状是 [num_edges, edge_in_dim]
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # aggr_out 的形状是 [num_nodes, out_dim]
        # x 的形状是 [num_nodes, node_in_dim]
        update_input = torch.cat([aggr_out, x], dim=-1)
        return self.update_mlp(update_input)


# ----------------------------------------------------------------------------
# 核心目标: 定义判别器 D
# ----------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    判别器 D 接收一个图 (或一批图), 并输出一个实数分数表示其真实性。
    """
    def __init__(self, node_feature_dim: int = 10, edge_feature_dim: int = 3, 
                 mpn_hidden_dims: list = [64, 128, 256], mlp_hidden_dims: list = [128, 64]):
        super().__init__()

        # --- 1. MPN层定义 ---
        self.mpn_layers = nn.ModuleList()
        
        # 第一层
        self.mpn_layers.append(
            CustomMessagePassingLayer(node_feature_dim, edge_feature_dim, mpn_hidden_dims[0])
        )
        
        # 循环添加中间层
        # 输入维度是上一层的输出维度
        for i in range(len(mpn_hidden_dims) - 1):
            self.mpn_layers.append(
                CustomMessagePassingLayer(mpn_hidden_dims[i], edge_feature_dim, mpn_hidden_dims[i+1])
            )
            
        # --- 2. 全局池化 ---
        self.pooling = global_mean_pool

        # --- 3. MLP分类器 ---
        mlp_layers = []
        # MLP的输入维度是MPN最后一层的输出维度
        last_mpn_dim = mpn_hidden_dims[-1]
        
        # 循环构建MLP层
        for dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(last_mpn_dim, dim))
            mlp_layers.append(nn.LeakyReLU(0.2, inplace=True))
            last_mpn_dim = dim # 更新维度用于下一层
            
        # 最终输出层: 输出一个实数分数，无激活函数
        mlp_layers.append(nn.Linear(last_mpn_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, data: PyGData.Batch) -> torch.Tensor:
        """
        前向传播函数。
        输入: data: PyG.Batch 对象，包含 x, edge_index, edge_attr, batch。
        """
        # --- 1. 解包数据 ---
        # x:     节点特征张量, 形状 (B*N, node_feature_dim)
        # edge_index: 边索引, 形状 (2, B*M)
        # edge_attr: 边特征张量, 形状 (B*M, edge_feature_dim)
        # batch:  批次索引, 形状 (B*N,)，指示每个节点属于哪个图
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # --- 2. 通过MPN网络 ---
        h = x
        for layer in self.mpn_layers:
            h = layer(h, edge_index, edge_attr)
        # 经过MPN后，h 的形状为 (B*N, mpn_hidden_dims[-1])
        
        # --- 3. 全局聚合 ---
        # 将每个图的节点表征聚合为单个图表征
        # h 形状 (B*N, 256) -> graph_embedding 形状 (B, 256)
        graph_embedding = self.pooling(h, batch)
        
        # --- 4. 最终评分 ---
        # score 形状 (B, 1)
        score = self.mlp(graph_embedding)
        
        return score

# ============================================================================
# 使用示例 (Usage Example)
# ============================================================================
if __name__ == '__main__':
    # --- 1. 定义模型参数 ---
    NODE_DIM = 10
    EDGE_DIM = 3
    MPN_DIMS = [64, 128, 256]
    MLP_DIMS = [128, 64]
    BATCH_SIZE = 4 # 模拟一个批次包含4个图

    # --- 2. 实例化判别器 ---
    discriminator = Discriminator(
        node_feature_dim=NODE_DIM,
        edge_feature_dim=EDGE_DIM,
        mpn_hidden_dims=MPN_DIMS,
        mlp_hidden_dims=MLP_DIMS
    )
    print("判别器模型结构:")
    print(discriminator)

    # --- 3. 创建一批模拟图数据 ---
    graph_list = []
    for _ in range(BATCH_SIZE):
        num_nodes = torch.randint(5, 15, (1,)).item() # 每个图的节点数随机
        num_edges = torch.randint(10, 20, (1,)).item() # 每个图的边数随机
        
        graph_list.append(
            PyGData.Data(
                x=torch.randn(num_nodes, NODE_DIM),
                edge_index=torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long),
                edge_attr=torch.randn(num_edges, EDGE_DIM)
            )
        )
    
    # 使用 DataLoader 自动将图列表打包成一个 Batch 对象
    loader = DataLoader(graph_list, batch_size=BATCH_SIZE)
    batch_data = next(iter(loader))
    
    print("\n输入数据 (Batch 对象):")
    print(batch_data)

    # --- 4. 前向传播测试 ---
    # 将数据移动到合适的设备 (如果可用)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # discriminator.to(device)
    # batch_data.to(device)
    
    # 禁用梯度计算以进行推理
    with torch.no_grad():
        output_score = discriminator(batch_data)

    print(f"\n模型输出分数 (形状: {output_score.shape}):")
    print(output_score)
    
    # --- 5. 验证输出形状 ---
    assert output_score.shape == (BATCH_SIZE, 1)
    print("\n✅ 输出形状验证成功!")