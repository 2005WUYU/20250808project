# 文件路径: src/models/discriminator.py (改进版)

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import erdos_renyi_graph # <--- [改进1] 导入更合理的图生成工具

# ----------------------------------------------------------------------------
# 内部模块: 自定义消息传递层 (带残差连接)
# ----------------------------------------------------------------------------
class CustomMessagePassingLayer(MessagePassing):
    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int):
        super().__init__(aggr='mean')
        
        self.message_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(out_dim + node_in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- [改进3] 为残差连接准备 ---
        # 如果输入和输出维度不同，需要一个线性层来变换维度以进行残差连接
        if node_in_dim != out_dim:
            self.residual_proj = nn.Linear(node_in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # 保存原始输入用于残差连接
        residual = self.residual_proj(x)
        
        # 消息传递和更新
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # --- [改进3] 加入残差连接 ---
        return out + residual

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        update_input = torch.cat([aggr_out, x], dim=-1)
        return self.update_mlp(update_input)


# ----------------------------------------------------------------------------
# 核心目标: 定义判别器 D (带LayerNorm)
# ----------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, node_feature_dim: int = 10, edge_feature_dim: int = 3, 
                 mpn_hidden_dims: list = [64, 128, 256], mlp_hidden_dims: list = [128, 64]):
        super().__init__()

        self.mpn_layers = nn.ModuleList()
        # --- [改进2] 添加LayerNorm层 ---
        self.mpn_norms = nn.ModuleList()
        
        current_dim = node_feature_dim
        for hidden_dim in mpn_hidden_dims:
            self.mpn_layers.append(
                CustomMessagePassingLayer(current_dim, edge_feature_dim, hidden_dim)
            )
            self.mpn_norms.append(nn.LayerNorm(hidden_dim)) # 每个MPN层后跟一个LayerNorm
            current_dim = hidden_dim
            
        self.pooling = global_mean_pool

        mlp_layers = []
        last_mpn_dim = mpn_hidden_dims[-1]
        for dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(last_mpn_dim, dim))
            mlp_layers.append(nn.LeakyReLU(0.2, inplace=True))
            last_mpn_dim = dim
            
        mlp_layers.append(nn.Linear(last_mpn_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    # --- [改进4] 更新类型注解 ---
    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # --- [改进2] 在MPN层之间应用LayerNorm ---
        h = x
        for i, layer in enumerate(self.mpn_layers):
            h = layer(h, edge_index, edge_attr)
            h = self.mpn_norms[i](h) # 应用LayerNorm
            # 注意：激活函数已在层内部，这里不再需要
        
        graph_embedding = self.pooling(h, batch)
        score = self.mlp(graph_embedding)
        return score

# ============================================================================
# 使用示例 (已改进)
# ============================================================================
if __name__ == '__main__':
    NODE_DIM = 10
    EDGE_DIM = 3
    MPN_DIMS = [64, 128, 256]
    MLP_DIMS = [128, 64]
    BATCH_SIZE = 4

    discriminator = Discriminator(
        node_feature_dim=NODE_DIM,
        edge_feature_dim=EDGE_DIM,
        mpn_hidden_dims=MPN_DIMS,
        mlp_hidden_dims=MLP_DIMS
    )
    print("判别器模型结构 (改进版):")
    print(discriminator)

    # --- [改进1] 使用 erdos_renyi_graph 生成更合理的图结构 ---
    graph_list = []
    for _ in range(BATCH_SIZE):
        num_nodes = torch.randint(20, 30, (1,)).item()
        # 生成不含自环和重复边的边索引
        edge_index = erdos_renyi_graph(num_nodes, p=0.2) 
        num_edges = edge_index.shape[1]
        
        graph_list.append(
            Data(
                x=torch.randn(num_nodes, NODE_DIM),
                edge_index=edge_index,
                edge_attr=torch.randn(num_edges, EDGE_DIM)
            )
        )
    
    loader = DataLoader(graph_list, batch_size=BATCH_SIZE)
    batch_data = next(iter(loader))
    
    print("\n输入数据 (Batch 对象):")
    print(batch_data)

    with torch.no_grad():
        output_score = discriminator(batch_data)

    print(f"\n模型输出分数 (形状: {output_score.shape}):")
    print(output_score)
    
    assert output_score.shape == (BATCH_SIZE, 1)
    print("\n✅ 输出形状验证成功!")