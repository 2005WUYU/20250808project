import torch

def calculate_derived_matrices(edge_index, num_nodes, device=None):
    """
    根据边索引矩阵计算邻接矩阵、度数矩阵和拉普拉斯矩阵。
    
    参数:
        edge_index (Tensor): 形状为 (2, M) 的边索引矩阵，每一列表示一条边的两个端点编号。
        num_nodes (int): 节点总数 N。
        device (str/torch.device, 可选): 所有矩阵的设备，默认为 None（自动继承 edge_index 的设备）。
        
    返回:
        dict:
            - adjacency_matrix (Tensor): 邻接矩阵，形状为 (N, N)。
            - degree_matrix (Tensor): 度数矩阵（对角阵），形状为 (N, N)。
            - laplacian_matrix (Tensor): 普通拉普拉斯矩阵，形状为 (N, N)。
    
    使用场景:
        数据加载时，模板图谱和训练样本初始化时调用，一次性计算所有基础矩阵。
    """
    if device is None:
        device = edge_index.device

    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1.0
    adjacency_matrix[edge_index[1], edge_index[0]] = 1.0  # 假设无向图

    degree = adjacency_matrix.sum(dim=1)
    degree_matrix = torch.diag(degree)

    laplacian_matrix = degree_matrix - adjacency_matrix

    return {
        'adjacency_matrix': adjacency_matrix,
        'degree_matrix': degree_matrix,
        'laplacian_matrix': laplacian_matrix
    }

def calculate_edge_features(X, edge_index):
    """
    根据节点特征矩阵（通常前三维为空间坐标）和边索引，计算每条边的空间特征。
    如 Δx, Δy, Δz 组成的边属性。

    参数:
        X (Tensor): 形状为 (N, F) 的节点特征矩阵，前3列应为空间坐标。
        edge_index (Tensor): 形状为 (2, M) 的边索引矩阵。

    返回:
        Tensor: 边特征矩阵，形状为 (M, 3)，每行为 [Δx, Δy, Δz]。
    
    使用场景:
        梯度惩罚/插值样本边属性生成、EF重建等。
    """
    src = edge_index[0]
    tgt = edge_index[1]
    delta = X[tgt, :3] - X[src, :3]
    return delta