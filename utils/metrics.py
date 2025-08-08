import numpy as np
from scipy.linalg import eigh

def get_graph_statistics(graph_batch, stats_to_compute=['edge_length', 'node_feature', 'laplacian_spectrum'], k_spectrum=30, sigma=1.0):
    """
    计算一批图的统计量分布，包括边长分布、节点特征分布、拉普拉斯谱分布。

    参数:
        graph_batch (list of dict): 每个元素为一个图对象，包含 'X'(节点特征), 'edge_index'(边索引) 等键。
        stats_to_compute (list[str]): 需要计算的统计量类型。
        k_spectrum (int): 拉普拉斯谱取前k个非零特征值，默认30。
        sigma (float): 计算加权拉普拉斯时的高斯核参数，默认1.0。

    返回:
        dict: 每类统计量名称为键，对应一维Numpy数组为值。
    
    使用场景:
        训练过程中监控模型生成样本的结构及属性分布，辅助评估GAN性能。
    """
    output = {}

    if 'edge_length' in stats_to_compute:
        edge_lengths = []
        for g in graph_batch:
            X = g['X']
            E = g['edge_index']
            coords = X[:, :3]
            src, tgt = E[0], E[1]
            dists = np.linalg.norm(coords[tgt] - coords[src], axis=1)
            edge_lengths.append(dists)
        output['edge_length'] = np.concatenate(edge_lengths)

    if 'node_feature' in stats_to_compute:
        features_per_dim = []
        for d in range(graph_batch[0]['X'].shape[1]):
            feats = []
            for g in graph_batch:
                feats.append(g['X'][:, d])
            features_per_dim.append(np.concatenate(feats))
        output['node_feature'] = features_per_dim  # 列表，每个维度一个数组

    if 'laplacian_spectrum' in stats_to_compute:
        spectrum = []
        for g in graph_batch:
            X = g['X']
            E = g['edge_index']
            N = X.shape[0]
            coords = X[:, :3]
            W = np.zeros((N, N), dtype=np.float32)
            src, tgt = E[0], E[1]
            deltas = coords[tgt] - coords[src]
            weights = np.exp(-np.sum(deltas ** 2, axis=1) / (2 * sigma ** 2))
            W[src, tgt] = weights
            W[tgt, src] = weights  # 对称
            D = np.diag(W.sum(axis=1))
            L = D - W
            eigvals = eigh(L, eigvals_only=True)
            eigvals = eigvals[np.argsort(eigvals)]  # 升序
            eigvals = eigvals[1:k_spectrum+1] if eigvals.shape[0] > k_spectrum+1 else eigvals[1:]
            spectrum.append(eigvals)
        output['laplacian_spectrum'] = np.concatenate(spectrum)
    return output