# src/data_loader/dataset.py

import os
import glob
import torch
from torch_geometric.data import Dataset
from typing import List, Callable, Optional

class BrainGraphDataset(Dataset):
    """
    用于加载预处理好的脑图谱图数据的 PyTorch Geometric 数据集。

    该数据集假定图数据已经被预处理并保存为单个的 .pt 文件，
    每个文件包含一个 torch_geometric.data.Data 对象。
    此类的主要职责是发现这些 .pt 文件并按需加载它们。
    """

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        """
        初始化数据集。

        Args:
            root (str): 数据集的根目录。按照 PyG 的惯例，
                        该目录下应包含一个名为 'processed' 的子目录，
                        其中存放着 .pt 文件。例如，如果文件在 'data/processed/'，
                        则 root 应设置为 'data'。
            transform (Callable, optional): 每次访问数据时应用的动态变换。默认为 None。
            pre_transform (Callable, optional): 数据在保存到磁盘前应用的变换
                                             （在此设计中已由外部脚本完成）。默认为 None。
        """
        super().__init__(root, transform, pre_transform)
        
        # 扫描 processed_dir 目录 (例如: data/processed)，获取所有 .pt 文件路径的列表并排序。
        # 排序可以确保每次运行时数据集的顺序都是一致的，这对于复现实验结果很重要。
        self.processed_file_paths: List[str] = sorted(glob.glob(os.path.join(self.processed_dir, '*.pt')))

    @property
    def raw_file_names(self) -> List[str]:
        """
        由于数据已预处理，我们不需要原始文件。返回一个空列表。
        """
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """
        返回在 `processed_dir` 目录中应该存在的文件名列表。
        这里我们返回扫描到的 .pt 文件的基本名称（不含路径）。
        这有助于 PyG 的内部检查机制。
        """
        return [os.path.basename(path) for path in self.processed_file_paths]

    def len(self) -> int:
        """
        返回数据集中图的总数。
        """
        return len(self.processed_file_paths)

    def get(self, idx: int):
        """
        根据索引获取一个数据样本。

        Args:
            idx (int): 要获取的数据样本的索引。

        Returns:
            torch_geometric.data.Data: 从磁盘加载的单个图数据对象。
        """
        # 1. 根据索引获取对应的文件路径
        path = self.processed_file_paths[idx]
        
        # 2. 使用 torch.load 从磁盘反序列化该文件，得到一个 PyG.Data 对象
        data = torch.load(path)
        
        return data

# --- 使用示例 ---
if __name__ == '__main__':
    # 为了能独立运行此文件进行测试，我们模拟一些环境
    from torch_geometric.data import Data
    
    # 1. 创建一个模拟的 data/processed 目录
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    # 2. 创建几个假的 .pt 数据文件
    for i in range(10):
        # 创建一个简单的图数据对象
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16) # 3个节点, 每个节点16个特征
        y = torch.tensor([i % 2], dtype=torch.float) # 标签
        
        sample_data = Data(x=x, edge_index=edge_index, y=y)
        sample_data.subject_id = f'sub_{i}' # 还可以添加自定义属性
        
        # 保存到 .pt 文件
        torch.save(sample_data, f'data/processed/graph_{i}.pt')

    print("创建了10个模拟的 .pt 文件在 'data/processed/' 目录下。")

    # 3. 初始化数据集
    # 注意 `root` 参数指向 `data` 目录
    brain_dataset = BrainGraphDataset(root='data')

    # 4. 测试数据集功能
    print(f"\n数据集大小 (len): {len(brain_dataset)}")

    # 获取第一个样本
    first_sample = brain_dataset.get(0)
    print(f"\n获取第一个样本 (get(0)):")
    print(first_sample)
    print(f"样本中的自定义属性 subject_id: {first_sample.subject_id}")


    # PyG 的 DataLoader 会自动使用这个数据集
    from torch_geometric.loader import DataLoader

    # batch_size=4 表示每个批次加载4个图
    loader = DataLoader(brain_dataset, batch_size=4, shuffle=True)

    print("\n使用 DataLoader 加载数据...")
    for batch in loader:
        print("--- New Batch ---")
        print(batch)
        print(f"批次中的图数量: {batch.num_graphs}")
        print(f"批次中的节点总数: {batch.num_nodes}")
        print(f"批次标签: {batch.y}")