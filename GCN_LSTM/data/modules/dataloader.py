import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse

def load_data(adjacency_path, degree_path, feature_path):
    # 加载邻接矩阵
    adjacency_matrix = pd.read_csv(adjacency_path, header=None)
    
    # 清洗非数值数据
    adjacency_matrix = adjacency_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
    adjacency_matrix = adjacency_matrix.values

    # 加载其他数据
    degree_matrix = pd.read_csv(degree_path, header=None).values
    feature_matrix = pd.read_csv(feature_path)['Average Waiting Time'].values.reshape(-1, 1)
    
    # 转换为 PyTorch 张量
    adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
    edge_index, edge_weight = dense_to_sparse(adjacency_tensor)

    return feature_tensor, edge_index, edge_weight
