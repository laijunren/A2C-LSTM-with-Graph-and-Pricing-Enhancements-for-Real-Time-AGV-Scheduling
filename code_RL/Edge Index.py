import torch

# 邻接矩阵
adjacency_matrix = torch.tensor([
    [0, 15, 0, 6, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [11, 4, 0, 3, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [4, 7, 0, 8, 0, 0, 0, 10],
    [7, 5, 0, 6, 0, 0, 0, 3],
    [5, 11, 0, 7, 0, 0, 0, 6],
    [15, 7, 0, 9, 0, 0, 0, 0],
], dtype=torch.float32)

# 特征矩阵
feature_matrix = torch.tensor([
    [830, 2100, 0],
    [1255, 1050, 0],
    [1690, 1080, 0],
    [1060, 1145, 0],
    [715, 1120, 0],
    [1450, 1020, 0],
    [1305, 1200, 0],
    [1100, 1600, 0],
], dtype=torch.float32)

# 转换邻接矩阵为 PyTorch Geometric 稀疏格式
from torch_geometric.utils import dense_to_sparse
edge_index, edge_weight = dense_to_sparse(adjacency_matrix)

print("Edge Index:")
print(edge_index)

print("\nFeature Matrix:")
print(feature_matrix)
