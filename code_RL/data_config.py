import numpy as np
import pandas as pd
import torch

# ------------------------------
#      从 Excel 文件加载任务数据，并构建图数据
# ------------------------------
# 设置任务文件路径（替换为实际路径）
TASKS_FILE_PATH = '/home/aaa/my_code/hospital-main/Simulator/Hospital_DRL/test_instances/d5_164.xlsx'
tasks = pd.read_excel(TASKS_FILE_PATH)

# 提取所有唯一节点（按照字典序排序）
unique_nodes = sorted(set(tasks['start_node']).union(set(tasks['end_node'])))
node_index = {node: idx for idx, node in enumerate(unique_nodes)}
num_nodes = len(unique_nodes)
print(f"加载任务共涉及 {num_nodes} 个节点。")

# 构建邻接矩阵：记录每对 (start_node, end_node) 出现的次数
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
for _, row in tasks.iterrows():
    start = row['start_node']
    end = row['end_node']
    adj_matrix[node_index[start], node_index[end]] += 1

# 计算度矩阵：采用出度（每个节点在邻接矩阵中所在行的和）
degree_values = adj_matrix.sum(axis=1)  # 一维数组，每个节点的出度
degree_matrix = np.diag(degree_values)

# ------------------------------
#      定义节点坐标信息（包含 X、Y、Z 坐标）
# ------------------------------
# 注意：节点名称应与任务数据中保持一致
node_positions = {
    "N021S00430": (830, 2100, 40),
    "N021S00436": (1255, 1050, 40),
    "N021S00441": (1690, 1080, 40),
    "N021S00695": (1060, 1145, 40),
    "N074S01111": (1100, 1600, 40),
    "N0401S00230": (1450, 1020, 0),
    "N0401S00280": (1305, 1200, 0),
    "N0251S00501": (715, 1120, -40)
}
# 根据 unique_nodes 顺序生成坐标矩阵（若某节点未定义则默认 (0,0,0)）
coordinates = []
for node in unique_nodes:
    if node in node_positions:
        coordinates.append(node_positions[node])
    else:
        coordinates.append((0, 0, 0))
coordinates = np.array(coordinates, dtype=np.float32)  # shape: (num_nodes, 3)

# 此处将节点坐标作为节点特征
feature_matrix = torch.tensor(coordinates, dtype=torch.float32)

# ------------------------------
#      固定的距离矩阵（8×8），数值按行排列
# ------------------------------
distance_matrix = np.array([
    [  0.0, 313.7, 153.5, 147.6, 182.5, 223.1, 120.8, 154.0],
    [313.7,   0.0,  61.3, 103.6,  54.1,  39.3, 207.6, 240.8],
    [153.5,  61.3,   0.0,  62.0,  56.3,  27.5,  47.3,  80.5],
    [147.6, 103.6,  62.0,   0.0,  95.6,  71.3,  41.4,  74.6],
    [182.5,  54.1,  56.3,  95.6,   0.0,   4.0,  76.3, 109.5],
    [223.1,  39.3,  27.5,  71.3,   4.0,   0.0, 116.9, 150.1],
    [120.8, 207.6,  47.3,  41.4,  76.3, 116.9,   0.0,  14.5],
    [154.0, 240.8,  80.5,  74.6, 109.5, 150.1,  14.5,   0.0]
], dtype=np.float32)

# 计算距离矩阵的统计信息（按行计算）
distance_mean = np.mean(distance_matrix, axis=1, keepdims=True).astype(np.float32)
distance_min  = np.min(distance_matrix, axis=1, keepdims=True).astype(np.float32)
distance_max  = np.max(distance_matrix, axis=1, keepdims=True).astype(np.float32)

# ------------------------------
#      将邻接矩阵转换为 PyG 所需格式
# ------------------------------
def adjacency_to_edge_index_and_weight(adj: np.ndarray, dist: np.ndarray = None):
    """
    将 numpy 邻接矩阵转为 (2, E) 的 edge_index，
    如果提供 dist 则利用 1.0/距离 作为边权重（防止距离为0），
    否则权重统一设为 1.0。
    """
    edge_list = []
    weights = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                edge_list.append([i, j])
                if dist is not None:
                    w = dist[i, j]
                    if w <= 1e-9:
                        w = 1e-6
                    weights.append(1.0 / w)
                else:
                    weights.append(1.0)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight

edge_index, edge_weight = adjacency_to_edge_index_and_weight(adj_matrix, distance_matrix)

# ------------------------------
#      导出数据变量供主代码使用
# ------------------------------
__all__ = [
    "adj_matrix", "degree_matrix", "feature_matrix", "distance_matrix",
    "distance_mean", "distance_min", "distance_max",
    "edge_index", "edge_weight", "unique_nodes", "node_index", "node_positions"
]
