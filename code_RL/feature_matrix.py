import numpy as np
import pandas as pd

# 邻接矩阵（表示节点之间的连接）
adjacency_matrix = np.array([
    [0, 15, 0, 6, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [11, 4, 0, 3, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [4, 7, 0, 8, 0, 0, 0, 10],
    [7, 5, 0, 6, 0, 0, 0, 3],
    [5, 11, 0, 7, 0, 0, 0, 6],
    [15, 7, 0, 9, 0, 0, 0, 0],
])

# 节点位置信息
nodes = {
    "N0251S00501": {"X": 715, "Y": 1120, "Z": 0},
    "N021S00430": {"X": 830, "Y": 2100, "Z": 0},
    "N021S00695": {"X": 1060, "Y": 1145, "Z": 0},
    "N021S00436": {"X": 1255, "Y": 1050, "Z": 0},
    "N021S00441": {"X": 1690, "Y": 1080, "Z": 0},
    "N074S01111": {"X": 1100, "Y": 1600, "Z": 0},
    "N0401S00230": {"X": 1450, "Y": 1020, "Z": 0},
    "N0401S00280": {"X": 1305, "Y": 1200, "Z": 0},
}

# 确保邻接矩阵的节点顺序与位置信息的顺序一致
node_order = [
    "N021S00430", "N021S00436", "N021S00441", "N021S00695",
    "N0251S00501", "N0401S00230", "N0401S00280", "N074S01111"
]

# 构建特征矩阵
feature_matrix = np.array([
    [nodes[node]["X"], nodes[node]["Y"], nodes[node]["Z"]] for node in node_order
])

# 转换为 DataFrame 以便查看
feature_df = pd.DataFrame(feature_matrix, columns=["X", "Y", "Z"], index=node_order)

# 打印邻接矩阵和特征矩阵
print("邻接矩阵：")
print(adjacency_matrix)
print("\n特征矩阵：")
print(feature_df)

# 保存特征矩阵为 CSV 文件
feature_df.to_csv("feature_matrix.csv", index_label="Node")
print("特征矩阵已保存为 'feature_matrix.csv'")
