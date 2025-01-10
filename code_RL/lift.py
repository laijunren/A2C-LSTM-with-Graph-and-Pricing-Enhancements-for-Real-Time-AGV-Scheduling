import pandas as pd
import numpy as np

# 从 Excel 文件中加载数据
# 假设你的文件名为 "node_connections.xlsx"，包含 "start_node" 和 "end_node" 两列
file_path = "/home/aaa/my_code/hospital-main/robot_data/robot_node.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 获取所有唯一的节点
nodes = sorted(set(df['start_node']).union(set(df['end_node'])))
node_to_index = {node: i for i, node in enumerate(nodes)}

# 初始化邻接矩阵
adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

# 填充邻接矩阵
for _, row in df.iterrows():
    start_index = node_to_index[row['start_node']]
    end_index = node_to_index[row['end_node']]
    adj_matrix[start_index, end_index] += 1  # 如果是无向图，增加对称赋值： adj_matrix[end_index, start_index] += 1

# 将邻接矩阵转换为 DataFrame 以便查看
adj_matrix_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)

# 保存为 CSV 文件
output_path = "adjacency_matrix2.csv"  # 替换为想要保存的路径
adj_matrix_df.to_csv(output_path, index=True)

# 打印结果
print(f"生成的邻接矩阵已保存为 {output_path}")
