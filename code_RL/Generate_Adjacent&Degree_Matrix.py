import pandas as pd
import numpy as np

# 加载任务表格
file_path = '/home/aaa/my_code/hospital-main/Simulator/Hospital_DRL/test_instances/d5_164.xlsx'  # 替换为实际路径
tasks = pd.read_excel(file_path)  # 使用 read_excel 函数读取 .xlsx 文件

# 提取所有唯一节点
unique_nodes = sorted(set(tasks['start_node']).union(set(tasks['end_node'])))
node_index = {node: idx for idx, node in enumerate(unique_nodes)}

# 初始化邻接矩阵
adj_matrix = np.zeros((len(unique_nodes), len(unique_nodes)), dtype=int)

# 填充邻接矩阵
for _, row in tasks.iterrows():
    start = row['start_node']
    end = row['end_node']
    adj_matrix[node_index[start], node_index[end]] += 1

# 转换为 DataFrame
adj_matrix_df = pd.DataFrame(adj_matrix, index=unique_nodes, columns=unique_nodes)

# 计算度矩阵
degree_values = adj_matrix_df.sum(axis=1)  # 每行求和表示出度
degree_matrix = np.diag(degree_values)
degree_matrix_df = pd.DataFrame(degree_matrix, index=unique_nodes, columns=unique_nodes)

# 输出邻接矩阵和度矩阵
print("邻接矩阵：")
print(adj_matrix_df)
print("度矩阵：")
print(degree_matrix_df)

# 设置保存路径
adj_matrix_output_path = '/home/aaa/my_code/hospital-main/Simulator/Hospital_DRL/output/adjacency_matrix.csv'  # 替换为实际路径
degree_matrix_output_path = '/home/aaa/my_code/hospital-main/Simulator/Hospital_DRL/output/degree_matrix.csv'  # 替换为实际路径

# 保存邻接矩阵
adj_matrix_df.to_csv(adj_matrix_output_path, index=True)
print(f"邻接矩阵已保存到: {adj_matrix_output_path}")

# 保存度矩阵
degree_matrix_df.to_csv(degree_matrix_output_path, index=True)
print(f"度矩阵已保存到: {degree_matrix_output_path}")


