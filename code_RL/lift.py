import numpy as np
import torch

# 电梯特征数据
lift_data = {
    'Lift Code': ['DTZ-00001', 'DTZ-00002', 'DTZ-00003', 'DTZ-00004', 'DTZ-00005', 'DTZ-00006'],
    'Average Waiting Time': [91.393061, 76.040360, 69.030981, 75.766086, 108.062206, 83.977494]
}

# 将电梯数据转换为 DataFrame
import pandas as pd
lift_df = pd.DataFrame(lift_data)

# 节点编码列表（假设与邻接矩阵对应）
node_codes = ['DTZ-00001', 'DTZ-00002', 'DTZ-00003', 'DTZ-00004', 'DTZ-00005', 'DTZ-00006']

# 创建 Lift Code 到节点索引的映射
lift_to_node_index = {code: idx for idx, code in enumerate(node_codes)}

# 初始化特征矩阵，假设每个节点只有一个特征（等待时间）
num_nodes = len(node_codes)
lift_features = np.zeros((num_nodes, 1))  # 一个特征（等待时间）

# 填充电梯等待时间到特征矩阵
for _, row in lift_df.iterrows():
    lift_code = row['Lift Code']
    waiting_time = row['Average Waiting Time']
    if lift_code in lift_to_node_index:  # 确保电梯编码存在于节点列表中
        node_index = lift_to_node_index[lift_code]
        lift_features[node_index, 0] = waiting_time

# 转换为 PyTorch 张量
lift_features_tensor = torch.tensor(lift_features, dtype=torch.float32)

print("Lift Features Tensor:")
print(lift_features_tensor)
