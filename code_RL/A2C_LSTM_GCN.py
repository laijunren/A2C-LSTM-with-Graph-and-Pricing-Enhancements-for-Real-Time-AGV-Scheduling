# import logging
# import math
# import os
# import random
# import time
# from datetime import datetime

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from Environment import make_env  # 你自己的环境

# seed = 2024
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True

# output_path = "./A2C_output/"
# STATE_DIM = 22    # 环境 state 的维度
# ACTION_DIM = 9    # 动作空间大小
# NUM_EPISODE = 4000  # 训练回合数(示例改小一点)
# A_HIDDEN = 256    # Actor LSTM隐藏层大小
# C_HIDDEN = 256    # Critic LSTM隐藏层大小
# a_lr = 1e-3
# c_lr = 1e-3

# # 定义邻接矩阵和节点特征
# adjacency_matrix = np.array([
#     [0, 15, 0, 6, 0, 0, 0, 8],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [11, 4, 0, 3, 0, 0, 0, 7],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [4, 7, 0, 8, 0, 0, 0, 10],
#     [7, 5, 0, 6, 0, 0, 0, 3],
#     [5, 11, 0, 7, 0, 0, 0, 6],
#     [15, 7, 0, 9, 0, 0, 0, 0],
# ])

# feature_matrix = torch.tensor([
#     [830., 2100., 40.],
#     [1255., 1050., 40.],
#     [1690., 1080., 40.],
#     [1060., 1145., 40.],
#     [715., 1120., -40.],
#     [1450., 1020., 0.],
#     [1305., 1200., 0.],
#     [1100., 1600., 40.],
# ], dtype=torch.float)

# # 定义度矩阵并提取对角线元素
# degree_matrix = np.array([
#     [29, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 25, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 29, 0, 0, 0],
#     [0, 0, 0, 0, 0, 21, 0, 0],
#     [0, 0, 0, 0, 0, 0, 29, 0],
#     [0, 0, 0, 0, 0, 0, 0, 31]
# ])
# degree_values = np.diag(degree_matrix)  # 提取对角线元素
# degree_values_tensor = torch.tensor(degree_values, dtype=torch.float).view(-1, 1)

# # 添加距离矩阵信息
# distance_matrix = np.array([
#     [0.0, 61.3, 62.0, 54.1, 313.7, 207.6, 240.8, 39.3],
#     [61.3, 0.0, 56.3, 95.6, 153.5, 47.3, 80.5, 27.5],
#     [62.0, 56.3, 0.0, 71.3, 147.6, 41.4, 74.6, 71.3],
#     [54.1, 95.6, 71.3, 0.0, 182.5, 76.3, 109.5, 4.0],
#     [313.7, 153.5, 147.6, 182.5, 0.0, 120.8, 154.0, 223.1],
#     [207.6, 47.3, 41.4, 76.3, 120.8, 0.0, 14.5, 116.9],
#     [240.8, 80.5, 74.6, 109.5, 154.0, 14.5, 0.0, 150.1],
#     [39.3, 27.5, 71.3, 4.0, 223.1, 116.9, 150.1, 0.0],
# ])

# # 提取每行的统计信息（平均值、最小值和最大值）
# distance_mean = torch.tensor(np.mean(distance_matrix, axis=1), dtype=torch.float).view(-1, 1)
# distance_min = torch.tensor(np.min(distance_matrix, axis=1), dtype=torch.float).view(-1, 1)
# distance_max = torch.tensor(np.max(distance_matrix, axis=1), dtype=torch.float).view(-1, 1)

# # 更新节点特征矩阵
# feature_matrix = torch.cat([feature_matrix, degree_values_tensor, distance_mean, distance_min, distance_max], dim=1)


# # 将邻接矩阵转为 PyG 的 edge_index
# def adjacency_to_edge_index(adj: np.ndarray):
#     """
#     将 numpy 邻接矩阵转换为 (2, E) 的 edge_index PyG格式
#     只保留非零权重的边（有向图）。
#     """
#     edge_list = []
#     for i in range(adj.shape[0]):
#         for j in range(adj.shape[1]):
#             if adj[i, j] != 0:
#                 edge_list.append([i, j])
#     if not edge_list:
#         return torch.empty((2, 0), dtype=torch.long)
#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#     return edge_index

# edge_index = adjacency_to_edge_index(adjacency_matrix)

# # 图相关信息
# num_node_features = feature_matrix.size(1)  # 更新为 7（位置特征3 + 度1 + 距离特征3）
# gcn_hidden_channels = 32                   # GCN隐藏层维度

# # 定义网络结构
# class ActorNetwork(nn.Module):
#     def __init__(self, state_dim, gcn_in_channels, gcn_hidden_channels, lstm_hidden_size, action_dim):
#         super(ActorNetwork, self).__init__()
#         # GCN部分
#         self.gcn1 = GCNConv(gcn_in_channels, gcn_hidden_channels)
#         self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)

#         # 把 "图卷积输出" + "环境状态" 进行拼接 => 作为 LSTM 的输入
#         self.lstm_input_dim = gcn_hidden_channels + state_dim

#         self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_size, batch_first=True)
#         self.fc = nn.Linear(lstm_hidden_size, action_dim)

#     def forward(self, x_graph, edge_index, x_state, hidden):
#         # 1) 图卷积
#         x_graph = self.gcn1(x_graph, edge_index)
#         x_graph = F.relu(x_graph)
#         x_graph = self.gcn2(x_graph, edge_index)
#         x_graph = F.relu(x_graph)

#         # 2) global mean pooling
#         x_graph = x_graph.mean(dim=0, keepdim=True)  # [1, gcn_hidden_channels]

#         # 3) 拼接图特征与环境状态
#         x_combined = torch.cat([x_graph, x_state], dim=-1)  # [1, gcn_hidden_channels + state_dim]

#         # 4) LSTM需要 (batch, seq, feature_dim)
#         x_combined = x_combined.unsqueeze(0)  # => [1, 1, (gcn_hidden_channels + state_dim)]
#         x_out, hidden = self.lstm(x_combined, hidden)  # => x_out: [1,1,lstm_hidden_size]

#         # 5) 全连接输出(action_dim)，再做 log_softmax
#         x_out = self.fc(x_out)  # => [1,1,action_dim]
#         x_out = F.log_softmax(x_out, dim=2)
#         return x_out, hidden

# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim, gcn_in_channels, gcn_hidden_channels, lstm_hidden_size):
#         super(ValueNetwork, self).__init__()
#         # GCN
#         self.gcn1 = GCNConv(gcn_in_channels, gcn_hidden_channels)
#         self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)

#         # 把图输出和状态拼接
#         self.lstm_input_dim = gcn_hidden_channels + state_dim
#         self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_size, batch_first=True)

#         # Critic输出维度=1
#         self.fc = nn.Linear(lstm_hidden_size, 1)

#     def forward(self, x_graph, edge_index, x_state, hidden):
#         # 1) GCN
#         x_graph = self.gcn1(x_graph, edge_index)
#         x_graph = F.relu(x_graph)
#         x_graph = self.gcn2(x_graph, edge_index)
#         x_graph = F.relu(x_graph)

#         # 2) global mean
#         x_graph = x_graph.mean(dim=0, keepdim=True)  # => [1, gcn_hidden_channels]

#         # 3) 拼接 state
#         x_combined = torch.cat([x_graph, x_state], dim=-1)  # => [1, gcn_hidden_channels + state_dim]

#         # 4) LSTM
#         x_combined = x_combined.unsqueeze(0)  # => [1, 1, (gcn_hidden_channels + state_dim)]
#         x_out, hidden = self.lstm(x_combined, hidden)

#         # 5) 全连接 -> 价值
#         x_out = self.fc(x_out)  # => [1,1,1]
#         return x_out, hidden

# # 其余部分保持不变（如 roll_out 和 main 函数）
# def calculate_reward(final_obs, optimal_theoretical):
#     """
#     final_obs: ( (batch_of_states), total_mpn )
#                其中 final_obs[0] 是若干工件的 (wait, setUp, execTime, lift)
#                final_obs[1] 是总的 mpn
#     optimal_theoretical: 理论最优值
#     """
#     rewards = []
#     total_mpn = final_obs[1]
#     if total_mpn <= 0.0:
#         total_mpn = 100000
#         print("error: simulator run out of time")
#     assert total_mpn > 0
#     makespan = math.log((total_mpn - optimal_theoretical) / 100, 1.3) - 2
#     for i in range(len(final_obs[0])):
#         wait, setUp, execTime, lift = final_obs[0][i]
#         a = math.log(wait * 0.01 + math.e) - 1
#         b = math.log(setUp * 0.02 + 2, 2) - 1
#         c = execTime * 0.002
#         d = lift * 0.005
#         rewards.append(-(a + b + c + d + makespan - 11) / 6)
#     return rewards

# def discount_reward(r, gamma, final_r):
#     discounted_r = np.zeros_like(r)
#     running_add = final_r
#     for t in reversed(range(0, len(r))):
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r


# # roll_out 时用 (feature_matrix, edge_index, 当前state)
# def roll_out(actor, env, random=False):
#     states = []
#     actions = []
#     done = False
#     final_r = 0
#     state = env.reset()  # 初始时环境返回的状态(长度=22)
#     reward = None

#     # 初始化 Actor 的 LSTM 隐状态
#     a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
#     a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)

#     while not done:
#         # 记录下当前 state
#         states.append(state)

#         # 把 state 转成张量 shape=(1, STATE_DIM)
#         env_state_tensor = torch.FloatTensor(state).unsqueeze(0)

#         # 前向：ActorNetwork(包含GCN + LSTM)
#         log_softmax_action, (a_hx, a_cx) = actor(
#             feature_matrix, edge_index, env_state_tensor, (a_hx, a_cx)
#         )

#         # 采样动作
#         prob = torch.exp(log_softmax_action).cpu().data.numpy()[0][0]  # shape: (ACTION_DIM,)
#         action = np.random.choice(ACTION_DIM, p=prob)

#         # 环境执行动作
#         done, next_state, reward = env.step(action)

#         # 将动作独热保存
#         one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
#         actions.append(one_hot_action)

#         # 更新 state
#         state = next_state

#     makespan = reward[1]
#     rewards = calculate_reward(reward, 7300)

#     return states, actions, rewards, final_r, state, makespan


# # 主训练循环：同样在训练时，把 (feature_matrix, edge_index, states_var[t]) 送给网络

# def main():
#     LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
#     logger = logging.getLogger('main')
#     logger.setLevel(logging.DEBUG)

#     # 创建环境
#     env = make_env()

#     # 实例化价值网络Critic
#     value_network = ValueNetwork(
#         state_dim=STATE_DIM,
#         gcn_in_channels=num_node_features,
#         gcn_hidden_channels=gcn_hidden_channels,
#         lstm_hidden_size=C_HIDDEN
#     )
#     value_network_optim = torch.optim.Adam(value_network.parameters(), lr=c_lr)

#     # 实例化策略网络Actor
#     actor_network = ActorNetwork(
#         state_dim=STATE_DIM,
#         gcn_in_channels=num_node_features,
#         gcn_hidden_channels=gcn_hidden_channels,
#         lstm_hidden_size=A_HIDDEN,
#         action_dim=ACTION_DIM
#     )
#     actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=a_lr)

#     # 记录训练过程数据
#     actor_loss_records = []
#     critic_loss_records = []
#     episode_rewards = []
#     episode_makespan = []

#     is_random = False
#     start_time = time.time()

#     for episode in range(1, NUM_EPISODE + 1):
#         # roll_out 采集一条完整序列
#         states, actions, rewards, final_r, current_state, mpn = roll_out(actor_network, env, is_random)

#         # 转成PyTorch张量
#         actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)  # [1, T, ACTION_DIM]
#         states_var = torch.Tensor(states).view(-1, STATE_DIM)                  # [T, STATE_DIM]
#         T = states_var.size(0)

#         # 训练 Actor
#         a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
#         a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
#         actor_network_optim.zero_grad()

#         log_softmax_actions_list = []
#         for t in range(T):
#             # 取第 t 步的 env state
#             env_state_t = states_var[t].unsqueeze(0)  # [1, STATE_DIM]

#             # 每一步都用 (feature_matrix, edge_index, env_state_t)
#             out, (a_hx, a_cx) = actor_network(feature_matrix, edge_index, env_state_t, (a_hx, a_cx))
#             log_softmax_actions_list.append(out)  # shape (1,1,ACTION_DIM)

#         # 拼接T步
#         log_softmax_actions = torch.cat(log_softmax_actions_list, dim=1)  # (1,T,ACTION_DIM)

#         # 训练 Critic
#         c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
#         c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
#         value_network_optim.zero_grad()

#         vs_list = []
#         for t in range(T):
#             env_state_t = states_var[t].unsqueeze(0)  # [1, STATE_DIM]
#             out, (c_hx, c_cx) = value_network(feature_matrix, edge_index, env_state_t, (c_hx, c_cx))
#             vs_list.append(out)

#         vs = torch.cat(vs_list, dim=1)  # (1,T,1)
#         vs_detached = vs.detach()       # 用于计算优势

#         # 计算折扣回报和优势
#         qs = torch.Tensor(discount_reward(rewards, 0.99, final_r))  # (T,)
#         qs = qs.view(1, -1, 1)  # (1, T, 1)
#         advantages = qs - vs_detached

#         # Actor loss
#         # log_softmax_actions: (1,T,ACTION_DIM)
#         # actions_var:         (1,T,ACTION_DIM)
#         probs = torch.sum(log_softmax_actions * actions_var, dim=2, keepdim=True)  # (1,T,1)
#         actor_network_loss = - torch.mean(probs * advantages)
#         actor_network_loss.backward()
#         torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
#         actor_network_optim.step()

#         # Critic loss
#         criterion = nn.MSELoss()
#         value_network_loss = criterion(vs, qs)
#         value_network_loss.backward()
#         torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
#         value_network_optim.step()

#         # 记录并打印
#         actor_loss_records.append(actor_network_loss.item())
#         critic_loss_records.append(value_network_loss.item())
#         episode_makespan.append(mpn)
#         episode_reward = np.sum(rewards)
#         episode_rewards.append(episode_reward)

#         logger.info(f"[Episode {episode}] makespan={mpn}, reward={episode_reward}")

#     end_time = time.time()
#     print('Training time:', end_time - start_time)

#     # 保存结果
#     os.makedirs(output_path, exist_ok=True)
#     date_time = datetime.now().strftime("%m_%d_%H_%M")

#     torch.save(actor_network.state_dict(), os.path.join(output_path, f"Actor_{date_time}.pth"))
#     torch.save(value_network.state_dict(), os.path.join(output_path, f"Critic_{date_time}.pth"))

#     with open(os.path.join(output_path, f'makespan_{date_time}.txt'), 'w') as f:
#         for r in episode_makespan:
#             f.write(str(r) + '\n')
#     with open(os.path.join(output_path, f'rewards_{date_time}.txt'), 'w') as f:
#         for r in episode_rewards:
#             f.write(str(r) + '\n')
#     with open(os.path.join(output_path, f'actor_loss_{date_time}.txt'), 'w') as f:
#         for loss in actor_loss_records:
#             f.write(f"{loss}\n")
#     with open(os.path.join(output_path, f'critic_loss_{date_time}.txt'), 'w') as f:
#         for loss in critic_loss_records:
#             f.write(f"{loss}\n")

# if __name__ == '__main__':
#     main()



import logging
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from Environment import make_env  # 你自己的环境

# ------------------------------
#         超参 & 固定随机种子
# ------------------------------
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

output_path = "./A2C_output/"
STATE_DIM = 22      # 环境 state 的维度
ACTION_DIM = 9      # 动作空间大小
NUM_EPISODE = 5000  # 训练回合数
A_HIDDEN = 512      # Actor LSTM隐藏层大小
C_HIDDEN = 512    # Critic LSTM隐藏层大小
a_lr = 1e-3
c_lr = 1e-3

# ------------------------------
#        定义图结构和特征
# ------------------------------
adjacency_matrix = np.array([
    [0, 15, 0,  6,  0,  0,  0,  8],
    [0,  0, 0,  0,  0,  0,  0,  0],
    [11, 4, 0,  3,  0,  0,  0,  7],
    [0,  0, 0,  0,  0,  0,  0,  0],
    [4,  7, 0,  8,  0,  0,  0, 10],
    [7,  5, 0,  6,  0,  0,  0,  3],
    [5, 11, 0,  7,  0,  0,  0,  6],
    [15,7,  0,  9,  0,  0,  0,  0],
])

feature_matrix = np.array([
    [830.,  2100.,   40.],
    [1255., 1050.,   40.],
    [1690., 1080.,   40.],
    [1060., 1145.,   40.],
    [715.,  1120.,  -40.],
    [1450., 1020.,    0.],
    [1305., 1200.,    0.],
    [1100., 1600.,   40.],
], dtype=np.float32)

# 度矩阵 (提取对角线元素)
degree_matrix = np.array([
    [29, 0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0, 25,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0, 29,  0,  0,  0],
    [0,  0,  0,  0,  0, 21,  0,  0],
    [0,  0,  0,  0,  0,  0, 29,  0],
    [0,  0,  0,  0,  0,  0,  0, 31]
])
degree_values = np.diag(degree_matrix).reshape(-1,1).astype(np.float32)

# 距离矩阵 (用于边权重 & 节点统计信息)
distance_matrix = np.array([
    [  0.0,  61.3,  62.0,  54.1, 313.7, 207.6, 240.8,  39.3],
    [ 61.3,   0.0,  56.3,  95.6, 153.5,  47.3,  80.5,  27.5],
    [ 62.0,  56.3,   0.0,  71.3, 147.6,  41.4,  74.6,  71.3],
    [ 54.1,  95.6,  71.3,   0.0, 182.5,  76.3, 109.5,   4.0],
    [313.7, 153.5, 147.6, 182.5,   0.0, 120.8, 154.0, 223.1],
    [207.6,  47.3,  41.4,  76.3, 120.8,   0.0,  14.5, 116.9],
    [240.8,  80.5,  74.6, 109.5, 154.0,  14.5,   0.0, 150.1],
    [ 39.3,  27.5,  71.3,   4.0, 223.1, 116.9, 150.1,   0.0],
], dtype=np.float32)

# 把 distance_matrix 的每行做一些统计特征
distance_mean = np.mean(distance_matrix, axis=1, keepdims=True).astype(np.float32)
distance_min  = np.min(distance_matrix,  axis=1, keepdims=True).astype(np.float32)
distance_max  = np.max(distance_matrix,  axis=1, keepdims=True).astype(np.float32)

# 拼回节点特征(3 + 1 + 3 = 7 列)，不做标准化
feature_matrix = np.concatenate([
    feature_matrix,
    degree_values,
    distance_mean,
    distance_min,
    distance_max
], axis=1)  # 得到 shape [8, 7]

# 转成 torch.Tensor
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)

num_node_features = feature_matrix.size(1)
gcn_hidden_channels = 32

# ------------------------------
#      将邻接矩阵转为 PyG 格式
#      并利用 distance 作为边权重
# ------------------------------
def adjacency_to_edge_index_and_weight(adj: np.ndarray, dist: np.ndarray):
    """
    将 numpy 邻接矩阵 adj 转为 (2, E) 的 edge_index，
    同时返回 edge_weight (shape=[E]) 用于 GCNConv。
    示例中用 1.0/distance (若 distance=0 则设为 1e-6)，
    你可以换成其它做法。
    """
    edge_list = []
    weights = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                edge_list.append([i, j])
                w = dist[i,j]
                if w <= 1e-9:
                    w = 1e-6
                weights.append(1.0 / w)  # 距离越小 => 权重越大

    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight

edge_index, edge_weight = adjacency_to_edge_index_and_weight(adjacency_matrix, distance_matrix)

# ------------------------------
#      定义 Actor/Critic 网络
# ------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, gcn_in_channels, gcn_hidden_channels, lstm_hidden_size, action_dim):
        super(ActorNetwork, self).__init__()
        self.gcn1 = GCNConv(gcn_in_channels, gcn_hidden_channels)
        self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)

        self.lstm_input_dim = gcn_hidden_channels + state_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, action_dim)

    def forward(self, x_graph, edge_index, edge_weight, x_state, hidden):
        # 1) GCN
        x_graph = self.gcn1(x_graph, edge_index, edge_weight=edge_weight)
        x_graph = F.relu(x_graph)
        x_graph = self.gcn2(x_graph, edge_index, edge_weight=edge_weight)
        x_graph = F.relu(x_graph)

        # 2) global mean pooling
        x_graph = x_graph.mean(dim=0, keepdim=True)  # [1, gcn_hidden_channels]

        # 3) 拼接图特征与环境状态
        x_combined = torch.cat([x_graph, x_state], dim=-1)

        # 4) LSTM (batch=1, seq=1)
        x_combined = x_combined.unsqueeze(0)  # => [1,1,lstm_input_dim]
        x_out, hidden = self.lstm(x_combined, hidden)

        # 5) 输出 action log_prob
        x_out = self.fc(x_out)  # => [1,1,action_dim]
        x_out = F.log_softmax(x_out, dim=2)
        return x_out, hidden

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, gcn_in_channels, gcn_hidden_channels, lstm_hidden_size):
        super(ValueNetwork, self).__init__()
        self.gcn1 = GCNConv(gcn_in_channels, gcn_hidden_channels)
        self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)

        self.lstm_input_dim = gcn_hidden_channels + state_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x_graph, edge_index, edge_weight, x_state, hidden):
        x_graph = self.gcn1(x_graph, edge_index, edge_weight=edge_weight)
        x_graph = F.relu(x_graph)
        x_graph = self.gcn2(x_graph, edge_index, edge_weight=edge_weight)
        x_graph = F.relu(x_graph)

        x_graph = x_graph.mean(dim=0, keepdim=True)
        x_combined = torch.cat([x_graph, x_state], dim=-1)

        x_combined = x_combined.unsqueeze(0)
        x_out, hidden = self.lstm(x_combined, hidden)
        x_out = self.fc(x_out)
        return x_out, hidden

# ------------------------------
#       其余部分（Reward等）
# ------------------------------
def calculate_reward(final_obs, optimal_theoretical):
    """
    final_obs: ( (batch_of_states), total_mpn )
               其中 final_obs[0] 是若干工件的 (wait, setUp, execTime, lift)
               final_obs[1] 是总的 mpn
    optimal_theoretical: 理论最优值
    """
    rewards = []
    total_mpn = final_obs[1]
    if total_mpn <= 0.0:
        total_mpn = 100000
        print("error: simulator run out of time")
    assert total_mpn > 0
    makespan = math.log((total_mpn - optimal_theoretical) / 100, 1.3) - 2
    for i in range(len(final_obs[0])):
        wait, setUp, execTime, lift = final_obs[0][i]
        a = math.log(wait * 0.01 + math.e) - 1
        b = math.log(setUp * 0.02 + 2, 2) - 1
        c = execTime * 0.002
        d = lift * 0.005
        rewards.append(-(a + b + c + d + makespan - 11) / 6)
    return rewards

def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# ------------------------------
#       roll_out 函数
# ------------------------------
def roll_out(actor, env, random=False):
    states = []
    actions = []
    done = False
    final_r = 0
    state = env.reset()  # 初始时环境返回的状态(长度=22)
    reward = None

    # 初始化 Actor 的 LSTM 隐状态
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)

    while not done:
        states.append(state)
        env_state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 前向：ActorNetwork(包含GCN + LSTM)
        log_softmax_action, (a_hx, a_cx) = actor(
            feature_matrix, edge_index, edge_weight,
            env_state_tensor, (a_hx, a_cx)
        )

        # 采样动作
        prob = torch.exp(log_softmax_action).cpu().data.numpy()[0][0]  # shape: (ACTION_DIM,)
        action = np.random.choice(ACTION_DIM, p=prob)

        # 环境执行动作
        done, next_state, reward = env.step(action)

        # 将动作独热保存
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        actions.append(one_hot_action)

        # 更新 state
        state = next_state

    makespan = reward[1]
    rewards = calculate_reward(reward, 7300)

    return states, actions, rewards, final_r, state, makespan

# ------------------------------
#       主训练循环
# ------------------------------
def main():
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # 创建环境
    env = make_env()

    # 实例化价值网络Critic
    value_network = ValueNetwork(
        state_dim=STATE_DIM,
        gcn_in_channels=num_node_features,
        gcn_hidden_channels=gcn_hidden_channels,
        lstm_hidden_size=C_HIDDEN
    )
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=c_lr)

    # 实例化策略网络Actor
    actor_network = ActorNetwork(
        state_dim=STATE_DIM,
        gcn_in_channels=num_node_features,
        gcn_hidden_channels=gcn_hidden_channels,
        lstm_hidden_size=A_HIDDEN,
        action_dim=ACTION_DIM
    )
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=a_lr)

    actor_loss_records = []
    critic_loss_records = []
    episode_rewards = []
    episode_makespan = []

    is_random = False
    start_time = time.time()

    for episode in range(1, NUM_EPISODE + 1):
        states, actions, rewards, final_r, current_state, mpn = roll_out(actor_network, env, is_random)

        actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)  # [1, T, ACTION_DIM]
        states_var = torch.Tensor(states).view(-1, STATE_DIM)  # [T, STATE_DIM]
        T = states_var.size(0)

        # 训练 Actor
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        actor_network_optim.zero_grad()

        log_softmax_actions_list = []
        for t in range(T):
            env_state_t = states_var[t].unsqueeze(0)
            out, (a_hx, a_cx) = actor_network(
                feature_matrix, edge_index, edge_weight,
                env_state_t, (a_hx, a_cx)
            )
            log_softmax_actions_list.append(out)

        log_softmax_actions = torch.cat(log_softmax_actions_list, dim=1)  # (1,T,ACTION_DIM)

        # 训练 Critic
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        value_network_optim.zero_grad()

        vs_list = []
        for t in range(T):
            env_state_t = states_var[t].unsqueeze(0)
            v_out, (c_hx, c_cx) = value_network(
                feature_matrix, edge_index, edge_weight,
                env_state_t, (c_hx, c_cx)
            )
            vs_list.append(v_out)

        vs = torch.cat(vs_list, dim=1)       # (1,T,1)
        vs_detached = vs.detach()            # 用于计算优势

        qs = torch.Tensor(discount_reward(rewards, 0.99, final_r))  # (T,)
        qs = qs.view(1, -1, 1)               # (1,T,1)
        advantages = qs - vs_detached        # (1,T,1)

        # Actor loss
        probs = torch.sum(log_softmax_actions * actions_var, dim=2, keepdim=True)  # (1,T,1)
        actor_network_loss = - torch.mean(probs * advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
        actor_network_optim.step()

        # Critic loss
        criterion = nn.MSELoss()
        value_network_loss = criterion(vs, qs)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
        value_network_optim.step()

        actor_loss_records.append(actor_network_loss.item())
        critic_loss_records.append(value_network_loss.item())
        episode_makespan.append(mpn)
        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)

        logger.info(f"[Episode {episode}] makespan={mpn}, reward={episode_reward}")

    end_time = time.time()
    print('Training time:', end_time - start_time)

    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    date_time = datetime.now().strftime("%m_%d_%H_%M")

    torch.save(actor_network.state_dict(), os.path.join(output_path, f"Actor_{date_time}.pth"))
    torch.save(value_network.state_dict(), os.path.join(output_path, f"Critic_{date_time}.pth"))

    with open(os.path.join(output_path, f'makespan_{date_time}.txt'), 'w') as f:
        for r in episode_makespan:
            f.write(str(r) + '\n')
    with open(os.path.join(output_path, f'rewards_{date_time}.txt'), 'w') as f:
        for r in episode_rewards:
            f.write(str(r) + '\n')
    with open(os.path.join(output_path, f'actor_loss_{date_time}.txt'), 'w') as f:
        for loss in actor_loss_records:
            f.write(f"{loss}\n")
    with open(os.path.join(output_path, f'critic_loss_{date_time}.txt'), 'w') as f:
        for loss in critic_loss_records:
            f.write(f"{loss}\n")

if __name__ == '__main__':
    main()
