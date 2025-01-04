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
# from Environment import make_env 

# ############################################
# # 1. 在此处定义你的邻接矩阵和特征矩阵
# ############################################
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
#     [830., 2100., 0.],
#     [1255., 1050., 0.],
#     [1690., 1080., 0.],
#     [1060., 1145., 0.],
#     [715., 1120., 0.],
#     [1450., 1020., 0.],
#     [1305., 1200., 0.],
#     [1100., 1600., 0.],
# ], dtype=torch.float)

# ############################################
# # 将邻接矩阵转为 edge_index
# ############################################
# def adjacency_to_edge_index(adj: np.ndarray):
#     """
#     将 numpy 邻接矩阵转换为 (2, E) 的 edge_index（PyG格式）
#     只保留非零权重的边（有向图）。
#     """
#     edge_list = []
#     for i in range(adj.shape[0]):
#         for j in range(adj.shape[1]):
#             if adj[i, j] != 0:
#                 edge_list.append([i, j])
#     if not edge_list:
#         # 如果全部是零，那就默认给个空张量
#         return torch.empty((2, 0), dtype=torch.long)
#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#     return edge_index

# # 将 adjacency_matrix 转换为 edge_index
# edge_index = adjacency_to_edge_index(adjacency_matrix)

# # GCN 相关配置
# num_node_features = feature_matrix.size(1)  # 这里是 3
# gcn_hidden_channels = 16  # 你可以自由调整

# ############################################
# # 其他超参数
# ############################################
# seed = 2024
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True

# output_path = "./A2C_output/"
# STATE_DIM = 22  # 环境 state 的维度(可能是 22)
# ACTION_DIM = 9
# NUM_EPISODE = 100  # 示意改小一点，调试时更快
# A_HIDDEN = 128
# C_HIDDEN = 128
# a_lr = 1e-3
# c_lr = 3e-3

# ############################################
# # 2. 定义 Actor 和 Critic 网络
# #   - 在 forward 中使用 GCNConv
# ############################################
# class ActorNetwork(nn.Module):
#     def __init__(self, in_size, hidden_size, out_size):
#         """
#         in_size:  环境 state 的维度(这里未真正使用)
#         hidden_size: LSTM 隐层大小
#         out_size: 动作维度
#         """
#         super(ActorNetwork, self).__init__()
#         self.gcn1 = GCNConv(num_node_features, gcn_hidden_channels)
#         self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)
#         # 这里暂时直接用 gcn_hidden_channels 做 LSTM 的输入
#         # 如果你想把环境的状态拼进去，需要在 forward 里进行拼接
#         self.lstm = nn.LSTM(gcn_hidden_channels, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, out_size)

#     def forward(self, x, edge_index, hidden):
#         """
#         x:       (num_nodes, num_node_features)
#         edge_index: (2, E)
#         hidden:  (h, c)
#         """
#         # GCN部分
#         x = self.gcn1(x, edge_index)
#         x = F.relu(x)
#         x = self.gcn2(x, edge_index)
#         x = F.relu(x)
#         # 这里简单做一个全局平均，将节点特征聚合成 (1, gcn_hidden_channels)
#         x = x.mean(dim=0, keepdim=True)  # [1, gcn_hidden_channels]

#         # LSTM部分 (batch=1, seq=1, feature_dim=gcn_hidden_channels)
#         x = x.unsqueeze(0)
#         x, hidden = self.lstm(x, hidden)
#         x = self.fc(x)  # (1, 1, out_size)
#         x = F.log_softmax(x, dim=2)
#         return x, hidden


# class ValueNetwork(nn.Module):
#     def __init__(self, in_size, hidden_size, out_size):
#         """
#         in_size:  环境 state 的维度(这里未真正使用)
#         hidden_size: LSTM 隐层大小
#         out_size: critic 输出维度(这里是 1)
#         """
#         super(ValueNetwork, self).__init__()
#         self.gcn1 = GCNConv(num_node_features, gcn_hidden_channels)
#         self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)
#         self.lstm = nn.LSTM(gcn_hidden_channels, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, out_size)

#     def forward(self, x, edge_index, hidden):
#         # GCN
#         x = self.gcn1(x, edge_index)
#         x = F.relu(x)
#         x = self.gcn2(x, edge_index)
#         x = F.relu(x)
#         # 全局平均
#         x = x.mean(dim=0, keepdim=True)
#         x = x.unsqueeze(0)  # (1, 1, gcn_hidden_channels)

#         # LSTM
#         x, hidden = self.lstm(x, hidden)
#         x = self.fc(x)  # (1, 1, 1)
#         return x, hidden

# ############################################
# # 3. 定义奖励函数、roll_out 等
# ############################################
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
#         # 这里和原公式一致
#         rewards.append(-(a + b + c + d + makespan - 11) / 6)
#     return rewards


# def discount_reward(r, gamma, final_r):
#     discounted_r = np.zeros_like(r)
#     running_add = final_r
#     for t in reversed(range(0, len(r))):
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r

# ############################################
# # 4. roll_out 时，这里为了演示 GCN 的使用，
# #   不再把 state 传给 actor，而是直接用
# #   feature_matrix, edge_index.
# #   如果你需要融合 state，可自行修改 forward。
# ############################################
# def roll_out(actor, env, random=False):
#     states = []
#     actions = []
#     done = False
#     final_r = 0
#     state = env.reset()  # 虽然拿到，但本示例暂不送进网络
#     reward = None

#     # LSTM 隐状态
#     a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
#     a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)

#     # 直接把 feature_matrix & edge_index 喂给 actor
#     while not done:
#         # 这里保留对环境 state 的存储，但不进神经网络
#         states.append(state)

#         # GCN + LSTM 产生动作分布
#         log_softmax_action, (a_hx, a_cx) = actor(feature_matrix, edge_index, (a_hx, a_cx))
#         # 从分布中采样动作
#         prob = torch.exp(log_softmax_action).cpu().data.numpy()[0][0]  # (ACTION_DIM,)
#         action = np.random.choice(ACTION_DIM, p=prob)

#         # 环境执行 step
#         done, next_state, reward = env.step(action)

#         # 将 one-hot 动作保存
#         one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
#         actions.append(one_hot_action)
#         state = next_state

#     makespan = reward[1]
#     # 计算奖励
#     rewards = calculate_reward(reward, 7300)

#     return states, actions, rewards, final_r, state, makespan

# def select_action_greedy(state):
#     """
#     如果需要根据 env state 的某种贪心，这里自定义；
#     本示例不多做改动
#     """
#     min_val = 10000
#     index = 0
#     for i in range(len(state) // 2):
#         if state[i * 2] < min_val:
#             min_val = state[i * 2]
#             index = i
#         elif state[i * 2] == min_val and np.random.rand() > 0.7:
#             min_val = state[i * 2]
#             index = i
#     return np.array(index)

# ############################################
# # 5. 主训练循环
# ############################################
# def main():
#     LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
#     logger = logging.getLogger('main')
#     logger.setLevel(logging.DEBUG)

#     # 初始化env
#     env = make_env()

#     # 初始化价值网络
#     value_network = ValueNetwork(
#         in_size=STATE_DIM,
#         hidden_size=C_HIDDEN,
#         out_size=1
#     )
#     value_network_optim = torch.optim.Adam(value_network.parameters(), lr=c_lr)

#     # 初始化策略网络
#     actor_network = ActorNetwork(
#         in_size=STATE_DIM,
#         hidden_size=A_HIDDEN,
#         out_size=ACTION_DIM
#     )
#     actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=a_lr)

#     actor_loss_records = []
#     critic_loss_records = []
#     episode_rewards = []
#     episode_makespan = []

#     is_random = False
#     start_time = time.time()

#     for episode in range(1, NUM_EPISODE + 1):
#         # 一次 roll_out
#         states, actions, rewards, final_r, current_state, mpn = roll_out(actor_network, env, is_random)

#         # 准备训练
#         actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)  # shape = [1, T, ACTION_DIM]
#         states_var = torch.Tensor(states).view(-1, STATE_DIM).unsqueeze(0)    # shape = [1, T, STATE_DIM]

#         # 训练 actor
#         a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
#         a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
#         # 注意：现在 actor 的前向网络需要 (feature_matrix, edge_index, hidden)，
#         # 但我们原来是把 states_var 放进去，这里为了和原逻辑匹配只写占位：
#         actor_network_optim.zero_grad()
#         # 由于实际前向已使用 feature_matrix, edge_index，这里只为了能 broadcast 对齐
#         # 简单重复 roll_out 步数次即可
#         # 不过，这样和真正将 "states_var" 送进 Actor 并不一致，仅保留原代码结构

#         T = states_var.size(1)
#         log_softmax_actions_list = []
#         for t in range(T):
#             # 同样地，在训练阶段也固定喂相同的图结构
#             out, (a_hx, a_cx) = actor_network(feature_matrix, edge_index, (a_hx, a_cx))
#             log_softmax_actions_list.append(out)  # (1,1,ACTION_DIM)

#         log_softmax_actions = torch.cat(log_softmax_actions_list, dim=1)  # (1,T,ACTION_DIM)
#         # 训练 Critic
#         c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
#         c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)

#         # 同理，对 Critic 做类似处理
#         value_network_optim.zero_grad()
#         vs_list = []
#         for t in range(T):
#             out, (c_hx, c_cx) = value_network(feature_matrix, edge_index, (c_hx, c_cx))
#             vs_list.append(out)

#         vs = torch.cat(vs_list, dim=1)  # (1, T, 1)
#         vs_detached = vs.detach()

#         # advantage
#         qs = torch.Tensor(discount_reward(rewards, 0.99, final_r))
#         qs = qs.view(1, -1, 1)  # (1, T, 1)
#         advantages = qs - vs_detached

#         # actor loss
#         # log_softmax_actions: (1, T, ACTION_DIM)
#         # actions_var: (1, T, ACTION_DIM)
#         probs = torch.sum(log_softmax_actions * actions_var, dim=2, keepdim=True)  # (1,T,1)
#         actor_network_loss = - torch.mean(probs * advantages)
#         actor_network_loss.backward()
#         torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
#         actor_network_optim.step()

#         # critic loss
#         criterion = nn.MSELoss()
#         value_network_loss = criterion(vs, qs)
#         value_network_loss.backward()
#         torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
#         value_network_optim.step()

#         # 记录
#         actor_loss_records.append(actor_network_loss.item())
#         critic_loss_records.append(value_network_loss.item())
#         episode_makespan.append(mpn)
#         episode_reward = np.sum(rewards)
#         episode_rewards.append(episode_reward)

#         logger.info(f'Episode {episode} completed, makespan: {mpn}, reward: {episode_reward}')

#     end_time = time.time()
#     print('Training time taken:', end_time - start_time)

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


# 定义邻接矩阵和节点特征

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

feature_matrix = torch.tensor([
    [830., 2100., 0.],
    [1255., 1050., 0.],
    [1690., 1080., 0.],
    [1060., 1145., 0.],
    [715., 1120., 0.],
    [1450., 1020., 0.],
    [1305., 1200., 0.],
    [1100., 1600., 0.],
], dtype=torch.float)


# 将邻接矩阵转为 PyG 的 edge_index

def adjacency_to_edge_index(adj: np.ndarray):
    """
    将 numpy 邻接矩阵转换为 (2, E) 的 edge_index（PyG格式）
    只保留非零权重的边（有向图）。
    """
    edge_list = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                edge_list.append([i, j])
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

edge_index = adjacency_to_edge_index(adjacency_matrix)

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

output_path = "./A2C_output/"
STATE_DIM = 22    # 环境 state 的维度
ACTION_DIM = 9    # 动作空间大小
NUM_EPISODE = 50  # 训练回合数(示例改小一点)
A_HIDDEN = 128    # Actor LSTM隐藏层大小
C_HIDDEN = 128    # Critic LSTM隐藏层大小
a_lr = 1e-3
c_lr = 3e-3

# 图相关信息
num_node_features = feature_matrix.size(1)  # 3
gcn_hidden_channels = 16                   # GCN隐藏层维度

# 定义网络结构
# 将 "环境state" + "图卷积输出" 融合
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, gcn_in_channels, gcn_hidden_channels, lstm_hidden_size, action_dim):
        super(ActorNetwork, self).__init__()
        # GCN部分
        self.gcn1 = GCNConv(gcn_in_channels, gcn_hidden_channels)
        self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)

        # 把 "图卷积输出" + "环境状态" 进行拼接 => 作为 LSTM 的输入
        self.lstm_input_dim = gcn_hidden_channels + state_dim

        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, action_dim)

    def forward(self, x_graph, edge_index, x_state, hidden):
        """
        x_graph:   (num_nodes, gcn_in_channels)   # 图的节点特征
        edge_index:(2, E)                         # 图的边索引
        x_state:   (batch=1, state_dim)           # 环境的状态(在roll_out时获取)
        hidden:    (h, c)                         # LSTM隐状态
        """
        # 1) 图卷积
        x_graph = self.gcn1(x_graph, edge_index)
        x_graph = F.relu(x_graph)
        x_graph = self.gcn2(x_graph, edge_index)
        x_graph = F.relu(x_graph)

        # 2) global mean pooling
        x_graph = x_graph.mean(dim=0, keepdim=True)  # [1, gcn_hidden_channels]

        # 3) 拼接图特征与环境状态
        # x_state => [1, state_dim]
        x_combined = torch.cat([x_graph, x_state], dim=-1)  # [1, gcn_hidden_channels + state_dim]

        # 4) LSTM需要 (batch, seq, feature_dim)
        x_combined = x_combined.unsqueeze(0)  # => [1, 1, (gcn_hidden_channels + state_dim)]
        x_out, hidden = self.lstm(x_combined, hidden)  # => x_out: [1,1,lstm_hidden_size]

        # 5) 全连接输出(action_dim)，再做 log_softmax
        x_out = self.fc(x_out)  # => [1,1,action_dim]
        x_out = F.log_softmax(x_out, dim=2)
        return x_out, hidden


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, gcn_in_channels, gcn_hidden_channels, lstm_hidden_size):
        super(ValueNetwork, self).__init__()
        # GCN
        self.gcn1 = GCNConv(gcn_in_channels, gcn_hidden_channels)
        self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)

        # 把图输出和状态拼接
        self.lstm_input_dim = gcn_hidden_channels + state_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_size, batch_first=True)

        # Critic输出维度=1
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x_graph, edge_index, x_state, hidden):
        # 1) GCN
        x_graph = self.gcn1(x_graph, edge_index)
        x_graph = F.relu(x_graph)
        x_graph = self.gcn2(x_graph, edge_index)
        x_graph = F.relu(x_graph)

        # 2) global mean
        x_graph = x_graph.mean(dim=0, keepdim=True)  # => [1, gcn_hidden_channels]

        # 3) 拼接 state
        x_combined = torch.cat([x_graph, x_state], dim=-1)  # => [1, gcn_hidden_channels + state_dim]

        # 4) LSTM
        x_combined = x_combined.unsqueeze(0)  # => [1, 1, (gcn_hidden_channels + state_dim)]
        x_out, hidden = self.lstm(x_combined, hidden)

        # 5) 全连接 -> 价值
        x_out = self.fc(x_out)  # => [1,1,1]
        return x_out, hidden


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


# roll_out 时用 (feature_matrix, edge_index, 当前state)
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
        # 记录下当前 state
        states.append(state)

        # 把 state 转成张量 shape=(1, STATE_DIM)
        env_state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 前向：ActorNetwork(包含GCN + LSTM)
        log_softmax_action, (a_hx, a_cx) = actor(
            feature_matrix, edge_index, env_state_tensor, (a_hx, a_cx)
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


# 主训练循环：同样在训练时，把 (feature_matrix, edge_index, states_var[t]) 送给网络

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

    # 记录训练过程数据
    actor_loss_records = []
    critic_loss_records = []
    episode_rewards = []
    episode_makespan = []

    is_random = False
    start_time = time.time()

    for episode in range(1, NUM_EPISODE + 1):
        # roll_out 采集一条完整序列
        states, actions, rewards, final_r, current_state, mpn = roll_out(actor_network, env, is_random)

        # 转成PyTorch张量
        actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)  # [1, T, ACTION_DIM]
        states_var = torch.Tensor(states).view(-1, STATE_DIM)                  # [T, STATE_DIM]
        T = states_var.size(0)

        # 训练 Actor
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        actor_network_optim.zero_grad()

        log_softmax_actions_list = []
        for t in range(T):
            # 取第 t 步的 env state
            env_state_t = states_var[t].unsqueeze(0)  # [1, STATE_DIM]

            # 每一步都用 (feature_matrix, edge_index, env_state_t)
            out, (a_hx, a_cx) = actor_network(feature_matrix, edge_index, env_state_t, (a_hx, a_cx))
            log_softmax_actions_list.append(out)  # shape (1,1,ACTION_DIM)

        # 拼接T步
        log_softmax_actions = torch.cat(log_softmax_actions_list, dim=1)  # (1,T,ACTION_DIM)

        # 训练 Critic
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        value_network_optim.zero_grad()

        vs_list = []
        for t in range(T):
            env_state_t = states_var[t].unsqueeze(0)  # [1, STATE_DIM]
            out, (c_hx, c_cx) = value_network(feature_matrix, edge_index, env_state_t, (c_hx, c_cx))
            vs_list.append(out)

        vs = torch.cat(vs_list, dim=1)  # (1,T,1)
        vs_detached = vs.detach()       # 用于计算优势

        # 计算折扣回报和优势
        qs = torch.Tensor(discount_reward(rewards, 0.99, final_r))  # (T,)
        qs = qs.view(1, -1, 1)  # (1, T, 1)
        advantages = qs - vs_detached

        # Actor loss
        # log_softmax_actions: (1,T,ACTION_DIM)
        # actions_var:         (1,T,ACTION_DIM)
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

        # 记录并打印
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
