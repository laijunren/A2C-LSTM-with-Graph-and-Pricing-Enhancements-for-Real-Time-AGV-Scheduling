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
from torch_geometric.nn import GCNConv

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
NUM_EPISODE = 3000  # 训练回合数
A_HIDDEN = 256      # Actor LSTM隐藏层大小
C_HIDDEN = 256      # Critic LSTM隐藏层大小
a_lr = 1e-3
c_lr = 1e-3

# ------------------------------
#     从 data_config 导入图数据配置
# ------------------------------
from data_config import feature_matrix, edge_index, edge_weight, unique_nodes, node_index

# 根据 feature_matrix 的维度确定图卷积网络的输入通道数
num_node_features = feature_matrix.size(1)
gcn_hidden_channels = 32

# ------------------------------
#      定义部分加载状态字典的函数
# ------------------------------
def load_partial_state(model, state_dict):
    model_dict = model.state_dict()
    # 筛选出键名存在且形状匹配的参数
    compatible_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    return len(compatible_dict), len(model_dict)

# 自动遍历目录，选择匹配率最高的checkpoint加载
def load_best_checkpoint(model, checkpoint_dir, keyword):
    best_ratio = 0
    best_file = None
    model_dict = model.state_dict()
    total_params = len(model_dict)
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') and keyword in file:
            ckpt_path = os.path.join(checkpoint_dir, file)
            try:
                state = torch.load(ckpt_path)
            except Exception as e:
                print(f"无法加载 {ckpt_path}: {e}")
                continue
            match_count = sum(1 for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape)
            ratio = match_count / total_params
            print(f"文件 {file} 匹配比例：{ratio:.2f}")
            if ratio > best_ratio:
                best_ratio = ratio
                best_file = ckpt_path
    if best_file:
        state = torch.load(best_file)
        match_count, total = load_partial_state(model, state)
        print(f"从 {best_file} 加载了 {match_count}/{total} 个匹配的参数")
    else:
        print(f"未找到匹配 {keyword} 的预训练模型，模型将从头开始训练")

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
        # 1) GCN计算节点表示
        x_graph = self.gcn1(x_graph, edge_index, edge_weight=edge_weight)
        x_graph = F.relu(x_graph)
        x_graph = self.gcn2(x_graph, edge_index, edge_weight=edge_weight)
        x_graph = F.relu(x_graph)
        # 2) Global mean pooling
        x_graph = x_graph.mean(dim=0, keepdim=True)  # [1, gcn_hidden_channels]
        # 3) 拼接图特征与环境状态
        x_combined = torch.cat([x_graph, x_state], dim=-1)
        # 4) LSTM (batch=1, seq=1)
        x_combined = x_combined.unsqueeze(0)  # => [1, 1, lstm_input_dim]
        x_out, hidden = self.lstm(x_combined, hidden)
        # 5) 输出动作的对数概率
        x_out = self.fc(x_out)  # => [1, 1, action_dim]
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
#       定义 Reward 计算及辅助函数
# ------------------------------
def calculate_reward(final_obs, optimal_theoretical):
    rewards = []
    total_mpn = final_obs[1]
    if total_mpn <= 0.0:
        total_mpn = 100000
        print("error: simulator run out of time")
    assert total_mpn > 0

    diff = total_mpn - optimal_theoretical
    if diff <= 0:
        makespan = -2
    else:
        makespan = math.log(diff / 100, 1.3) - 2

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
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def roll_out(actor, env, random=False):
    states = []
    actions = []
    done = False
    final_r = 0
    state = env.reset()  # 初始时环境返回状态 (长度=22)
    reward = None

    # 初始化 Actor 的 LSTM 隐状态
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)

    while not done:
        states.append(state)
        env_state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 前向：ActorNetwork (包含 GCN + LSTM)
        log_softmax_action, (a_hx, a_cx) = actor(
            feature_matrix, edge_index, edge_weight,
            env_state_tensor, (a_hx, a_cx)
        )

        # 采样动作
        prob = torch.exp(log_softmax_action).cpu().data.numpy()[0][0]  # shape: (ACTION_DIM,)
        action = np.random.choice(ACTION_DIM, p=prob)

        # 环境执行动作
        done, next_state, reward = env.step(action)

        # 将动作转换为 one-hot 格式
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        actions.append(one_hot_action)

        state = next_state

    makespan = reward[1]
    rewards = calculate_reward(reward, 7300)
    return states, actions, rewards, final_r, state, makespan

# ------------------------------
#     保存训练结果的辅助函数
# ------------------------------
def save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                          actor_loss_records, critic_loss_records, episode_rewards, episode_makespan):
    os.makedirs(output_path, exist_ok=True)
    date_time = datetime.now().strftime("%m_%d_%H_%M")

    # 保存模型
    torch.save(actor_network.state_dict(), os.path.join(output_path, f"Actor_{date_time}.pth"))
    torch.save(value_network.state_dict(), os.path.join(output_path, f"Critic_{date_time}.pth"))

    # 保存优化器状态
    torch.save(actor_network_optim.state_dict(), os.path.join(output_path, f"Actor_optimizer_{date_time}.pth"))
    torch.save(value_network_optim.state_dict(), os.path.join(output_path, f"Critic_optimizer_{date_time}.pth"))

    # 保存训练过程记录
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
    print(f"训练结果已保存到 {output_path}")

# ------------------------------
#         主训练循环
# ------------------------------
def main():
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # 创建环境
    env = make_env()

    # 实例化 Critic 网络
    value_network = ValueNetwork(
        state_dim=STATE_DIM,
        gcn_in_channels=num_node_features,
        gcn_hidden_channels=gcn_hidden_channels,
        lstm_hidden_size=C_HIDDEN
    )
    value_network_optim = torch.optim.AdamW(value_network.parameters(), lr=c_lr)

    # 实例化 Actor 网络
    actor_network = ActorNetwork(
        state_dim=STATE_DIM,
        gcn_in_channels=num_node_features,
        gcn_hidden_channels=gcn_hidden_channels,
        lstm_hidden_size=A_HIDDEN,
        action_dim=ACTION_DIM
    )
    actor_network_optim = torch.optim.AdamW(actor_network.parameters(), lr=a_lr)

    # 自动加载匹配率最高的预训练模型（部分加载）
    checkpoint_dir = "/home/aaa/my_code/hospital-main/A2C_output/181"  # 替换为你的checkpoint存放目录
    load_best_checkpoint(actor_network, checkpoint_dir, keyword="Actor")
    load_best_checkpoint(value_network, checkpoint_dir, keyword="Critic")

    actor_loss_records = []
    critic_loss_records = []
    episode_rewards = []
    episode_makespan = []

    is_random = False
    start_time = time.time()

    try:
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

    except KeyboardInterrupt:
        logger.info("训练过程中检测到中断，正在保存中间结果...")
        save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                              actor_loss_records, critic_loss_records, episode_rewards, episode_makespan)
        return
    except Exception as e:
        logger.error(f"训练过程中发生异常: {e}，正在保存中间结果...")
        save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                              actor_loss_records, critic_loss_records, episode_rewards, episode_makespan)
        raise e

    end_time = time.time()
    print('Training time:', end_time - start_time)

    # 正常结束后保存最终结果
    save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                          actor_loss_records, critic_loss_records, episode_rewards, episode_makespan)

if __name__ == '__main__':
    main()
