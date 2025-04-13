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

from Environment import make_env
# 设置随机种子以保证实验的可重复性
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

output_path = "./A2C_output/" # 输出路径
STATE_DIM = 22 # 18 + 4 
ACTION_DIM = 9 # 
NUM_EPISODE = 3000 # 训练的总回合数
A_HIDDEN = 128 # Actor 网络的隐藏层大小
C_HIDDEN = 128 # Critic 网络的隐藏层大小
a_lr = 1e-4 # Actor 网络的学习率
c_lr = 1e-4 # Critic 网络的学习率

def load_partial_state(model, state_dict):
    """
    仅加载那些键名存在且形状匹配的参数
    """
    model_dict = model.state_dict()
    compatible_dict = {k: v for k, v in state_dict.items() 
                       if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    return len(compatible_dict), len(model_dict)

def load_best_checkpoint(model, checkpoint_dir, keyword):
    """
    遍历 checkpoint_dir 中所有文件，选择文件名中包含 keyword 且匹配比例最高的文件加载
    """
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
            match_count = sum(1 for k, v in state.items() 
                              if k in model_dict and model_dict[k].shape == v.shape)
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
# 原有加载模型的函数（若需要指定文件可用）
# ------------------------------
def load_model(model, filepath):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        model.eval()  # 设置为评估模式
        print(f"模型已从 {filepath} 加载")
    else:
        print(f"未找到模型文件 {filepath}, 将从头开始训练")

class ActorNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ActorNetwork, self).__init__() 
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True) # LSTM 用于处理序列数据
        self.fc = nn.Linear(hidden_size, out_size) # 全连接层，用于生成动作的概率分布

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden) # LSTM 处理输入
        x = self.fc(x) # 输出动作分布
        x = F.log_softmax(x, 2) # 对输出取对数 softmax，得到动作的 log 概率分布
        return x, hidden


class ValueNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ValueNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True) # LSTM 用于处理序列数据
        self.fc = nn.Linear(hidden_size, out_size) # 全连接层，用于生成状态值

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden) # LSTM 处理输入
        x = self.fc(x) # 输出状态值
        return x, hidden


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


def roll_out(actor, env, random=False):
    states = [] # 用于记录每个时间步的状态
    actions = [] # 用于记录每个时间步的动作
    reward = None # 存储当前时间步的奖励
    done = False # episode是否结束
    final_r = 0 
    state = env.reset() # 重置环境，获得初始状态
    # 初始化 Actor 网络的 LSTM 隐藏状态和细胞状态
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)

    while not done: # 每一次循环就是一个时间步
        states.append(state) # 当前时间步的状态
         # 使用 Actor 网络计算动作的 log 概率分布
        log_softmax_action, (a_hx, a_cx) = actor(torch.FloatTensor([state]).unsqueeze(0), (a_hx, a_cx))
        # 根据概率分布采样动作
        action = np.random.choice(ACTION_DIM, p=torch.exp(log_softmax_action).cpu().data.numpy()[0][0])

        # if random:
        #     action = select_action_greedy(state)
        # 将动作转化为one_hot_action
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]  # 动作执行后，环境返回的下一个状态、奖励和结束标志

        done, next_state, reward = env.step(action) # 执行动作，获得新的状态、奖励和是否结束标志

        actions.append(one_hot_action)  # 保存当前动作
        state = next_state  # 更新状态 

    makespan = reward[1] # 提取 makespan
    rewards = calculate_reward(reward, 7300) # 计算奖励
    # print(f"Done: {done}, Final_r (before calculation): {final_r}")

    return states, actions, rewards, final_r, state, makespan


# Select action based on the state
def select_action_greedy(state):
    min = 10000
    index = 0
    for i in range(len(state) // 2):
        if state[i * 2] < min:
            min = state[i * 2]
            index = i
        elif state[i * 2] == min and np.random.rand() > 0.7:
            min = state[i * 2]
            index = i
    return np.array(index)


def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        
    return discounted_r


def main():
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # Initialize the environment
    env = make_env()

    # Initialize the networks first
    value_network = ValueNetwork(in_size=STATE_DIM, hidden_size=C_HIDDEN, out_size=1)
    actor_network = ActorNetwork(in_size=STATE_DIM, hidden_size=A_HIDDEN, out_size=ACTION_DIM)

    # Now load previously saved checkpoints (if any)
    checkpoint_dir = "/home/aaa/my_code/hospital-main/A2C_output/pth"
    load_best_checkpoint(actor_network, checkpoint_dir, "Actor")
    load_best_checkpoint(value_network, checkpoint_dir, "Critic")

    # Create the optimizers for each network
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=c_lr)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=a_lr)


    # 初始化日志记录变量
    actor_loss_records = [] # 记录 Actor 网络的损失
    critic_loss_records = [] # 记录 Critic 网络的损失
    episode_rewards = [] # 每回合的总奖励
    episode_makespan = [] # 每回合的 makespan
    test_episode_rewards = []
    test_episode_makespan = []
    is_random = False
    start_time = time.time() # 记录起始时间
    # 开始训练
    for episode in range(1, NUM_EPISODE + 1):  #调用 roll_out 函数，执行当前 episode
        states, actions, rewards, final_r, current_state, mpn = roll_out(actor_network, env, is_random)
        print(f"Episode{episode}")

        # 数据处理
        actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)
        states_var = torch.Tensor(states).view(-1, STATE_DIM).unsqueeze(0)

        # Actor 网络的训练
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        actor_network_optim.zero_grad()  # 更新网络，计算损失并进行反向传播
        log_softmax_actions, _ = actor_network(states_var, (a_hx, a_cx))  # 计算 log_softmax_actions 和 状态值 vs
        vs, _ = value_network(states_var, (c_hx, c_cx))
        vs.detach()

        # advantage 计算累积奖励 (Q值)
        qs = torch.Tensor(discount_reward(rewards, 0.99, final_r))
        qs = qs.view(1, -1, 1)
        advantages = qs - vs # 计算优势函数 A(s_i, a_i) = Q(s_i, a_i) - V(s_i)
        # print("Advantages:", advantages)
        # log_softmax_actions * actions_var动作独热编码extract the prob
        probs = torch.sum(log_softmax_actions * actions_var, 2).view(1, -1, 1)
        # print("Log probabilities for actions (probs):", probs)
        actor_network_loss = - torch.mean(probs * advantages)
        # print("Actor network loss:", actor_network_loss)
        actor_network_loss.backward()  # 反向传播和优化：通过 .backward() 和优化器 .step() 来更新策略网络的参数
        torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
        actor_network_optim.step()  # 策略梯度更新

        # Critic 网络的训练
        value_network_optim.zero_grad()  # 清除梯度
        target_values = qs # Calculate target values (yi) 
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        values, _ = value_network(states_var, (c_hx, c_cx))

        criterion = nn.MSELoss() # 使用均方误差损失函数
        value_network_loss = criterion(values, target_values) # 计算TD误差
        value_network_loss.backward() # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5) # 梯度裁剪
        value_network_optim.step() # 优化Critic网络

        actor_loss_records.append(actor_network_loss)
        critic_loss_records.append(value_network_loss)
        episode_makespan.append(mpn)
        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)

        logger.info(f'Episode {episode} completed, makespan: {mpn}, reward: {episode_reward}')


    end_time = time.time()
    print('time taken', end_time - start_time)

    os.makedirs(output_path, exist_ok=True)
    date_time = datetime.now().strftime("%m_%d_%H_%M")

    torch.save(actor_network.state_dict(), output_path + "Actor_" + date_time + ".pth")
    torch.save(value_network.state_dict(), output_path + "Critic_" + date_time + ".pth")

    with open(output_path + 'makespan_' + date_time + '.txt', 'w') as f:
        for r in episode_makespan:
            f.write(str(r) + '\n')

    with open(output_path + 'rewards_' + date_time + '.txt', 'w') as f:
        for r in episode_rewards:
            f.write(str(r) + '\n')

    # Saving loss records
    with open(output_path + 'actor_loss_' + date_time + '.txt', 'w') as f:
        for loss in actor_loss_records:
            f.write(f"{loss}\n")

    with open(output_path + 'critic_loss_' + date_time + '.txt', 'w') as f:
        for loss in critic_loss_records:
            f.write(f"{loss}\n")


if __name__ == '__main__':
    main()
