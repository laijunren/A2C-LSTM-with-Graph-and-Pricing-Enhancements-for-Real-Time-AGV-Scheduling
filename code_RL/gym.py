import logging
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

output_path = "./A2C_output/"
STATE_DIM = 3
ACTION_DIM = 2
NUM_EPISODE = 800
NUM_RANDOM_EPISODE = 0
A_HIDDEN = 128
C_HIDDEN = 128
a_lr = 1e-3
c_lr = 3e-3


class ActorNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ActorNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        x = F.log_softmax(x, 2)
        return x, hidden


class ValueNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ValueNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


def roll_out(actor, env, random=False):
    states = []
    actions = []
    rewards = []
    done = False
    final_r = 0
    state, _ = env.reset()
    state = np.delete(state, 1)
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)

    while not done:
        states.append(state)
        log_softmax_action, (a_hx, a_cx) = actor(torch.FloatTensor([state]).unsqueeze(0), (a_hx, a_cx))
        # action = Categorical(log_softmax_action).sample().squeeze(0).numpy()[0]
        action = np.random.choice(ACTION_DIM, p=torch.exp(log_softmax_action).cpu().data.numpy()[0][0])

        # 动作独热编码
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]

        env.render()
        next_state, reward, done, halt, info = env.step(action)
        next_state = np.delete(next_state, 1)

        actions.append(one_hot_action)
        rewards.append(reward)
        state = next_state

    return states, actions, rewards, final_r, state


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

    # 初始化env
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = env.unwrapped

    # 初始化价值网络
    value_network = ValueNetwork(in_size=STATE_DIM, hidden_size=C_HIDDEN, out_size=1)
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=c_lr)

    # 初始化动作网络
    actor_network = ActorNetwork(in_size=STATE_DIM, hidden_size=A_HIDDEN, out_size=ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=a_lr)

    actor_loss_records = []
    critic_loss_records = []
    episode_rewards = []
    is_random = False
    start_time = time.time()

    for episode in range(1, NUM_EPISODE + 1):
        states, actions, rewards, final_r, current_state = roll_out(actor_network, env, is_random)

        # rollout结束后的初态
        actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)
        states_var = torch.Tensor(states).view(-1, STATE_DIM).unsqueeze(0)

        # 训练动作网络
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        actor_network_optim.zero_grad()
        log_softmax_actions, _ = actor_network(states_var, (a_hx, a_cx))
        vs, _ = value_network(states_var, (c_hx, c_cx))
        vs.detach()

        # advantage
        qs = torch.Tensor(discount_reward(rewards, 0.99, final_r))
        qs = qs.view(1, -1, 1)
        advantages = qs - vs
        # log_softmax_actions * actions_var动作独热编码extract the prob
        probs = torch.sum(log_softmax_actions * actions_var, 2).view(1, -1, 1)
        actor_network_loss = - torch.mean(probs * advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
        actor_network_optim.step()

        # 训练价值网络
        value_network_optim.zero_grad()
        target_values = qs
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        values, _ = value_network(states_var, (c_hx, c_cx))

        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
        value_network_optim.step()

        actor_loss_records.append(actor_network_loss)
        critic_loss_records.append(value_network_loss)
        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)

        logger.info(f'Episode {episode} completed, reward: {episode_reward}')

    env.close()
    end_time = time.time()
    print('time taken', end_time - start_time)

    os.makedirs(output_path, exist_ok=True)

    with open(output_path + 'rewards_0.txt', 'w') as f:
        for r in episode_rewards:
            f.write(str(r) + '\n')

    # Saving loss records
    with open(output_path + 'actor_loss_0.txt', 'w') as f:
        for loss in actor_loss_records:
            f.write(f"{loss}\n")

    with open(output_path + 'critic_loss_0.txt', 'w') as f:
        for loss in critic_loss_records:
            f.write(f"{loss}\n")


if __name__ == '__main__':
    main()
