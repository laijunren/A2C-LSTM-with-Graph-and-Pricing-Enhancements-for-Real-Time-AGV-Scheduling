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
from Pricing import pricing

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

output_path = "./A2C_output/"
STATE_DIM = 26  # 18 + 4 + 4
ACTION_DIM = 9
NUM_EPISODE = 5000
A_HIDDEN = 256
C_HIDDEN = 256
a_lr = 1e-4
c_lr = 1e-4
weight_decay = 1e-4  # Add weight decay for regularization


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


def integrate_pricing_to_state(state, map):
    """Integrate Pricing optimization results into the state."""
    dual_values = pricing(map)
    return state + list(dual_values)


def calculate_reward(final_obs, optimal_theoretical, map):
    rewards = []
    total_mpn = final_obs[1]
    if total_mpn <= 0.0:
        total_mpn = 100000
        print("error: simulator run out of time")
    assert total_mpn > 0
    makespan = math.log((total_mpn - optimal_theoretical) / 100, 1.3) - 2
    dual_values = pricing(map)  # Add pricing optimization results

    for i in range(len(final_obs[0])):
        wait, setUp, execTime, lift = final_obs[0][i]
        dual_value_effect = sum(dual_values) * 0.01
        a = math.log(wait * 0.01 + math.e) - 1
        b = math.log(setUp * 0.02 + 2, 2) - 1
        c = execTime * 0.002
        d = lift * 0.005
        rewards.append(-(a + b + c + d + makespan + dual_value_effect - 11) / 6)
    return rewards


def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def roll_out(actor, env, map, random=False):
    states = []
    actions = []
    reward = None
    done = False
    final_r = 0
    state = env.reset()
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)

    while not done:
        # Integrate Pricing optimization into state
        state = integrate_pricing_to_state(state, map)
        states.append(state)

        log_softmax_action, (a_hx, a_cx) = actor(torch.FloatTensor([state]).unsqueeze(0), (a_hx, a_cx))
        action = np.random.choice(ACTION_DIM, p=torch.exp(log_softmax_action).cpu().data.numpy()[0][0])

        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        done, next_state, reward = env.step(action)

        state = next_state
        actions.append(one_hot_action)

    makespan = reward[1]
    rewards = calculate_reward(reward, 7300, map)

    return states, actions, rewards, final_r, state, makespan


def main():
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    env = make_env()
    map = [[1997, 950, 0], [1112, 946, 0], [1515, 947, 0]]  # Real-world location coordinates

    value_network = ValueNetwork(in_size=STATE_DIM, hidden_size=C_HIDDEN, out_size=1)
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=c_lr, weight_decay=weight_decay)

    actor_network = ActorNetwork(in_size=STATE_DIM, hidden_size=A_HIDDEN, out_size=ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=a_lr, weight_decay=weight_decay)

    actor_loss_records = []
    critic_loss_records = []
    episode_rewards = []
    episode_makespan = []
    start_time = time.time()

    for episode in range(1, NUM_EPISODE + 1):
        states, actions, rewards, final_r, current_state, mpn = roll_out(actor_network, env, map)

        actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)
        states_var = torch.Tensor(states).view(-1, STATE_DIM).unsqueeze(0)

        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)

        # Update actor network
        actor_network_optim.zero_grad()
        log_softmax_actions, _ = actor_network(states_var, (a_hx, a_cx))
        vs, _ = value_network(states_var, (c_hx, c_cx))
        vs.detach()

        qs = torch.Tensor(discount_reward(rewards, 0.99, final_r)).view(1, -1, 1)
        advantages = qs - vs
        probs = torch.sum(log_softmax_actions * actions_var, 2).view(1, -1, 1)
        actor_network_loss = -torch.mean(probs * advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
        actor_network_optim.step()

        # Update value network
        value_network_optim.zero_grad()
        criterion = nn.MSELoss()
        target_values = qs
        values, _ = value_network(states_var, (c_hx, c_cx))
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
        value_network_optim.step()

        actor_loss_records.append(actor_network_loss.item())
        critic_loss_records.append(value_network_loss.item())
        episode_makespan.append(mpn)
        episode_rewards.append(np.sum(rewards))

        logger.info(f'Episode {episode} completed, makespan: {mpn}, reward: {np.sum(rewards)}')

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
