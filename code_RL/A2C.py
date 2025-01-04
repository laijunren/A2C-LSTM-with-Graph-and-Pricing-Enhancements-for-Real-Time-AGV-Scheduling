import math

import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        # nn.initializer.set_global_initializer(nn.initializer.XavierNormal(), nn.initializer.Constant(value=0.))
        hidden_dim1 = 256
        hidden_dim2 = 64

        self.actor = nn.Sequential(
            # nn.Flatten(0),
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            # nn.Flatten(0),
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        probs = self.actor(state)
        dist = Categorical(probs)
        return dist, value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    returns = []
    R = next_value
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train_net(optimizer, log_probs, values, returns, entropy):
    log_probs = torch.stack(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)
    advantage = returns - values
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()
    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return actor_loss, critic_loss


def select_action(action_probs):
    if not torch.all(torch.isfinite(action_probs[0])) or torch.any(action_probs[0] < 0):
        print("Warning: Invalid action probabilities detected. Using a random action.")
        action = np.random.randint(0, 9)
    else:
        action = torch.multinomial(action_probs, 1).item()
    return action


# Select action based on the state
def select_action_greedy(state):
    min = 10000
    index = 0
    for i in range(len(state)):
        if (state[i] < min):
            min = state[i]
            index = i
    return index


def calculate_reward(final_obs, optimal_theoretical):
    rewards = []
    total_mpn = final_obs[1]
    if total_mpn <= 0.0:
        total_mpn = 100000
        print("error: simulator run out of time")
    assert total_mpn > 0
    makespan = math.log((total_mpn - optimal_theoretical) / 100, 1.5) - 1.5
    for i in range(len(final_obs[0])):
        wait, setUp, execTime, lift = final_obs[0][i]
        a = math.log(wait * 0.04 + math.e) - 1
        b = setUp * 0.02
        c = execTime * 0.003
        d = lift * 0.01
        # print(a, b, c, d, makespan)
        rewards.append(-(a + b + c + d + makespan - 13))
    return rewards


def extract_feature(obs):
    states = []
    for o in obs:
        state = []
        for arr in obs['agvInfo']:
            state.append(arr[0])
        states.append(state)
    return states
