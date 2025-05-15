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

# Set random seed for reproducibility
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

output_path = "./A2C_output/"
STATE_DIM = 26  # 18 original features + 4 elevator + 4 pricing
ACTION_DIM = 9
NUM_EPISODE = 3000
A_HIDDEN = 128
C_HIDDEN = 128
a_lr = 1e-4
c_lr = 1e-4

# ----------------------------------
# Utility: Load only matching parameters from a checkpoint
def load_partial_state(model, state_dict):
    model_dict = model.state_dict()
    compatible_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    return len(compatible_dict), len(model_dict)

# Load best matching checkpoint based on keyword
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
                print(f"Failed to load {ckpt_path}: {e}")
                continue
            match_count = sum(1 for k, v in state.items()
                              if k in model_dict and model_dict[k].shape == v.shape)
            ratio = match_count / total_params
            print(f"File {file} match ratio: {ratio:.2f}")
            if ratio > best_ratio:
                best_ratio = ratio
                best_file = ckpt_path
    if best_file:
        state = torch.load(best_file)
        match_count, total = load_partial_state(model, state)
        print(f"Loaded {match_count}/{total} matching parameters from {best_file}")
    else:
        print(f"No matching pretrained model found for keyword {keyword}. Training from scratch.")

# Load model from specific file
def load_model(model, filepath):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        model.eval()
        print(f"Model loaded from {filepath}")
    else:
        print(f"Model file {filepath} not found. Training from scratch.")

# ----------------------------------
# Define network structures

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

# ----------------------------------
# State transformation and reward design

# Integrate pricing results into state representation
def integrate_pricing_to_state(state, map):
    dual_values = pricing(map)
    return state + list(dual_values)

# Reward function with pricing signal
def calculate_reward(final_obs, optimal_theoretical, map):
    rewards = []
    total_mpn = final_obs[1]
    if total_mpn <= 0.0:
        total_mpn = 100000
        print("Error: simulator timed out")
    assert total_mpn > 0

    diff = total_mpn - optimal_theoretical
    makespan = -2 if diff <= 0 else math.log(diff / 100, 1.3) - 2

    dual_values = pricing(map)
    for i in range(len(final_obs[0])):
        wait, setUp, execTime, lift = final_obs[0][i]
        dual_value_effect = sum(dual_values) * 0.01
        a = math.log(wait * 0.01 + math.e) - 1
        b = math.log(setUp * 0.02 + 2, 2) - 1
        c = execTime * 0.002
        d = lift * 0.005
        rewards.append(-(a + b + c + d + makespan + dual_value_effect - 11) / 6)
    return rewards

# Compute discounted return
def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Perform one trajectory rollout
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

# ----------------------------------
# Save model parameters and logs

def save_training_results(actor_network, value_network, actor_optimizer, critic_optimizer,
                          actor_loss_records, critic_loss_records, episode_rewards, episode_makespan):
    """Save model weights, optimizer states, and logs"""
    os.makedirs(output_path, exist_ok=True)
    suffix = datetime.now().strftime("%m_%d_%H_%M")

    torch.save(actor_network.state_dict(), os.path.join(output_path, f"Actor_{suffix}.pth"))
    torch.save(value_network.state_dict(), os.path.join(output_path, f"Critic_{suffix}.pth"))
    torch.save(actor_optimizer.state_dict(), os.path.join(output_path, f"Actor_optim_{suffix}.pth"))
    torch.save(critic_optimizer.state_dict(), os.path.join(output_path, f"Critic_optim_{suffix}.pth"))

    with open(os.path.join(output_path, f"makespan_{suffix}.txt"), 'w') as f:
        for r in episode_makespan:
            f.write(str(r) + '\n')

    with open(os.path.join(output_path, f"rewards_{suffix}.txt"), 'w') as f:
        for r in episode_rewards:
            f.write(str(r) + '\n')

    with open(os.path.join(output_path, f"actor_loss_{suffix}.txt"), 'w') as f:
        for loss in actor_loss_records:
            f.write(f"{loss}\n")

    with open(os.path.join(output_path, f"critic_loss_{suffix}.txt"), 'w') as f:
        for loss in critic_loss_records:
            f.write(f"{loss}\n")
    print(f"Training results saved. Suffix: {suffix}")

# ----------------------------------
# Main training loop

def main():
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    env = make_env()
    map = [[1997, 950, 0], [1112, 946, 0], [1515, 947, 0]]  # Real coordinates

    value_network = ValueNetwork(in_size=STATE_DIM, hidden_size=C_HIDDEN, out_size=1)
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=c_lr)

    actor_network = ActorNetwork(in_size=STATE_DIM, hidden_size=A_HIDDEN, out_size=ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=a_lr)

    checkpoint_dir = "/home/aaa/my_code/hospital-main/A2C_output/pth"
    load_best_checkpoint(actor_network, checkpoint_dir, "Actor")
    load_best_checkpoint(value_network, checkpoint_dir, "Critic")

    actor_loss_records = []
    critic_loss_records = []
    episode_rewards = []
    episode_makespan = []
    start_time = time.time()

    for episode in range(1, NUM_EPISODE + 1):
        states, actions, rewards, final_r, current_state, mpn = roll_out(actor_network, env, map)
        print(f"Episode {episode}")

        actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)
        states_var = torch.Tensor(states).view(-1, STATE_DIM).unsqueeze(0)

        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)

        # Update Actor
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

        # Update Critic
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
    print('Time taken:', end_time - start_time)

    save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                          actor_loss_records, critic_loss_records, episode_rewards, episode_makespan)

if __name__ == '__main__':
    main()
