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

from torch_geometric.nn import GATConv

from Environment import make_env 


# Set seeds for reproducibility
# ------------------------------
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# Define hyperparameters and output path
output_path = "./A2C_output/"
STATE_DIM = 22      
ACTION_DIM = 9      
NUM_EPISODE = 3000  
A_HIDDEN = 256      
C_HIDDEN = 256     
a_lr = 1e-4
c_lr = 1e-4

# ------------------------------
from data_config import feature_matrix, edge_index, edge_weight, unique_nodes, node_index

num_node_features = feature_matrix.size(1)
gcn_hidden_channels = 32

# Function to load partially matching model state
def load_partial_state(model, state_dict):
    model_dict = model.state_dict()
    compatible_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    return len(compatible_dict), len(model_dict)

# Automatically load the checkpoint with the highest match ratio
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
            match_count = sum(1 for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape)
            ratio = match_count / total_params
            print(f"Match ratio for {file}: {ratio:.2f}")
            if ratio > best_ratio:
                best_ratio = ratio
                best_file = ckpt_path
    if best_file:
        state = torch.load(best_file)
        match_count, total = load_partial_state(model, state)
        print(f"Loaded {match_count}/{total} matching parameters from {best_file}")
    else:
        print(f"No suitable pretrained model found with keyword {keyword}. Training from scratch.")

# Actor network definition
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, gcn_in_channels, gcn_hidden_channels, lstm_hidden_size, action_dim):
        super(ActorNetwork, self).__init__()
        self.gcn1 = GATConv(gcn_in_channels, gcn_hidden_channels, heads=1, concat=False, edge_dim=1)
        self.gcn2 = GATConv(gcn_hidden_channels, gcn_hidden_channels, heads=1, concat=False, edge_dim=1)
        self.lstm_input_dim = gcn_hidden_channels + state_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, action_dim)

    def forward(self, x_graph, edge_index, edge_weight, x_state, hidden):
        x_graph = self.gcn1(x_graph, edge_index, edge_attr=edge_weight)
        x_graph = F.relu(x_graph)
        x_graph = self.gcn2(x_graph, edge_index, edge_attr=edge_weight)
        x_graph = F.relu(x_graph)
        x_graph = x_graph.mean(dim=0, keepdim=True)  # Global mean pooling
        x_combined = torch.cat([x_graph, x_state], dim=-1)
        x_combined = x_combined.unsqueeze(0)  # => [1, 1, lstm_input_dim]
        x_out, hidden = self.lstm(x_combined, hidden)
        x_out = self.fc(x_out)  # => [1, 1, action_dim]
        x_out = F.log_softmax(x_out, dim=2)
        return x_out, hidden

# Critic network definition
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, gcn_in_channels, gcn_hidden_channels, lstm_hidden_size):
        super(ValueNetwork, self).__init__()
        self.gcn1 = GATConv(gcn_in_channels, gcn_hidden_channels, heads=1, concat=False, edge_dim=1)
        self.gcn2 = GATConv(gcn_hidden_channels, gcn_hidden_channels, heads=1, concat=False, edge_dim=1)
        self.lstm_input_dim = gcn_hidden_channels + state_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x_graph, edge_index, edge_weight, x_state, hidden):
        x_graph = self.gcn1(x_graph, edge_index, edge_attr=edge_weight)
        x_graph = F.relu(x_graph)
        x_graph = self.gcn2(x_graph, edge_index, edge_attr=edge_weight)
        x_graph = F.relu(x_graph)
        x_graph = x_graph.mean(dim=0, keepdim=True)
        x_combined = torch.cat([x_graph, x_state], dim=-1)
        x_combined = x_combined.unsqueeze(0)
        x_out, hidden = self.lstm(x_combined, hidden)
        x_out = self.fc(x_out)
        return x_out, hidden

# Reward function
def calculate_reward(final_obs, optimal_theoretical):
    rewards = []
    total_mpn = final_obs[1]
    if total_mpn <= 0.0:
        total_mpn = 100000
        print("error: simulator run out of time")
    assert total_mpn > 0
    diff = total_mpn - optimal_theoretical
    makespan = -2 if diff <= 0 else math.log(diff / 100, 1.3) - 2
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

# Rollout function for episode interaction
def roll_out(actor, env, random=False):
    states = []
    actions = []
    done = False
    final_r = 0
    state = env.reset()  
    reward = None

    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)

    while not done:
        states.append(state)
        env_state_tensor = torch.FloatTensor(state).unsqueeze(0)
        log_softmax_action, (a_hx, a_cx) = actor(
            feature_matrix, edge_index, edge_weight,
            env_state_tensor, (a_hx, a_cx)
        )
        prob = torch.exp(log_softmax_action).cpu().data.numpy()[0][0]
        action = np.random.choice(ACTION_DIM, p=prob)
        done, next_state, reward = env.step(action)
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        actions.append(one_hot_action)
        state = next_state

    makespan = reward[1]
    rewards = calculate_reward(reward, 7300)
    return states, actions, rewards, final_r, state, makespan

# Save training metrics and models
def save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                          actor_loss_records, critic_loss_records, episode_rewards, episode_makespan):
    os.makedirs(output_path, exist_ok=True)
    date_time = datetime.now().strftime("%m_%d_%H_%M")
    torch.save(actor_network.state_dict(), os.path.join(output_path, f"Actor_{date_time}.pth"))
    torch.save(value_network.state_dict(), os.path.join(output_path, f"Critic_{date_time}.pth"))
    torch.save(actor_network_optim.state_dict(), os.path.join(output_path, f"Actor_optimizer_{date_time}.pth"))
    torch.save(value_network_optim.state_dict(), os.path.join(output_path, f"Critic_optimizer_{date_time}.pth"))
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
    print(f"Training results saved to {output_path}")

# Main training loop
def main():
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    env = make_env()

    value_network = ValueNetwork(STATE_DIM, num_node_features, gcn_hidden_channels, C_HIDDEN)
    value_network_optim = torch.optim.AdamW(value_network.parameters(), lr=c_lr)

    actor_network = ActorNetwork(STATE_DIM, num_node_features, gcn_hidden_channels, A_HIDDEN, ACTION_DIM)
    actor_network_optim = torch.optim.AdamW(actor_network.parameters(), lr=a_lr)

    checkpoint_dir = "/home/aaa/my_code/hospital-main/A2C_output/pth"
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

            actions_var = torch.Tensor(actions).view(-1, ACTION_DIM).unsqueeze(0)
            states_var = torch.Tensor(states).view(-1, STATE_DIM)
            T = states_var.size(0)

            # Train Actor
            a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
            a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
            actor_network_optim.zero_grad()

            log_softmax_actions_list = []
            for t in range(T):
                env_state_t = states_var[t].unsqueeze(0)
                out, (a_hx, a_cx) = actor_network(feature_matrix, edge_index, edge_weight, env_state_t, (a_hx, a_cx))
                log_softmax_actions_list.append(out)
            log_softmax_actions = torch.cat(log_softmax_actions_list, dim=1)

            # Train Critic
            c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
            c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
            value_network_optim.zero_grad()

            vs_list = []
            for t in range(T):
                env_state_t = states_var[t].unsqueeze(0)
                v_out, (c_hx, c_cx) = value_network(feature_matrix, edge_index, edge_weight, env_state_t, (c_hx, c_cx))
                vs_list.append(v_out)
            vs = torch.cat(vs_list, dim=1)
            vs_detached = vs.detach()

            qs = torch.Tensor(discount_reward(rewards, 0.99, final_r)).view(1, -1, 1)
            advantages = qs - vs_detached

            probs = torch.sum(log_softmax_actions * actions_var, dim=2, keepdim=True)
            actor_network_loss = -torch.mean(probs * advantages)
            actor_network_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
            actor_network_optim.step()

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
        logger.info("Training interrupted. Saving intermediate results...")
        save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                              actor_loss_records, critic_loss_records, episode_rewards, episode_makespan)
        return
    except Exception as e:
        logger.error(f"Exception during training: {e}. Saving intermediate results...")
        save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                              actor_loss_records, critic_loss_records, episode_rewards, episode_makespan)
        raise e

    end_time = time.time()
    print('Training time:', end_time - start_time)

    save_training_results(actor_network, value_network, actor_network_optim, value_network_optim,
                          actor_loss_records, critic_loss_records, episode_rewards, episode_makespan)

if __name__ == '__main__':
    main()
