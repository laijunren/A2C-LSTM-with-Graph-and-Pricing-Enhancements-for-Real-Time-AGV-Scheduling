import os
import time
import random
from collections import deque
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import logging

from Utils import *
from bonsalt import BonsaltEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()

        hidden_dim1 = 256
        hidden_dim2 = 128
        hidden_dim3 = 64

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Select action using epsilon-greedy policy
def select_action(state, policy_net, epsilon):
    tensor_state = torch.tensor(state, dtype=torch.float32).to(device)
    # Check dynamic task queue size
    index = 0
    for lst in state:


        if lst == 1. and lst == 1. and lst == 1.:
            break
        index += 1
    if random.random() > epsilon:
        with torch.no_grad():
            action_probs = policy_net(tensor_state)[0]
            action = torch.multinomial(action_probs[:index], 1).item()
    else:
        action = np.random.randint(0, index)
    return action


# DQN Training Step
def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    # Extracting batches of states, actions, rewards, and next_states
    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_states_batch = torch.cat(batch[3])
    done_batch = torch.tensor(batch[4], dtype=torch.float32, device=device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states using target network
    next_state_values = target_net(next_states_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma * (1 - done_batch)) + reward_batch

    # Compute loss
    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # hyperparameters
    input_dim = 80
    action_dim = 20
    total_episodes = 5000
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = total_episodes
    TARGET_UPDATE = 10
    MEMORY_CAPACITY = 20000

    # Initialize policy and target networks
    policy_net = DQN(input_dim, action_dim).to(device)
    target_net = DQN(input_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
    memory = ReplayBuffer(MEMORY_CAPACITY)

    env = BonsaltEnv('http://localhost:5000', 'rl-hospital', 'not used')

    # for visualization
    episode_rewards = []
    DQN_loss = []
    start_time = time.time()
    epsilon = EPS_START

    for episode in range(total_episodes):
        terminated = False
        # obs (Feature): agvInfo 4*9, taskInfoQueue 7*20 (taskid, pickupNode XYZ, deliveryNode XYZ)
        # ons (Reward): makeSpan 1, reward 3*164 (waitingTime, setupTime, executionTime)
        obs = env.reset()
        agv_id = select_agv(obs['agvInfo'])
        # State: 20 * (agvDistanceToTasks, agvVerticalDistanceToTasks, distanceCostOfTasks, verticalDistanceCostOfTasks)
        state = extract_feature(obs, agv_id)
        episode_memory = []

        while not terminated:
            # epsilon decay
            action = select_action(state, policy_net, epsilon)
            if 'taskInfoQueue' in obs:
                task_info_queue = obs['taskInfoQueue']
            else:
                task_info_queue = []  # 或者其他默认值

            action = check_action(task_info_queue, action)

            (terminated, obs) = env.step({'agv_id': agv_id, 'task_id': obs['taskInfoQueue'][action][0]})
            agv_id = select_agv(obs['agvInfo'])
            next_state = extract_feature(obs, agv_id)

            action_tensor = torch.tensor([action], dtype=torch.long).to(device)
            tensor_state = torch.tensor(state, dtype=torch.float32).to(device)
            tensor_next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            # store memory without immediate reward
            episode_memory.append((tensor_state, action_tensor, None, tensor_next_state, terminated))
            state = next_state

        epsilon = EPS_END + (EPS_START - EPS_END) * max(1 - episode / EPS_DECAY, 0)

        if terminated:
            print('reward', obs['makespan'])
            episode_rewards.append(obs['makespan'])
            rewards = calculate_reward(obs)

            # store memory with immediate reward to replay buffer
            for i, trans in enumerate(episode_memory):
                state, action, _, next_state, done = trans
                memory.push(state, action, torch.tensor([rewards[i]], device=device, dtype=torch.float32), next_state,
                            done)

            # Perform training
            loss = optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA)
            DQN_loss.extend(loss)

        # Update the target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        logger.info(f'Episode {episode} completed')
        print('epsilon', epsilon)

    end_time = time.time()
    print('time taken ', (end_time - start_time) / 3600, ' hours')

    output_path = "./DQN_output/"
    os.makedirs(output_path, exist_ok=True)
    date_time = datetime.now().strftime("_%m_%d_%H_%M")

    save_model(policy_net, output_path + "policy_net.pth")
    save_model(target_net, output_path + "target_net.pth")

    with open(output_path + 'rewards' + date_time + '.txt', 'w') as f:
        for r in episode_rewards:
            f.write(str(r) + '\n')

    # Saving loss records
    with open(output_path + 'DQN_loss' + date_time + '.txt', 'w') as f:
        for loss in DQN_loss:
            f.write(f"{loss}\n")
