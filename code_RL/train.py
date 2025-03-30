import logging
import os
import random
import sys
import time
from datetime import datetime
import torch.optim as optim

from A2C import *
from Environment import *

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_path = "./A2C_output/"
input_dim = 22
action_dim = 9
lr = 1e-4
total_episodes = 3000
num_envs = 1
num_agv = 164

model = ActorCritic(input_dim, action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr)
# if os.path.exists(output_path + "A2C.pth"):
#     model.load_state_dict(torch.load(output_path + "A2C.pth"))
#     print('Model Loaded')

env = make_env()

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)  # 设置日志记录的级别为INFO，而非DEBUG

actor_loss_records = []
critic_loss_records = []
episode_rewards = []
episode_makespan = []
start_time = time.time()


def run_episode(model, env):
    log_probs = []
    values = []
    reward = None
    entropy = 0
    done = False
    next_state = env.reset()

    while not done:
        state = torch.FloatTensor(next_state).to(device)
        dist, value = model(state)
        action = dist.sample([1]).squeeze(0)
        done, next_state, reward = env.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)

    state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(state)
    return log_probs, values, entropy, reward, next_value


masks = torch.ones(num_agv)
masks[-1] = 0
for episode in range(total_episodes):
    log_probs, values, entropy, rewards, next_value = run_episode(model, env)
    makespan = rewards[1]

    rewards = torch.FloatTensor(calculate_reward(rewards, 7500)).to(device)

    returns = compute_returns(next_value, rewards, masks)

    actor_loss, critic_loss = train_net(optimizer, log_probs, values, returns, entropy)

    actor_loss_records.append(actor_loss)
    critic_loss_records.append(critic_loss)
    # makespan = np.average([rewards[i][1] for i in range(num_envs)])
    episode_makespan.append(makespan)
    episode_reward = torch.sum(rewards).item() / num_envs
    episode_rewards.append(episode_reward)

    logger.info(f'Episode {episode} completed, makespan: {makespan}, reward: {episode_reward}')

end_time = time.time()
print('time taken', end_time - start_time)

os.makedirs(output_path, exist_ok=True)
date_time = datetime.now().strftime("%m_%d_%H_%M")

torch.save(model.state_dict(), output_path + "A2C_" + date_time + ".pth")

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

sys.exit(0)