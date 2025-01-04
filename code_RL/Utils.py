import math
import torch
import numpy as np


# Save model
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))


# Save model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)


# Calculate reward from the final observation
# def calculate_reward(final_obs, optimal_theoretical):
#     makespan = math.log2((final_obs['makespan'] - optimal_theoretical) / 100 + 2) - 1
#     assert makespan > 0
#     combined_times = []
#     for wait, setUp, execTime in final_obs['reward']:
#         a = math.log(wait * 0.05 + math.e) - 1
#         b = setUp * 0.02
#         c = execTime * 0.002
#         # print(a, b, c, makespan)
#         combined_times.append(-(a + b + c + makespan))
#
#     return combined_times


def select_agv(agv_info):
    agv_id = np.random.rand(0, 9)
    for i in range(0, 9):
        if agv_info[i][0] == 0:
            agv_id = i
            break
    return agv_id


def extract_feature(obs):
    state = []
    for arr in obs['state']:
        state += arr
    return state


# def extract_feature(obs, agv_id):
#     x, y, z = obs['agvInfo'][agv_id][1], obs['agvInfo'][agv_id][2], obs['agvInfo'][agv_id][3]
#     state = []
#     for lst in obs['taskInfoQueue']:
#         if lst[0] == 0 and lst[1] == 0:
#             state.append([1, 1, 1, 1])
#         else:
#             state.append([manhattan_distance(lst[1], x, lst[2], y) / 2000, (lst[3] - z) / 80,
#                           manhattan_distance(lst[1], lst[4], lst[2], lst[5]) / 2000, (lst[6] - z) / 80])
#     return state


def manhattan_distance(x1, x2, y1, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def select_action_random(task_info):
    task_index = np.random.randint(0, 20)
    up_boundary = 21
    while task_info[task_index][0] == 0 and task_info[task_index][1] == 0:
        up_boundary -= 5
        task_index = np.random.randint(0, up_boundary)
    return task_index


def check_action(task_info, task_index):
    if task_info[task_index][0] == 0 and task_info[task_index][1] == 0:
        print('Warning: Task not available, Use random strategy.')
        return select_action_random(task_info)
    return task_index


def main():
    # 初始化env
    env = gym.make("CartPole-v1")
    init_state = env.reset()
    init_state = np.delete(init_state, 1)  # 删掉cart velocity这一维度

    # 初始化价值网络
    value_network = ValueNetwork(in_size=STATE_DIM, hidden_size=C_HIDDEN, out_size=1)
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=0.005)

    # 初始化动作网络
    actor_network = ActorNetwork(in_size=STATE_DIM, hidden_size=A_HIDDEN, out_size=ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=0.001)

    steps = []
    task_episodes = []
    test_results = []

    for episode in range(NUM_EPISODE):
        # 完成一轮rollout
        states, actions, rewards, final_r, current_state = roll_out(actor_network, env, EPISODE_LEN, value_network,
                                                                    init_state)
        # states.shape = [epi_len,3],list

        # rollout结束后的初态
        init_state = current_state
        actions_var = Variable(torch.Tensor(actions).view(-1, ACTION_DIM)).unsqueeze(0)
        states_var = Variable(torch.Tensor(states).view(-1, STATE_DIM)).unsqueeze(0)

        # 训练动作网络
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);

        actor_network_optim.zero_grad()
        # print(states_var.unsqueeze(0).size())
        log_softmax_actions, (a_hx, a_cx) = actor_network(states_var, (a_hx, a_cx))
        vs, (c_hx, c_cx) = value_network(states_var, (c_hx, c_cx))  # 给出状态价值估计
        vs.detach()  # 不参与求梯度

        # 计算Q(s,a)和Advantage函数
        qs = Variable(torch.Tensor(discount_reward(rewards, 0.99, final_r)))
        qs = qs.view(1, -1, 1)
        advantages = qs - vs
        # print('adv,',advantages.shape)
        # log_softmax_actions * actions_var是利用独热编码特性取出对应action的对数概率
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions * actions_var, 1) * advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(), 0.5)
        actor_network_optim.step()

        # 训练价值网络
        value_network_optim.zero_grad()
        target_values = qs
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
        values, (c_hx, c_cx) = value_network(states_var, (c_hx, c_cx))

        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(), 0.5)
        value_network_optim.step()

        # Testing
        if (episode + 1) % 50 == 0:
            result = 0
            test_task = gym.make("CartPole-v1")
            for test_epi in range(10):  # 测试10个episode
                state = test_task.reset()
                state = np.delete(state, 1)

                a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
                a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
                c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
                c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);

                for test_step in range(500):  # 每个episode最长500frame

                    log_softmax_actions, (a_hx, a_cx) = actor_network(Variable(torch.Tensor([state]).view(1, 1, 3)),
                                                                      (a_hx, a_cx))
                    softmax_action = torch.exp(log_softmax_actions)

                    # print(softmax_action.data)
                    action = np.argmax(softmax_action.data.numpy()[0])
                    next_state, reward, done, _ = test_task.step(action)
                    next_state = np.delete(next_state, 1)

                    result += reward
                    state = next_state
                    if done:
                        break
            print("episode:", episode + 1, "test result:", result / 10.0)
            steps.append(episode + 1)
            test_results.append(result / 10)
    plt.plot(steps, test_results)
    plt.savefig('training_score.png')


def roll_out(actor_network, env, episode_len, value_network, init_state):
    '''
    rollout最长1000frames
    返回：
    状态序列，不包括终态
    动作序列，独热编码
    奖励序列，不包括终态奖励
    state：游戏环境初始化后的初始状态
    '''
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state  # 初始状态
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);  # 初始化隐状态
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
    c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
    c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);

    for j in range(episode_len):
        states.append(state)
        log_softmax_action, (a_hx, a_cx) = actor_network(Variable(torch.Tensor([state]).unsqueeze(0)), (a_hx, a_cx))
        # 这个部分可以用torch Categorical来实现
        # from torch.distributions import Categorical
        softmax_action = torch.exp(log_softmax_action)  # 对数softmax取指数，保证大于0
        action = np.random.choice(ACTION_DIM, p=softmax_action.cpu().data.numpy()[0][0])

        # 动作独热编码
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]

        next_state, reward, done, _ = env.step(action)
        next_state = np.delete(next_state, 1)
        # fix_reward = -10 if done else 1

        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state  # final_state和state是一回事
        state = next_state
        if done:
            is_done = True
            state = env.reset()
            state = np.delete(state, 1)
            a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
            a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
            c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
            c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);

            # 打印episode总分
            print(j + 1)
            break
    if not is_done:  # 1000frame后如果episode还未结束，就用VNet估计终态价值c_out
        c_out, (c_hx, c_cx) = value_network(Variable(torch.Tensor([final_state])), (c_hx, c_cx))
        final_r = c_out.cpu().data.numpy()  # 如果episode正常结束，final_r=0表示终态cart失去控制得0分
    return states, actions, rewards, final_r, state