import subprocess
import threading

from bonsalt import BonsaltEnv
from Pricing import pricing


def extract_state(obs):
    state = []
    for arr in obs['state']:
        state.append(arr[0])
        state.append(arr[1] * 1e-3)
    state+=pricing([obs['nodes'][9]])
    return state


def extract_reward(obs):
    return [obs['reward'], obs['makespan']]


class MyEnv():
    env: BonsaltEnv = None

    def __init__(self, env) -> None:
        self.env = env

    def step(self, action: int):
        (done, obs) = self.env.step({'agv_id': int(action)})
        next_state = extract_state(obs)
        reward = extract_reward(obs)
        return done, next_state, reward

    def reset(self) -> []:
        state = extract_state(self.env.reset())
        return state

    def close(self):
        self.env.close()


def simulator():
    subprocess.run("./../Simulator/Hospital_DRL/start-simulator.sh", shell=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def make_env(seed=None):
    if seed is not None:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)

    threading.Thread(target=simulator).start()
    return MyEnv(BonsaltEnv('http://localhost:5000', 'rl-hospital', 'not used'))


class VecEnv():
    env_list: [BonsaltEnv] = None
    nenvs: int = 0

    def __init__(self, env_fns) -> None:
        self.nenvs = len(env_fns)
        self.env_list = env_fns

    def step(self, actions: [int]):
        done = False
        next_states = []
        rewards = []
        for i in range(self.nenvs):
            (done, obs) = self.env_list[i].step({'agv_id': int(actions[i])})
            next_states.append(extract_state(obs))
            rewards.append(extract_reward(obs))
        return done, next_states, rewards

    def reset(self) -> [dict]:
        states = []
        for i in range(self.nenvs):
            states.append(extract_state(self.env_list[i].reset()))
        return states

    def close(self):
        for env in self.env_list:
            env.close()