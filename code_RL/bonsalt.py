from typing import Any, Tuple
import logging
import requests


class BrainState():
    def episode_start(config: dict):
        d = {
            'type': 'EpisodeStart',
            # 'sessionId': session_id,
            # 'sequenceId': sequence_id,
            'episodeStart': {
                'config': config
            }
        }
        return d

    def episode_step(action: dict):
        d = {
            'type': 'EpisodeStep',
            # 'sessionId': session_id,
            # 'sequenceId': sequence_id,
            'episodeStep': {
                'action': action
            }
        }
        return d

    def episode_finish():
        d = {
            'type': 'EpisodeFinish',
            # 'sessionId': session_id,
            # 'sequenceId': sequence_id,
        }
        return d

    def idle(callback_time: int):
        d = {
            'type': 'EpisodeStart',
            # 'sessionId': session_id,
            # 'sequenceId': sequence_id,
        }

        if callback_time is not None:
            d['idle'] = {
                'callbackTime': callback_time
            }

        return d

    def unregister():
        d = {
            'type': 'EpisodeStart',
            # 'sessionId': session_id,
            # 'sequenceId': sequence_id,
            'unregister': {
                'reason': 'Unspecified'
            }
        }
        return d


class BrainClient():
    logger = logging.getLogger('BonsaltClient')

    host: str = 'http://localhost:5000'
    workspace: str = None
    access_key: str = None
    session_id: str = None

    sequence_id = 2
    state: dict = None
    event: dict = None

    def __init__(self, host: str, workspace: str, access_key: str) -> None:
        self.host = host
        self.workspace = workspace
        self.access_key = access_key
        self.sequence_id = 2

    def register(self) -> None:
        endpoint = f'{self.host}/v2/Workspaces/{self.workspace}/BrainSessions'
        r = requests.post(endpoint)

        if r.status_code != 201:
            self.logger.warning('failed to register brain to platform')
            pass

        data = r.json()
        self.session_id = data['sessionId']
        self.logger.info(f'brain {self.session_id} registered to platform')

    def advance(self, state) -> Any:
        state['sessionId'] = self.session_id
        state['sequenceId'] = self.sequence_id

        endpoint = f'{self.host}/v2/Workspaces/{self.workspace}/BrainSessions/{self.session_id}/Advance'
        r = requests.post(endpoint, json=state)

        if r.status_code != 200:
            self.logger.warning('bad response from platform')
            pass

        self.event = r.json()
        # self.sequence_id = self.event['sequenceId']

        self.logger.debug(
            f'[{self.session_id}] seq = {self.sequence_id}, event = {self.event["type"]}')

        if self.sequence_id == self.event['sequenceId']:
            self.sequence_id += 1

        return self.event

    def unregister(self) -> None:
        endpoint = f'{self.host}/v2/Workspaces/{self.workspace}/BrainSessions/{self.session_id}'
        r = requests.delete(endpoint)

        if r.status_code != 204:
            self.logger.warning('failed to unregister brain from platform')
            pass

        self.logger.info(f'brain {self.session_id} unregistered from platform')

    def heartbeat(self) -> None:
        pass
        endpoint = f'{self.host}/v2/Workspaces/{self.workspace}/BrainSessions/{self.session_id}/HeartBeat'
        r = requests.get(endpoint)


class BonsaltEnv():
    client: BrainClient = None

    def __init__(self, host: str, workspace: str, access_key: str) -> None:
        self.client = BrainClient(host, workspace, access_key)
        self.client.register()

    def step(self, action: dict) -> Tuple[bool, dict]:
        while True:
            event = self.client.advance(BrainState.episode_step(action))
            # print(f"Debug: Step event -> {event}")  # 添加调试打印
            if event['type'] in ['EpisodeStart', 'EpisodeStep', 'EpisodeFinish']:
                sim = event['simulatorState']
                return (sim['halted'], sim['state'])

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> dict:
        while True:
            event = self.client.advance(BrainState.episode_start(options))
            # print(f"Debug: Reset event -> {event}")  # 添加调试打印
            if event['type'] == 'EpisodeStart':
                return event['simulatorState']['state']

    def close(self):
        self.client.unregister()


# class BonsaltGymnasiumEnv(Env):
#     client: BrainClient = None

#     def __init__(self, host: str, workspace: str, access_key: str) -> None:
#         super().__init__()
#         self.client = BrainClient(host, workspace, access_key)
#         self.client.register()

#     def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
#         return super().step(action)

#     def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
#         return super().reset(seed=seed, options=options)

#     def close(self):
#         self.client.unregister()
#         return super().close()


if __name__ == '__main__':
    import time

    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
    logging.getLogger('BonsaltClient').setLevel(logging.DEBUG)

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    env = BonsaltEnv('http://localhost:5000', 'rl-hospital', 'not used')
   

    index = 0
    while True:
        logger.info(f'episode {index}')
        index += 1

        terminated = False
        obs = env.reset()
        while not terminated:
            (terminated, obs) = env.step({
                'nResA': 20,
                'nResB': 20,
                'processTime': 1,
                'conveyorSpeed': 15,
            })

        time.sleep(5)
