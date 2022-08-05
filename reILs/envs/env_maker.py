import gym
import dmc2gym as dmc

from typing import Dict
from gym.wrappers import RecordVideo

def make_env(env_name: str, env_config: Dict):
    env_type='dmc' if '_' in env_name else 'gym'
    seed = env_config['seed']
    episode_length = env_config['episode_length']

    if env_type == 'dmc':
        domain, task = tuple(env_name.split('_'))
        env = dmc.make(
            domain_name=domain,
            task_name=task,
            seed=seed,
            episode_length = episode_length
        )
        env.action_space.seed(seed)
    else:
        raise ValueError("Not supported env type, avaiable env_type = [dm_control]")

    return env
    
