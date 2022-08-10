import gym
from gym.wrappers import NormalizeObservation
import dmc2gym as dmc
from typing import Dict

def make_env(env_name: str, env_config: Dict):
    env_type='dmc' if '_' in env_name else 'gym'
    seed = env_config['seed']

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
        env = gym.make(env_name)
        env = NormalizeObservation(env)
        env.seed(seed)
        env.action_space.seed(seed)
    # raise ValueError("Not supported env type, avaiable env_type = [dm_control]")
    return env
