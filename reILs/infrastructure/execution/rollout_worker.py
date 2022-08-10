import random
import numpy as np
import torch
import os
import warnings
import ray
from typing import Optional, List, Dict, Union, Callable
from reILs.infrastructure.execution import RolloutSaver
from reILs.infrastructure.datas import Batch

def update_gloabl_seed(
    seed: Optional[int] = None,
    worker_id: int=0,
) -> None:
    """Seed global modules such as random, numpy, torch.
    This is useful for debugging and testing.
    Argsw
        seed: An optional int seed. If None, will not do
            anything.
    """
    if seed is None:
        return

    computed_seed: int = worker_id * 1000 + seed
    # Python random module.
    random.seed(computed_seed)
    # Numpy.
    np.random.seed(computed_seed)
    # Torch.
    torch.manual_seed(computed_seed)

def update_env_seed(
    env,
    seed: Optional[int] = None,
    worker_id: int=0,
):
    """Set a deterministic random seed on environment.
    NOTE: this may not work with remote environments (issue #18154).
    """
    if not seed:
        return

    # A single RL job is unlikely to have more than 10K
    # rollout workers.
    computed_seed: int = worker_id * 1000 + seed

    # Gym.env.
    # This will silently fail for most OpenAI gyms
    # (they do nothing and return None per default)
    if not hasattr(env, "seed"):
        warnings.wran("Env doesn't support env.seed(): {}".format(env))
    else:
        env.seed(computed_seed)

class RolloutWorker:

    """Common experience collection class.
    This class wraps a policy instance and an environment class to
    collect experiences from the environment. You can create many replicas of
    this class as Ray actors to scale RL training."""

    @classmethod
    def as_remote(
        cls,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[Union[int, float]] = None,
        memory: Optional[int] = None,
        object_store_memory: Optional[int] = None,
        resources: Optional[dict] = None,
    ) -> type:
        """Returns RolloutWorker class as a `@ray.remote using given options`.
        The returned class can then be used to instantiate ray actors.
        Args:
            num_cpus: The number of CPUs to allocate for the remote actor.
            num_gpus: The number of GPUs to allocate for the remote actor.
                This could be a fraction as well.
            memory: The heap memory request for the remote actor.
            object_store_memory: The object store memory for the remote actor.
            resources: The default custom resources to allocate for the remote
                actor.
        Returns:
            The `@ray.remote` decorated RolloutWorker class.
        """
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def __init__(
        self,
        *,
        worker_id: int = 0,
        env_maker: Callable[["env_name", "env_config"], "env"],
        policy_maker: Callable[["policy_name", "policy_config"], "policy"],
        config: Dict,
    ):
        """
        Initializes a RolloutWorker instance.

        Args:
            env_creator: Function that returns a gym.Env given an 
                wrapped configuration.
            worker_id: For remote workers, this should be set to a
                non-zero and unique value. This id is passed to created envs
                through EnvContext so that envs can be configured per worker.
            config: Config to pass to worker (consists of env_config,
                policy_config)
        """
        self.worker_id = worker_id
        self.config = config
        update_gloabl_seed(config.get('seed'), worker_id)
        self.policy = policy_maker(
            policy_name = config.get('policy_name'),
            policy_config = config.get('policy_config')
            )
        self.env = env_maker(
            env_name = config.get('env_name'),
            env_config = config.get('env_config')
        )
        update_env_seed(self.env, config.get('seed'), worker_id)
        self.saver = RolloutSaver(save_info=True)

    def sample(self) -> Batch:
        step_list = []
        env, policy = self.env, self.policy
        done, ep_rew, ep_len, step_list = False, 0.0, 0, []
        obs = env.reset()
        while not done:
            act = policy.get_action(obs)
            next_obs, rew, done, info = env.step(act)

            ep_rew += rew
            step_return = Batch(
                obs=obs, act=act, next_obs=next_obs,
                rew=rew, done=done, info=info,
                ep_len=ep_len, ep_rew=ep_rew,
            )
            step_list.append(step_return)
            obs = next_obs
            ep_len += 1

        batch = Batch.stack(step_list)
        return batch

    def set_weights(self, weights):
        self.policy.set_weights(weights)

    def get_weights(self):
        return self.policy.get_weights()

    def stop(self):
        """Releases all resources used by this RolloutWorker."""
        self.env.close()
