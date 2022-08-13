import ray
import numpy as np

from typing import Optional, List, Dict, Union, Callable
from reILs.infrastructure.execution import RolloutWorker
from ray.actor import ActorHandle

class WorkerSet:

    """
    Set of RolloutWorkers with n @ray.remote workers and zero or one local worker.
    Where: n >= 0.
    """
    def __init__(
        self,
        *,
        num_workers: int = 0,
        env_maker: Callable[["env_name", "env_config"], "env"] = None,
        policy_maker: Callable[["policy_name", "policy_config"], "policy"] = None,
        config: Dict = None,
        local_worker: bool = True,
        local_mode: bool = False,
        _setup: bool = True,
    ):
        """Initializes a WorkerSet instance.
        Args:
            num_workers: Number of remote rollout workers to create.
            env_maker: Function that returns env given env config.
            policy_maker: Function that returns policy given policy config.
            config: Config to pass to workset and each worker (consists of
            env_config, policy_config)
            local_worker: Whether to create a local (non @ray.remote) worker
                in the returned set as well (default: True). If `num_workers`
                is 0, always create a local worker.
            _setup: Whether to setup workers. This is only for testing.
        """
        self._num_workers = num_workers
        self._env_maker = env_maker
        self._policy_maker = policy_maker
        self._config = config
        self._cls = RolloutWorker.as_remote().remote
        self._remote_workers = []

        if _setup:
            # Create a number of @ray.remote workers.
            self.add_workers(num_workers)

        if num_workers == 0 or local_worker:
            # Create a local worker, if needed.
            # If num_workers > 0 and we don't have an env on the local worker,
            # get the observation- and action spaces for each policy from
            # the first remote worker (which does have an env).
            self._local_worker = self._make_worker(
                cls = RolloutWorker,
                worker_id = 0,
                env_maker = self._env_maker,
                policy_maker = self._policy_maker,
                config = self._config
            ) 

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                local_mode=local_mode,
            )

    def local_worker(self) -> RolloutWorker:
        """Returns the local rollout worker."""
        return self._local_worker

    def remote_workers(self) -> List[ActorHandle]:
        """Returns a list of remote rollout workers."""
        return self._remote_workers

    def sync_weights(self):
        """Syncs model weights from the local worker to all remote workers. """

        # Only sync if we have remote workers or `from_worker` is provided.
        weights = None
        if len(self.remote_workers()):

            # sync nerual network params
            weights = self.local_worker().get_weights()
            # Put weights only once into object store and use same object
            # ref to synch to all workers.
            weights_ref = ray.put(weights)
            # Sync to all remote workers in this WorkerSet.
            for to_worker in self.remote_workers():
                to_worker.set_weights.remote(weights_ref)

            # sync obs_rms from remote envs to local env
            if hasattr(self.local_worker().env, 'obs_rms'):
                obs_statistics = ray.get(
                    [worker.get_obs_statistics.remote() for worker in self.remote_workers()]
                )
                obs_mean, obs_var = np.mean(obs_statistics, axis=0)
                self.local_worker().env.obs_rms.mean = obs_mean
                self.local_worker().env.obs_rms.var = obs_var


    def add_workers(self, num_workers: int):
        """Creates and adds a number of remote workers to this worker set.
        Can be called several times on the same WorkerSet to add more
        RolloutWorkers to the set.
        Args:
            num_workers: The number of remote Workers to add to this
                WorkerSet.
        Raises:
            RayError: If any of the constructed remote workers is not up and running
            properly
        """
        old_num_workers = len(self._remote_workers)
        self._remote_workers.extend(
            [
                self._make_worker(
                    cls=self._cls,
                    worker_id = old_num_workers + i + 1,
                    env_maker = self._env_maker,
                    policy_maker = self._policy_maker,
                    config = self._config,
                )
                for i in range(num_workers)
            ]
        )

    def stop(self):
        try:
            self.local_worker.stop()
            tids = [w.stop.remote() for w in self.remote_workers]
            ray.get(tids)
        except Exception:
            logging.warning("Failed to stop workers!")
        finally:
            for w in self.remote_workers():
                w.__ray_terminate__.remote()

    def _make_worker(
        self,
        cls: Callable,
        worker_id,
        env_maker: Callable[["env_name", "env_config"], "env"] = None,
        policy_maker: Callable[["policy_name", "policy_config"], "policy"] = None,
        config: Dict = None,
    ):
        worker = cls(
            worker_id=worker_id,
            env_maker=env_maker,
            policy_maker=policy_maker,
            config=config
        )
        return worker
