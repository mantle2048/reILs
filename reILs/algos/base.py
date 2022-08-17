from typing import Dict,Union,List,Tuple
from torch.optim.lr_scheduler import LambdaLR
import ray

from reILs.envs import make_env
from reILs.policies import make_policy
from reILs.infrastructure.execution import WorkerSet
from reILs.infrastructure.datas import ReplayBuffer, Batch
from reILs.infrastructure.utils.statistics import RunningMeanStd
from reILs.infrastructure.utils.lr_scheduler import PiecewiseSchedule, MultipleLRSchedulers

class OnAgent:

    def __init__(self, config: Dict):

        # init params
        self.config = config
        self.workers = WorkerSet(
            num_workers = config['num_workers'],
            env_maker = make_env,
            policy_maker = make_policy,
            config = config
        )
        self.env = self.workers.local_worker().env
        self.policy = self.workers.local_worker().policy
        self.replay_buffer = ReplayBuffer(config['step_per_itr'])
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self.lr_schedulers = \
            self.create_lr_scheduler(self.config.get('lr_schedule', {}))

    def process_fn(self, batch: Union[List[Batch], Batch]) -> Batch:

        raise NotImplementedError

    def train(self, batch_size: int, repeat: int) -> Dict:
        """
            Training an agent refers to updating its policy using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """
        raise NotImplementedError

    def create_lr_scheduler(self, schedule_dict: Dict[str, List[List]]):
        lr_schedulers = MultipleLRSchedulers()
        for k, v in schedule_dict.items():
            assert k in self.policy.optimizers.keys(), f'Not found corresponding {k} optimizer'
            sche = PiecewiseSchedule(endpoints=schedule_dict[k])
            lr_schedulers.update({k: LambdaLR(self.policy.optimizers[k], lr_lambda=sche)})
        return lr_schedulers

    def sample(self, batch_size: int) -> Batch:
        return self.replay_buffer.sample(batch_size)

    def add_to_replay_buffer(self, batch: Batch):
        self.replay_buffer.add_batch(batch)

    def get_statistics(self):
        statistics = {}
        if self.config.get('obs_norm'):
            statistics.update(ray.get(self.workers.remote_workers()[0].get_statistics.remote()))
        return statistics

    def set_statistics(self, statistics: Dict):
        self.workers.local_worker().set_statistics(statistics)

    def set_weights(self, weights):
        self.policy.set_weights(weights)

    def get_weights(self):
        return self.policy.get_weights()
