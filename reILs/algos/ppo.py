import numpy as np
import reILs.algos as algos

from scipy.signal import lfilter
from typing import Dict,Union,List,Tuple

from numba import njit
from reILs.envs import make_env
from reILs.policies import make_policy
from reILs.infrastructure.datas import ReplayBuffer, Batch
from reILs.infrastructure.execution import WorkerSet
from reILs.infrastructure.utils import utils
from reILs.infrastructure.utils import pytorch_util as ptu
from reILs.infrastructure.utils.statistics import RunningMeanStd


class PPOAgent:

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

        self.gamma = config.get('gamma')
        self.standardize_advantages = \
                config.get('standardize_advantages')
        self.gae_lambda = config.get('gae_lambda')
        self.target_kl = config.get('target_kl')
        self.recompute_adv = config.get('recompute_adv')
        self.rew_norm = config.setdefault('rew_norm', False)
        self.adv_norm = config.setdefault('adv_norm', False)
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8

    def process_fn(self, batch_list: List[Batch]) -> Batch:
        batch = Batch.cat(batch_list)
        batch = self.get_log_prob(batch)
        batch = self.get_returns_and_advs(batch)
        return batch

    def get_returns_and_advs(self, batch: Batch) -> Batch:
        obss, next_obss, rews = batch.obs, batch.next_obs, batch.rew
        dones, terminals = batch.done.copy(), batch.terminal.copy()
        returns, advs = self.estimate_returns_and_advantages(
            obss, next_obss,
            rews, dones, terminals
        )
        batch['returns'] = returns
        batch['adv'] = advs
        return batch

    def get_log_prob(self, batch: Batch):
        log_probs =  self.policy._get_log_prob(batch.obs, batch.act)
        batch['log_prob'] = log_probs
        return batch

    def train(self, batch_size: int, repeat: int) -> Dict:

        """
            Training a PPO agent refers to updating its policy using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """
        train_logs = []
        for step in range(repeat):
            batch = self.sample(0)
            if self.recompute_adv and step > 0:
                batch = self.get_returns_and_advs(batch)
            for minibatch in batch.split(batch_size, merge_last=True):
                if self.adv_norm:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / std  # per-batch norm
                train_log = self.policy.update(minibatch)
                train_logs.append(train_log)
            if self.target_kl is not None and train_log['KL Divergence'] > self.target_kl:
                break

        self.workers.sync_weights()


        return train_logs

    def estimate_returns_and_advantages(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        rews: np.ndarray,
        dones: np.ndarray,
        terminals: np.ndarray
    )-> Tuple[np.ndarray, np.ndarray]:
        obss_v = self.policy.run_baseline_prediction(obss)
        next_obss_v = self.policy.run_baseline_prediction(next_obss)
        if self.rew_norm:
            obss_v =  obss_v * np.sqrt(self.ret_rms.var + self._eps)
            next_obss_v = next_obss_v * np.sqrt(self.ret_rms.var + self._eps)
        # Value mask
        next_obss_v[terminals] = 0.0
        # truncted episode
        dones[-1] = True
        advs = _gae_return(
            obss_v, next_obss_v,
            rews, dones,
            self.gamma, self.gae_lambda
        )
        unnormalized_returns = advs + obss_v # λ return
        if self.rew_norm:
            returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            returns = unnormalized_returns
        return returns, advs


    def sample(self, batch_size: int) -> Batch:
        return self.replay_buffer.sample(batch_size)

    def add_to_replay_buffer(self, batch: Batch):
        self.replay_buffer.add_batch(batch)

    def resume(self):
        pass


@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns
