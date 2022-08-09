import numpy as np
from scipy.signal import lfilter
from typing import Dict,Union,List

from .ppo_policy import PPOPolicy
from reILs.envs.env_maker import make_env
from reILs.algos.policy_maker import make_policy
from reILs.infrastructure.datas import ReplayBuffer
from reILs.infrastructure.execution import WorkerSet
from reILs.infrastructure.utils import utils
from reILs.infrastructure.utils import pytorch_util as ptu


class PPOAgent:

    def __init__(self, config: Dict):

        # init params
        self.config = config
        self.workers = WokerSet(
            num_workers = config['num_workers'],
            env_maker = make_env,
            policy_maker = make_policy,
            config = config
        )
        self.env = self.workers.local_worker().env
        self.policy = self.workers.local_worker().policy
        self.replay_buffer = ReplayBuffer(config['itr_size'])

        self.gamma = config.setdefault('gamma', 0.99)
        self.standardize_advantages = \
                config.setdefault('standardize_advantages', True)
        self.gae_lambda = config.setdefault('gae_lambda', 0.99)
        self.target_kl = config.setdefault('target_kl', 0.2)

    def process_fn(self, batch_list: List[Batch]) -> Batch:
        rew_list = [batch.rew for batch in batch_list]
        full_batch = Batch.cat(batch_list)
        obss, acts = full_batch.obs, full_batch.act
        dones = full_batch.done

        # step 1: calculate q values of each (s_t, a_t) point, using rewards [r_1, ..., r_t, ..., r_T]
        full_batch['q_value'] = self.calculate_q_values(rews_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        full_batch['adv'] = self.estimate_advantages(obss, rews_list, q_values, dones)

        # step 3: obtain log prob that correspond to each (s_t, a_t) point
        full_batch['log_prob'] = self.get_log_prob(obss, acts)
        return full_batch

    def train(self, batch_size: int, repeat: int) -> Dict:

        """
            Training a PPO agent refers to updating its policy using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """
        train_logs = []
        for _ in range(repeat):
            batch = self.sample(batch_size)
            # use all datapoints (s_t, a_t, q_t, adv_t) to update the PG actor/policy
            ## HINT: `train_log` should be returned by the actor update method
            train_log = self.policy.update(batch)
            train_logs.append(train_log)

            if self.target_kl is not None and train_log['KL Divergence'] > 1.5 * self.target_kl:
                break

        return train_logs

    def sample(self, batch_size: int) -> Batch:
        return self.replay_buffer.sample(batch_size)

    def add_to_replay_buffer(self, batch: Batch):
        self.replay_buffer.add_batch(batch)

    def resume(self):
        pass

    def calculate_q_values(self, rews_list: List[np.ndarray]):

        """
            Monte Carlo estimation of the Q function.
        """
        # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full trajectory
        # In other words: value of (s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        q_values = np.concatenate([self._discounted_cumsum(self.gamma, rews) for rews in rews_list])

        return q_values

    def estimate_advantages(self, obss, rews_list, q_values, dones):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """
        # Estimate the advantage when use_baseline is True,
        # by querying the neural network that you're using to learn the baseline
        baselines_standardized = self.policy.run_baseline_prediction(obss)
        ## ensure that the baseline and q_values have the same dimensionality
        ## to prevent silent broadcasting errors
        assert baselines_standardized.ndim == q_values.ndim
        ## baseline was trained with standardized q_values, so ensure that the predictions
        ## have the same mean and standard deviation as the current batch of q_values
        baselines =  \
                utils.de_standardize(baselines_standardized, np.mean(q_values), np.std(q_values))

        if self.gae_lambda is not None:
            ### append a dummy T+1 value for simpler recursive calculation
            baselines = np.append(baselines, [0])

            ### combine rews_list into a single array
            rews = np.concatenate(rews_list)

            ### create empty numpy array to populate with GAE advantage
            ### estimates, with dummy T+1 value for simpler recursive calculation
            advs = np.zeros_like(baselines)

            deltas = rews + self.gamma * baselines[1:] * (1 - dones) - baselines[:-1]

            for i in reversed(range(obss.shape[0])):
                advs[i] = self.gamma * self.gae_lambda * advs[i+1] * (1 - dones[i]) + deltas[i]

            ### remove dummy advantage
            advs = advs[:-1]

        else:
            ### compute advantage estimates using q_values and baselines
            advs = q_values - baselines

        # Normalize the resulting advantages
        ## standardize the advantages to have a mean of zero
        ## and a standard deviation of one
        ## HINT: there is a `standardize` function in `infrastructure.utils`
        advs = utils.standardize(advs, np.mean(advs), np.std(advs))

        return advs

    def get_log_prob(self, obss: np.ndarray, acts: np.ndarray):
        return self.policy._get_log_prob(obss, acts)

    def _discounted_return(self, discount, rewards: List) -> np.ndarray:
        """
            Helper function
            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T
            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        discounted_returns = np.ones_like(rewards) * self._discounted_cumsum(discount, rewards)[0]
        return discounted_returns

    def _discounted_cumsum(self, discount, rewards: List) -> np.ndarray:
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        discounted_cursums = lfilter([1], [1, -discount], rewards[::-1], axis=0)[::-1]
        return discounted_cursums
