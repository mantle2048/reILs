import torch
import itertools
import numpy as np

from typing import Any, Dict, List, Optional
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

from reILs.infrastructure.datas import Batch
from reILs.infrastructure.utils import pytorch_util as ptu
from reILs.infrastructure.utils import utils

class PPOPolicy(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        # init params
        self.config = config

        self.obs_dim = config['obs_dim']
        self.act_dim = config['act_dim']
        self.layers = config['layers']
        self.discrete = config['discrete']
        self.lr = config['lr']
        self.entropy_coeff = config['entropy_coeff']
        self.grad_clip = config['grad_clip']
        self.epsilon = config['epsilon']
        self.optimizers = {}

        # discrete or continus
        if self.discrete:
            self.logits_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers)
            self.logits_net.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(
                params = self.logits_net.parameters(),
                lr = self.lr)
        else:
            self.logits_net = None
            self.mean_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers,
                                          )
            self.logstd = nn.Parameter(
                    -0.5 * torch.ones(self.act_dim, dtype=torch.float32, device=ptu.device))

            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                params = itertools.chain(self.mean_net.parameters(),[self.logstd]),
                lr = self.lr)
        self.optimizers.update(Pi=self.optimizer)

        # init baseline
        self.baseline = ptu.build_mlp(
            input_size=self.obs_dim,
            output_size=1,
            layers=self.layers
        )
        self.baseline.to(ptu.device)
        self.baseline_optimizer = optim.Adam(
            self.baseline.parameters(),
            self.lr,
        )
        self.optimizers.update(Baseline=self.baseline_optimizer)

        self.apply(ptu.init_weights)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        ptu.scale_last_layer(self.logits_net if self.logits_net else self.mean_net)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        '''
            query the policy with observation(s) to get selected action(s)
        '''
        if len(obs.shape) == 1:
            obs = obs[None]

        obs = ptu.from_numpy(obs.astype(np.float32))

        act_dist = self.forward(obs)

        act = act_dist.sample()

        act = act.squeeze()
        # if self.discrete and act.shape != ():
        #     act = act.squeeze()

        return ptu.to_numpy(act)

    def update(
        self,
        batch: Batch = None,
        **kwargs: Any
    )-> Dict[str, float]:
        '''
            Update the policy using ppo-clip surrogate object
        '''
        obss = ptu.from_numpy(batch.obs)
        acts = ptu.from_numpy(batch.act)
        log_pi_old = ptu.from_numpy(batch.log_prob)
        advs = ptu.from_numpy(batch.adv)

        act_dist = self.forward(obss)
        log_pi = act_dist.log_prob(acts)
        entropy = act_dist.entropy().mean()

        ratio = torch.exp(log_pi - log_pi_old)
        surr1 = ratio * advs
        surr2 = ratio.clamp(
            1.0-self.epsilon, 1.0+self.epsilon
        ) * advs
        surrogate_obj = torch.min(surr1, surr2)
        loss = -torch.mean(surrogate_obj) - self.entropy_coeff * entropy

        # Userful extral info
        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        log_ratio = log_pi - log_pi_old
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipped = ratio.gt(1+self.epsilon) | ratio.lt(1-self.epsilon)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean()

        # optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        targets = ptu.from_numpy(batch.returns)
        baseline_preds = self.baseline(obss).flatten()

        ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
        ## [ N ] versus shape [ N x 1 ]
        ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
        assert baseline_preds.shape == targets.shape

        ## compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
        ## HINT: use `F.mse_loss`
        baseline_loss = F.mse_loss(baseline_preds, targets)

        # optimize `baseline_loss` using `self.baseline_optimizer`
        # HINT: remember to `zero_grad` first
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

        train_log = {}
        train_log['Training loss'] = ptu.to_numpy(loss)
        train_log['Entropy'] = ptu.to_numpy(entropy)
        train_log['KL Divergence'] = ptu.to_numpy(approx_kl)
        train_log['Clip Frac'] = ptu.to_numpy(clipfrac)
        train_log['Baseline loss'] = ptu.to_numpy(baseline_loss)
        return train_log

    def forward(self, obs: torch.Tensor):
        '''
        This function defines the forward pass of the network.
        You can return anything you want, but you should be able to differentiate
        through it. For example, you can return a torch.FloatTensor. You can also
        return more flexible objects, such as a
        `torch.distributions.Distribution` object. It's up to you!
        '''
        if self.discrete:
            logits_na = self.logits_net(obs)
            act_dist = distributions.Categorical(logits = logits_na)

        else:
            mean_na = self.mean_net(obs)
            std_na = torch.exp(self.logstd)
            act_dist = distributions.MultivariateNormal(loc=mean_na, scale_tril=torch.diag(std_na))
            # helpful: difference between multivariatenormal and normal sample/batch/event shapes:
            # https://bochang.me/blog/posts/pytorch-distributions/
            # https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/

        return act_dist

    def run_baseline_prediction(self, obs: np.ndarray):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array
            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]
        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]

    def save(self, filepath=None):
        torch.save(self.state_dict(), filepath)

    def set_weights(self, weights: Dict):
        self.load_state_dict(weights)

    def get_weights(self) -> Dict:
        return {k: v.cpu().detach() for k, v in self.state_dict().items()}

    def _get_log_prob(self, obss, acts):
        obss = ptu.from_numpy(obss)
        acts = ptu.from_numpy(acts)
        act_dist = self.forward(obss)
        return ptu.to_numpy(act_dist.log_prob(acts))

