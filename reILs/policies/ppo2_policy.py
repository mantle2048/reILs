import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb,

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

        # discrete or continus
        if self.discrete:
            self.logits_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers)
            net_a = Net(
                self.obs_dim,
                hidden_sizes=args.layers,
                activation=nn.Tanh,
                device=args.device,
            )
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
                                            layers=self.layers)
            self.logstd = nn.Parameter(
                    -0.5 * torch.ones(self.act_dim, dtype=torch.float32, device=ptu.device))

            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                params = itertools.chain(self.mean_net.parameters(),[self.logstd]),
                lr = self.lr)

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

        self.apply(ptu.init_weights)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        ptu.scale_last_layer(self.logits_net if self.logits_net else self.mean_net)
