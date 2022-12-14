import time
import gym
import torch
import numpy as np
import random

from gym import wrappers
from typing import Dict
from pyvirtualdisplay import Display
from reILs.envs import make_env
from reILs.infrastructure.loggers import setup_logger
from reILs.infrastructure.utils import pytorch_util as ptu
from reILs.infrastructure.execution.evaluate import evaluate
from reILs.infrastructure.execution.collect import collect
from reILs.infrastructure.utils.lr_scheduler import PiecewiseSchedule

# how many rollouts to save as videos
MAX_NVIDEO = 2

class RL_Trainer(object):

    def __init__(self, config: Dict):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.config = config
        self.logger = setup_logger(**self.config['logger_config'])
        self.virtual_disp = Display(visible=False, size=(1400,900))
        self.virtual_disp.start()

        # Set random seed
        seed = self.config.setdefault('seed', 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Init GPU
        ptu.init_gpu(
            use_gpu=not self.config['no_gpu'],
            gpu_id=self.config['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        dummy_env = make_env(self.config['env_name'], self.config['env_config'])

        # Is this env continuous, or self.discrete?
        discrete = isinstance(dummy_env.action_space, gym.spaces.Discrete or gym.spaces.MultiDiscrete)
        # Are the observations images?
        img = len(dummy_env.observation_space.shape) > 2
        self.config['policy_config']['discrete'] = discrete

        # Observation and action sizes
        obs_dim = dummy_env.observation_space.shape if img else dummy_env.observation_space.shape[0]
        act_dim = dummy_env.action_space.n if discrete else dummy_env.action_space.shape[0]
        self.config['policy_config']['obs_dim'] = obs_dim
        self.config['policy_config']['act_dim'] = act_dim
        print("Env: ", dummy_env)
        print("Observation Dim: ", obs_dim)
        print("Action Dim: ", act_dim)
        dummy_env.close()
        del dummy_env

        #############
        ## AGENT
        #############
        agent_class = self.config['agent_class']
        self.agent = agent_class(self.config)
        print("Lr Scheduler: ", self.agent.lr_schedulers)

    ####################################
    ####################################

    def run_training_loop(self, n_itr):
        """
        param n_itr:  number of iterations
        """
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(1, n_itr + 1):

            ## decide if tabular should be logged
            self._refresh_logger_flags(itr)

            ## collect trajectories, to be used for training
            train_batch_list = collect(
                agent=self.agent,
                num_episodes=self.config.get('episode_per_itr', None),
                num_steps=self.config.get('step_per_itr', None),
            )
            train_batch = self.agent.process_fn(train_batch_list)
            self.total_envsteps += len(train_batch.rew)

            ## add collected data to replay buffer
            self.agent.add_to_replay_buffer(train_batch)

            ## train agent (using sampled data from replay buffer)
            train_log = self.agent.train(
                batch_size = self.config.get('batch_size', len(train_batch)),
                repeat = self.config.get('repeat_per_itr', 1),
            )

            ###########################
            ## log and save config_json
            ###########################
            if itr == 1:
                self.logger.log_variant('config.json', self.config)

            ## log/save
            if self.logtabular:
                ## perform tabular and video
                self.perform_logging(itr, train_log)

            if self.logparam:
                self.logger.save_itr_params(itr, self.agent.get_weights())
                self.logger.save_extra_data(
                    self.agent.get_statistics(),
                    file_name='statistics.pkl',
                ) 
        self.logger.close()

    ####################################
    ####################################

    def perform_logging(self, itr, train_log):

        # collect eval trajectories, for logging
        print("\nCollecting rollouts for eval...")
        eval_batch_list = evaluate(self.agent, num_episodes=10)

        # save eval rollouts as videos
        if self.logvideo:
            print('\nCollecting and saving video rollouts')
            video_batch_list = evaluate(agent=self.agent, num_episodes=MAX_NVIDEO, render=True)
            ## save train/eval videos
            print('\nSaving rollouts as videos...')
            self.logger.log_paths_as_videos(
                video_batch_list, itr, video_title='rollouts'
            )

        #######################

        # save eval tabular
        if self.logtabular:
            # returns, for logging
            eval_returns = [batch["rew"].sum() for batch in eval_batch_list]

            # episode lengths, for logging
            eval_ep_lens = [len(batch) for batch in eval_batch_list]

            # decide what to log
            self.logger.record_tabular("Itr", itr)

            self.logger.record_tabular_misc_stat("EvalReward", eval_returns)

            self.logger.record_tabular("EvalEpLen", np.mean(eval_ep_lens))
            self.logger.record_tabular("TotalEnvInteracts", self.total_envsteps)
            self.logger.record_tabular("Time", (time.time() - self.start_time) / 60)
            self.logger.record_dict(train_log)

            self.logger.dump_tabular(with_prefix=True, with_timestamp=False)

    def _refresh_logger_flags(self, itr):

        ## decide if videos should be rendered/logged at this iteration
        if self.config.get('video_log_freq', None) \
                and itr % self.config['video_log_freq'] == 0:
            self.logvideo = True
        else:
            self.logvideo = False

        if self.config.get('tabular_log_freq', None) \
                and itr % self.config['tabular_log_freq'] == 0:
            self.logtabular = True
        else:
            self.logtabular = False

        if self.config.get('param_log_freq', None) \
                and itr % self.config['param_log_freq'] == 0:
            self.logparam = True
        else:
            self.logparam = False
