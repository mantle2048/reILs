
from typing import Dict
from reILs.rl_trainer import RL_Trainer
from reRLs.algos.ppo import PPOAgent

def get_config(args):
    #####################
    ## SET AGENT CONFIGS
    #####################
    config['policy_name'] = 'ppo'

    # policy args
    policy_config = {
        'policy_name': config['policy_name'],
        'layers': config['layers'],
        'lr': config['lr'],
        'lr_decay': config['lr_decay'],
        'epsilon': config['epsilon'],
    }

    # logger args
    logger_config = {
        'exp_prefix': f"{config['policy_name']}_{config['env_name']}",
        'seed': config['seed'],
        'snapshot_mode': config['snapshot_mode'],
    }

    # env args
    env_config = {
        'env_name': config['env_name'],
        'seed': config['seed'],
    }

    agent_config = {
        'agent_class': PPOAgent,
        'policy_name': policy_name,
        'env_name': env_name,
        'policy_config': policy_config,
        'env_config': env_config,
        'logger_config': logger_config,
        'standardize_advantages': not(config.get('dont_standardize_advantages', False)),
        'gae_lambda': config.get('gae_lambda', None),
        'target_kl': config.get('target_kl', None),
        'gamma': config.get('gamma', None),
        'entropy_coeff': config.get('entropy_coeff', None),
        'grad_clip': config.get('grad_clip', None),
        'n_itr': config.get('n_itr', None),
        'num_workers': config.get('num_workers', None),
        'no_gpu': config.get('no_gpu', None),
        'which_gpu': config.get('which_gpu', None),
        'batch_size': config.get('batch_size', None),
        'repeat_per_itr': config.get('repeat_per_itr', None),
        'step_per_itr': config.get('step_per_itr', None),
        'episode_per_itr': config.get('episode_per_itr', None),
        'video_log_freq': config.get('video_log_freq', None),
        'tabulate_log_freq': config.get('tabulate_log_freq', None),
        'param_log_freq': config.get('param_log_freq', None),
    }

def get_parser():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', type=str, default='CartPole-v1')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--which-gpu', default=0)
    parser.add_argument('--snapshot-mode', type=str, default="last")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-itr', '-n', type=int, default=10)
    parser.add_argument('--step-per-itr', type=int, default=5000) #steps collected per train iteration
    parser.add_argument('--repeat-per-itr', type=int, default=5) #steps collected per train iteration
    parser.add_argument('--batch-size', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--tabular-log-freq', type=int, default=1)
    parser.add_argument('--video-log-freq', type=int, default=None)
    parser.add_argument('--param-log-freq', type=int, default=None)
    parser.add_argument('--obs-norm', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy_coeff', type=float, default=0.)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=None)
    parser.add_argument('--dont-standardize-advantages', '-dsa', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=[64,64])
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--target_kl', type=float, default=None)
    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    # convert to dictionary
    config = vars(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=config['n_itr'])


if __name__ == '__main__':
    main()
