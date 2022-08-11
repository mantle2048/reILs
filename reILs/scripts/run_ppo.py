import argparse

from typing import Dict
from reILs.rl_trainer import RL_Trainer
from reILs.algos import PPOAgent

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='CartPole-v1')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--which-gpu', default=0)
    parser.add_argument('--snapshot-mode', type=str, default="last")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-itr', '-n', type=int, default=100)
    parser.add_argument('--step-per-itr', type=int, default=5000) #steps collected per train iteration
    parser.add_argument('--repeat-per-itr', type=int, default=5) #steps collected per train iteration
    parser.add_argument('--batch-size', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--tabular-log-freq', type=int, default=1)
    parser.add_argument('--video-log-freq', type=int, default=None)
    parser.add_argument('--param-log-freq', type=int, default=10)
    parser.add_argument('--obs-norm', action='store_true')
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coeff', type=float, default=0.)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=None)
    parser.add_argument('--dont-standardize-advantages', '-dsa', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', action='store_true')
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=[64,64])
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--target_kl', type=float, default=None)
    parser.add_argument("--recompute-adv", action='store_true')
    parser.add_argument("--rew-norm", action='store_true')
    parser.add_argument("--adv-norm", action='store_true')
    return parser

def get_config(args: argparse.Namespace) -> Dict:
    #####################
    ## SET AGENT CONFIGS
    #####################
    args.policy_name = 'ppo'

    # policy args
    policy_config = {
        'policy_name': args.policy_name,
        'layers': args.layers,
        'lr': args.lr,
        'lr_decay': args.lr_decay,
        'epsilon': args.epsilon,
        'entropy_coeff': args.entropy_coeff,
        'grad_clip': args.grad_clip,
    }

    # logger args
    logger_config = {
        'exp_prefix': f"{args.policy_name}_{args.env_name}",
        'seed': args.seed,
        'snapshot_mode': args.snapshot_mode,
    }

    # env args
    env_config = {
        'env_name': args.env_name,
        'obs_norm': args.obs_norm,
        'seed': args.seed,
    }

    agent_config = vars(args)

    agent_config.update(
        agent_class=PPOAgent,
        policy_config=policy_config,
        env_config=env_config,
        logger_config=logger_config
    )
    return agent_config

def main():

    parser = get_parser()
    args = parser.parse_args([])

    # convert to dictionary
    agent_config = get_config(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=100)


if __name__ == '__main__':
    main()
