from reILs.scripts.run_ppo import get_parser, get_config, main
from reILs.rl_trainer import RL_Trainer
from reILs.algos import PPOAgent
# import ray
# ray.init(
#     ignore_reinit_error=True,
#     local_mode=True,
# )
# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args([
        '--env-name',
        'Ant-v3',
        '--seed',
        '1',
        '--n-itr',
        '501',
        '--num-workers',
        '2',
        '--step-per-itr',
        '2000',
        '--repeat-per-itr',
        '10',
        '--batch-size',
        '100',
        '--tabular-log-freq',
        '5',
        '--gae-lambda',
        '0.95',
        '--lr',
        '3e-4',
        '--obs-norm',
        '--rew-norm',
        '--recompute-adv'
        # '--adv-norm',

    ])

    # convert to dictionary
    agent_config = get_config(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=agent_config['n_itr'])




