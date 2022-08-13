from reILs.scripts.run_gail import get_parser, get_config, main
from reILs.rl_trainer import RL_Trainer
from reILs.algos import GAILAgent
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
        '--disc-lr',
        '2.5e-5',
        '--disc-update-num',
        '2',
        '--env-name',
        'HalfCheetah-v2',
        '--seed',
        '0',
        '--n-itr',
        '601',
        '--num-workers',
        '10',
        '--step-per-itr',
        '5000',
        '--lr-decay',
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
        '--entropy-coeff',
        '0.001',
        # '--recompute-adv',
        # '--ret-norm',
        # '--adv-norm',

    ])

    # convert to dictionary
    agent_config = get_config(args)

    ################
    # RUN TRAINING #
    ################

    rl_trainer = RL_Trainer(agent_config)
    rl_trainer.run_training_loop(n_itr=agent_config['n_itr'])




