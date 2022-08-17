import shelve
import pickle
import joblib
import argparse
import json
import numpy as np
import os
import os.path as osp
import torch
import reILs
import ray

from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
from pyvirtualdisplay import Display
from collections import defaultdict

from reILs import algos
from reILs.user_config import LOCAL_DIR, LOCAL_RENDER_CONFIG
from reILs.infrastructure.loggers import VideoRecorder
from reILs.infrastructure.execution import RolloutSaver, synchronous_parallel_sample, WorkerSet 
from reILs.infrastructure.datas import Batch
from reILs.infrastructure.utils import utils
from reILs.infrastructure.utils import pytorch_util as ptu
from reILs.infrastructure.utils.gym_util import get_max_episode_steps

"yanked and modified from https://github.com/ray-project/ray/blob/130b7eeaba/rllib/evaluate.py"

EXAMPLE_USAGE = """
Example usage via executable:
    ./evaluate.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
Example usage w/o checkpoint (for testing purposes):
    ./evaluate.py --run PPO --env CartPole-v0 --episodes 500
"""

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent given a checkpoint model.",
        )
    parser.add_argument(
        "--exp-dir",
        type=str,
        nargs='?',
        help="exp_dir from which to roll out.",
        )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run ray in local mode for easier debugging."
        )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment while rollouting"
        )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        )
    parser.add_argument(
        '--no-gpu',
        action='store_true'
    )
    parser.add_argument(
        '--which-gpu',
        default=0
    )
    parser.add_argument(
        "--steps",
        default=0,
        help="Number of timesteps to roll out. Rollout will also stop if "
        "`--episodes` limit is reached first. A value of 0 means no "
        "limitation on the number of timesteps run.",
        )
    parser.add_argument(
        "--episodes",
        default=2,
        help="Number of complete episodes to roll out. Rollout will also stop "
        "if `--steps` (timesteps) limit is reached first. A value of 0 means "
        "no limitation on the number of episodes run.",
        )
    parser.add_argument(
        "--save-info",
        action="store_true",
        help="Save the info field generated by the step() method, "
        "as well as the action, observations, rewards and done fields.",
        )
    parser.add_argument(
        "--track-progress",
        action="store_true",
        help="Write progress to a temporary file (updated "
        "after each episode). An output filename must be set using --out; "
        "the progress file will live in the same folder.",
    )
    return parser


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # If num_episodes is set, stop if limit reached.
    if num_episodes and episodes >= num_episodes:
        return False
    # If num_steps is set, stop if limit reached.
    elif num_steps and steps >= num_steps:
        return False
    # Otherwise, keep going.
    return True

def run(args):
    # Load configuration from exp_dir.
    exp_dir = osp.join(LOCAL_DIR, args.exp_dir)
    config_dir = osp.join(exp_dir, 'config.json')
    params_dir = osp.join(exp_dir, 'params')
    statistics_dir = osp.join(exp_dir, 'statistics.pkl')

    params = shelve.open(osp.join(params_dir, 'params'))

    virtual_disp = Display(visible=False, size=(1400,900))
    virtual_disp.start()

    if not osp.exists(config_dir) or not osp.exists(params_dir):
        raise ValueError(
            f"Could not find params or config.json in exp_dir: {exp_dir}!"
            )
    with open(config_dir, 'r') as f:
        config = json.load(f)
        config['num_workers'] = args.num_workers

    if osp.exists(statistics_dir):
        with open(statistics_dir, 'rb') as f:
            statistics = joblib.load(f)
    else:
        statistics = {}

    # Init GPU
    ptu.init_gpu(
        use_gpu=not config.get('no_gpu'),
        gpu_id=config.get('which_gpu'),
    )
    agent_class = eval(config.get('agent_class')['$class'])
    agent = agent_class(
        config=config
    )
    agent.set_weights(ptu.map_location(params['last'], ptu.device))
    agent.set_statistics(statistics)

    ray.init(
        ignore_reinit_error=True,
        local_mode=args.local_mode,
    )
    num_steps = int(args.steps)
    num_episodes = int(args.episodes)
    # Do the actual rollout.
    with RolloutSaver(
        exp_dir,
        num_steps=num_steps,
        num_episodes=num_episodes,
        track_progress=args.track_progress,
        save_info=args.save_info,
    ) as saver:
        eval_batch_list = evaluate(agent, num_episodes, num_steps, saver, render=args.render)

        for idx, batch in enumerate(eval_batch_list):
            video_frames = batch.pop('img_obs')
            video_name = f"rollout-{idx}"
            utils.save_video(saver.rollout_dir, video_name, video_frames)

def evaluate(
    agent,
    num_episodes: int=None,
    num_steps: int=None,
    saver: RolloutSaver=None,
    render: bool=False,
) -> List[Batch]:
    if saver is None:
        saver = RolloutSaver()

    assert hasattr(agent, 'workers') \
        and isinstance(agent.workers, WorkerSet), \
        f'Agent: {agent} must have workers to evaluate.'

    # no saver, just evaluate the agent performance.
    if agent.workers.remote_workers():
        eval_batch_list = synchronous_parallel_sample(
            remote_workers=agent.workers.remote_workers(),
            max_steps=num_steps,
            max_episodes=num_episodes,
            concat=False,
        )
        return eval_batch_list
    # with saver, save the rollout to drives
    else:
        local_worker = agent.workers.local_worker()
        env, policy = local_worker.env, local_worker.policy

        steps = 0
        episodes = 0
        eval_batch_list = []
        img_obs = []
        max_step = get_max_episode_steps(env)
        while keep_going(steps, num_steps, episodes, num_episodes):
            done, ep_rew, ep_len, step_list = False, 0.0, 0, []
            terminal = False
            obs = env.reset()
            if render:
                img_obs = env.render(**LOCAL_RENDER_CONFIG)
            while not done and keep_going(steps, num_steps, episodes, num_episodes):
                act = policy.get_action(obs)
                next_obs, rew, done, info = env.step(act)
                ep_rew += rew
                ep_len += 1
                if done and ep_len != max_step:
                    terminal = True
                step_return = Batch(
                    obs=obs, act=act, next_obs=next_obs, rew=rew,
                    done=done, info=info, img_obs=img_obs,
                    ep_len=ep_len, ep_rew=ep_rew, terminal=terminal
                )
                step_list.append(step_return)
                if render:
                    img_obs = env.render(**LOCAL_RENDER_CONFIG)
                obs = next_obs
                steps += 1
            if done:
                episodes += 1

            batch = Batch.stack(step_list)
            eval_batch_list.append(batch)
    for idx, batch in enumerate(eval_batch_list):
        print(f"Rollout {idx + 1} Rew: ", batch.rew.sum())
        saver.store(batch)
    return eval_batch_list
