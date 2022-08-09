import shelve
import argparse
import json
import numpy as np
import os
import os.path as osp
import torch

from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict

from reILs.infrastructure.datas import Batch, convert_batch_to_dict
from reILs.infrastructure.utils import utils

class RolloutSaver:
    """
    Utility class for storing rollouts.
    Each rollout is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:

    with shelve.open('rollouts.pkl') as rollouts:
       for episode_id in range(rollouts["episodes"]):
          rollout = rollouts[str(episode_id)]

    If outfile is None, this class does nothing.

    """

    def __init__(
        self,
        exp_dir=None,
        num_steps=None,
        num_episodes=None,
        track_progress=False,
        save_info=False,
    ):
        self._exp_dir = exp_dir
        self._track_progress = track_progress

        self._outfile = None # exp_dir/rollouts/rollouts.pkl
        self._progressfile = None # exp_dir/rollouts/__progress_rollouts
        self._shelf = None

        self._episodes = 0
        self._steps = 0
        self._num_episodes = num_episodes
        self._num_steps = num_steps
        self._save_info = save_info

        if self._exp_dir:
            self._rollout_dir = osp.join(self._exp_dir, 'rollouts')
            os.makedirs(self._rollout_dir, exist_ok=True)
            self._video_dir = osp.join(self._rollout_dir, 'videos')
            os.makedirs(self._video_dir, exist_ok=True)
            self._outfile = osp.join(self._rollout_dir, 'rollouts.pkl') 

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return output.parent / ("__progress_" + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    @property
    def is_invalid(self):
        return self._exp_dir is None

    def __enter__(self):
        if self._outfile:
            self._shelf = shelve.open(self._outfile)
        if self._track_progress:
            self._progressfile = self._get_tmp_progress_filename().open(mode='w')
        return self
    
    def __exit__(self):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf['episodes'] = self._episodes
            self._shelf.close()
        if self._track_progress:
            # Remove the temp progress file:
            self._get_tmp_progress_filename().unlink()
            self._progressfile = None

    def _get_progress(self):
        if self._num_episodes:
            return f"{self._episodes} / {self._num_episodes} episodes completed"
        elif self._num_steps:
            return f"{self._steps} / {self._num_steps} steps completed"
        else:
            return f"{self._episodes} episodes completed"

    def store(self, batch: Batch):
        if self._outfile:
            video_frames = batch.pop('img_obs', None)
            self._shelf[str(self._episodes)] = convert_batch_to_dict(batch)
            if img_obs:
                video_name = f"rollout-{self._episodes}"
                video_file = osp.join(self._video_dir, video_name) 
                utils.write_gif(video_name, video_frames)
        self._episodes += 1
        if self._progressfile:
            self._progressfile.seek(0) # Move the pointer on top of the file
            self._progressfile.wtire(self._get_progress() + "\n")
            self._progressfile.flush()
