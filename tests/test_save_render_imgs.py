import os
import gym
import os.path as osp
import numpy as np
from pyvirtualdisplay import Display
from PIL import Image as im

EP_LEN = 10

def rollout(env):
    env.reset()
    img_obss = [env.render(mode='rgb_array')]
    for _ in range(EP_LEN):
        act = np.zeros(env.action_space.shape)
        next_obs, rew, done, _ = env.step(act)
        img_obs = env.render(mode='rgb_array')
        img_obss.append(img_obs)
        if done: break
    return img_obss

def save_img_obss(img_obss):
    imgdir_path = osp.join(os.getcwd(), 'imgs')
    print(imgdir_path)
    os.makedirs(imgdir_path, exist_ok=True)
    for i, img_obs in enumerate(img_obss):
        img_name = f'frame_{i}.png'
        img_path = osp.join(imgdir_path, img_name)
        img_data = im.fromarray(img_obs)
        img_data.save(img_path)
        print(f"Saving Frame {i} ... ...")

if __name__ == '__main__':
    virtual_disp = Display(visible=False, size=(1400,900))
    virtual_disp.start()
    env_name = 'HalfCheetah-v3'
    env = gym.make(env_name)
    img_obss = rollout(env)
    save_img_obss(img_obss)
    
