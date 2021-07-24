import random
from pathlib import Path
import time

import hydra
import numpy as np
import pybullet as p


"""
This script loads a rendered episode and replays it using the recorded actions.
Optionally, gaussian noise can be added to the actions.
"""

@hydra.main(config_path="../../conf", config_name="tabletoptrial")
def run_env(cfg):

    env = hydra.utils.instantiate(cfg.env,show_gui=True, use_vr=False, use_scene_info=True)

    #root_dir = '/home/niket/Desktop/DL_lab/Project/VREnv/actions.npz'
    #data = np.load(root_dir)
    #x = (data['actions'])
    root_dir = '/home/niket/Desktop/DL_lab/Project/Grasp_000000.npz'
    data = np.load(root_dir)
    y = data['rel_actions']

    # root_dir = '/home/niket/Desktop/DL_lab/Project/actions.npz'
    # data = np.load(root_dir)
    # x = (data['actions'])

    for i in range(168):
        env.render()
        # actions = np.array([0, 0, 1, 0, 1, 0,1],dtype=float)
        actions = y[i]
        o, _, _, info = env.step(actions)
        # if i%10==0:
        #     env.reset()
        # time.sleep(0.1)

    # rob=np.array([0.2,0.6,0.75,3.1,.32,-1.7,0,-1.46,-1.48,1.5,-2.43,-1.8,1.85,-1.23,1.0])
    # rob=np.array([0.3, 0.15, 0.6,0, 0, 1.5707963,-1.4653239989567401, 1.4817260599394233, 1.525733453119315, -2.435192704551518, -1.809600862016179, 1.8558817172819435, -1.23183583862092,0, 0.04])
    # env.reset(robot_obs=rob)
    #env.reset()

if __name__ == "__main__":
    run_env()
