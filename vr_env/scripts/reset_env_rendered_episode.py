from copy import deepcopy
import glob
import os
from pathlib import Path
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from vr_env.scripts.point_cloud_processing import PointCloud
from vr_env.envs.tasks import Tasks

"""
This script loads a rendered episode and replays it using the recorded actions.
Optionally, gaussian noise can be added to the actions.
"""


def noise(action, pos_std=0.01, rot_std=1):
    """
    adds gaussian noise to position and orientation.
    units are m for pos and degree for rot
    """
    pos, orn, gripper = action
    rot_std = np.radians(rot_std)
    pos_noise = np.random.normal(0, pos_std, 3)
    rot_noise = p.getQuaternionFromEuler(np.random.normal(0, rot_std, 3))
    pos, orn = p.multiplyTransforms(pos, orn, pos_noise, rot_noise)
    return pos, orn, gripper


def angle_between(a, b):
    if b > a:
        if b - a > np.pi:
            return b - a - (2 * np.pi)
        else:
            return b - a
    else:
        if b - a < -np.pi:
            return b - a + (2 * np.pi)
        else:
            return b - a


def create_relative_action(action, obs):
    action_diff = action[:6] - obs

    rel_pos = np.clip(action_diff[:3], -0.02, 0.02) / 0.02
    # rel_pos = action_diff[:3]
    rel_orn = np.array([angle_between(a, b) for a, b in zip(obs[3:6], action[3:6])])
    rel_orn = np.clip(rel_orn, -0.05, 0.05) / 0.05

    gripper = action[-1]
    return np.concatenate([rel_pos, rel_orn, [gripper]])


@hydra.main(config_path="../../conf", config_name="tabletoptrial")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)

    # root_dir = Path("/home/tmaske/Tushar/uni/dll/Project/dl_lab_21_pcl/Single_episode_apple/rendered_frames/2021-07-15/16-02-38_test")
    root_dir = Path("/home/nayana/Desktop/DLLab/Project/VREnv/rendereddata/2021-07-19/11-20-36")
    # root_dir = Path("/home/tmaske/Tushar/uni/dll/Project/tabletop_ep0_half/ep0")
    ep_start_end_ids = np.sort(np.load(root_dir / "ep_start_end_ids.npy"), axis=0)
    rel_actions = []
    # tasks = hydra.utils.instantiate(cfg.tasks)
    # prev_info = None
    #body = ((0.35571227476014555, 0.6516900576194229, 0.6305517313735546),
           #(-0.011053665891940501, 0.010430665963868284, 0.017593170002749375, 0.9997297124959095))
    #p.resetBasePositionAndOrientation(bodyUniqueId=1, posObj=body[0], ornObj=body[1], physicsClientId=0)
    for s, e in ep_start_end_ids:
        for i in range(s, e + 1):
            file = root_dir / f"episode_{i:06d}.npz"
            data = np.load(file)
            # if (i - s) % 32 == 0:
            #     print(f"reset {i}")
            #     time.sleep(0.5)
            #     env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
            if i == 0:
                print(f"reset {i}")
                time.sleep(0.5)
                env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
            # action = np.split(data["actions"], [3, 6])
            action = data["rel_actions"]
            pc = data["point_cloud"]
            # PointCloud.viewPointCloud(pcd=pc)
            print(action)
            # action = noise(action)

            # rel_actions.append(create_relative_action(data["actions"], data["robot_obs"][:6]))
            # action = create_relative_action(data["actions"], data["robot_obs"][:6])
            # tcp_pos, tcp_orn = p.getLinkState(env.robot.robot_uid, env.robot.tcp_link_id, physicsClientId=env.cid)[:2]
            # tcp_orn = p.getEulerFromQuaternion(tcp_orn)
            # action2 = create_relative_action(data["actions"], np.concatenate([tcp_pos, tcp_orn]))
            o, _, _, info = env.step(action)

            # if i > s:
            #     print(tasks.get_task_info(prev_info, info))
            # else:
            #     prev_info = deepcopy(info)
            time.sleep(0.1)
    rel_actions = np.array(rel_actions)
    for j in range(rel_actions.shape[1]):
        plt.figure(j)
        plt.hist(rel_actions[:, j], bins=10)
        plt.plot()
        plt.show()


if __name__ == "__main__":
    run_env()
