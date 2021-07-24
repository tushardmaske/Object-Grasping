from copy import deepcopy
import logging
from math import pi
import os
from pathlib import Path
import pickle
import pkgutil
import sys
import time
import random

import cv2
import gym
import gym.utils
import gym.utils.seeding
import hydra
import numpy as np
from omegaconf import OmegaConf
import pybullet as p
import pybullet_utils.bullet_client as bc

import vr_env
from vr_env.utils.utils import FpsController, get_git_commit_hash, timeit

# A logger for this file
log = logging.getLogger(__name__)

REPO_BASE = Path(__file__).parents[2]


class PlayTableSimEnv(gym.Env):
    def __init__(
            self,
            distribution,
            num_objects,
            robot_cfg,
            global_scaling,
            seed,
            use_vr,
            data_path,
            bullet_time_step,
            cameras,
            show_gui,
            scene,
            euler_obs,
            use_scene_info,
            use_egl,
            control_freq=30,

    ):
        self.p = p
        # for calculation of FPS
        self.t = time.time()
        self.prev_time = time.time()
        self.fps_controller = FpsController(bullet_time_step)
        self.use_vr = use_vr
        self.show_gui = show_gui
        self.euler_obs = euler_obs
        self.use_scene_info = use_scene_info
        self.cid = -1
        self.ownsPhysicsClient = False
        self.use_egl = use_egl
        self.control_freq = control_freq
        self.action_repeat = int(bullet_time_step // control_freq)
        render_width = max([cam.width for cam in cameras]) if cameras else None
        render_height = max([cam.height for cam in cameras]) if cameras else None
        self.initialize_bullet(bullet_time_step, render_width, render_height)
        self.robot = hydra.utils.instantiate(robot_cfg, cid=self.cid)
        self.global_scaling = global_scaling
        self.selected_object = {}
        if os.path.isabs(data_path):
            self.data_path = Path(data_path)
        else:
            self.data_path = REPO_BASE / data_path
        self.p.setAdditionalSearchPath(self.data_path.as_posix())
        self.num_objects = num_objects
        self.distribution = distribution

        self.allobjects = OmegaConf.to_container(scene.objects)
        self.allobjects_list = []
        for name in self.allobjects:
            if name == 'table':
                continue
            else:
                self.allobjects_list.append(name)

        self.objects = {}
        self.sample_random_objects()
        self.objects['table'] = self.allobjects['table']

        self.scene_joints = {}
        self.object_num = len(self.objects.keys())
        # Load Env
        self.load_scene()
        self.reset_scene()
        # init cameras after scene is loaded to have robot id available
        self.cameras = [
            hydra.utils.instantiate(camera, cid=self.cid, robot_id=self.robot.robot_uid, objects=self.objects)
            for camera in cameras
        ]
        self.np_random = None
        self.seed(seed)
        log.info(f"Using VREnv with commit {get_git_commit_hash(Path(vr_env.__file__))}.")

    # From pybullet gym_manipulator_envs code
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_manipulator_envs.py
    def initialize_bullet(self, bullet_time_step, render_width, render_height):
        if self.cid < 0:
            self.ownsPhysicsClient = True
            if self.use_vr:
                self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
                    sys.exit(1)
                self.p.setRealTimeSimulation(enableRealTimeSimulation=1, physicsClientId=cid)
            elif self.show_gui:
                self.p = bc.BulletClient(connection_mode=p.GUI)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to connect to GUI.")
            elif self.use_egl:
                options = f"--width={render_width} --height={render_height}"
                self.p = p
                cid = self.p.connect(p.DIRECT, options=options)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=cid)
                egl = pkgutil.get_loader("eglRenderer")
                log.info("Loading EGL plugin (may segfault on misconfigured systems)...")
                if egl:
                    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                else:
                    plugin = p.loadPlugin("eglRendererPlugin")
                if plugin < 0:
                    log.error("\nPlugin Failed to load!\n")
                    sys.exit()
                log.info("Successfully loaded egl plugin")
            else:
                self.p = bc.BulletClient(connection_mode=p.DIRECT)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to start DIRECT bullet mode.")
            log.info(f"Connected to server with id: {cid}")

            self.cid = cid
            self.p.resetSimulation(physicsClientId=self.cid)
            self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            log.info(f"Connected to server with id: {self.cid}")
            self.p.setTimeStep(1.0 / bullet_time_step, physicsClientId=self.cid)
            return cid

    def close(self):
        if self.ownsPhysicsClient:
            print("disconnecting id %d from server" % self.cid)
            if self.cid >= 0:
                self.p.disconnect(physicsClientId=self.cid)
            self.cid = -1
        else:
            print("does not own physics client id")

    def render(self, mode="human"):
        """render is gym compatibility function"""
        rgb_obs, depth_obs = self.get_camera_obs()
        point_cloud = self.cameras[0].distance_map_to_point_cloud(depth_obs, 75, 200, 200)
        if mode == "human":
            for i in range(len(self.cameras)):
                img = rgb_obs[i][:, :, ::-1]
                point_cloud = self.cameras[i].distance_map_to_point_cloud(depth_obs[i], 75, 200, 200)
                # cv2.imshow("Depth map"+str(i), depth_obs[i])
                cv2.imshow("simulation cam" + str(i), img)
            #  cv2.imshow("Point cloud"+str(i), point_cloud)
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return rgb_obs[0]
        else:
            raise NotImplementedError

    def reset_scene(self, scene_obs=None):
        """Reset objects and doors to initial position."""
        if scene_obs is None:
            for i in range(self.num_objects):
                p.removeBody(i)

            self.sample_random_objects()
            self.sample_random_obj_pose()
            # print(self.objects.items())

            ##load new objects
            # for name, obj in self.objects.items()[1:self.num_objects+1]:
            for name, obj in self.objects.items():
                if name == 'table':
                    continue
                else:
                    orn = self.p.getQuaternionFromEuler(obj["initial_orn"])
                    obj["file"] = self.data_path / obj["file"]
                    uid = self.p.loadURDF(
                        obj["file"].as_posix(),
                        obj["initial_pos"],
                        orn,
                        globalScaling=self.global_scaling,
                        physicsClientId=self.cid,
                    )
                    obj["uid"] = uid
                    obj["global_scaling"] = self.global_scaling

            for _, obj in self.objects.items():
                if not obj["fixed"]:
                    self.p.resetBasePositionAndOrientation(
                        obj["uid"],
                        obj["initial_pos"],
                        self.p.getQuaternionFromEuler(obj["initial_orn"]),
                        physicsClientId=self.cid,
                    )
            for cur_joint_infos in self.scene_joints.values():
                self.p.resetJointState(
                    cur_joint_infos["object_id"],
                    cur_joint_infos["joint_index"],
                    cur_joint_infos["initial_state"],
                    physicsClientId=self.cid,
                )
        else:
            # an object pose is composed of position (3) and orientation (4 for quaternion)  / (3 for euler)
            n_sj = len(self.scene_joints)
            assert len(scene_obs.shape) == 1 and (
                    scene_obs.shape[0] == 7 * self.object_num + n_sj or scene_obs.shape[0] == 6 * self.object_num + n_sj
            )
            assert len(scene_obs.shape) == 1
            for i, cur_joint_infos in enumerate(self.scene_joints.values()):
                self.p.resetJointState(
                    cur_joint_infos["object_id"], cur_joint_infos["joint_index"], scene_obs[i], physicsClientId=self.cid
                )
            # check if quaternion or euler orientation
            step = 6 if scene_obs.shape[0] == 6 * self.object_num + n_sj else 7
            for obj in self.objects.values():
                if obj["fixed"]:
                    continue
                i = obj["uid"]
                pos = scene_obs[n_sj + i * step: n_sj + (i * step) + 3]
                orn = scene_obs[n_sj + (i * step) + 3: n_sj + (i * step) + step]
                if step == 6:
                    orn = p.getQuaternionFromEuler(orn)
                self.p.resetBasePositionAndOrientation(i, pos, orn, physicsClientId=self.cid)

    def get_scene_info(self):
        """
        get dictionary of information about the objects in the scene
        self.objects:
            obj1:
                file: path to urdf
                initial_pos: [x, y, z]
                initial_orn: [x, y, z]  # euler angles
                fixed: true if object has a fixed base
                joints:
                    joint1:
                        joint_index: int
                        initial_state: float  # revolute
                        current_state: float
                    ...
                current_pos: [x, y, z]
                current_orn: [x, y, z, w]  # quaternion
                contacts: output of pybullet getContactPoints(...)
                links:  # key exists only if object has num_joints > 0
                    link1: link_id  #  name: id
            ...
        """
        for obj in self.objects.values():
            pos, orn = self.p.getBasePositionAndOrientation(obj["uid"], physicsClientId=self.cid)
            obj["current_pos"] = pos
            obj["current_orn"] = orn
            obj["contacts"] = self.p.getContactPoints(bodyA=obj["uid"], physicsClientId=self.cid)
            if "joints" in obj:
                for joint in obj["joints"].values():
                    joint_state = self.p.getJointState(obj["uid"], joint["joint_index"], physicsClientId=self.cid)
                    joint["current_state"] = joint_state
            if self.p.getNumJoints(obj["uid"] > 0):
                # save link names and ids in dictionary
                if "links" not in obj:
                    obj["links"] = {
                        self.p.getJointInfo(obj["uid"], i, physicsClientId=self.cid)[12].decode("utf-8"): i
                        for i in range(self.p.getNumJoints(obj["uid"], physicsClientId=self.cid))
                    }
                    obj["links"]["base_link"] = -1
        return deepcopy(self.objects)

    def get_scene_obs(self):
        """Return state information of the doors, drawers and shelves."""
        door_state = []
        object_poses = []
        for _, obj in self.objects.items():
            if "joints" in obj:
                for joint in obj["joints"].values():
                    joint_state = self.p.getJointState(obj["uid"], joint["joint_index"], physicsClientId=self.cid)
                    door_state.append(float(joint_state[0]))
            pos, orn = self.p.getBasePositionAndOrientation(obj["uid"], physicsClientId=self.cid)
            if self.euler_obs:
                orn = p.getEulerFromQuaternion(orn)
            object_poses = np.append(object_poses, np.concatenate([pos, orn]))
        return np.concatenate([door_state, object_poses])

    def get_scene_obs_labels(self):
        scene_joint_labels = [
            f"scene_o{cur_joint_infos.object_id}_j{cur_joint_infos.joint_index}_{cur_joint_name}"
            for cur_joint_name, cur_joint_infos in self.scene_joints.items()
        ]

        object_labels = []
        axes = ["pos_x", "pos_y", "pos_z", "orn_x", "orn_y", "orn_z", "orn_w"]
        if self.euler_obs:
            axes = axes[:-1]
        for _, obj in self.objects.items():
            for s in axes:
                object_labels.append(f"scene_o{obj['uid']}_{s}")

        labels = scene_joint_labels + object_labels
        n_scene_obs = len(self.get_scene_obs())
        if len(labels) != n_scene_obs:
            log.warning(
                f"number of scene joints returned from get_scene_obs ({n_scene_obs} "
                f"does not match config {len(labels)}"
            )
            return [f"scene_info_{i}" for i in range(n_scene_obs)]

        return labels

    def reset(self, robot_obs=None, scene_obs=None):
        self.reset_scene(scene_obs)
        self.robot.reset(robot_obs)
        self.p.stepSimulation(physicsClientId=self.cid)
        return self.get_obs()

    def sample_random_objects(self):

        for j in range(self.num_objects):
            i = random.randint(0, len(self.allobjects_list) - 1)
            chosen = self.allobjects_list[i]
            self.objects[f'ob{str(j)}'] = self.allobjects[chosen]

    def sample_random_obj_pose(self, name='all'):

        if self.distribution == 'uniform':
            if name == 'all':
                for obj_name, obj in self.objects.items():
                    if obj_name == 'table':
                        continue
                    else:
                        self.objects[str(obj_name)]['initial_pos'][0] = random.uniform(-0.1, 0.75)
                        self.objects[str(obj_name)]['initial_pos'][1] = random.uniform(0.5, 1)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def load_scene(self):
        log.info("Resetting simulation")
        self.p.resetSimulation(physicsClientId=self.cid)
        log.info("Setting gravity")
        self.p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        # self.selected_object = {'ob': self.objects['box], 'table' : objects['table']}
        # self.sample_random_obj_pose()

        for name, obj in self.objects.items():
            orn = self.p.getQuaternionFromEuler(obj["initial_orn"])
            obj["file"] = self.data_path / obj["file"]
            uid = self.p.loadURDF(
                obj["file"].as_posix(),
                obj["initial_pos"],
                orn,
                globalScaling=self.global_scaling,
                physicsClientId=self.cid,
            )
            obj["uid"] = uid
            obj["global_scaling"] = self.global_scaling
            if "joints" in obj:
                for joint_name, joint in obj["joints"].items():
                    # get joint_index by name (to prevent index errors when additional joints are added)
                    joint_index = next(
                        i
                        for i in range(p.getNumJoints(obj["uid"]))
                        if p.getJointInfo(obj["uid"], i)[1].decode("utf-8") == joint_name
                    )
                    joint["joint_index"] = joint_index
                    self.scene_joints[joint_name] = {
                        "object_id": obj["uid"],
                        "joint_index": joint_index,
                        "initial_state": joint["initial_state"],
                    }
        self._init_scene_joints()

        self.p.loadURDF(os.path.join(self.data_path, "plane.urdf"), physicsClientId=self.cid)
        self.robot.load()

    def load_object(self):
        # p.removeBody(self.num_objects)
        for i in range(self.num_objects):
            p.removeBody(i)

        print(self.objects.items())

        ##load new objects
        # for name, obj in self.objects.items()[1:self.num_objects+1]:
        for name, obj in self.objects.items():
            if name == 'table':
                continue
            else:
                orn = self.p.getQuaternionFromEuler(obj["initial_orn"])
                obj["file"] = self.data_path / obj["file"]
                uid = self.p.loadURDF(
                    obj["file"].as_posix(),
                    obj["initial_pos"],
                    orn,
                    globalScaling=self.global_scaling,
                    physicsClientId=self.cid,
                )
                obj["uid"] = uid
                obj["global_scaling"] = self.global_scaling

    def _init_scene_joints(self):
        max_force = 4
        for joint_info in self.scene_joints.values():
            self.p.setJointMotorControl2(
                joint_info["object_id"],
                joint_info["joint_index"],
                controlMode=p.VELOCITY_CONTROL,
                force=max_force,
                physicsClientId=self.cid,
            )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def get_camera_obs(self):
        assert self.cameras is not None
        rgb_obs = []
        depth_obs = []
        for cam in self.cameras:
            rgb, depth = cam.render()
            rgb_obs.append(rgb)
            depth_obs.append(depth)
        return rgb_obs, depth_obs

    def get_obs(self):
        """Collect camera, robot and scene observations."""
        rgb_obs, depth_obs = self.get_camera_obs()
        obs = {"rgb_obs": rgb_obs, "depth_obs": depth_obs}
        obs.update(self.get_state_obs())
        return obs

    def get_state_obs(self):
        """
        Collect state observation dict
        --state_obs
            --robot_obs
                --robot_state_full
                    -- [tcp_pos, tcp_orn, gripper_opening_width]
                --gripper_opening_width
                --arm_joint_states
                --gripper_action}
            --scene_obs
        """
        robot_obs, robot_info = self.robot.get_observation()
        scene_obs = self.get_scene_obs()
        obs = {"robot_obs": robot_obs, "scene_obs": scene_obs}
        return obs

    def get_info(self):
        _, robot_info = self.robot.get_observation()
        info = {"robot_info": robot_info}
        if self.use_scene_info:
            info["scene_info"] = self.get_scene_info()
        return info

    def step(self, action):
        # in vr mode real time simulation is enabled, thus p.stepSimulation() does not have to be called manually
        if self.use_vr:
            log.debug(f"SIM FPS: {(1 / (time.time() - self.t)):.0f}")
            self.t = time.time()
            current_time = time.time()
            delta_t = current_time - self.prev_time
            if delta_t >= (1.0 / self.control_freq):
                log.debug(f"Act FPS: {1 / delta_t:.0f}")
                self.prev_time = time.time()
                self.robot.apply_action(action)
            self.fps_controller.step()
        # for RL call step simulation repeat
        else:
            self.robot.apply_action(action)
            for i in range(self.action_repeat):
                self.p.stepSimulation(physicsClientId=self.cid)
        obs = self.get_obs()
        info = self.get_info()
        # obs, reward, done, info
        return obs, 0, False, info


class TableSimEnv(PlayTableSimEnv):
    def __init__(self, robot_cfg, global_scaling, seed, use_vr, data_path, bullet_time_step, cameras, show_gui, scene):
        super().__init__(robot_cfg, global_scaling, seed, use_vr, data_path, bullet_time_step, cameras, show_gui, scene)

    def _init_scene_joints(self):
        pass


def get_env(dataset_path, show_gui=True):
    from pathlib import Path

    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(Path(dataset_path) / ".hydra" / "merged_config.yaml")
    env = hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True)
    return env


@hydra.main(config_path="../../conf", config_name="config_data_collection")
def test_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)
    env.reset()
    obj_dict = {}
    while True:
        p.stepSimulation()
        time.sleep(0.01)
    # dir = '/tmp/recordings/16-29-36'
    # for i in range(200):
    #     with open(os.path.join(dir, f"action_{i}.pickle"), 'rb') as file:
    #         action = pickle.load(file)
        action = np.array([0., 0, 0, 0, 0, 0, 1])
        o, _, _, info = env.step(action)
        if info['scene_info']['ob0']['file'] not in obj_dict:
            x = info['scene_info']['ob0']['file']
            y = np.array(info['scene_info']['ob0']['current_pos'])
            obj_dict[info['scene_info']['ob0']['file']] = np.array(info['scene_info']['ob0']['current_pos'])
        else:
            # print(obj_dict[info['scene_info']['ob0']['file']][2] - np.array(info['scene_info']['ob0']['current_pos'])[2])
            d = {file: abs(np.array(info['scene_info']['ob0']['current_pos'])[2] - h[2]) for file, h in obj_dict.items()}
            file = min(d.items(), key=lambda x: x[1])[0]
            print(file == info['scene_info']['ob0']['file'])
        env.reset()



if __name__ == "__main__":
    # from hydra.experimental import compose, initialize
    # initialize(config_path="../../conf", job_name="test_app")
    # cfg = compose(config_name="config_data_collection")
    test_env()
    print("ok")
