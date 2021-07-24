import logging
import os
from typing import Any, Dict, Tuple, Union

import gym
from lfp.datasets.utils.episode_utils import process_actions, process_depth, process_rgb, process_state
import numpy as np
from omegaconf import DictConfig
import torch

from vr_env.envs.play_table_env import get_env
from vr_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id

logger = logging.getLogger(__name__)


class PlayLMPWrapper(gym.Wrapper):
    def __init__(self, dataset_loader, device, show_gui=False):
        self.set_egl_device(device)
        env = get_env(dataset_loader.abs_datasets_dir, show_gui=show_gui)
        super(PlayLMPWrapper, self).__init__(env)
        self.observation_space_keys = dataset_loader.observation_space
        self.transforms = dataset_loader.transforms
        self.proprio_state = dataset_loader.proprio_state
        self.device = device
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        logger.info(f"Initialized PlayTableEnv for device {self.device}")

    @staticmethod
    def set_egl_device(device):
        assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
        cuda_id = device.index if device.type == "cuda" else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to VREnv README"
            )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def transform_observation(self, obs: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        rgb_obs = {f"rgb_{name}": img for name, img in zip([cam.name for cam in self.env.cameras], obs["rgb_obs"])}
        depth_obs = {
            f"depth_{name}": img for name, img in zip([cam.name for cam in self.env.cameras], obs["depth_obs"])
        }

        state_obs = process_state(obs, self.observation_space_keys, self.transforms, self.proprio_state)
        rgb_obs = process_rgb(rgb_obs, self.observation_space_keys, self.transforms)
        depth_obs = process_depth(depth_obs, self.observation_space_keys, self.transforms)

        state_obs = state_obs.to(self.device)
        rgb_obs = tuple(rgb_ob.to(self.device) for rgb_ob in rgb_obs)  # type: ignore
        depth_obs = tuple(depth_ob.to(self.device) for depth_ob in depth_obs)  # type: ignore

        return {"rgb_obs": rgb_obs, "state_obs": state_obs, "depth_obs": depth_obs}

    def step(
        self, action_tensor: torch.Tensor
    ) -> Tuple[Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]], int, bool, Dict]:
        if self.relative_actions:
            action = action_tensor.squeeze().cpu().detach().numpy()
            assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error("actions are required to have length 8 (for euler angles) or 9 (for quaternions)")
                raise NotImplementedError
            action = np.split(action_tensor.squeeze().cpu().detach().numpy(), slice_ids)
        action[-1] = 1 if action[-1] > 0 else -1
        o, r, d, i = self.env.step(action)

        obs = self.transform_observation(o)
        return obs, r, d, i

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        scene_obs: Any = None,
        robot_obs: Any = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        if reset_info is not None:
            obs = self.env.reset(
                robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
                scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            )
        elif scene_obs is not None or robot_obs is not None:
            obs = self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        else:
            obs = self.env.reset()

        return self.transform_observation(obs)

    def get_info(self):
        return self.env.get_info()
