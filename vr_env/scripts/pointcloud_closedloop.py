from pathlib import Path
import torch
import yaml
import hydra
import numpy as np
from pointnet.pointnet2.Pointnet import PointNetEncoderDecoder
from vr_env.scripts.point_cloud_processing import PointCloud
import open3d as o3d

"""
This script loads a rendered episode and replays it using the recorded actions.
Optionally, gaussian noise can be added to the actions.
"""


@hydra.main(config_path="../../conf", config_name="tabletoptrial")
def run_env(cfg):

    with open("../../../pointnet/pointnet2/config/Hyperparam.yaml", "r") as yamlfile:
        HP_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    hp = hydra_params_to_dotdict(HP_config)

    # To visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud')
    viewPCD = o3d.geometry.PointCloud()

    load_pos_orn_from_file = cfg['load_pos_orn_from_file']
    load_model = cfg['load_model']
    data = np.load(load_pos_orn_from_file)
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)
    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
    # env.reset()

    Net = PointNetEncoderDecoder(hp).cuda()
    Net.load_state_dict(torch.load(load_model))
    Net.eval()
    camera = env.cameras[0]
    PC = PointCloud()
    accumulatedPC = np.array([])
    rgb, depth = None, None
    for i in range(500):
        rgb, depth = camera.render()
        env.render()
        min_depth = np.min(depth[np.nonzero(depth)])
        print(min_depth)

        if min_depth <= 0.11:
            for i in range(20):
                action_grasp = np.array([0, 0, 0, 0, 0, 0, -1], dtype=float)
                env.step(action_grasp)
            for i in range(50):
                action_up = np.array([0, 0, 1, 0, 0, 0, 0], dtype=float)
                env.step(action_up)
            data = np.load(load_pos_orn_from_file)
            scene_obs = data["scene_obs"]
            robot_obs = data["robot_obs"]
            env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
            # env.reset()
            PC.accumulatedPC = np.array([])
            rgb, depth = camera.render()

        view_matrix = np.array(camera.view_matrix).reshape((4, 4)).T
        T_Robot_cam = np.linalg.inv(view_matrix)
        T_Robot_cam[:3, 1:3] *= -1
        PC.T_Robot_cam = T_Robot_cam

        organized_pc = camera.distance_map_to_point_cloud(depth, camera.fov, camera.width, camera.height)
        # Filter Point Cloud (Xi) and get only object point set from point cloud
        filtered_pc = organized_pc[np.where(depth != 0)]
        accumulatedPC = PC.transform_pointcloud_matrix(filtered_pc)
        # if accumulatedPC.size == 0:
        # if accumulatedPC.shape != (0, 3):
        while accumulatedPC.shape[0] < hp["num_points"]:
            accumulatedPC = np.concatenate((accumulatedPC, accumulatedPC), axis=0)
        index = np.random.choice(range(accumulatedPC.shape[0]), size=hp["num_points"], replace=False)
        SampledPC = accumulatedPC[index, :]

        # To visualize the point cloud
        viewPCD.points = o3d.utility.Vector3dVector(SampledPC)
        vis.add_geometry(viewPCD)
        vis.update_geometry(viewPCD)
        vis.poll_events()
        vis.update_renderer()

        # PC.viewPointCloud(SampledPC)

        SampledPC = torch.Tensor(SampledPC).unsqueeze(0)

        action = Net(SampledPC.cuda()).detach().cpu().numpy()
        action = np.squeeze(action)
        action = np.append(action, [1], axis=0)
        o, _, _, info = env.step(action)

    vis.destroy_window()


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v
        return res

    return _to_dot_dict(hparams)


if __name__ == "__main__":
    run_env()