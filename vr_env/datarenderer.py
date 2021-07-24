#!/usr/bin/python3
import time
from collections import deque
import logging
from multiprocessing import Process
import os
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import pybullet as p
import tacto
import cv2
import vr_env.io_utils.storage as storage
import vr_env.io_utils.vr_input as vr_input
from vr_env.utils import utils
from vr_env.utils.utils import print_ok
from vr_env.scripts.point_cloud_processing import PointCloud
episodeNumber = 0
# A logger for this file
log = logging.getLogger(__name__)

BODY_INFO_LABELS = (
        ["obj_id"] + [f"obj_pos_{ax}" for ax in ("x", "y", "z")] + [f"obj_orn_euler_{ax}" for ax in ("x", "y", "z")]
)


@hydra.main(config_path="../conf", config_name="config_rendering")
def main(cfg):
    global episodeNumber
    log.info("pyBullet Data Renderer")
    log.info("Determining maximum frame")

    recording_dir_main = (Path(hydra.utils.get_original_cwd()) / cfg.load_dir).absolute()
    files = os.listdir(recording_dir_main)
    episodeNumber = 0
    for f in files:
        print(f)
        recording_dir = Path(recording_dir_main) / f
        max_frames = storage.count_frames(recording_dir)
        log.info(f"Found continuous interval of {max_frames} frames without gaps")

        num_prev_rendered_episodes = num_previously_rendered_episodes()
        if num_prev_rendered_episodes == 0:
            playback_cfg = build_rendering_config(recording_dir, rendering_config=cfg)
        else:
            playback_cfg = load_rendering_config()

        log.info("Initialization done!")
        log.info(f"Starting {cfg.processes} processes")

        if playback_cfg.set_static_cam:
            playback_cfg = set_static_cams_from_gui(playback_cfg, recording_dir, max_frames)

        if cfg.processes != 1:
            if playback_cfg.show_gui:
                log.warning("Multiprocess rendering requires headless mode, setting cfg.show_gui = False")
                playback_cfg.show_gui = False
            # in order to distribute the rendering to multiple processes, predetermine the lengths of the
            # (rendered) episodes and to which (recording) file ids the episode start and end correspond
            # a rendered episode does not contain the done frame, thus length(render_episode) = length(recording_episode) -1
            episode_lengths, render_start_end_ids = storage.get_episode_lengths(cfg.load_dir, max_frames)
            episode_lengths = episode_lengths[:1]
            render_start_end_ids = render_start_end_ids[:1]

            if cfg.processes > len(episode_lengths):
                log.warning(f"Trying to use more processes ({cfg.processes}) than episodes ({len(episode_lengths)}).")
                log.warning(f"Reducing number of processes to {len(episode_lengths)}.")
                cfg.processes = len(episode_lengths)
            # distribute the episodes equally to processes
            split_indices = np.array_split(np.array(render_start_end_ids), cfg.processes, axis=0)
            # every process renders the interval [proc_start_ids, proc_end_ids)
            proc_start_ids = [split_indices[proc_num][0][0] for proc_num in range(cfg.processes)]
            proc_end_ids = [split_indices[proc_num][-1][1] for proc_num in range(cfg.processes)]
            # predetermine starting episode indices for multiple processes
            if cfg.max_episode_frames == 1:
                proc_ep_ids = np.cumsum(
                    [0] + list(map(np.sum, np.array_split(np.array(episode_lengths), cfg.processes, axis=0)))[:-1]
                )
            else:
                proc_ep_ids = np.cumsum(
                    [0]
                    + [
                          np.sum(
                              list(map(lambda x: x // cfg.max_episode_frames + bool(x % cfg.max_episode_frames), eps)))
                          for eps in np.array_split(np.array(episode_lengths), cfg.processes, axis=0)
                      ][:-1]
                )
            proc_ep_ids += num_prev_rendered_episodes
            processes = [
                Process(
                    target=worker_run,
                    args=(
                        recording_dir,
                        playback_cfg,
                        proc_num,
                        proc_start_ids[proc_num],
                        proc_end_ids[proc_num],
                        proc_ep_ids[proc_num],
                    ),
                    name=f"Worker {proc_num}",
                )
                for proc_num in range(cfg.processes)
            ]
            deque(map(lambda proc: proc.start(), processes))
            deque(map(lambda proc: proc.join(), processes))
            if playback_cfg.max_episode_frames == 1:
                save_ep_lens(episode_lengths, num_prev_rendered_episodes)
        else:

            load_dir = cfg.load_dir + "/" + f
            episode_lengths, render_start_end_ids = storage.get_episode_lengths(load_dir, max_frames)
            episode_lengths = episode_lengths[:1]
            render_start_end_ids = render_start_end_ids[0:1]
            # max_frames = render_start_end_ids[0][1]
            # start_frame = render_start_end_ids[0][0]
            start_frame = 0
            worker_run(recording_dir, playback_cfg, 0, start_frame, max_frames, num_prev_rendered_episodes)

    print_ok("All workers done")


def build_rendering_config(recording_dir, rendering_config):
    merged_conf = omegaconf.OmegaConf.load(Path(recording_dir) / ".hydra" / "config.yaml")

    merged_conf = omegaconf.OmegaConf.merge(merged_conf, rendering_config)
    # for conf_selection in ('env.cameras', 'env.use_vr', 'processes',
    #                        'static_camera', 'gripper_camera'):
    #     override_conf = omegaconf.OmegaConf.select(rendering_config, conf_selection)
    #     omegaconf.OmegaConf.update(merged_conf, conf_selection, override_conf, merge=False)
    hydra.core.utils._save_config(merged_conf, "merged_config.yaml", Path(os.getcwd(), ".hydra"))
    return merged_conf


def load_rendering_config():
    conf = omegaconf.OmegaConf.load(Path(os.getcwd()) / ".hydra" / "merged_config.yaml")
    conf.set_static_cam = False
    return conf


def num_previously_rendered_episodes():
    return len(list(Path(os.getcwd()).glob("*.npz")))


def save_ep_lens(episode_lengths, num_prev_episodes):
    if num_prev_episodes > 0:
        previous_ep_lens = np.load("ep_lens.npy")
        episode_lengths = np.concatenate((previous_ep_lens, episode_lengths))
    np.save("ep_lens.npy", episode_lengths)
    end_ids = np.cumsum(episode_lengths) - 1
    start_ids = [0] + list(end_ids + 1)[:-1]
    ep_start_end_ids = list(zip(start_ids, end_ids))
    np.save("ep_start_end_ids.npy", ep_start_end_ids)


def save_episode(
        counter,
        rgbs,
        depths,
        actions,
        robot_obs,
        scene_obs,
        body_infos,
        cam_names,
        actions_labels,
        state_obs_labels_robot,
        state_obs_labels_scene,
        **additional_infos,
):
    assert len(rgbs) == len(depths)
    assert len(robot_obs[0]) == len(state_obs_labels_robot)
    assert len(scene_obs[0]) == len(state_obs_labels_scene)
    assert len(actions[0]) == len(actions_labels)
    if cam_names is not None:
        # n_frames x n_cameras x ...
        assert len(rgbs[0]) == len(cam_names)
    obj_names = []
    obj_pos_orn = []
    if len(body_infos):
        obj_names = body_infos[0]["obj_names"]  # constant
        # n_frames x n_objects x ...
        obj_pos_orn = [e["obj_pos_orn"] for e in body_infos]
        assert len(obj_pos_orn[0][0]) == len(BODY_INFO_LABELS)
    rgb_entries = {f"rgb_{cam_name}": np.asarray([rgb[i] for rgb in rgbs]) for i, cam_name in enumerate(cam_names)}
    depths_entries = {
        f"depth_{cam_name}": np.asarray([depth[i] for depth in depths]) for i, cam_name in enumerate(cam_names)
    }
    np.savez_compressed(
        f"episode_{counter:04d}.npz",
        cam_names=cam_names,
        actions=actions,
        actions_labels=actions_labels,
        robot_obs=robot_obs,
        scene_obs=scene_obs,
        state_obs_labels_robot=state_obs_labels_robot,
        state_obs_labels_scene=state_obs_labels_scene,
        obj_names=obj_names,
        obj_pos_orn=obj_pos_orn,
        obj_pos_orn_labels=BODY_INFO_LABELS,
        **rgb_entries,
        **depths_entries,
        **additional_infos,
    )


def save_step(counter, rgbs, depths, actions, robot_obs, scene_obs, point_cloud, cam_names, **additional_infos):
    global episodeNumber
    rgb_entries = {f"rgb_{cam_name}": rgbs[i] for i, cam_name in enumerate(cam_names)}
    depths_entries = {f"depth_{cam_name}": depths[i] for i, cam_name in enumerate(cam_names)}
    if actions[-1] == 0:
        actions[-1] = -1
    # print(utils.to_relative_action(actions, robot_obs))

    # if point_cloud.size != 0:
    name1 = "EP_"+str(episodeNumber)
    name2 = f"_{counter:06d}.npz"
    # print("Point_cloud", point_cloud.size)
    np.savez_compressed(
        name1+name2,
        actions=actions,
        rel_actions=utils.to_relative_action(actions, robot_obs),
        robot_obs=robot_obs,
        scene_obs=scene_obs,
        point_cloud=point_cloud,
        **rgb_entries,
        **depths_entries,
        **additional_infos,
    )


def state_to_action(info):
    """
    save action as [tcp_pos, tcp_orn_quaternion, gripper_action]
    """
    tcp_pos = info["robot_info"]["tcp_pos"]
    tcp_orn = info["robot_info"]["tcp_orn"]
    gripper_action = info["robot_info"]["gripper_action"]
    action = np.concatenate([tcp_pos, tcp_orn, [gripper_action]])
    return action


def _extract_button_events(vr_events):
    events = []
    for e in vr_events:
        e = vr_input.VrEvent(*e)
        button_event = {
            f"{name}_{event_type}": getattr(state, event_type)
            for event_type in ("was_triggered", "was_released")
            for name, state in e.button_dicts.items()
        }
        if any(button_event.values()):
            events.append(button_event)
    return events


def set_static_cams_from_gui(cfg, load_dir, max_frames):
    import cv2

    assert cfg.env.show_gui
    env = hydra.utils.instantiate(cfg.env)
    env.reset()
    frame = 0
    log.info("--------------------------------------------------")
    log.info("Use Debug GUI to change the position of the camera")
    log.info("Use Render_view_window for keyboard input")
    log.info("Press A or D to move through frames")
    log.info("Press Q or E to skip through frames")
    log.info("Press S to set camera position")
    log.info("Press ENTER to save the set camera position")
    log.info("Press ESC to skip setting position for current camera")
    for cam_index, cam in enumerate(cfg.cameras):
        if "static" in cam._target_:
            # initialize variables
            look_from = cam.look_from
            look_at = cam.look_at
            while True:
                file_path = os.path.abspath(os.path.join(load_dir, f"{str(frame).zfill(12)}.pickle"))
                state_ob, vr_events, state_time, done, body_info, info = storage.load(file_path, env.cid)
                env.p.stepSimulation()
                frame_rgbs, frame_depths = env.get_camera_obs()
                rgb_static = frame_rgbs[cam_index]
                cv2.imshow("Render_view_window", cv2.resize(rgb_static, (500, 500))[:, :, ::-1])
                k = cv2.waitKey(10) % 256
                if k == ord("a"):
                    frame -= 1
                    frame = np.clip(frame, 0, max_frames - 1)
                if k == ord("d"):
                    frame += 1
                    frame = np.clip(frame, 0, max_frames - 1)
                if k == ord("q"):
                    frame -= 100
                    frame = np.clip(frame, 0, max_frames - 1)
                if k == ord("e"):
                    frame += 100
                    frame = np.clip(frame, 0, max_frames - 1)
                if k == 13:  # Enter
                    cam.look_from = look_from
                    cam.look_at = look_at
                    log.info(f"Set look_from of camera {cam_index} to {look_from}")
                    log.info(f"Set look_at of camera {cam_index} to {look_at}")
                    break
                if k == 27:  # ESC
                    log.info(f"Do no change position of camera {cam_index}")
                    break
                # if k == ord("s"):
                look_from, look_at = env.cameras[cam_index].set_position_from_gui()
    hydra.core.utils._save_config(cfg, "merged_config.yaml", Path(os.getcwd(), ".hydra"))
    return cfg


def worker_run(load_dir, rendering_cfg, proc_num, start_frame, stop_frame, episode_index):
    global episodeNumber
    if rendering_cfg.max_episode_frames == 1 and rendering_cfg.processes == 1:
        num_prev_rendered_episodes = episode_index
    log.info(f"[{proc_num}] Starting worker {proc_num}")
    env = hydra.utils.instantiate(rendering_cfg.env)
    cam_infos = [
        list(cam.get("look_from", [np.nan] * 3))
        + list(cam.get("look_at", [np.nan] * 3))
        + [cam.get("gripper_cam_link", -1)]
        for cam in rendering_cfg.env["cameras"]
    ]
    cam_info_labels = (
            [f"look_from_{ax}" for ax in ("x", "y", "z")] + [f"look_at_{ax}" for ax in ("x", "y", "z")] + [
        "robot_cam_link"]
    )
    # label definitions for arrays in export
    if rendering_cfg.robot.euler_obs:
        action_labels = (
                [f"tcp_pos_{ax}" for ax in ("x", "y", "z")]
                + [f"tcp_orn_{ax}" for ax in ("x", "y", "z")]
                + ["gripper_action"]
        )
    else:
        action_labels = (
                [f"tcp_pos_{ax}" for ax in ("x", "y", "z")] + [f"tcp_orn_quat_{i}" for i in range(4)] + [
            "gripper_action"]
        )
    log.info(f"[{proc_num}] Entering Loop")
    # episode_index = 0
    frame_counter = 0
    rgbs, depths, actions, robot_obs, scene_obs, body_infos, frame_times = [], [], [], [], [], [], []
    episode_lengths = []
    # collect button events and write them to CSV later (e.g., to use it to segment recording)
    button_events = []
    # obj_names, obj_heights = [], []
    # allobjects = rendering_cfg['scene']['objects']
    # for i in enumerate(allobjects):
    #     obj_heights.append(allobjects[i[1]]['initial_pos'][2])
    #     obj_names.append(allobjects[i[1]]['file'])
    obj_heights = [0.02, 0.61, 0.63, 0.65, 0.69, 0.62, 0.624]
    obj_names = ['table/hightable.urdf', '029_plate/google_16k/textured.urdf', '013_apple/google_16k/textured.urdf',
                 'OrgangeMarmelad/meshes/OrangeMarmelade_25k_tex.urdf', '003_cracker_box/google_16k/textured.urdf',
                 '012_strawberry/google_16k/textured.urdf', '025_mug/google_16k/textured.urdf']

    PC = PointCloud()
    acc = None
    pcs_acc, act_l, relAct_l = [], [], []
    reg_num = 3000
    load_ob = True
    save = True
    drop_frames = False
    for frame in range(start_frame, stop_frame):
        file_path = os.path.abspath(os.path.join(load_dir, f"{str(frame).zfill(12)}.pickle"))
        state_ob, vr_events, state_time, done, body_info, info = storage.load(file_path, env.cid)
        # print("frame: ", frame)
        if not drop_frames or done or load_ob:
            if load_ob:
                PC.accumulatedPC = np.array([])
                print("Height: ", body_info["obj_pos_orn"][0][3])
                d = [abs(x - body_info["obj_pos_orn"][0][3]) for x in obj_heights]
                file = obj_names[d.index(min(d))]
                env.objects["ob0"]["file"] = env.data_path.as_posix() + '/' + file
                env.objects["ob0"]["initial_pos"] = body_info["obj_pos_orn"][0][1:4]
                env.objects["ob0"]["initial_orn"] = body_info["obj_pos_orn"][0][4:7]
                env.load_object()
                episodeNumber += 1
                print("Episode: ", episodeNumber)
                drop_frames = False

            load_ob = done

            camera = env.cameras[0]
            rgb, depth = camera.render()
            # cv2.imshow("Depth map12", depth)
            rgb_obs, depth_obs = env.get_camera_obs()
            cv2.imshow("rgb map", rgb_obs[0][:, :, ::-1])
            cv2.waitKey(1)
            view_matrix = np.array(camera.view_matrix).reshape((4, 4)).T
            T_Robot_cam = np.linalg.inv(view_matrix)
            T_Robot_cam[:3, 1:3] *= -1
            PC.T_Robot_cam = T_Robot_cam

            organized_pc = camera.distance_map_to_point_cloud(depth, camera.fov, camera.width, camera.height)
            # Filter Point Cloud (Xi) and get only object point set from point cloud
            filtered_pc = organized_pc[np.where(depth != 0)]
            if frame_counter > 3:
                accTemp = PC.transform_pointcloud_matrix(filtered_pc)
                if accTemp.shape != (0, 3) and frame_counter > 3:
                    save = True
                    while accTemp.shape[0] < reg_num:
                        accTemp = np.concatenate((accTemp, accTemp), axis=0)
                    center_indexes = np.random.choice(
                        range(accTemp.shape[0]), size=reg_num, replace=False
                    )
                    acc = accTemp[center_indexes, :]
                else:
                    save = False
                    acc = np.array([])
            else:
                save = False
                acc = np.array([])

            if done:
                acc = np.array([])
            pcs_acc.append(acc)
            action = state_to_action(info)
            robot_obs.append(state_ob["robot_obs"])
            scene_obs.append(state_ob["scene_obs"])
            frame_times.append(state_time)
            if rendering_cfg.save_body_infos:
                body_infos.append(body_info)

            new_button_events = _extract_button_events(vr_events)
            for new_event in new_button_events:
                new_event["frame_no"] = frame
                new_event["time"] = state_time
                button_events.append(new_event)

            # action is robot state of next frame
            if frame_counter > 0:
                actions.append(action)
            # in order to get correct force measurements with digit sensor, we have to reapply the gripper action
            env.robot.control_gripper(info["robot_info"]["gripper_action"])
            env.p.stepSimulation()

            frame_rgbs, frame_depths = env.get_camera_obs()
            rgbs.append(frame_rgbs)
            depths.append(frame_depths)
            # for terminal states save current robot state as action
            frame_counter += 1
            log.debug(f"episode counter {episode_index} frame counter {frame_counter} done {done}")
            if (done or frame_counter >= rendering_cfg.max_episode_frames) and rendering_cfg.max_episode_frames > 1:
                actions.append(action)
                # to be consistent with single frame rendering, do not save done frame
                if done:
                    [x.pop() for x in [rgbs, depths, actions, robot_obs, scene_obs, body_infos, frame_times, pcs_acc]]
                    PC.accumulatedPC = np.array([])

                state_obs_labels_robot = env.robot.get_observation_labels()
                state_obs_labels_scene = env.get_scene_obs_labels()

                save_episode(
                    episode_index,
                    rgbs,
                    depths,
                    actions,
                    robot_obs,
                    scene_obs,
                    body_infos=body_infos,
                    actions_labels=action_labels,
                    state_obs_labels_robot=state_obs_labels_robot,
                    state_obs_labels_scene=state_obs_labels_scene,
                    cam_names=[cam.name for cam in env.cameras],
                    frame_times=frame_times,
                    cam_infos=cam_infos,
                    cam_infos_labels=cam_info_labels,
                )
                episode_index += 1
                frame_counter = 0
                rgbs, depths, actions, robot_obs, scene_obs, body_infos, frame_times, pcs_acc = [], [], [], [], [], [], [], []
                PC.accumulatedPC = np.array([])

            elif rendering_cfg.max_episode_frames == 1 and frame_counter > 1:
                # if frame_counter == 40:
                #     PC.viewPointCloud(pcs_acc[0])
                if actions[0][-1]==0:
                    drop_frames = True
                    print("Drop frame: EP_" + str(episodeNumber) + "_Frame_"+str(frame_counter))
                else:
                    drop_frames = False
                save_step(
                    episode_index,
                    rgbs.pop(0),
                    depths.pop(0),
                    actions.pop(0),
                    robot_obs.pop(0),
                    scene_obs.pop(0),
                    pcs_acc.pop(0),
                    cam_names=[cam.name for cam in env.cameras],
                )
                episode_index += 1
                if done:
                    episode_lengths.append(frame_counter - 1)
                    frame_counter = 0
                    PC.accumulatedPC = np.array([])
                    rgbs, rgbs_gripper, depths, depths_gripper, actions, robot_obs, scene_obs, body_infos, frame_times, pcs_acc = (
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        []
                    )

        else:
            frame_counter += 1

        log.debug(f"[{proc_num}] Rendered frame {frame}")

    assert done

    if rendering_cfg.max_episode_frames == 1 and rendering_cfg.processes == 1:
        save_ep_lens(episode_lengths, num_prev_rendered_episodes)
    pd.DataFrame(button_events).to_csv(f"button_events_{proc_num}.csv", index=False)

    p.disconnect(env.cid)
    log.info(f"[{proc_num}] Finishing worker {proc_num}")


if __name__ == "__main__":
    main()
