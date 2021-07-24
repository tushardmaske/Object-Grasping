import hashlib
import itertools
import os
import pickle
import time

import pybullet as p

from vr_env.utils.utils import print_fail

VERSION_NUM = 3


def load(filename, cid):
    """
    Loads world state custom pickled data format and restores position, orientations and joint values.

    Args:
        filename: file to load from.
        cid: Bullet physics client ID

    Returns:
        observation
    """

    with open(filename, "rb") as file:
        data = pickle.load(file)

    body_data = data["body_data"]
    for entry in body_data:
        bodyId = entry["id"]
        body = entry["body"]
        joints = entry["joints"]
        p.resetBasePositionAndOrientation(bodyUniqueId=bodyId, posObj=body[0], ornObj=body[1], physicsClientId=cid)
        numJoints = len(joints)
        if numJoints > 0:
            for i in range(numJoints):
                jointData = joints[i]
                p.resetJointState(
                    bodyUniqueId=bodyId,
                    jointIndex=i,
                    targetValue=jointData[0],
                    targetVelocity=jointData[1],
                    physicsClientId=cid,
                )
    body_infos = {
        "obj_names": [str(e["info"][1], "utf-8") for e in body_data],
        "obj_pos_orn": [
            [e["id"]] + list(e["body"][0]) + list(p.getEulerFromQuaternion(e["body"][1]))
            for i, e in enumerate(body_data)
        ],
    }

    return data["state_obs"], data["vr_events"], data["time"], data["done"], body_infos, data["info"]


def save(filename, vrEvents):
    """
    Saves the state in a custom pickled data format.

    Args:
        filename: file to save to.
        vrEvents: vrEvents to attach to data.

    Returns:
        None
    """

    # serialize data for each frame in one file (bodyInfo, bodyPosOrn, bodyJointStates)
    data = freeze_data()
    data["vr_events"] = vrEvents
    serialize_data(data, filename)


def save_header(filename, global_scaling):
    """
    Saves the header file for preload

    Args:
        filename: path to file

    Returns:
        None
    """
    num_bodies = p.getNumBodies()
    data = {
        "body_data": [None for _ in range(num_bodies)],
        "version": VERSION_NUM,
        "num_bodies": num_bodies,
        "date": time.time_ns(),
        "global_scaling": global_scaling,
    }
    for body in range(num_bodies):
        bodyData = {
            "id": body,
            "info": p.getBodyInfo(body),
        }

        for i in range(p.getNumUserData(body)):
            userDataId, key, bodyId, linkIndex, visualShapeIndex = p.getUserDataInfo(body, i)
            val = p.getUserData(userDataId)
            bodyData[key] = val

        data["body_data"][body] = bodyData

    with open(filename, "wb") as file:
        pickle.dump(data, file)


def freeze_data():
    """
    Extract data from pybullet

    Returns:
        data from pybullet
    """

    data = {"vr_events": [], "body_data": [], "version": VERSION_NUM, "time": time.time_ns() / (10 ** 9)}
    for body in range(p.getNumBodies()):
        bodyData = {"id": body, "info": p.getBodyInfo(body), "body": p.getBasePositionAndOrientation(body)}
        jointData = ()
        numJoints = p.getNumJoints(body)
        if numJoints > 0:
            jointData = p.getJointStates(body, list(range(numJoints)))

        bodyData["joints"] = jointData

        data["body_data"].append(bodyData)
    return data


def serialize_data(filename, data):
    """
    Serialize data and write to file

    Args:
        filename: str of file name
        data: data to write

    Returns:
        None
    """

    with open(filename, "wb") as file:
        pickle.dump(data, file)


def count_frames(directory):
    """
    counts the number of consecutive pickled frames in directory

    Args:
        directory: str of directory

    Returns:
         0 for none, otherwise >0
    """

    for i in itertools.count(start=0):
        pickle_file = os.path.join(directory, f"{str(i).zfill(12)}.pickle")
        if not os.path.isfile(pickle_file):
            return i


def get_episode_lengths(load_dir, num_frames):
    episode_lengths = []
    render_start_end_ids = [[0]]
    i = 0
    for frame in range(num_frames):
        file_path = os.path.abspath(os.path.join(load_dir, f"{str(frame).zfill(12)}.pickle"))
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            done = data["done"]
            if not done:
                i += 1
            else:
                episode_lengths.append(i)
                render_start_end_ids[-1].append(frame + 1)
                render_start_end_ids.append([frame + 1])
                i = 0
    render_start_end_ids = render_start_end_ids[:-1]
    return episode_lengths, render_start_end_ids
