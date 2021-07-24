from functools import partial
from typing import Dict, Set

import numpy as np
from omegaconf import ListConfig
from scipy.spatial.transform import Rotation as R

from vr_env.utils.utils import timeit


class Tasks:
    def __init__(self, tasks):
        """
        A task is defined as a specific change between the start_info and end_info dictionaries.
        Use config file in conf/tasks/ to define tasks using the base task functions defined in this class
        """
        # register task functions from config file
        self.tasks = {name: partial(getattr(self, args[0]), *args[1:]) for name, args in dict(tasks).items()}
        # dictionary mapping from task name to task id
        self.task_to_id = {name: i for i, name in enumerate(self.tasks.keys())}
        # dictionary mapping from task id to task name
        self.id_to_task = {i: name for i, name in enumerate(self.tasks.keys())}

    def get_task_info(self, start_info: Dict, end_info: Dict) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        returns set with achieved tasks
        """
        # call functions that are registered in self.tasks
        return {
            task_name
            for task_name, function in self.tasks.items()
            if function(start_info=start_info, end_info=end_info)
        }

    def get_task_info_for_set(self, start_info: Dict, end_info: Dict, task_filter: Set) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        task_filter: set with task names to check
        returns set with achieved tasks
        """
        # call functions that are registered in self.tasks
        return {
            task_name
            for task_name, function in self.tasks.items()
            if task_name in task_filter and function(start_info=start_info, end_info=end_info)
        }

    @property
    def num_tasks(self):
        return len(self.tasks)

    @staticmethod
    def rotate_object(obj_name, z_degrees, x_y_threshold=30, z_threshold=180, start_info=None, end_info=None):
        """
        Returns True if the object with obj_name was rotated more than z_degrees degrees around the z-axis while not
        being rotated more than x_y_threshold degrees around the x or y axis.
        z_degrees is negative for clockwise rotations and positive for counter-clockwise rotations.
        """
        start_orn = R.from_quat(start_info["scene_info"][obj_name]["current_orn"])
        end_orn = R.from_quat(end_info["scene_info"][obj_name]["current_orn"])
        rotation = end_orn * start_orn.inv()
        x, y, z = rotation.as_euler("xyz", degrees=True)
        if z_degrees > 0:
            return z_degrees < z < z_threshold and abs(x) < x_y_threshold and abs(y) < x_y_threshold
        else:
            return z_degrees > z > -z_threshold and abs(x) < x_y_threshold and abs(y) < x_y_threshold

    @staticmethod
    def push_object(obj_name, x_direction, y_direction, start_info, end_info):
        """
        Returns True if the object with 'obj_name' was moved more than 'x_direction' meters in x direction
        (or 'y_direction' meters in y direction analogously).
        Note that currently x and y pushes are mutually exclusive, meaning that one of the arguments has to be 0.
        The sign matters, e.g. pushing an object to the right when facing the table coincides with a movement in
        positive x-direction.
        """
        assert x_direction * y_direction == 0 and x_direction + y_direction != 0
        start_pos = np.array(start_info["scene_info"][obj_name]["current_pos"])
        end_pos = np.array(end_info["scene_info"][obj_name]["current_pos"])
        pos_diff = end_pos - start_pos

        start_contacts = set(c[2] for c in start_info["scene_info"][obj_name]["contacts"])
        end_contacts = set(c[2] for c in end_info["scene_info"][obj_name]["contacts"])
        robot_uid = {start_info["robot_info"]["uid"]}

        # computing set difference to check if object had surface contact (excluding robot) at both times
        surface_contact = len(start_contacts - robot_uid) and len(end_contacts - robot_uid)
        if not surface_contact:
            return False

        if x_direction > 0:
            return pos_diff[0] > x_direction
        elif x_direction < 0:
            return pos_diff[0] < x_direction

        if y_direction > 0:
            return pos_diff[1] > y_direction
        elif y_direction < 0:
            return pos_diff[1] < y_direction

    @staticmethod
    def lift_object(obj_name, z_direction, surface_body=None, surface_link=None, start_info=None, end_info=None):
        """
        Returns True if the object with 'obj_name' was grasped by the robot and lifted more than 'z_direction' meters.
        """
        assert z_direction > 0
        start_pos = np.array(start_info["scene_info"][obj_name]["current_pos"])
        end_pos = np.array(end_info["scene_info"][obj_name]["current_pos"])
        pos_diff = end_pos - start_pos
        z_diff = pos_diff[2]

        robot_uid = start_info["robot_info"]["uid"]
        start_contacts = set(c[2] for c in start_info["scene_info"][obj_name]["contacts"])
        end_contacts = set(c[2] for c in end_info["scene_info"][obj_name]["contacts"])

        surface_criterion = True
        if surface_body and surface_link is None:
            surface_uid = start_info["scene_info"][surface_body]["uid"]
            surface_criterion = surface_uid in start_contacts
        elif surface_body and surface_link:
            surface_uid = start_info["scene_info"][surface_body]["uid"]
            surface_link_id = start_info["scene_info"][surface_body]["links"][surface_link]
            start_contacts_links = set((c[2], c[4]) for c in start_info["scene_info"][obj_name]["contacts"])
            surface_criterion = (surface_uid, surface_link_id) in start_contacts_links

        return (
            z_diff > z_direction
            and robot_uid not in start_contacts
            and robot_uid in end_contacts
            and len(end_contacts) == 1
            and surface_criterion
        )

    @staticmethod
    def place_object(dest_body, dest_link=None, start_info=None, end_info=None):
        """
        Returns True if the object that the robot has currently lifted is placed on the body 'dest_body'
        (on 'dest_link' if provided).
        The robot may not touch the object after placing.
        """
        robot_uid = start_info["robot_info"]["uid"]

        robot_contacts_start = set(c[2] for c in start_info["robot_info"]["contacts"])
        robot_contacts_end = set(c[2] for c in end_info["robot_info"]["contacts"])
        if not len(robot_contacts_start) == 1:
            return False
        obj_uid = list(robot_contacts_start)[0]

        if obj_uid in robot_contacts_end:
            return False
        obj_name = [k for k, v in start_info["scene_info"].items() if v["uid"] == obj_uid][0]

        dest_uid = end_info["scene_info"][dest_body]["uid"]

        object_contacts_start = set(c[2] for c in start_info["scene_info"][obj_name]["contacts"])
        if dest_link is None:
            object_contacts_end = set(c[2] for c in end_info["scene_info"][obj_name]["contacts"])
            return (
                robot_uid in object_contacts_start
                and len(object_contacts_start) == 1
                and dest_uid in object_contacts_end
            )
        else:
            dest_link_id = end_info["scene_info"][dest_body]["links"][dest_link]
            end_contacts_links = set((c[2], c[4]) for c in end_info["scene_info"][obj_name]["contacts"])
            return (
                robot_uid in object_contacts_start
                and len(object_contacts_start) == 1
                and (dest_uid, dest_link_id) in end_contacts_links
            )

    @staticmethod
    def push_object_into(obj_name, src_body, dest_body, start_info=None, end_info=None):
        """
        obj_name is either a list of object names or a string
        Returns True if the object / any of the objects changes contact from src_body to dest_body.
        The robot may neither touch the object at start nor end.
        """
        if isinstance(obj_name, (list, ListConfig)):
            return any(Tasks.push_object_into(ob, src_body, dest_body, start_info, end_info) for ob in obj_name)
        robot_uid = start_info["robot_info"]["uid"]

        src_uid = start_info["scene_info"][src_body]["uid"]
        dest_uid = end_info["scene_info"][dest_body]["uid"]

        start_contacts = set(c[2] for c in start_info["scene_info"][obj_name]["contacts"])
        end_contacts = set(c[2] for c in end_info["scene_info"][obj_name]["contacts"])
        return (
            robot_uid not in start_contacts | end_contacts
            and len(start_contacts) == 1
            and src_uid in start_contacts
            and dest_uid in end_contacts
        )

    @staticmethod
    def move_door_abs(obj_name, joint_name, start_threshold, end_threshold, start_info, end_info):
        """
        Returns True if the joint specified by 'obj_name' and 'joint_name' (e.g. a door or drawer)
        is moved from at least 'start_threshold' to 'end_threshold'.
        """
        start_joint_state = start_info["scene_info"][obj_name]["joints"][joint_name]["current_state"][0]
        end_joint_state = end_info["scene_info"][obj_name]["joints"][joint_name]["current_state"][0]

        if start_threshold < end_threshold:
            return start_joint_state < start_threshold < end_threshold < end_joint_state
        elif start_threshold > end_threshold:
            return start_joint_state > start_threshold > end_threshold > end_joint_state
        else:
            raise ValueError

    @staticmethod
    def move_door_rel(obj_name, joint_name, threshold, start_info, end_info):
        """
        Returns True if the joint specified by 'obj_name' and 'joint_name' (e.g. a door or drawer)
        is moved from at least 'start_threshold' to 'end_threshold'.
        """
        start_joint_state = start_info["scene_info"][obj_name]["joints"][joint_name]["current_state"][0]
        end_joint_state = end_info["scene_info"][obj_name]["joints"][joint_name]["current_state"][0]

        return (
            0 < threshold < end_joint_state - start_joint_state or 0 > threshold > end_joint_state - start_joint_state
        )
