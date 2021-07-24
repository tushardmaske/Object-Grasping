import logging
import math
import os

import numpy as np
import pybullet as p
import quaternion

from vr_env.robot.mixed_ik import MixedIK
from vr_env.utils.utils import timeit

# A logger for this file
log = logging.getLogger(__name__)
GRIPPER_FORCE = 200


class Robot:
    def __init__(
        self,
        filename,
        base_position,
        base_orientation,
        initial_joint_positions,
        max_joint_force,
        limit_angle,
        constraint_force,
        arm_joint_ids,
        gripper_joint_ids,
        gripper_joint_limits,
        tcp_link_id,
        end_effector_link_id,
        gripper_orientation_offset,
        cid,
        use_nullspace,
        max_velocity,
        use_ik_fast,
        euler_obs,
        max_rel_pos=0.02,
        max_rel_orn=0.05,
        magic_scaling_factor_pos=1,
        magic_scaling_factor_orn=1,
        use_target_pose=True,
        **kwargs,
    ):
        log.info("Loading robot")
        self.cid = cid
        self.filename = filename
        self.use_nullspace = use_nullspace
        self.max_velocity = max_velocity
        self.use_ik_fast = use_ik_fast
        self.base_position = base_position
        self.base_orientation = p.getQuaternionFromEuler(base_orientation)
        self.arm_joint_ids = arm_joint_ids
        self.initial_joint_positions = initial_joint_positions
        self.gripper_joint_ids = gripper_joint_ids
        self.max_joint_force = max_joint_force
        self.gripper_joint_limits = gripper_joint_limits
        self.tcp_link_id = tcp_link_id
        # Setup constraint
        self.constraintForce = constraint_force
        self.constraintEnabled = False
        # internal constraint
        self._constraintEnabled = False
        self.gripper_orientation_offset = quaternion.from_euler_angles(gripper_orientation_offset)
        self.prev_ee_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_uid = None
        self.use_ik = True
        self.end_effector_link_id = end_effector_link_id
        self.gripper_action = 1
        self.ll = self.ul = self.jr = self.rp = None
        self.ll_real = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.ul_real = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.mixed_ik = None
        self.euler_obs = euler_obs
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.magic_scaling_factor_pos = magic_scaling_factor_pos
        self.magic_scaling_factor_orn = magic_scaling_factor_orn
        self.target_pos = None
        self.target_orn = None
        self.use_target_pose = use_target_pose

    def load(self):
        self.robot_uid = p.loadURDF(
            fileName=self.filename,
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            useFixedBase=True,
            physicsClientId=self.cid,
        )
        # for i in range(p.getNumJoints(self.robot_uid)):
        #     p.changeVisualShape(self.robot_uid, i, rgbaColor=[1, 0, 0, 1])
        # create a constraint to keep the fingers centered
        c = p.createConstraint(
            self.robot_uid,
            self.gripper_joint_ids[0],
            self.robot_uid,
            self.gripper_joint_ids[1],
            jointType=p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.cid,
        )
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50, physicsClientId=self.cid)
        num_dof = p.computeDofCount(self.robot_uid)
        # lower limits for null space (todo: set them to proper range)
        self.ll = [-7] * num_dof
        # upper limits for null space (todo: set them to proper range)
        self.ul = [7] * num_dof
        # joint ranges for null space (todo: set them to proper range)
        self.jr = [7] * num_dof
        # restposes for null space
        self.rp = list(self.initial_joint_positions) + [self.gripper_joint_limits[1]] * 2
        self.reset()
        self.mixed_ik = MixedIK(
            self.robot_uid,
            self.cid,
            self.ll_real,
            self.ul_real,
            self.base_position,
            self.base_orientation,
            self.tcp_link_id,
            self.ll,
            self.ul,
            self.jr,
            self.rp,
            self.use_ik_fast,
            threshold_pos=0.03,
            threshold_orn=0.1,
            weights=(10, 8, 6, 6, 2, 2, 1),
            num_angles=30,
        )

    def reset(self, robot_state=None):
        if robot_state is None:
            for i, id in enumerate(self.arm_joint_ids):
                p.resetJointState(self.robot_uid, id, self.initial_joint_positions[i], physicsClientId=self.cid)
        else:
            joint_indices = [i for i, x in enumerate(self.get_observation_labels()) if x.startswith("robot_joint")]
            joint_states = robot_state[joint_indices]
            gripper_opening_width = robot_state[self.get_observation_labels().index("gripper_opening_width")]
            assert len(joint_states) == len(self.arm_joint_ids)
            # TODO: find out why it is needed to call both resetJointState and setMotorControl2
            for i, id in enumerate(self.arm_joint_ids):
                p.resetJointState(self.robot_uid, id, joint_states[i], physicsClientId=self.cid)
                p.setJointMotorControl2(
                    bodyIndex=self.robot_uid,
                    jointIndex=id,
                    controlMode=p.POSITION_CONTROL,
                    force=self.max_joint_force,
                    targetPosition=joint_states[i],
                    maxVelocity=self.max_velocity,
                    physicsClientId=self.cid,
                )

            p.resetJointState(
                self.robot_uid, self.gripper_joint_ids[0], gripper_opening_width / 2, physicsClientId=self.cid
            )
            p.setJointMotorControl2(
                bodyIndex=self.robot_uid,
                jointIndex=self.gripper_joint_ids[0],
                controlMode=p.POSITION_CONTROL,
                force=GRIPPER_FORCE,
                targetPosition=gripper_opening_width / 2,
                maxVelocity=1,
                physicsClientId=self.cid,
            )
            p.resetJointState(
                self.robot_uid, self.gripper_joint_ids[1], gripper_opening_width / 2, physicsClientId=self.cid
            )
            p.setJointMotorControl2(
                bodyIndex=self.robot_uid,
                jointIndex=self.gripper_joint_ids[1],
                controlMode=p.POSITION_CONTROL,
                force=GRIPPER_FORCE,
                targetPosition=gripper_opening_width / 2,
                maxVelocity=1,
                physicsClientId=self.cid,
            )
            tcp_pos, tcp_orn = p.getLinkState(self.robot_uid, self.tcp_link_id, physicsClientId=self.cid)[:2]
            if self.euler_obs:
                tcp_orn = p.getEulerFromQuaternion(tcp_orn)
            self.target_pos = np.array(tcp_pos)
            self.target_orn = np.array(tcp_orn)

    def get_observation(self):
        """
        returns:
        - robot_state: ndarray (16,)
            - tcp_pos: robot_state[:3]
            - tcp_orn: robot_state[3:7] (quat) / [3:6] (euler)
            - gripper_opening_width: robot_state[7:8] (quat) / [6:7] (euler)
            - arm_joint_states: robot_state[8:15] (quat) / [7:14] (euler)
            - gripper_action: robot_state[15:] (quat) / [14:] (euler)
        - robot_info: Dict
        """
        tcp_pos, tcp_orn = p.getLinkState(self.robot_uid, self.tcp_link_id, physicsClientId=self.cid)[:2]
        if self.euler_obs:
            tcp_orn = p.getEulerFromQuaternion(tcp_orn)
        gripper_opening_width = (
            p.getJointState(self.robot_uid, self.gripper_joint_ids[0], physicsClientId=self.cid)[0]
            + p.getJointState(self.robot_uid, self.gripper_joint_ids[1], physicsClientId=self.cid)[0]
        )
        arm_joint_states = []
        for i in self.arm_joint_ids:
            arm_joint_states.append(p.getJointState(self.robot_uid, i, physicsClientId=self.cid)[0])
        robot_state = np.array([*tcp_pos, *tcp_orn, gripper_opening_width, *arm_joint_states, self.gripper_action])
        robot_info = {
            "tcp_pos": tcp_pos,
            "tcp_orn": tcp_orn,
            "gripper_opening_width": gripper_opening_width,
            "arm_joint_states": arm_joint_states,
            "gripper_action": self.gripper_action,
            "uid": self.robot_uid,
            "contacts": p.getContactPoints(bodyA=self.robot_uid, physicsClientId=self.cid),
        }
        return robot_state, robot_info

    def get_observation_labels(self):
        tcp_pos_labels = [f"tcp_pos_{ax}" for ax in ("x", "y", "z")]
        if self.euler_obs:
            tcp_orn_labels = [f"tcp_orn_{ax}" for ax in ("x", "y", "z")]
        else:
            tcp_orn_labels = [f"tcp_orn_{ax}" for ax in ("x", "y", "z", "w")]
        return [
            *tcp_pos_labels,
            *tcp_orn_labels,
            "gripper_opening_width",
            *[f"robot_joint_{i}" for i in self.arm_joint_ids],
            "gripper_action",
        ]

    def relative_to_absolute(self, action):
        assert len(action) == 7
        rel_pos, rel_orn, gripper = np.split(action, [3, 6])
        rel_pos *= self.max_rel_pos * self.magic_scaling_factor_pos
        rel_orn *= self.max_rel_orn * self.magic_scaling_factor_orn
        if self.use_target_pose:
            self.target_pos += rel_pos
            self.target_orn += rel_orn
            return self.target_pos, self.target_orn, gripper
        else:
            tcp_pos, tcp_orn = p.getLinkState(self.robot_uid, self.tcp_link_id, physicsClientId=self.cid)[:2]
            tcp_orn = p.getEulerFromQuaternion(tcp_orn)
            abs_pos = np.array(tcp_pos) + rel_pos
            abs_orn = np.array(tcp_orn) + rel_orn
            return abs_pos, abs_orn, gripper

    def apply_action(self, action):
        if not len(action) == 3:
            action = self.relative_to_absolute(action)
        target_ee_pos, target_ee_orn, self.gripper_action = action

        assert len(target_ee_pos) == 3
        assert len(target_ee_orn) in (3, 4)
        # automatically transform euler actions to quaternion
        if len(target_ee_orn) == 3:
            target_ee_orn = p.getQuaternionFromEuler(target_ee_orn)

        if not isinstance(self.gripper_action, int) and len(self.gripper_action) == 1:
            self.gripper_action = self.gripper_action[0]
        # assert self.gripper_action in (-1, 1)
        #assert self.gripper_action in (0, 1)

        # #
        # cam_rot = p.getMatrixFromQuaternion(target_ee_orn)
        # cam_rot = np.array(cam_rot).reshape(3, 3)
        # cam_rot_x, cam_rot_y, cam_rot_z = cam_rot[:, 0], cam_rot[:, 1], cam_rot[:, 2]
        # p.addUserDebugLine(target_ee_pos, target_ee_pos + cam_rot_x, lineWidth=3, lineColorRGB=[1,0,0])
        # p.addUserDebugLine(target_ee_pos, target_ee_pos +cam_rot_y, lineWidth=3, lineColorRGB=[0,1,0])
        # p.addUserDebugLine(target_ee_pos, target_ee_pos +cam_rot_z, lineWidth=3, lineColorRGB=[0,0,1])
        #
        # tcp_pos, tcp_orn = p.getLinkState(self.robotId, self.tcp_link_id)[:2]
        # tcp_euler = p.getEulerFromQuaternion(tcp_orn)
        # p.addUserDebugLine([0,0,0], target_ee_pos, lineWidth=8, lineColorRGB=[0,1,0])
        # p.addUserDebugLine([0,0,0], p.getLinkState(self.robot_uid, 6)[4], lineWidth=3, lineColorRGB=[1,0,0])
        # p.addUserDebugLine([0,0,0], p.getLinkState(self.robot_uid, 13)[4], lineWidth=3, lineColorRGB=[0,1,0])
        # target_ee_pos, target_ee_orn = self.tcp_to_ee(target_ee_pos, target_ee_orn)
        # p.addUserDebugLine([0,0,0], target_ee_pos, lineWidth=8, lineColorRGB=[1,0,0])
        if self.use_ik:
            jnt_ps = self.mixed_ik.get_ik(target_ee_pos, target_ee_orn)
            for i in range(self.end_effector_link_id):
                # p.resetJointState(self.robot_uid, i, jnt_ps[i])
                p.setJointMotorControl2(
                    bodyIndex=self.robot_uid,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    force=self.max_joint_force,
                    targetPosition=jnt_ps[i],
                    maxVelocity=self.max_velocity,
                    physicsClientId=self.cid,
                )
        else:
            force = self.constraintForce
            if not (self.constraintEnabled and self._constraintEnabled):
                force = 0
            p.changeConstraint(
                self.constraintId, target_ee_pos, target_ee_orn, maxForce=force, physicsClientId=self.cid
            )

        self.control_gripper(self.gripper_action)

    def control_gripper(self, gripper_action):
        if gripper_action == 1:
            gripper_finger_position = self.gripper_joint_limits[1]
        else:
            gripper_finger_position = self.gripper_joint_limits[0]
        for id in self.gripper_joint_ids:
            p.setJointMotorControl2(
                bodyIndex=self.robot_uid,
                jointIndex=id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=gripper_finger_position,
                force=GRIPPER_FORCE,
                maxVelocity=1,
                physicsClientId=self.cid,
            )

    def __str__(self):
        return f"{self.filename} : {self.__dict__}"
