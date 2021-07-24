import numpy as np
import pybullet as p

from vr_env.camera.camera import Camera
import open3d as o3d
import cv2

class GripperCamera(Camera):
    def __init__(
        self, fov, aspect, nearval, farval, width, height, robot_id, gripper_cam_link, cid, name, objects=None
    ):
        self.robot_uid = robot_id
        self.gripper_cam_link = gripper_cam_link
        self.fov = fov
        self.aspect = aspect
        self.nearval = nearval
        self.farval = farval
        self.width = width
        self.height = height
        self.cid = cid
        self.name = name
        self.view_matrix = None

    def render(self):
        camera_ls = p.getLinkState(
            bodyUniqueId=self.robot_uid, linkIndex=self.gripper_cam_link, physicsClientId=self.cid
        )
        camera_pos, camera_orn = camera_ls[:2]
        cam_rot = p.getMatrixFromQuaternion(camera_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
        # camera: eye position, target position, up vector

        view_matrix = p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z)
        self.view_matrix = view_matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect, nearVal=self.nearval, farVal=self.farval
        )
        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            physicsClientId=self.cid,
        )
        rgb_img, depth_img = self.process_rgbd(image, self.nearval, self.farval)

        (width, height, rgbPixels, depthPixels, segmentationMaskBuffer) = image
        # For now, we have only one object and we get value 1 for the object in our segmented mask
        target_ob = 0

        ind = np.argwhere(segmentationMaskBuffer != target_ob)
        for i in ind:
            depth_img[i[0]][i[1]]=0

        return rgb_img, depth_img
