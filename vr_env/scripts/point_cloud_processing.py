import numpy as np
import pybullet as p

from vr_env.camera.camera import Camera
import open3d as o3d
import cv2


class PointCloud:
    def __init__(
            self):
        self.T_Robot_cam = np.array([])
        self.accumulatedPC = np.array([])

    def transform_pointcloud_matrix(self, filtered_pc):
        # filtered_pc (Xi): shape(N*3)
        # T_Robot_cam (Ti): shape(4*4) Robot to Camera #transformation matrix

        # filtered_pc_4D = shape(4*N) -> reshaped in order to multiply with transformation matrix
        filtered_pc_4D = np.concatenate([filtered_pc.T, np.ones((1, filtered_pc.shape[0]))], axis=0)

        # Inverse of Ti
        # T_cam_Robot = np.linalg.inv(T_Robot_cam)
        # T_cam_Robot = (T_Robot_cam)
        # inv(Ti) * Xi
        transformed_pc = self.T_Robot_cam @ filtered_pc_4D

        # Uncomment following lines to visualize individual point cloud (at every instance)
        # tranformedPCD = o3d.geometry.PointCloud()
        # tranformedPCD.points = o3d.utility.Vector3dVector(transformed_pc[:3, :].T)
        # o3d.visualization.draw_geometries([tranformedPCD])

        # Accumulate transformed Point clouds
        if self.accumulatedPC.size == 0:
            # we don't need to concatenate first pcd
            self.accumulatedPC = transformed_pc
        else:
            self.accumulatedPC = np.concatenate((self.accumulatedPC, transformed_pc), axis=1)

        finalAcc = np.linalg.inv(self.T_Robot_cam) @ self.accumulatedPC

        return finalAcc[:3, :].T

    def viewPointCloud(self, pcd):
        # pcd : shape(4,N), x,y and distances separated
        tranformedPCD = o3d.geometry.PointCloud()
        tranformedPCD.points = o3d.utility.Vector3dVector(pcd)
        # Uncomment following IF -> you want un-transformed pcd
        o3d.visualization.draw_geometries([tranformedPCD])