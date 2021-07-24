# examples/Python/Basic/working_with_numpy.py

import copy
import random

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


def transformPoinCloud(point_cloud, T_Robot_cam):
    # point cloud: shape(200*200*3)
    # T_Robot_cam: Robot to Camera #transformation matrix

    # xyz : shape(40000,3), x,y and distances separated
    xyz = point_cloud[np.where(np.ones((200, 200)))]

    # another way of geting xyz
    # xyz = np.zeros((np.size(x), 3))
    # xyz[:, 0] = np.reshape(x, -1)
    # xyz[:, 1] = np.reshape(y, -1)
    # xyz[:, 2] = np.reshape(distances, -1)

    # newxyz = 4*40000 -> reshaped in order to multiply with transformation matrix
    new_xyz = np.concatenate([xyz.T, np.ones((1, xyz.shape[0]))], axis=0)

    # transform
    tranformed_xyz = T_Robot_cam @ new_xyz

    # convert back to pcd form
    tranformedPCD = o3d.geometry.PointCloud()
    tranformedPCD.points = o3d.utility.Vector3dVector(tranformed_xyz[:3, :].T)

    # Uncomment following IF -> you want un-transformed pcd
    # tranformedPCD.points = o3d.utility.Vector3dVector(xyz)

    return tranformedPCD


def transform_pointcloud_matrix(filtered_pc, T_Robot_cam):
    # filtered_pc (Xi): shape(N*3)
    # T_Robot_cam (Ti): shape(4*4) Robot to Camera #transformation matrix

    # filtered_pc_4D = shape(4*N) -> reshaped in order to multiply with transformation matrix
    filtered_pc_4D = np.concatenate([filtered_pc.T, np.ones((1, filtered_pc.shape[0]))], axis=0)

    # Inverse of Ti
    # T_cam_Robot = np.linalg.inv(T_Robot_cam)
    # T_cam_Robot = (T_Robot_cam)
    # inv(Ti) * Xi
    tranformed_pc = T_Robot_cam @ filtered_pc_4D

    # Uncomment following lines to visualize individual point cloud (at every instance)
    tranformedPCD = o3d.geometry.PointCloud()
    tranformedPCD.points = o3d.utility.Vector3dVector(tranformed_pc[:3, :].T)
    # o3d.visualization.draw_geometries([tranformedPCD])

    return tranformed_pc


def transform_pointcloud_matrix_o3d(filtered_pc, T_Robot_cam):
    # filtered_pc (Xi): shape(N*3)
    # T_Robot_cam (Ti): shape(4*4) Robot to Camera #transformation matrix

    # filtered_pc_4D = shape(4*N) -> reshaped in order to multiply with transformation matrix
    filtered_pc_4D = np.concatenate([filtered_pc.T, np.ones((1, filtered_pc.shape[0]))], axis=0)

    # Inverse of Ti
    # T_cam_Robot = np.linalg.inv(T_Robot_cam)
    # T_cam_Robot = (T_Robot_cam)
    # inv(Ti) * Xi
    transformed_pc = T_Robot_cam @ filtered_pc_4D

    # Uncomment following lines to visualize individual point cloud (at every instance)
    transformedPCD = o3d.geometry.PointCloud()
    transformedPCD.points = o3d.utility.Vector3dVector(transformed_pc[:3, :].T)
    transformedPCD.paint_uniform_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
    # o3d.visualization.draw_geometries([tranformedPCD])

    return transformed_pc, transformedPCD


def viewPointCloud(pcd):
    # pcd : shape(4,N), x,y and distances separated
    tranformedPCD = o3d.geometry.PointCloud()
    tranformedPCD.points = o3d.utility.Vector3dVector(pcd[:3, :].T)
    # Uncomment following IF -> you want un-transformed pcd
    o3d.visualization.draw_geometries([tranformedPCD])


def viewPointCloud_color(pcds):
    # pcds: Open3D point cloud
    o3d.visualization.draw_geometries(pcds)


if __name__ == "__main__":
    # generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print('xyz')
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("sync.ply")
    o3d.visualization.draw_geometries([pcd_load])

    # convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)

    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    o3d.io.write_image("sync.png", img)
    o3d.visualization.draw_geometries([img])
