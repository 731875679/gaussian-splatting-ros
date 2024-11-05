import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt

# 新的相机内参
depth_fx = 520.9
depth_fy = 521.0
depth_cx = 325.1
depth_cy = 249.7
depth_factor = 5000.0

# 畸变系数
distortion_coeffs = np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])

# 读取深度图和RGB图像
depth_folder = 'dataset/depth'
rgb_folder = 'dataset/rgb'
groundtruth_file = 'dataset/groundtruth.txt'

# 读取相机位姿真值
camera_poses = {}
with open(groundtruth_file, 'r') as f:
    for line in f:
        timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, line.strip().split())
        camera_poses[timestamp] = np.array([tx, ty, tz, qx, qy, qz, qw])

# 存储所有点云数据
all_points = []
all_colors = []
camera_rectangles = []  # 用于保存相机矩形对象

# 创建相机矩形
def create_camera_rectangle():
    rectangle = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.05, depth=0.02)
    rectangle.paint_uniform_color([0.1, 0.9, 0.1])  # 绿色相机矩形
    return rectangle

# 畸变矫正函数
def undistort_image(rgb_image):
    height, width = rgb_image.shape[:2]
    # 相机矩阵
    camera_matrix = np.array([[depth_fx, 0, depth_cx],
                               [0, depth_fy, depth_cy],
                               [0, 0, 1]])
    
    # 使用OpenCV的畸变矫正函数
    undistorted_image = cv2.undistort(rgb_image, camera_matrix, distortion_coeffs)
    return undistorted_image

# 处理每张图像
for depth_img_name in sorted(os.listdir(depth_folder)):
    timestamp = float(depth_img_name[:-4])  # 去掉.png
    if timestamp not in camera_poses:
        continue

    # 读取深度图和对应的RGB图像
    depth_image = o3d.io.read_image(os.path.join(depth_folder, depth_img_name))
    depth_data = np.asarray(depth_image)
    rgb_img_name = f"{int(timestamp)}.png"
    rgb_image = o3d.io.read_image(os.path.join(rgb_folder, rgb_img_name))
    rgb_data = np.asarray(rgb_image)

    # 进行畸变矫正
    undistorted_data = undistort_image(rgb_data)

    # 生成3D点云
    height, width = depth_data.shape
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            Z = depth_data[v, u] / depth_factor
            if Z > 0:  # 只考虑有效深度
                X = (u - depth_cx) * Z / depth_fx
                Y = (v - depth_cy) * Z / depth_fy
                points.append([X, Y, Z])
                color = undistorted_data[v, u] / 255.0  # 归一化颜色
                colors.append(color)

    # 获取当前相机的绝对位姿
    current_pose = camera_poses[timestamp]
    tx, ty, tz, qx, qy, qz, qw = current_pose

    # 使用四元数构造旋转矩阵
    r = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = r.as_matrix()

    # 将点云转换到世界坐标系
    points = np.array(points)
    points = np.dot(rotation_matrix, points.T).T + np.array([tx, ty, tz])

    # 设置当前点云的点和颜色
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 将当前点云合并到总点云
    all_points.extend(points)
    all_colors.extend(colors)

    # 创建相机矩形并应用旋转和平移
    camera_rectangle = create_camera_rectangle()
    camera_rectangle.rotate(rotation_matrix, center=(0, 0, 0))  # 应用旋转
    camera_rectangle.translate([tx, ty, tz])  # 应用平移
    camera_rectangles.append(camera_rectangle)

    # 显示畸变前后的图像
    # if rgb_data is not None and undistorted_data is not None:
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.title("Distorted Image")
    #     plt.imshow(cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB))  # 转换BGR到RGB
    #     plt.axis('off')

    #     plt.subplot(1, 2, 2)
    #     plt.title("Undistorted Image")
    #     plt.imshow(cv2.cvtColor(undistorted_data, cv2.COLOR_BGR2RGB))  # 转换BGR到RGB
    #     plt.axis('off')

    #     plt.show()

# 创建最终合并的点云
final_point_cloud = o3d.geometry.PointCloud()
final_point_cloud.points = o3d.utility.Vector3dVector(all_points)
final_point_cloud.colors = o3d.utility.Vector3dVector(all_colors)

# 可视化所有点云和相机矩形
o3d.visualization.draw_geometries([final_point_cloud] + camera_rectangles, window_name="Merged Point Cloud with Camera Rectangles")
