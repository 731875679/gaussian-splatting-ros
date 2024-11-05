import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import cv2  # 用于保存图像

# 文件夹路径
tum_folder = 'tum'
sparse_folder = os.path.join(tum_folder, 'sparse/0')
images_folder = os.path.join(tum_folder, 'images')  # 用于保存RGB图像
depth_folder = os.path.join(tum_folder, 'depth')    # 用于保存深度图像

# 创建文件夹
os.makedirs(sparse_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)

# 相机内参
camera_id = 1
depth_fx = 520.9
depth_fy = 521.0
depth_cx = 325.1
depth_cy = 249.7
depth_factor = 5000.0

# 图像的宽和高
image_width = 640
image_height = 480

# groundtruth文件路径
groundtruth_file = 'dataset/groundtruth.txt'

# 存储相机位姿
camera_poses = {}
with open(groundtruth_file, 'r') as f:
    for line in f:
        timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, line.strip().split())
        camera_poses[timestamp] = np.array([tx, ty, tz, qx, qy, qz, qw])

# 保存cameras.txt文件
with open(os.path.join(sparse_folder, 'cameras.txt'), 'w') as f:
    # 假设所有相机内参一致，模型为PINHOLE
    params = [depth_fx, depth_fy, depth_cx, depth_cy]
    f.write(f"{camera_id} PINHOLE {image_width} {image_height} {' '.join(map(str, params))}\n")

# 初始化点云数据和保存格式相关变量
all_points = []
all_colors = []
points3D_data = {}  # 保存三维点数据，结构为 {point3D_id: [X, Y, Z, R, G, B, error, TRACK[]]}
images_data = []  # 保存每张图像的数据
image_id = 1  # 图像ID从1开始
point3D_id = 1  # 3D点ID从1开始
point_id_map = {}  # 用于保存每个2D点对应的3D点ID

# 读取每张深度图并生成点云
for depth_img_name in sorted(os.listdir('dataset/depth')):
    timestamp = float(depth_img_name[:-4])  # 去掉.png
    if timestamp not in camera_poses:
        continue

    # 读取深度图和RGB图像
    depth_image = o3d.io.read_image(os.path.join('dataset/depth', depth_img_name))
    depth_data = np.asarray(depth_image)
    rgb_img_name = f"{int(timestamp)}.png"
    rgb_image = o3d.io.read_image(os.path.join('dataset/rgb', rgb_img_name))
    rgb_data = np.asarray(rgb_image)

    # 保存RGB和深度图像到指定的文件夹中
    cv2.imwrite(os.path.join(images_folder, rgb_img_name), cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))  # RGB图像
    cv2.imwrite(os.path.join(depth_folder, depth_img_name), depth_data)  # 深度图像

    # 获取当前相机的绝对位姿
    current_pose = camera_poses[timestamp]
    tx, ty, tz, qx, qy, qz, qw = current_pose

    # 使用四元数构造旋转矩阵
    r = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = r.as_matrix()

    # 生成3D点云
    height, width = depth_data.shape
    img_2D_points = []  # 保存当前图像的2D点
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            Z = depth_data[v, u] / depth_factor
            if Z > 0:  # 只考虑有效深度
                X = (u - depth_cx) * Z / depth_fx
                Y = (v - depth_cy) * Z / depth_fy
                point = np.dot(rotation_matrix, [X, Y, Z]) + np.array([tx, ty, tz])  # 转换到世界坐标系

                color = rgb_data[v, u] / 255.0  # 归一化颜色
                points.append(point)
                colors.append(color)

                # 保存三维点与其对应的二维像素点
                if point3D_id not in points3D_data:
                    points3D_data[point3D_id] = [point[0], point[1], point[2], color[0], color[1], color[2], 0, []]

                # 记录TRACK中的信息，(IMAGE_ID, POINT2D_IDX)
                points3D_data[point3D_id][7].append((image_id, len(img_2D_points)))
                img_2D_points.append([u, v, point3D_id])  # 记录2D点和对应的3D点ID
                point3D_id += 1

    # 保存每张图像的位姿及其2D-3D映射关系
    images_data.append({
        'image_id': image_id,
        'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
        'tx': tx, 'ty': ty, 'tz': tz,
        'camera_id': camera_id,
        'name': rgb_img_name,
        '2d_points': img_2D_points
    })
    image_id += 1

# 保存相机位姿并转换为COLMAP格式
relative_poses = {}
for image in images_data:
    # 获取绝对位姿
    tx, ty, tz = image['tx'], image['ty'], image['tz']
    qw, qx, qy, qz = image['qw'], image['qx'], image['qy'], image['qz']

    # 使用四元数构造旋转矩阵
    r = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = r.as_matrix()

    # 构建齐次变换矩阵 Rt
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = rotation_matrix
    Rt[:3, 3] = np.array([tx, ty, tz])
    Rt[3, 3] = 1.0

    # 计算从世界坐标系到相机坐标系的变换
    W2C = np.linalg.inv(Rt)

    # 相机的相对位置和旋转
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]

    # 将相对位姿保存
    images_data[image['image_id'] - 1]['pos'] = pos
    images_data[image['image_id'] - 1]['rot'] = rot

# 保存 images.txt 文件
with open(os.path.join(sparse_folder, 'images.txt'), 'w') as f:
    for image in images_data:
        pos = image['pos']
        rot = image['rot']

        # 将旋转矩阵转换为四元数
        rot_quat = R.from_matrix(rot).as_quat()
        qw, qx, qy, qz = rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]

        # 写入第一行: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        f.write(f"{image['image_id']} {qw} {qx} {qy} {qz} "
                f"{pos[0]} {pos[1]} {pos[2]} {image['camera_id']} {image['name']}\n")
        
        # 写入第二行: 2D点和对应的3D点ID
        points_2d_str = " ".join([f"{x} {y} {pid}" for x, y, pid in image['2d_points']])
        f.write(f"{points_2d_str}\n")

# 保存 points3D.txt 文件，添加TRACK[]信息
with open(os.path.join(sparse_folder, 'points3D.txt'), 'w') as f:
    for pid, (X, Y, Z, R, G, B, error, track) in points3D_data.items():
        # 将TRACK中的(IMAGE_ID, POINT2D_IDX)转为字符串
        track_str = " ".join([f"{img_id} {point2d_idx}" for img_id, point2d_idx in track])
        f.write(f"{pid} {X} {Y} {Z} {int(R*255)} {int(G*255)} {int(B*255)} {error} {track_str}\n")
