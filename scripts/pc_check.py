import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# 相机参数验证
def validate_cameras(cameras_file, expected_params):
    with open(cameras_file, 'r') as f:
        for line in f:
            # 按空格分割行
            tokens = line.strip().split()

            # 检查是否有至少8个字段
            if len(tokens) < 8:
                print(f"Line doesn't have enough parameters: {line}")
                continue

            # 解包前8个字段
            camera_id, model, width, height, fx, fy, cx, cy = tokens[:8]

            try:
                # 转换数据类型
                width, height = float(width), float(height)
                fx, fy, cx, cy = map(float, [fx, fy, cx, cy])

                # 校验参数
                if (width, height, fx, fy, cx, cy) == expected_params:
                    print(f"Camera {camera_id}: Parameters match.")
                else:
                    print(f"Camera {camera_id}: Parameters do not match.")
            
            except ValueError as e:
                print(f"Error parsing camera parameters: {e}")
                continue
# 验证 images.txt 中的 2D 点和 3D 点映射是否正确
def validate_images(images_file, points3D_file):
    points3D_map = {}  # 加载3D点
    with open(points3D_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            point3D_id = float(data[0])
            points3D_map[point3D_id] = data[1:4]  # X, Y, Z

    with open(images_file, 'r') as f:
        image_id = None
        for line in f:
            if len(line.split()) == 10:  # IMAGE_ID行
                image_id = float(line.split()[0])
            else:  # 2D点行
                points = line.strip().split()
                for i in range(0, len(points), 3):
                    u, v, point3D_id = map(float, points[i:i+3])
                    if point3D_id not in points3D_map:
                        print(f"Error: 3D point {point3D_id} in image {image_id} not found in points3D.txt.")
                        return False
        print("Images and 3D points mapping is correct.")
    return True

# 验证 points3D.txt 中的 TRACK[] 是否正确映射到 images.txt 中
def validate_points3D(points3D_file, images_file):
    images_map = {}
    with open(images_file, 'r') as f:
        image_id = None
        for line in f:
            if len(line.split()) == 10:  # IMAGE_ID行
                image_id = float(line.split()[0])
                images_map[image_id] = []
            else:  # 2D点行
                points = line.strip().split()
                images_map[image_id] = [(float(points[i]), float(points[i+1])) for i in range(0, len(points), 3)]
    
    with open(points3D_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            point3D_id = float(data[0])
            track = data[8:]  # TRACK部分
            for i in range(0, len(track), 2):
                img_id = float(track[i])
                point2d_idx = float(track[i+1])
                if img_id not in images_map or point2d_idx >= len(images_map[img_id]):
                    print(f"Error: 3D point {point3D_id} references invalid 2D point (Image {img_id}, Point {point2d_idx}).")
                    return False
    print("3D points to 2D points mapping is correct.")
    return True

# 绘制相机位姿矩形
# 绘制相机位姿矩形
def create_camera_frustum(position, rotation, scale=0.05, color=[1, 0, 0]):
    # 定义一个简单的矩形框表示相机的平面
    vertices = np.array([
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]  # 相机矩形的四个顶点 (x, y, 0)
    ]) * scale  # 根据 scale 缩放矩形大小

    # 将矩形的顶点应用旋转和平移变换
    vertices = np.dot(rotation, vertices.T).T + position

    # 定义矩形的边
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    # 使用 Open3D 创建线段集合来表示相机矩形
    camera_frustum = o3d.geometry.LineSet()
    camera_frustum.points = o3d.utility.Vector3dVector(vertices)
    camera_frustum.lines = o3d.utility.Vector2iVector(lines)

    # 添加线的颜色，颜色是红色 (R, G, B)
    colors = [color for _ in lines]  # 为每条边设置颜色
    camera_frustum.colors = o3d.utility.Vector3dVector(colors)

    return camera_frustum

# 可视化点云和相机位姿
def visualize_point_cloud_with_cameras(points3D_file, images_file):
    point_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    
    with open(points3D_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            X, Y, Z = map(float, data[1:4])
            R_val, G, B = map(float, data[4:7])
            points.append([X, Y, Z])
            colors.append([R_val / 255.0, G / 255.0, B / 255.0])
    
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 从 images.txt 中读取相机位姿并生成矩形
    camera_frustums = []  # 用于存储相机矩形的集合
    with open(images_file, 'r') as f:
        for line in f:
            if len(line.split()) == 10:  # IMAGE_ID行
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line.strip().split()

                # 转换为浮点数
                tx, ty, tz = map(float, [tx, ty, tz])
                qw, qx, qy, qz = map(float, [qw, qx, qy, qz])

                # 四元数转换为旋转矩阵
                r = R.from_quat([qx, qy, qz, qw])
                rotation_matrix = r.as_matrix()

                # 创建齐次变换矩阵 (TWC: 从世界到相机的变换)
                Rt = np.zeros((4, 4))
                Rt[:3, :3] = rotation_matrix
                Rt[:3, 3] = np.array([tx, ty, tz])
                Rt[3, 3] = 1.0

                # 计算从相机到世界的变换 (TCW)
                TCW = np.linalg.inv(Rt)

                # 相机在世界坐标系下的位置和旋转
                camera_position_world = TCW[:3, 3]  # 提取平移部分
                camera_rotation_world = TCW[:3, :3]  # 提取旋转部分

                # 创建相机矩形并添加到集合
                camera_frustum = create_camera_frustum(camera_position_world, camera_rotation_world)
                camera_frustums.append(camera_frustum)

    # 可视化点云和相机矩形
    o3d.visualization.draw_geometries([point_cloud] + camera_frustums, window_name="3D Point Cloud with Cameras", width=800, height=600)

# 主函数
if __name__ == '__main__':
    # 文件路径
    cameras_file = 'tum-fg2-desk-orb/sparse/0/cameras.txt'
    images_file = 'tum-fg2-desk-orb/sparse/0/images.txt'
    points3D_file = 'tum-fg2-desk-orb/sparse/0/points3D.txt'
    
    # 文件路径
    # cameras_file = 'tum/sparse/0/cameras.txt'
    # images_file = 'tum/sparse/0/images.txt'
    # points3D_file = 'tum/sparse/0/points3D.txt'
    
    # 期望的相机参数 (width, height, fx, fy, cx, cy)
    expected_params = (640, 480, 520.9, 521.0, 325.1, 249.7)

    print("Validating cameras.txt:")
    validate_cameras(cameras_file, expected_params)

    print("\nValidating images.txt and 3D point mapping:")
    if validate_images(images_file, points3D_file):
        print("2D-3D point correspondence is correct in images.txt.")

    print("\nValidating points3D.txt and its TRACK[] mappings:")
    if validate_points3D(points3D_file, images_file):
        print("TRACK[] correspondence is correct in points3D.txt.")

    print("\nVisualizing the 3D point cloud:")
    visualize_point_cloud_with_cameras(points3D_file, images_file)
