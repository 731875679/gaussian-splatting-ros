import collections
import time
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
import os
import rospy
import numpy as np
import struct
from sensor_msgs.msg import Image as RosImage, PointCloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from threading import Lock, Thread
import open3d as o3d
import cv2
from tf.transformations import quaternion_matrix
from scene.dataset_readers import getNerfppNorm, SceneInfo, CameraInfo, BasicPointCloud
from utils.graphics_utils import focal2fov
from threading import Lock, Thread, Event
import open3d as o3d

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    
Cameraintr = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

class KeyframeProcessor:
    def __init__(self, path):
        self.path = path
        self.cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
        self.all_positions = np.empty((0, 3), dtype=np.float32)
        self.all_colors = np.empty((0, 3), dtype=np.float32)
        self.all_normals = np.empty((0, 3), dtype=np.float32)
        self.camera_info_dict = {}
        self.bridge = CvBridge()
        self.data_lock = Lock()
        self.frame_data_cache = {}

        self.init_subscribers()
        # 启动ROS spin线程
        self.spin_thread = Thread(target=self.spin_ros)
        self.spin_thread.start()
        rospy.loginfo("KeyframeProcessor initialized.")
        
        # 初始化其他属性
        self.update_event = Event()
        self.update_thread = Thread(target=self.monitor_updates)
        self.update_thread.start()
        
        self.last_msg_time = rospy.get_time()
        self.timeout_duration = 10  # 超时的秒数
        # 启动一个定时器来监测超时
        self.timeout_timer = rospy.Timer(rospy.Duration(1.0), self.check_timeout)
        self.shutdown = False
        
    def monitor_updates(self):
        while True:
            self.update_event.wait()  # 等待更新事件被设置
            scene_info = self.create_scene_info()  # 执行更新操作
            if scene_info:
                rospy.loginfo("SceneInfo updated successfully.")
            self.update_event.clear()  # 重置事件，准备下一次更新
            
    def trigger_scene_info_update(self):
        self.update_event.set()  # 设置事件以触发更新 
                       
    def parse_frame_id(self, frame_id_str):
        parts = frame_id_str.split('_')
        if len(parts) != 2:
            rospy.logwarn("Invalid frame_id format: %s", frame_id_str)
            return None, None
        try:
            frame_id = int(parts[0])
            is_loop_closure = parts[1] == '1'
            return frame_id, is_loop_closure
        except ValueError:
            rospy.logwarn("Invalid frame_id value: %s", frame_id_str)
            return None, None
        
    def rgb_callback(self, rgb_msg, pose_msg):
        self.last_msg_time = rospy.get_time()  # 更新最后消息接收时间

        try:
            frame_id_str = pose_msg.header.frame_id
            frame_id, is_loop_closure = self.parse_frame_id(frame_id_str) 
        except ValueError:
            rospy.logwarn("Invalid frame_id: %s", rgb_msg.header.frame_id)
            return

        with self.data_lock:
            self.frame_data_cache[frame_id] = {
                'rgb': rgb_msg,
                'pose': pose_msg,
                'is_loop_closure':is_loop_closure
            }
            self.process_image_and_pose(frame_id)
            
    def process_image_and_pose(self, frame_id):
        rgb_msg = self.frame_data_cache[frame_id].get('rgb')
        pose_msg = self.frame_data_cache[frame_id].get('pose')
        is_loop_closure = self.frame_data_cache[frame_id].get('is_loop_closure')

        # 处理位姿信息
        tvec = np.array([pose_msg.pose.position.x,
                    pose_msg.pose.position.y,
                    pose_msg.pose.position.z])
        qvec = np.array([pose_msg.pose.orientation.w,
                    pose_msg.pose.orientation.x,
                    pose_msg.pose.orientation.y,
                    pose_msg.pose.orientation.z])
        
        R = np.transpose(qvec2rotmat(qvec))
        T = np.array(tvec)
        
        if is_loop_closure:
            self.update_pose_only(frame_id, R, T)
        else:
            self.create_new_camera(frame_id, rgb_msg, R, T)

        # 删除 RGB 信息，保留位姿信息
        if 'rgb' in self.frame_data_cache[frame_id]:
            del self.frame_data_cache[frame_id]['rgb']

    def create_new_camera(self, frame_id, rgb_msg, R, T):
        cam_intrinsics = self.read_intrinsics_text(self.cameras_intrinsic_file, frame_id)
        intr = cam_intrinsics.get(frame_id)
        if intr is None:
            rospy.logwarn("No intrinsics found for frame %d", frame_id)
            return
        height = intr.height
        width = intr.width

        uid = intr.id
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            rospy.logerr("Unsupported camera model for frame_id: %d", frame_id)
            return

        image_name = f"{frame_id}.png"
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except Exception as e:
            rospy.logwarn("Failed to convert ROS Image to CV2 for frame_id: %d, error: %s", frame_id, e)
            return
        
        image_dir = os.path.join(self.path, "images_keyframe")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        image_path = os.path.join(image_dir, image_name)
        
        # try:
        #     cv2.imwrite(image_path, cv_image)
        #     # rospy.loginfo("Image saved to %s", image_path)
        # except Exception as e:
        #     rospy.logwarn("Failed to save image for frame_id: %d, error: %s", frame_id, e)  
                  
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params={},
                              image_path=image_path, image_name=image_name, depth_path="",
                              width=width, height=height, is_test=False)
        
        self.camera_info_dict[frame_id] = cam_info
        
    def read_intrinsics_text(self, path, frame_id):
        cameras = {}
        try:
            with open(path, "r") as fid:
                for line in fid:
                    line = line.strip()
                    if len(line) > 0 and line[0] != "#":
                        elems = line.split()
                        camera_id = frame_id
                        model = elems[1]
                        assert model == "PINHOLE", "Camera model is not PINHOLE"
                        width = int(elems[2])
                        height = int(elems[3])
                        params = np.array(tuple(map(float, elems[4:])))
                        cameras[camera_id] = Cameraintr(id=camera_id, model=model,
                                                       width=width, height=height,
                                                       params=params)
        except Exception as e:
            rospy.logerr("Error reading intrinsics: %s", e)
        return cameras
        
    def update_pose_only(self, frame_id, R, T):
        # 检查缓存中是否存在该帧的位姿信息
        pose_data = self.frame_data_cache.get(frame_id)
        if pose_data is None:
            rospy.logwarn("No pose message found for frame %d, skipping update.", frame_id)
            return

        # 获取位姿信息
        pose_msg = pose_data.get('pose')
        if pose_msg is None:
            rospy.logwarn("No pose message found for frame %d, skipping update.", frame_id)
            return

        # 更新位姿
        cam_info = self.camera_info_dict.get(frame_id)
        if cam_info:
            self.camera_info_dict[frame_id].R = R
            self.camera_info_dict[frame_id].T = T
            
            rospy.loginfo("Updated pose for frame_id: %d", frame_id)
        else:
            rospy.logwarn("No CameraInfo found for frame_id: %d", frame_id)
            
    def point_cloud_callback(self, msg):
        self.last_msg_time = rospy.get_time()  # 更新最后消息接收时间

        try:
            frame_id = int(msg.header.frame_id)
        except ValueError:
            rospy.logwarn("Invalid frame_id in PointCloud2: %s", msg.header.frame_id)
            return

        with self.data_lock:
            point_step = msg.point_step
            width = msg.width

            positions = np.zeros((width, 3), dtype=np.float32)
            colors = np.zeros((width, 3), dtype=np.float32)

            for i in range(width):
                offset = i * point_step
                x, y, z = struct.unpack_from('fff', msg.data, offset)
                positions[i] = [x, y, z]

                rgb_int = struct.unpack_from('I', msg.data, offset + 16)[0]
                r = (rgb_int >> 16) & 0xFF
                g = (rgb_int >> 8) & 0xFF
                b = rgb_int & 0xFF
                colors[i] = [r / 255.0, g / 255.0, b / 255.0]

            normals = self.compute_normals(positions, k=30)

            self.all_positions = np.vstack((self.all_positions, positions))
            self.all_colors = np.vstack((self.all_colors, colors))
            self.all_normals = np.vstack((self.all_normals, normals))
            
    def compute_normals(self, points, k=20):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        normals = np.asarray(cloud.normals)
        return normals
    
    def init_subscribers(self):
        rospy.loginfo("Initializing ROS subscribers.")
        
        # 同步 RGB 和 Pose 消息
        rgb_sub = message_filters.Subscriber("/slam/keyframe_image", RosImage)
        pose_sub = message_filters.Subscriber("/slam/keyframe_pose", PoseStamped)

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, pose_sub], 10, 0.1)
        ts.registerCallback(self.rgb_callback)

        # 点云消息单独处理
        rospy.Subscriber("/slam/keyframe_point3d", PointCloud2, self.point_cloud_callback)
        
    def spin_ros(self):
        """ 运行spin，监听ROS消息 """
        rospy.spin()
        
    def create_scene_info(self):        
        # 打印点云的大小        
        while (len(self.all_positions) < 200):
            rospy.logwarn("Not enough cameras available. Waiting more ...")
            time.sleep(0.2)  # 等待0.5秒后重试
        try:
            point_cloud = BasicPointCloud(points=self.all_positions, 
                                    colors=self.all_colors, 
                                    normals=self.all_normals)
        except Exception as e:
            rospy.logerr("Error creating BasicPointCloud: %s", e)
            return None

        # 使用Open3D保存点云数据到PLY文件
        ply_path = os.path.join(self.path, "sparse/0/points3D.ply")
        # 检查文件夹是否存在
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        
        # 创建新的点云对象
        new_point_cloud = o3d.geometry.PointCloud()
        new_point_cloud.points = o3d.utility.Vector3dVector(self.all_positions)
        new_point_cloud.colors = o3d.utility.Vector3dVector(self.all_colors)
        new_point_cloud.normals = o3d.utility.Vector3dVector(self.all_normals)
        
        # 如果PLY文件已存在，先读取已有点云，再追加新的点云
        if os.path.exists(ply_path):
            try:
                existing_point_cloud = o3d.io.read_point_cloud(ply_path)
                combined_point_cloud = existing_point_cloud + new_point_cloud
                # rospy.loginfo("PointCloud merged with existing PLY data.")
            except Exception as e:
                rospy.logerr("Error reading existing PLY file: %s", e)
                return None
        else:
            combined_point_cloud = new_point_cloud

        # 写入合并后的点云到文件
        try:
            o3d.io.write_point_cloud(ply_path, combined_point_cloud, write_ascii=True)
            # rospy.loginfo("PointCloud saved to %s", ply_path)
        except Exception as e:
            rospy.logerr("Error writing PointCloud to PLY file: %s", e)
            return None

        # 打印训练和测试相机的信息
        train_cameras_info = [cam for cam in self.camera_info_dict.values() if not cam.is_test]
        test_cameras_info = [cam for cam in self.camera_info_dict.values() if cam.is_test]
                        
        try:
            nerf_normalization = getNerfppNorm(train_cameras_info)
        except Exception as e:
            rospy.logerr("Error computing nerf_normalization: %s", e)
            return None
        
        try:
            scene_info = SceneInfo(point_cloud=point_cloud,
                                train_cameras=train_cameras_info,
                                test_cameras=test_cameras_info,
                                nerf_normalization=nerf_normalization,
                                ply_path=ply_path,
                                is_nerf_synthetic=False)
            # rospy.loginfo("SceneInfo created with %d points, %d train cameras, and %d test cameras.", 
            #             point_cloud.points.shape[0], len(train_cameras_info), len(test_cameras_info))
            # # 清空当前三维点信息
            # self.all_positions = np.empty((0, 3), dtype=np.float32)
            # self.all_colors = np.empty((0, 3), dtype=np.float32)
            # self.all_normals = np.empty((0, 3), dtype=np.float32)
            
            return scene_info
        
        except Exception as e:
            rospy.logerr("Error creating SceneInfo: %s", e)
        return None
    
    def check_timeout(self, event):
        current_time = rospy.get_time()
        if current_time - self.last_msg_time > self.timeout_duration:
            self.shutdown = True
            rospy.logwarn("No messages received for %d seconds. Shutting down.", self.timeout_duration)
            rospy.signal_shutdown("Timeout: No messages received")          
              
if __name__ == "__main__":
    rospy.init_node("keyframe_processing_node")
    processor = KeyframeProcessor(path="/home/wang/catkin_ws/src/wla-gaussian/tum-fg2-desk-orb")
    while(True):
        if input("输入'1'创建SceneInfo或其他命令: ") == "1":
            processor.trigger_scene_info_update()