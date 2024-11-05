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
        self.frame_counter = 0

        self.init_subscribers()
        # 启动ROS spin线程
        self.spin_thread = Thread(target=self.spin_ros)
        self.spin_thread.start()
        rospy.loginfo("KeyframeProcessor initialized.")
        
        # 初始化其他属性
        self.update_event = Event()
        self.update_thread = Thread(target=self.monitor_updates)
        self.update_thread.start()
        
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
            self.frame_counter += 1
            self.process_image_and_pose(frame_id)
            
    def process_image_and_pose(self, frame_id):
        rgb_msg = self.frame_data_cache[frame_id].get('rgb')
        pose_msg = self.frame_data_cache[frame_id].get('pose')
        is_loop_closure = self.frame_data_cache[frame_id].get('is_loop_closure')
        
        T = np.array([pose_msg.pose.position.x,
                      pose_msg.pose.position.y,
                      pose_msg.pose.position.z])
        q = np.array([pose_msg.pose.orientation.w,
                      pose_msg.pose.orientation.x,
                      pose_msg.pose.orientation.y,
                      pose_msg.pose.orientation.z])
        R_full = quaternion_matrix(q)
        R = R_full[:3, :3]
        if is_loop_closure:
            self.update_pose_only(frame_id, R, T)
        else:
            self.create_new_camera(frame_id, rgb_msg, R, T)

        del self.frame_data_cache[frame_id]

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
        image_path = os.path.join(self.path, "images_keyframe", image_name)
        cv2.imwrite(image_path, cv_image)
        
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
    
    def update_pose_only(self, frame_id, R=None, T=None):
        if R is None or T is None:
            pose_data = self.frame_data_cache.get(frame_id, {}).get('pose', (None, None))
            pose_msg = pose_data[0] if pose_data else None
            if pose_msg is None:
                rospy.logwarn("No pose message found for frame %d", frame_id)
                return
            T = np.array([pose_msg.pose.position.x,
                          pose_msg.pose.position.y,
                          pose_msg.pose.position.z])
            q = np.array([pose_msg.pose.orientation.w,
                          pose_msg.pose.orientation.x,
                          pose_msg.pose.orientation.y,
                          pose_msg.pose.orientation.z])
            R_full = quaternion_matrix(q)
            R = R_full[:3, :3]
            rospy.loginfo("Pose extracted from pose_msg for frame_id: %d", frame_id)

        cam_info = self.camera_info_dict.get(frame_id)
        if cam_info:
            cam_info.R = R
            cam_info.T = T
        else:
            rospy.logwarn("No CameraInfo found for frame_id: %d", frame_id)
            
    def point_cloud_callback(self, msg):
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
        try:
            point_cloud = BasicPointCloud(points=self.all_positions, 
                                        colors=self.all_colors, 
                                        normals=self.all_normals)
        except Exception as e:
            rospy.logerr("Error creating BasicPointCloud: %s", e)
            return None

        # 打印训练和测试相机的信息
        while True:
            # 打印训练和测试相机的信息
            train_cameras_info = [cam for cam in self.camera_info_dict.values() if not cam.is_test]
            test_cameras_info = [cam for cam in self.camera_info_dict.values() if cam.is_test]
                        
            if len(train_cameras_info) < 10:
                rospy.logwarn("No camera information available. Waiting for cameras to be input...")
                time.sleep(0.5)  # 等待5秒后重试
                continue
            else:
                break   
            
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
                                ply_path="",
                                is_nerf_synthetic=False)
            rospy.loginfo("SceneInfo created with %d points, %d train cameras, and %d test cameras.", 
                        point_cloud.points.shape[0], len(train_cameras_info), len(test_cameras_info))
            return scene_info
        
        except Exception as e:
            rospy.logerr("Error creating SceneInfo: %s", e)
        return None
                
if __name__ == "__main__":
    rospy.init_node("keyframe_processing_node")
    processor = KeyframeProcessor(path="/home/wang/catkin_ws/src/wla-gaussian/tum-fg2-desk-orb")
    while(True):
        if input("输入'1'创建SceneInfo或其他命令: ") == "1":
            processor.trigger_scene_info_update()