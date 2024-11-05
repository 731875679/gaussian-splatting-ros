import rospy
import numpy as np
import open3d as o3d
import tf.transformations as tf_trans
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class PoseVisualizer:
    def __init__(self):
        self.poses = []
        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.camera_pose = np.identity(4)
        self.bridge = CvBridge()
        rospy.Subscriber("/slam/keyframe", PoseStamped, self.pose_callback)

    def pose_callback(self, pose_msg):
        position = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        orientation = [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
                       pose_msg.pose.orientation.z, pose_msg.pose.orientation.w]
        transformation_matrix = tf_trans.quaternion_matrix(orientation)
        transformation_matrix[:3, 3] = position

        # Invert the transformation matrix to get T_WC
        inverse_matrix = np.linalg.inv(transformation_matrix)

        self.poses.append(inverse_matrix)
        self.update_visualization()

    def update_visualization(self):
        self.pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
        self.pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
        for pose in self.poses:
            new_pcd = o3d.geometry.PointCloud(self.pcd)
            new_pcd.transform(pose)
            self.pcd += new_pcd

        if not self.vis.is_visible():
            self.vis.create_window()

        self.vis.clear_geometries()
        self.vis.add_geometry(self.pcd)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

if __name__ == '__main__':
    rospy.init_node('pose_visualizer_node')
    visualizer = PoseVisualizer()
    rospy.spin()