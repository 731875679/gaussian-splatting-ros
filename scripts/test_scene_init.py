#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import threading
import time
import rospy
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import threading
import numpy as np
from scene.ros_loader import KeyframeProcessor
import open3d as o3d  # 新增
import sys
import torch
from argparse import ArgumentParser
from train_ros_test import training
from arguments import ModelParams, OptimizationParams, PipelineParams

class Scene:
    gaussians: GaussianModel
    args: ModelParams
    use_ros: bool

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], use_ros=True):
        """param path: Path to colmap scene main folder."""
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args
        self.use_ros = use_ros
            
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if use_ros:
            self.processor = KeyframeProcessor(path = args.source_path)
            scene_info = self.processor.create_scene_info()                        
        else: 
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp, use_ros)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
            else:
                assert False, "Could not recognize scene type!"
            
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def update_scene_info(self, event=None):
        """更新 scene_info 并刷新相关的相机信息。"""
        if not self.use_ros or not self.processor:
            print("ROS 未启用或 processor 未初始化。")
            return
        
        new_scene_info = self.processor.create_scene_info()
        if not new_scene_info:
            print("Failed to update scene_info.")
            return

        # 只更新相机信息，而不清空之前的相机数据
        self.cameras_extent = new_scene_info.nerf_normalization["radius"]
        
        # 更新训练和测试相机
        for resolution_scale in [1.0]:
            print("Reloading Training Cameras")
            self.train_cameras[resolution_scale].extend(cameraList_from_camInfos(new_scene_info.train_cameras, resolution_scale, self.args, new_scene_info.is_nerf_synthetic, False))
            print("Reloading Test Cameras")
            self.test_cameras[resolution_scale].extend(cameraList_from_camInfos(new_scene_info.test_cameras, resolution_scale, self.args, new_scene_info.is_nerf_synthetic, True))

        # 更新点云数据，清空当前三维点信息，只添加新的
        self.gaussians.create_from_pcd(new_scene_info.point_cloud, new_scene_info.train_cameras, self.cameras_extent)

        print("scene_info updated successfully.")
        
def periodic_update(scene, interval):
    def update():
        scene.update_scene_info()
        print("场景信息更新完成。")
        threading.Timer(interval, update).start()
    update()
    
def main():
    
    parser = ArgumentParser(description="Test Scene Initialization")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # Parse existing arguments from ModelParams, OptimizationParams, PipelineParams
    args = parser.parse_args()
    
    # 直接在代码中修改 model_path
    args.model_path = "/home/wang/catkin_ws/src/wla-gaussian/tum-fg2-desk-orb"
    
    # Initialize ROS node before any ROS-related operations
    rospy.init_node('test_scene_init', anonymous=True)

    # Extract parameters
    dataset = lp.extract(args)
    optimization = op.extract(args)
    pipeline = pp.extract(args)
    
    # Initialize GaussianModel with the extracted optimizer parameters
    gaussians = GaussianModel(dataset.sh_degree, optimization.optimizer_type)
    
    # Initialize Scene
    scene = Scene(args, gaussians=gaussians, use_ros=True)
    
    # 测试获取训练相机
    train_cameras = scene.getTrainCameras()
    print("训练相机:", train_cameras)

    # 测试获取测试相机
    test_cameras = scene.getTestCameras()
    print("测试相机:", test_cameras)

    # 测试更新场景信息
    scene.update_scene_info()
    print("场景信息更新完成。")
    
    # 设置 ROS 定时器每10秒调用一次update_scene_info
    timer = rospy.Timer(rospy.Duration(10), scene.update_scene_info)
    
    # 测试保存功能
    scene.save(iteration=1000)
    print("场景已保存。")
    
    # 保持主线程运行
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("停止定时更新。")
        timer.shutdown()
        sys.exit()

if __name__ == "__main__":
    main()