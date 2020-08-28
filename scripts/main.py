#!/usr/bin/env python

import rosbag
import message_filters
import cv_bridge
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import rospy
import math
from cdcpd_shape_completion.read_bagfile.helper import imgs2pc
from sensor_msgs.msg import Image, CameraInfo
from shape_completion_training.voxelgrid.conversions import pointcloud_to_voxelgrid, to_2_5D
from shape_completion_training.model.model_runner import ModelRunner
from rviz_voxelgrid_visuals import conversions
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from skimage import measure
import tensorflow as tf

rgb_list = []
depth_list = []
camera_info_list = []
bridge = cv_bridge.CvBridge()

is_online = False
is_simulation = True

    # group_defaults = {
    #     "NormalizingAE": # NormalizingAE/August_11_02-26-33_c94a291391
    #         {
    #             'num_latent_layers': 200,
    #             'flow': 'Flow/August_09_14-36-09_1486a44186',
    #             'network': 'NormalizingAE',
    #             'use_flow_during_inference': False
    #         },
    #     "VAE": # VAE/August_11_22-48-49_c94a291391
    #         {
    #             'num_latent_layers': 200,
    #             'network': 'VAE'
    #         },
    #     "VAE_GAN": # VAE_GAN/August_12_20-32-01_c94a291391/
    #         {
    #             'num_latent_layers': 200,
    #             'network': 'VAE_GAN',
    #             'learning_rate': 0.0001,
    #             'discriminator_learning_rate': 0.00005,
    #         },
    #     "Flow": # Flow/August_09_14-36-09_1486a44186
    #         {
    #             'batch_size': 1500,
    #             'network': 'RealNVP',
    #             'dim': 24,
    #             'num_masked': 12,
    #             'learning_rate': 1e-5,
    #             'translation_pixel_range_x': 10,
    #             'translation_pixel_range_y': 10,
    #             'translation_pixel_range_z': 10,
    #         },3 
    #     "FlowYCB": # FlowYCB/August_09_14-38-16_1486a44186
    #         {
    #             'batch_size': 1500,
    #             'network': 'RealNVP',
    #             'dim': 24,
    #             'num_masked': 12,
    #             'learning_rate': 1e-5,
    #             'translation_pixel_range_x': 20,
    #             'translation_pixel_range_y': 10,
    #             'translation_pixel_range_z': 10,
    #             'dataset': 'ycb',
    #         },
    #     "3D_rec_gan": # 3D_rec_gan/August_13_20-31-41_c94a291391
    #         {
    #             'batch_size': 4,
    #             'dataset': 'shapenet',
    #             'network': '3D_rec_gan',
    #             "learning_rate": 0.0001,
    #             "gan_learning_rate": 0.00005,
    #             "num_latent_layers": 2000,
    #             "is_u_connected": True,
    #         },
    #     "NormalizingAE_YCB": # NormalizingAE_YCB/August_15_12-49-28_c94a291391
    #         {
    #             'num_latent_layers': 200,
    #             'flow': 'FlowYCB/August_09_14-38-16_1486a44186',
    #             'network': 'NormalizingAE',
    #             'use_flow_during_inference': False,
    #             'dataset': 'ycb',
    #             'translation_pixel_range_x': 15,
    #             'translation_pixel_range_y': 10,
    #             'translation_pixel_range_z': 10,
    #             'apply_slit_occlusion': True,
    #         },
    #     "VAE_YCB": # VAE_YCB/August_15_20-22-32_c94a291391
    #         {
    #             'num_latent_layers': 200,
    #             'network': 'VAE',
    #             'dataset': 'ycb',
    #             'apply_slit_occlusion': True,
    #             'translation_pixel_range_x': 15,
    #             'translation_pixel_range_y': 10,
    #             'translation_pixel_range_z': 10,
    #         },
    #     "VAE_GAN_YCB": # VAE_GAN_YCB/August_16_04-27-12_c94a291391
    #         {
    #             'num_latent_layers': 200,
    #             'network': 'VAE_GAN',
    #             'learning_rate': 0.0001,
    #             'discriminator_learning_rate': 0.00005,
    #             'dataset': 'ycb',
    #             'apply_slit_occlusion': True,
    #             'translation_pixel_range_x': 15,
    #             'translation_pixel_range_y': 10,
    #             'translation_pixel_range_z': 10,
    #         },
    #     "3D_rec_gan_YCB": # 3D_rec_gan_YCB/August_16_13-15-52_c94a291391
    #         {
    #             'batch_size': 4,
    #             'dataset': 'shapenet',
    #             'network': '3D_rec_gan',
    #             "learning_rate": 0.0001,
    #             "gan_learning_rate": 0.00005,
    #             "num_latent_layers": 2000,
    #             "is_u_connected": True,
    #             'dataset': 'ycb',
    #             'apply_slit_occlusion': True,
    #             'translation_pixel_range_x': 15,
    #             'translation_pixel_range_y': 10,
    #             'translation_pixel_range_z': 10,
    #         },
    #     "NormalizingAE_YCB_noise": # NormalizingAE_YCB_noise/August_17_04-17-26_c94a291391
    #         {
    #             'num_latent_layers': 200,
    #             'flow': 'FlowYCB/August_09_14-38-16_1486a44186',
    #             'network': 'NormalizingAE',
    #             'use_flow_during_inference': False,
    #             'dataset': 'ycb',
    #             'translation_pixel_range_x': 15,
    #             'translation_pixel_range_y': 10,
    #             'translation_pixel_range_z': 10,
    #             'apply_slit_occlusion': True,
    #             'apply_depth_sensor_noise': True,
    #         },
    #     "NormalizingAE_noise": # NormalizingAE_noise/August_17_14-29-51_c94a291391
    #         {
    #             'num_latent_layers': 200,
    #             'flow': 'Flow/August_09_14-36-09_1486a44186',
    #             'network': 'NormalizingAE',
    #             'use_flow_during_inference': False,
    #             'apply_depth_sensor_noise': True,
    #         },
    # }

trial_path = "/home/deformtrack/catkin_ws/src/probabilistic_shape_completion/shape_completion_training/trials/3D_rec_gan_YCB/August_16_13-15-52_c94a291391"

params = {
    'num_latent_layers': 200,
    # 'translation_pixel_range_x': 10,
    # 'translation_pixel_range_y': 10,
    # 'translation_pixel_range_z': 10,
    # 'use_final_unet_layer': False,
    # 'simulate_partial_completion': False,
    # 'simulate_random_partial_completion': False,
    'network': '3D_rec_gan_YCB',
    # 'network': 'VAE_GAN',
    # 'network': 'Augmented_VAE',
    # 'network': 'Conditional_VCNN',
    # 'network': 'NormalizingAE',
    'batch_size': 16,
    # 'learning_rate': 1e-3,
    # 'flow': 'Flow/August_09_14-36-09_1486a44186'
}

model_runner = ModelRunner(training=True, trial_path=trial_path, params=params)

def CameraInfo2np(info):
    K = np.zeros((3, 3))
    for row in range(3):
        for col in range(3):
            K[row, col] = info.K[row*3+col]
    return K

def numpy2multiarray(np_array):
    multiarray = Float32MultiArray()
    multiarray.layout.dim = [MultiArrayDimension('dim%d' % i,
                                                 np_array.shape[i],
                                                 np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)];
    multiarray.data = np_array.astype(float).reshape([1, -1])[0].tolist();
    return multiarray


def callback(rgb, depth, camera_info):
    rgb_list.append(bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough"))
    depth_list.append(bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough"))
    camera_info_list.append(CameraInfo2np(camera_info))

def color_segmentation():
    mask_list = []
    # blue segmentation
    hsv_lower = (220, 0.2, 0.2)
    hsv_upper = (240, 1.0, 1.0)
    # green segmentation
    # hsv_lower = (100, 0.2, 0.2)
    # hsv_upper = (140, 1.0, 1.0)
    # red segmentation
    # hsv_lower_1 = (0, 0.2, 0.2)
    # hsv_upper_1 = (20, 1.0, 1.0)
    # hsv_lower_2 = (340, 0.2, 0.2)
    # hsv_upper_2 = (360, 1.0, 1.0)
    for rgb in rgb_list:
        rgb_f = np.float32(rgb)
        rgb_f = rgb_f / 255.0
        hsv = cv.cvtColor(rgb_f, cv.COLOR_BGR2HSV)
        # print(hsv[0, 0])
		# red segmentation
        # mask1 = cv.inRange(hsv, hsv_lower_1, hsv_upper_1)
        # mask2 = cv.inRange(hsv, hsv_lower_2, hsv_upper_2)
        # mask = cv.bitwise_or(mask1, mask2)
        # other segmentation
        mask = cv.inRange(hsv, hsv_lower, hsv_upper)
        mask_list.append(mask)
    return mask_list

# def np2floatmsg():

def read_bagfile(rosbag_name):
    cam_topic = "camera_info"
    rgb_topic = "image_color_rect"
    depth_topic = "image_depth_rect"

    rgb_sub = message_filters.Subscriber(rgb_topic, Image)
    depth_sub = message_filters.Subscriber(depth_topic, Image)
    info_sub = message_filters.Subscriber(cam_topic, CameraInfo)

    bag = rosbag.Bag(rosbag_name, 'r')

    ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, info_sub], 10)
    ts.registerCallback(callback)

    for topic, msg, t in bag.read_messages(topics=[cam_topic, rgb_topic, depth_topic]):
        if topic == cam_topic:
            info_sub.signalMessage(msg);
        if topic == rgb_topic:
            rgb_sub.signalMessage(msg);
        if topic == depth_topic:
            depth_sub.signalMessage(msg);

    bag.close()

def append_bagfile(in_bag_name, out_bag_name, msgs):
    in_bag = rosbag.Bag(in_bag_name, 'r')
    out_bag = rosbag.Bag(out_bag_name, 'w')

    truth_topic = "groud_truth"
    gripper_velocity_topic = "gripper_velocity"
    gripper_info_topic = "gripper_info"
    gripper_config_topic = "gripper_config"
    cam_topic = "camera_info"
    rgb_topic = "image_color_rect"
    depth_topic = "image_depth_rect"
    verts_topic = "comp_vertices"
    faces_topic = "comp_faces"
    normals_topic = "comp_normals"

    for topic, msg, t in in_bag.read_messages(topics=[cam_topic, rgb_topic, depth_topic, truth_topic, gripper_config_topic, gripper_velocity_topic, gripper_info_topic]):
        out_bag.write(topic, msg, t)

    if is_simulation:
        for frame in range(250):
            t = rospy.Time(frame+1)
            out_bag.write(verts_topic, msgs[0], t)
            out_bag.write(faces_topic, msgs[1], t)
            out_bag.write(normals_topic, msgs[2], t)

    in_bag.close()
    out_bag.close()

def read_live():
    print("NOT IMPLEMENTED")

def main():
    origin = (-0.7, -0.7, 2.0) # "left most" point
    shape = (64, 64, 64) # how many grids there
    shape_inp = (1, 64, 64, 64, 1)
    scale = 1.4/64.0 # how large the grid is
    rospy.init_node("cdcpd_shape_completion")
    pub_incomp = rospy.Publisher('incomp', VoxelgridStamped, queue_size=1)
    pub_comp = rospy.Publisher('comp', VoxelgridStamped, queue_size=1)
    if not is_online:
        in_rosbag_name = "/home/deformtrack/catkin_ws/src/cdcpd_test_blender/dataset/rope_edge_cover_1.bag"
        out_rosbag_name = "/home/deformtrack/catkin_ws/src/cdcpd_test_blender/dataset/rope_edge_cover_1_comp.bag"
        read_bagfile(in_rosbag_name)
        mask_list = color_segmentation()
        pc = imgs2pc(rgb_list[0], depth_list[0], camera_info_list[0], mask_list[0])
        voxelgrid = pointcloud_to_voxelgrid(pc, scale=scale, origin=origin, shape=shape)
        known_occ = np.zeros(voxelgrid.shape)
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    if voxelgrid[x, y, z] == 1:
                        known_occ[63-z, y, x] = 1
        known_occ = voxelgrid
        voxel_inp = np.zeros(shape_inp)
        free_inp = np.zeros(shape_inp)
        known_free = 1.0 - known_occ
        voxel_inp[0, :, :, :, 0] = known_occ
        free_inp[0, :, :, :, 0] = known_free
        inp = {
            'known_occ': voxel_inp,
            'known_free': free_inp,
        }
        completion = model_runner.model(inp)
        # print("completion vg:")
        comp_np = (completion['predicted_occ'].numpy())[0, :, :, :, 0]
        comp_np[comp_np < 0.0001] = 0
        comp_np[comp_np >= 0.0001] = 1

        # print(comp_np.shape)
        # print(comp_np.sum())
        # print(comp_np)

        pub_incomp.publish(conversions.vox_to_voxelgrid_stamped(known_occ, # Numpy or Tensorflow
                                                                scale=scale, # Each voxel is a 1cm cube
                                                                frame_id='world', # In frame "world", same as rviz fixed frame
                                                                origin=origin)) # Bottom left corner
        pub_comp.publish(conversions.vox_to_voxelgrid_stamped(comp_np, # Numpy or Tensorflow
                                                              scale=scale, # Each voxel is a 1cm cube
                                                              frame_id='world', # In frame "world", same as rviz fixed frame
                                                              origin=origin)) # Bottom left corner

        verts, faces, normals, values = measure.marching_cubes_lewiner(comp_np, 0)
        verts = verts * scale + origin
        verts_msg = numpy2multiarray(np.transpose(verts))
        faces_msg = numpy2multiarray(np.transpose(faces))
        normals_msg = numpy2multiarray(np.transpose(normals))
        app_msgs = [verts_msg, faces_msg, normals_msg]
        append_bagfile(in_rosbag_name, out_rosbag_name, app_msgs)
        print("completion finished")
    else:
        read_live()

if __name__ == "__main__":
    main()
