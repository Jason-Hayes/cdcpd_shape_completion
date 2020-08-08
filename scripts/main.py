#!/usr/bin/env python

import rosbag
import message_filters
import cv_bridge
import cv2 as cv
import numpy as np
import rospy
import math
from cdcpd_shape_completion.read_bagfile import imgs2pc
from sensor_msgs.msg import Image, CameraInfo
from shape_completion_training.voxelgrid.conversions import pointcloud_to_voxelgrid
from shape_completion_training.model.model_runner import ModelRunner
from rviz_voxelgrid_visuals import conversions
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from skimage import measure

rgb_list = []
depth_list = []
camera_info_list = []
mask_list = []
bridge = cv_bridge.CvBridge()

pixel_len = 0.0000222222
unit_scaling = 0.001

is_online = False

trial_path = ""

model_runner = ModelRunner(training=True, trial=trial_path)

def CameraInfo2np(info):
    K = np.zeros((3, 3))
    for row in range(3):
        for col in range(3):
            K[row, col] = info.K[row*3+col]
    return K

def callback(rgb, depth, camera_info):
    rgb_list.append(bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough"))
    depth_list.append(bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough"))
    camera_info_list.append(CameraInfo2np(camera_info))

def color_segmentation():
    # blue segmentation
    # hsv_lower = (220, 0.2, 0.2)
    # hsv_upper = (240, 1.0, 1.0)
    # green segmentation
    # hsv_lower = (100, 0.2, 0.2)
    # hsv_upper = (140, 1.0, 1.0)
    # red segmentation
    hsv_lower_1 = (0, 0.2, 0.2)
    hsv_upper_1 = (20, 1.0, 1.0)
    hsv_lower_2 = (340, 0.2, 0.2)
    hsv_upper_2 = (360, 1.0, 1.0)
    for rgb in rgb_list:
        rgb_f = np.float32(rgb)
        rgb_f = rgb_f / 255.0
        hsv = cv.cvtColor(rgb_f, cv.COLOR_BGR2HSV)
        # print(hsv[0, 0])
        mask1 = cv.inRange(hsv, hsv_lower_1, hsv_upper_1)
        mask2 = cv.inRange(hsv, hsv_lower_2, hsv_upper_2)
        mask = cv.bitwise_or(mask1, mask2)
        mask_list.append(mask)

def np2floatmsg():

def read_bagfile():
    cam_topic = "camera_info"
    rgb_topic = "image_color_rect"
    depth_topic = "image_depth_rect"
    rosbag_name = "/home/deformtrack/catkin_ws/src/cdcpd_test_blender/dataset/edge_cover_4.bag"

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

def read_live():
    print("WARNING")

def main():
    origin = (-2.0, -1.0, 5.0) # "left most" point
    shape = (100, 100, 100) # how many grids there
    scale = 0.1 # how large the grid is
    rospy.init_node("cdcpd_shape_completion")
    # pub = rospy.Publisher('grid', VoxelgridStamped, queue_size=1)
    if not is_online:
        read_bagfile()
        color_segmentation()
        for i in range(len(rgb_list)):
            pc = imgs2pc(rgb_list[i], depth_list[i], camera_info_list[i], mask_list[i])
            voxelgrid = pointcloud_to_voxelgrid(pc, scale=scale, origin=origin, shape=shape)
            completion = model_runner.model(voxelgrid)
            verts, faces, normals, values = measure.marching_cubes(completion['predicted_occ'], 0)
            # pub.publish(conversions.vox_to_voxelgrid_stamped(voxelgrid, # Numpy or Tensorflow
            #                                          scale=scale, # Each voxel is a 1cm cube
            #                                          frame_id='world', # In frame "world", same as rviz fixed frame
            #                                          origin=origin)) # Bottom left corner
    else:
        read_live()

if __name__ == "__main__":
    main()
