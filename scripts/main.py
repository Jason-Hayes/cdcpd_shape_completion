#!/usr/bin/env python

import rosbag
import message_filters
from cdcpd_shape_completion.read_bagfile import imgs2pc
from sensor_msgs.msg import Image, CameraInfo

rgb_list = [];
depth_list = [];
camera_info_list = [];

def callback(rgb, depth, camera_info):
    rgb_list.append(rgb);
    depth_list.append(depth);
    camera_info_list.append(camera_info);


def main():
    cam_topic = "camera_info"
    rgb_topic = "image_color_rect"
    depth_topic = "image_depth_rect"
    rosbag_name = "/home/deformtrack/catkin_ws/src/cdcpd_test_blender/dataset.bag"

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

if __name__ == "__main__":
    main()
