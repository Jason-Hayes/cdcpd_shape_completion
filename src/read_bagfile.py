#!/usr/bin/env python

import numpy as np
import rosbag
import rospy
import math

pixel_len = 0.0000222222
unit_scaling = 0.001

def imgs2pc(rgb, depth, intrinsics, mask):
    // rgb: cv2
    // depth: cv2
    // intrinsics: np.array
    // mask: cv2

    // return: np.array
    center_x = intrinsics[0, 2]
    center_y = intrinsics[1, 2]

    constant_x = 1.0 / (intrinsics[0, 0] * pixel_len)
    constant_y = 1.0 / (intrinsics[1, 1] * pixel_len)
    bad_point = math.nan

    pc = np.zeros((0, 0))

    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            d = depth[v, u]

            if not d == bad_point:
                x = (u - center_x) * pixel_len * depth * unit_scaling * constant_x
                y = (v - center_y) * pixel_len * depth * unit_scaling * constant_y
                z = depth * unit_scaling

                if mask[v, u]:
                    pc = np.concatenate((pc, np.array([[x, y, z]])))
    return pc
