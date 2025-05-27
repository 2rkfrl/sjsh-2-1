import math

import numpy as np

from movenet import KeypointGroup, Keypoint


def calculate_angle_coords(p1, p2, p3):
    v1 = np.abs(p1 - p2)  # vector 1: points from p1 to p2
    v2 = np.abs(p2 - p3)  # vector 1: points from p1 to p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    if angle is None:
        return 0
    return np.degrees(angle)


def calculate_angle(keypoints, keypoint1: Keypoint, keypoint2: Keypoint, keypoint3: Keypoint):
    return calculate_angle_coords(keypoints[keypoint1.value], keypoints[keypoint2.value], keypoints[keypoint3.value])

def calculate_average_velocity(previous_keypoints, current_keypoints, keypoints_front_full):
    if previous_keypoints is None:
        return 0
    reference_length = calculate_y_distance(keypoints_front_full, KeypointGroup.NOSE , KeypointGroup.HIP)

    displacement = (current_keypoints - previous_keypoints) / reference_length
    displacement_x = np.mean(displacement[1])
    displacement_y = np.mean(displacement[0])

    return displacement_x, displacement_y


def calculate_average_speed(previous_keypoints, current_keypoints, keypoints_front_full):
    """Calculates displacement between ticks"""
    vx, vy = calculate_average_velocity(previous_keypoints, current_keypoints, keypoints_front_full)

    return math.sqrt(vx ** 2 + vy ** 2)


def get_center_coords(keypoints, keypoints_group: KeypointGroup):
    if keypoints_group.value == 0:
        return keypoints[0][0:2]
    keypoints_center = [(keypoints[keypoints_group.value * 2 -1][1] + keypoints[keypoints_group.value * 2][1]) /2,(keypoints[keypoints_group.value * 2 -1][0] + keypoints[keypoints_group.value * 2][0]) /2]
    return np.array(keypoints_center)


def calculate_x_distance(keypoints, keypoint_group1: KeypointGroup, keypoint_group2: KeypointGroup):
    group1_center = get_center_coords(keypoints, keypoint_group1)
    group2_center = get_center_coords(keypoints, keypoint_group2)

    x_distance = math.fabs(group1_center[0] - group2_center[0])
    return x_distance


def calculate_y_distance(keypoints, keypoint_group1: KeypointGroup, keypoint_group2: KeypointGroup):
    group1_center = get_center_coords(keypoints, keypoint_group1)
    group2_center = get_center_coords(keypoints, keypoint_group2)

    y_distance = math.fabs(group1_center[1] - group2_center[1])
    return y_distance


def calculate_distance(keypoints, keypoint_group1: KeypointGroup, keypoint_group2: KeypointGroup):
    group1_center = get_center_coords(keypoints, keypoint_group1)
    group2_center = get_center_coords(keypoints, keypoint_group2)

    distance = math.sqrt((group1_center[0] - group2_center[0]) ** 2 + (group1_center[1] - group2_center[1]) ** 2)
    return distance
def calculate_coorddistance_samegroup(keypoints, keypoints_group : KeypointGroup, keypoints_coord):
    a = keypoints[keypoints_group*2 -1][keypoints_coord]
    b = keypoints[keypoints_group*2][keypoints_coord]
    return math.fabs(a-b)
def calculate_distance_samegroup(keypoints, keypoints_group: KeypointGroup):
    x = calculate_coorddistance_samegroup(keypoints, keypoints_group, 1)
    y = calculate_coorddistance_samegroup(keypoints, keypoints_group, 0)
    return math.sqrt(x**2 + y**2)
class RBContext:
    def __init__(self):
        self.settings = {}
