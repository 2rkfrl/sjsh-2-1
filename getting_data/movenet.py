from enum import Enum
import numpy as np
import tensorflow as tf


# KEYPOINT_GROUP * 2 + (- 1, 0)
class KeypointGroup(Enum):
    NOSE = 0
    EYE = 1
    EAR = 2
    SHOULDER = 3
    ELBOW = 4
    WRIST = 5
    HIP = 6
    KNEE = 7
    ANKLE = 8


class Keypoint(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

KEYPOINT_INDEX_TO_NAME = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

KEYPOINT_NAME_TO_INDEX = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


def filter_not(list_keypoints):
    """Returns an index list of keypoints that weren't passed to list_keypoints

    Args:
        list_keypoints: List of keypoint names

    Returns:
        Index list of keypoints that isn't contained in list_keypoints
    """

    index_list = np.array(range(17))
    index_list = np.delete(
        index_list,
        list(map(lambda keypoint_name: KEYPOINT_NAME_TO_INDEX[keypoint_name], list_keypoints)))

    return index_list


def load_model():
    """Loads Movenet Model"""
    interpreter = tf.lite.Interpreter(model_path="/Users/songsumin/Desktop/adjusted/movenet_thunder.tflite")
    interpreter.allocate_tensors()
    return interpreter


def parse_keypoints(image: np.ndarray, interpreter):
    """Returns a list of keypoints parsed from the given image.
    Movenet model should be loaded before call of this function

    Args:
        image: Image to parse keypoints from

    Returns:
        A list of raw keypoints data, referred as "frame" in this project
    """

    # preprocess image before passing it to movenet

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    if keypoints.size == 0:
        raise ValueError("No keypoints detected")
    keypoints = keypoints[:,:,:,:2]
    keypoints = keypoints[0,0]
    return keypoints