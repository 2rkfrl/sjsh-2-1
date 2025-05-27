from movenet import *
import cv2
import csv
from util import calculate_average_speed
from util import get_center_coords

def preprocess_image(image):
    input_size = 256
    processed_image = np.array(image)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    processed_image = cv2.resize(processed_image, (input_size, input_size))
    processed_image = processed_image.astype(np.uint8)
    processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image

def labeling():
    previous_keypoints = None
    cap = cv2.VideoCapture("standing.mp4")  
    interpreter = load_model()
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.flip(frame, 1)
        image = preprocess_image(frame)
        speed_list = []
        keypoints = parse_keypoints(image, interpreter)
        if previous_keypoints is not None:
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.NOSE), get_center_coords(keypoints, KeypointGroup.NOSE), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.EYE), get_center_coords(keypoints, KeypointGroup.EYE), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.EAR), get_center_coords(keypoints, KeypointGroup.EAR), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.SHOULDER), get_center_coords(keypoints, KeypointGroup.SHOULDER), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.ELBOW), get_center_coords(keypoints, KeypointGroup.ELBOW), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.WRIST), get_center_coords(keypoints, KeypointGroup.WRIST), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.HIP), get_center_coords(keypoints, KeypointGroup.HIP), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.KNEE), get_center_coords(keypoints, KeypointGroup.KNEE), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.ANKLE), get_center_coords(keypoints, KeypointGroup.ANKLE), keypoints))
            with open('data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([speed_list])
        previous_keypoints = keypoints
        processed_image = cv2.resize(frame, (256, 256))
        for i in range(17):
            x, y = keypoints[i][1]*256, keypoints[i][0]*256
            processed_image = cv2.circle(processed_image, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow('Recognition', processed_image)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def labeling():
    previous_keypoints = None
    cap = cv2.VideoCapture("standing.mp4")  
    interpreter = load_model()
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.flip(frame, 1)
        image = preprocess_image(frame)
        speed_list = []
        keypoints = parse_keypoints(image, interpreter)
        if previous_keypoints is not None:
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.NOSE), get_center_coords(keypoints, KeypointGroup.NOSE), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.EYE), get_center_coords(keypoints, KeypointGroup.EYE), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.EAR), get_center_coords(keypoints, KeypointGroup.EAR), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.SHOULDER), get_center_coords(keypoints, KeypointGroup.SHOULDER), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.ELBOW), get_center_coords(keypoints, KeypointGroup.ELBOW), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.WRIST), get_center_coords(keypoints, KeypointGroup.WRIST), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.HIP), get_center_coords(keypoints, KeypointGroup.HIP), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.KNEE), get_center_coords(keypoints, KeypointGroup.KNEE), keypoints))
            speed_list.append(calculate_average_speed(get_center_coords(previous_keypoints, KeypointGroup.ANKLE), get_center_coords(keypoints, KeypointGroup.ANKLE), keypoints))
            with open('data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([speed_list])
        previous_keypoints = keypoints
        processed_image = cv2.resize(frame, (256, 256))
        for i in range(17):
            x, y = keypoints[i][1]*256, keypoints[i][0]*256
            processed_image = cv2.circle(processed_image, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow('Recognition', processed_image)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()