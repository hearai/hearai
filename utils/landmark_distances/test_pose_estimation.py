import argparse
from operator import index
import os
import pandas as pd
import re
import sys
from os import listdir

import cv2
import numpy as np

import matplotlib.pyplot as plt
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calcl1(pt1, pt2):
    return np.linalg.norm(pt1-pt2, ord=1)


def calcl2(pt1, pt2):
    return np.linalg.norm(pt1-pt2, ord=2)


def get_landmark_xy(output, category, images, landmarks):
    if not output:
        output = {'category': [], 'norm': [], 'value': []}
    for idx, file in enumerate(images):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue
        pt1 = np.asarray([
            results.pose_landmarks.landmark[landmarks[0]].x * image_width,
            results.pose_landmarks.landmark[landmarks[0]].y * image_height,
        ])
        pt2 = np.asarray([
            results.pose_landmarks.landmark[landmarks[1]].x * image_width,
            results.pose_landmarks.landmark[landmarks[1]].y * image_height,
        ])
        output['category'].append(category)
        output['norm'].append('l1')
        output['value'].append(calcl1(pt1, pt2))

        output['category'].append(category)
        output['norm'].append('l2')
        output['value'].append(calcl2(pt1, pt2))

    return output


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="path to a directory containing images", default="")
    parser.add_argument("--output_dir", help="destination path for storing outputs", default=".")
    args = parser.parse_args()


    image_dir = args.image_dir
    CATEGORIES = [x[0] for x in os.walk(image_dir)]

    ext = ["png", "jpg", "jpeg", "bmp"]
    IMAGE_FILES = {}
    for c in CATEGORIES[1:]:
        IMAGE_FILES[c] = [os.path.join(c, f) for f in os.listdir(c) if f.endswith(tuple(ext))]

    LANDMARKS = {
        0: 'nose',
        1: 'left_eye_inner',
        2: 'left_eye',
        3: 'left_eye_outer',
        4: 'right_eye_inner',
        5: 'right_eye',
        6: 'right_eye_outer',
        7: 'left_ear',
        8: 'right_ear',
        9: 'mouth_left',
        10: 'mouth_right',
        11: 'left_shoulder',
        12: 'right_shoulder',
        13: 'left_elbow',
        14: 'right_elbow',
        15: 'left_wrist',
        16: 'right_wrist',
        17: 'left_pinky',
        18: 'right_pinky',
        19: 'left_index',
        20: 'right_index',
        21: 'left_thumb',
        22: 'right_thumb',
        23: 'left_hip',
        24: 'right_hip',
        25: 'left_knee',
        26: 'right_knee',
        27: 'left_ankle',
        28: 'right_ankle',
        29: 'left_heel',
        30: 'right_heel',
        31: 'left_foot_index',
        32: 'right_foot_index',
    }
    print(LANDMARKS)
    points = input('Please provide two points to use for calculations (comma separted indices): ')
    expression = '^[0-9]+,[0-9]+$'
    assert re.match(expression, points), f'Expected two points separated by comma'
    pt1 = int(points.split(',')[0])
    pt2 = int(points.split(',')[1])
    ladmarks = (pt1, pt2)

    output = {}

    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:

        for k, v in IMAGE_FILES.items():
            output = get_landmark_xy(output, k, v, ladmarks)

    df = pd.DataFrame.from_dict(output)
    norm = df.groupby(['category', 'norm']).agg([np.mean, np.std])
    print(norm)

    #l1 only
    dfl1 = df[df["norm"] == 'l1']
    fig, ax = plt.subplots()
    plt.title(f'L1 distance between {LANDMARKS[pt1]} and {LANDMARKS[pt2]} for different HamnNoSys category labels')
    average_distances = dfl1.groupby(['category']).agg([np.mean, np.std])['value']
    average_distances.plot.barh(color=['b'], ax=ax, xerr="std", width=0.1, alpha=0.6)
    fig.tight_layout()
    plt.show()

    #l2 only
    dfl2 = df[df["norm"] == 'l2']
    fig, ax = plt.subplots()
    plt.title(f'L2 distance between {LANDMARKS[pt1]} and {LANDMARKS[pt2]} for different HamnNoSys category labels')
    average_distances = dfl2.groupby(['category']).agg([np.mean, np.std])['value']
    average_distances.plot.barh(color=['b'], ax=ax, xerr="std", width=0.1, alpha=0.6)
    fig.tight_layout()
    plt.show()




