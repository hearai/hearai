import argparse
import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2 as cv
import mediapipe as mp


POSE_LANDMARKS_NAMES = [name.name for name in mp.solutions.holistic.PoseLandmark]
HAND_LANDMARKS_NAMES = [name.name for name in mp.solutions.holistic.HandLandmark]


def get_landmarks_columns_names(landmarks_names, prefix='Landmark'):
    output = []

    for landmark_name in landmarks_names:
        x_name = prefix + "." + str(landmark_name) + ".x"
        y_name = prefix + "." + str(landmark_name) + ".y"
        z_name = prefix + "." + str(landmark_name) + ".z"
        v_name = prefix + "." + str(landmark_name) + ".v"

        output.append(x_name)
        output.append(y_name)
        output.append(z_name)
        output.append(v_name)

    return output


def get_pose_columns_names():
    output = get_landmarks_columns_names(POSE_LANDMARKS_NAMES, prefix='Pose')
    return output


def get_left_hand_columns_names():
    output = get_landmarks_columns_names(HAND_LANDMARKS_NAMES, prefix='Left_hand')
    return output


def get_right_hand_columns_names():
    output = get_landmarks_columns_names(HAND_LANDMARKS_NAMES, prefix='Right_hand')
    return output


def get_face_columns_names():
    output = get_landmarks_columns_names(range(468), prefix='Face')
    return output


def get_landmarks_coordinates_row(landmarks):
    row = []
    for landmark in landmarks:
        row.append(landmark.x)
        row.append(landmark.y)
        row.append(landmark.z)
        row.append(landmark.visibility)
    return row


def video_to_landmarks(video):
    video_n_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv.CAP_PROP_FPS)
    video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    print('\tFPS:    ' + str(video_fps))
    print('\tFrames: ' + str(video_n_frames))
    print('\tSize:   ' + str(video_width) + 'x' + str(video_height))

    face_frames_ls = []
    pose_frames_ls = []
    left_hand_frames_ls = []
    right_hand_frames_ls = []

    counter = 1
    start_time = time.time()

    with mp_holistic.Holistic(model_complexity=1,
                              smooth_landmarks=True,
                              refine_face_landmarks=False,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        success, frame = video.read()

        while success:
            results = holistic.process(frame)

            # Extract face
            try:
                face_landmarks = results.face_landmarks.landmark
                face_row = get_landmarks_coordinates_row(face_landmarks)
            except:
                face_row = face_nans_row

            face_frames_ls.append(face_row)

            # Extract pose
            try:
                pose_landmarks = results.pose_landmarks.landmark
                pose_row = get_landmarks_coordinates_row(pose_landmarks)
            except:
                pose_row = pose_nans_row

            pose_frames_ls.append(pose_row)

            # Extract left hand
            try:
                left_hand_landmarks = results.left_hand_landmarks.landmark
                left_hand_row = get_landmarks_coordinates_row(left_hand_landmarks)
            except:
                left_hand_row = left_hand_nans_row

            left_hand_frames_ls.append(left_hand_row)

            # Extract right hand
            try:
                right_hand_landmarks = results.right_hand_landmarks.landmark
                right_hand_row = get_landmarks_coordinates_row(right_hand_landmarks)
            except:
                right_hand_row = right_hand_nans_row

            right_hand_frames_ls.append(right_hand_row)

            if counter % 25 == 0:
                current_time = time.time()
                current_fps = counter / (current_time - start_time)
                print(str(counter) + '/' + str(video_n_frames) + ' average FPS = ' + str(current_fps))

            counter += 1
            success, frame = video.read()

    video_face = pd.DataFrame(face_frames_ls, columns=face_columns_names)
    video_pose = pd.DataFrame(pose_frames_ls, columns=pose_columns_names)
    video_left_hand = pd.DataFrame(left_hand_frames_ls, columns=left_hand_columns_names)
    video_right_hand = pd.DataFrame(right_hand_frames_ls, columns=right_hand_columns_names)

    final_time = time.time()
    final_fps = (counter - 1) / (final_time - start_time)
    print(str(counter - 1) + '/' + str(video_n_frames) + "Final FPS = " + str(final_fps))
    return video_face, video_pose, video_left_hand, video_right_hand


if __name__ == "__main__":
    face_columns_names = get_face_columns_names()
    pose_columns_names = get_pose_columns_names()
    left_hand_columns_names = get_left_hand_columns_names()
    right_hand_columns_names = get_right_hand_columns_names()

    face_nans_row = [np.nan for _ in range(len(face_columns_names))]
    pose_nans_row = [np.nan for _ in range(len(pose_columns_names))]
    left_hand_nans_row = [np.nan for _ in range(len(left_hand_columns_names))]
    right_hand_nans_row = [np.nan for _ in range(len(right_hand_columns_names))]

    mp_holistic = mp.solutions.holistic

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to save outputs")
    args = parser.parse_args()
    print(args.input)

    video_path = args.input
    video_file_basename = os.path.basename(video_path)
    video_file_main_name = os.path.splitext(video_file_basename)[0]

    video = cv.VideoCapture(video_path)

    print('Processing file ' + video_path)

    video_face_df, video_pose_df, video_left_hand_df, video_right_hand_df = video_to_landmarks(video)

    dfs = [video_face_df, video_pose_df, video_left_hand_df, video_right_hand_df]
    dfs_filenames = [video_file_main_name + "_face.csv",
                     video_file_main_name + "_pose.csv",
                     video_file_main_name + "_left_hand.csv",
                     video_file_main_name + "_right_hand.csv"]

    for i in range(len(dfs)):
        df = dfs[i]
        filename = dfs_filenames[i]
        path = args.output+filename
        print(path)
        df.to_csv(path)

    print('Output saved to files:')
    print(dfs_filenames)
