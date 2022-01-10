import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv
import mediapipe as mp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to save outputs")
    args = parser.parse_args()
    print(args.input)

    mp_holistic = mp.solutions.holistic
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    pose_landmarks_names = [name.name for name in mp_holistic.PoseLandmark]
    hand_landmarks_names = [name.name for name in mp_holistic.HandLandmark]

    holistic = mp_holistic.Holistic()


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
        output = get_landmarks_columns_names(pose_landmarks_names, prefix='Pose')
        return output

    def get_left_hand_columns_names():
        output = get_landmarks_columns_names(hand_landmarks_names, prefix='Left_hand')
        return output

    def get_right_hand_columns_names():
        output = get_landmarks_columns_names(hand_landmarks_names, prefix='Right_hand')
        return output

    def get_face_columns_names():
        output = get_landmarks_columns_names(range(468), prefix='Face')
        return output


    video_path = args.input
    video = cv.VideoCapture(video_path)

    success, frame = video.read()
    counter = 0

    face_columns_names = get_face_columns_names()
    pose_columns_names = get_pose_columns_names()
    left_hand_columns_names = get_left_hand_columns_names()
    right_hand_columns_names = get_right_hand_columns_names()

    face_nan_df = pd.DataFrame([[np.nan for _ in range(len(face_columns_names))]],
                               columns=face_columns_names)
    pose_nan_df = pd.DataFrame([[np.nan for _ in range(len(pose_columns_names))]],
                               columns=pose_columns_names)
    left_hand_nan_df = pd.DataFrame([[np.nan for _ in range(len(left_hand_columns_names))]],
                                    columns=left_hand_columns_names)
    right_hand_nan_df = pd.DataFrame([[np.nan for _ in range(len(right_hand_columns_names))]],
                                     columns=right_hand_columns_names)

    video_face_df = pd.DataFrame(columns=face_columns_names)
    video_pose_df = pd.DataFrame(columns=pose_columns_names)
    video_left_hand_df = pd.DataFrame(columns=left_hand_columns_names)
    video_right_hand_df = pd.DataFrame(columns=right_hand_columns_names)


    while success:
        results = holistic.process(frame)
        # Extract face
        try:
            face_landmarks = results.face_landmarks.landmark
            
            face_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face_landmarks]).flatten()
            face_row = np.expand_dims(face_row, 0)
            
            video_face_df = video_face_df.append(pd.DataFrame(face_row, columns=face_columns_names), ignore_index=True)
        except:
            video_face_df = video_face_df.append(face_nan_df, ignore_index=True)
            
        
        # Extract pose
        try:
            pose_landmarks = results.pose_landmarks.landmark
            
            pose_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten()
            pose_row = np.expand_dims(pose_row, 0)

            video_pose_df = video_pose_df.append(pd.DataFrame(pose_row, columns=pose_columns_names), ignore_index=True)
        except:
            video_pose_df = video_pose_df.append(pose_nan_df, ignore_index=True)
            
        
        # Extract left hand
        try:
            left_hand_landmarks = results.left_hand_landmarks.landmark
            
            left_hand_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand_landmarks]).flatten()
            left_hand_row = np.expand_dims(left_hand_row, 0)
            
            video_left_hand_df = video_left_hand_df.append(pd.DataFrame(left_hand_row, columns=left_hand_columns_names), ignore_index=True)
        except:
            video_left_hand_df = video_left_hand_df.append(left_hand_nan_df)
            
        
        # Extract right hand
        try:
            right_hand_landmarks = results.right_hand_landmarks.landmark
            
            right_hand_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand_landmarks]).flatten()
            right_hand_row = np.expand_dims(right_hand_row, 0)
            
            video_right_hand_df = video_right_hand_df.append(pd.DataFrame(right_hand_row, columns=right_hand_columns_names), ignore_index=True)
        except:
            video_right_hand_df = video_right_hand_df.append(right_hand_nan_df)
            
        success, frame = video.read()
        counter += 1

    dfs = [video_face_df, video_pose_df, video_left_hand_df, video_right_hand_df]
    dfs_filenames = ["face.csv", "pose.csv", "left_hand.csv", "right_hand.csv"]

    for i in range(len(dfs)):
        df = dfs[i]
        filename = dfs_filenames[i]
        path = args.output+filename
        print(path)
        df.to_csv(path)