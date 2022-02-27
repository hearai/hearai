import argparse
import enum
import json
import math
import os
import numpy as np
import pandas as pd
import cv2 as cv
import mediapipe as mp
from tqdm import tqdm

TESTING_PHASE = True

POSE_LANDMARKS_NAMES = [name.name for name in mp.solutions.holistic.PoseLandmark]
HAND_LANDMARKS_NAMES = [name.name for name in mp.solutions.holistic.HandLandmark]


def generate_connections_list(connections):
    connections_list = []

    for first, second in connections:
        required_size = max(first, second) + 1
        if len(connections_list) < required_size:
            for i in range(len(connections_list), required_size):
                connections_list.append(None)

        to_first = connections_list[first]
        to_second = connections_list[second]

        if to_first is None:
            connections_list[first] = second
        else:
            connections_list[first] = min(to_first, second)
        if to_second is None:
            connections_list[second] = first
        else:
            connections_list[second] = min(to_second, first)

    return connections_list


POSE_CONNECTIONS = generate_connections_list(mp.solutions.holistic.POSE_CONNECTIONS)
HAND_CONNECTIONS = generate_connections_list(mp.solutions.holistic.HAND_CONNECTIONS)
FACE_CONNECTIONS = generate_connections_list(mp.solutions.holistic.FACEMESH_TESSELATION)


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


def get_landmarks_names_from_df(df: pd.DataFrame):
    landmarks_names = [cn[:(len(cn) - 2)] for cn in df.columns if cn[-2:] == '.x']
    return landmarks_names


def get_connection_name(landmark_name, connections_from, landmarks_enum=None):
    if landmarks_enum is None:
        return str(connections_from[int(landmark_name)])

    landmark_index = landmarks_enum[landmark_name].value
    connected_from_index = connections_from[landmark_index]
    return landmarks_enum(connected_from_index).name


def remove_column_prefix(df: pd.DataFrame, prefix: str):
    df.rename(columns=lambda s: s[len(prefix):], inplace=True)
    return df


def add_polar_coordinates(landmarks_df: pd.DataFrame, connections_from, landmarks_enum=None):
    extended_landmarks_df = pd.DataFrame()
    landmarks_names = get_landmarks_names_from_df(landmarks_df)
    for landmark in landmarks_names:
        connected_landmark = get_connection_name(landmark, connections_from, landmarks_enum)
        if connected_landmark is not None and connected_landmark != 'None':
            x = landmarks_df[landmark + '.x']
            y = landmarks_df[landmark + '.y']
            z = landmarks_df[landmark + '.z']
            v = landmarks_df[landmark + '.v']

            x_from = landmarks_df[connected_landmark + '.x']
            y_from = landmarks_df[connected_landmark + '.y']
            z_from = landmarks_df[connected_landmark + '.z']
            unavailability_from = landmarks_df[connected_landmark + '.v']

            delta_x = x - x_from
            delta_y = y - y_from
            delta_z = z - z_from

            r = np.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
            cos_theta = delta_z / r
            cos_phi = delta_x / (r * np.sqrt(1 - cos_theta * cos_theta))

            extended_landmarks_df[landmark + '.x'] = x
            extended_landmarks_df[landmark + '.y'] = y
            extended_landmarks_df[landmark + '.z'] = z
            extended_landmarks_df[landmark + '.v'] = v
            extended_landmarks_df[landmark + ".spherical"] = unavailability_from
            extended_landmarks_df[landmark + ".r"] = r
            extended_landmarks_df[landmark + ".cos_theta"] = cos_theta
            extended_landmarks_df[landmark + ".cos_phi"] = cos_phi

    return extended_landmarks_df


def process_single_json_file(json_file,
                             output_directory):

    file_name = json_file[0:(len(json_file) - 16)]

    file_main_name = os.path.basename(file_name)

    face_df = pd.read_csv(file_name + '_face.csv').iloc[:, 1:]
    pose_df = pd.read_csv(file_name + '_pose.csv').iloc[:, 1:]
    left_hand_df = pd.read_csv(file_name + '_left_hand.csv').iloc[:, 1:]
    right_hand_df = pd.read_csv(file_name + '_right_hand.csv').iloc[:, 1:]

    face_df = remove_column_prefix(face_df, 'Face.')
    pose_df = remove_column_prefix(pose_df, 'Pose.')
    left_hand_df = remove_column_prefix(left_hand_df, 'Left_hand.')
    right_hand_df = remove_column_prefix(right_hand_df, 'Right_hand.')

    face_extended = add_polar_coordinates(face_df, FACE_CONNECTIONS)
    pose_extended = add_polar_coordinates(pose_df, POSE_CONNECTIONS, mp.solutions.holistic.PoseLandmark)
    left_hand_extended = add_polar_coordinates(left_hand_df, HAND_CONNECTIONS, mp.solutions.holistic.HandLandmark)
    right_hand_extended = add_polar_coordinates(right_hand_df, HAND_CONNECTIONS, mp.solutions.holistic.HandLandmark)

    os.makedirs(output_directory, exist_ok=True)

    if TESTING_PHASE:
        dfs = [face_extended, pose_extended, left_hand_extended, right_hand_extended]
        dfs_filenames = [file_main_name + "_face_extended.csv",
                         file_main_name + "_pose_extended.csv",
                         file_main_name + "_left_hand_extended.csv",
                         file_main_name + "_right_hand_extended.csv"]
        for i in range(len(dfs)):
            df = dfs[i]
            filename = dfs_filenames[i]
            path = os.path.join(output_directory, filename)
            df.to_csv(path)

    face_array = face_extended.to_numpy()
    pose_array = pose_extended.to_numpy()
    left_hand_array = left_hand_extended.to_numpy()
    right_hand_array = right_hand_extended.to_numpy()

    output_file = os.path.join(output_directory, file_main_name + ".npz")

    np.savez_compressed(output_file,
                        face=face_array,
                        pose=pose_array,
                        left_hand=left_hand_array,
                        right_hand=right_hand_array)

    print('Arrays with landmarks saved to file ' + output_file)


class FileWithDirectory:
    def __init__(self, d, f_name):
        self.directory = d
        self.file_with_path = os.path.join(d, f_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to save outputs")
    args = parser.parse_args()

    files_with_directories = []

    root_directory = ''
    output_root_directory = args.output

    if os.path.isdir(args.input):
        root_directory = args.input
        for directory, _, files in os.walk(root_directory, topdown=True):
            for file in files:
                if file[-16:] == '_properties.json':
                    files_with_directories.append(FileWithDirectory(directory, file))
    elif os.path.isfile(args.input):
        files_with_directories = [FileWithDirectory('.', args.input)]
    else:
        raise Exception("Incorrect input file name or directory!")

    for fwd in files_with_directories:
        subdirectory = fwd.directory[len(root_directory):]
        output_directory = os.path.join(output_root_directory, subdirectory)

        # try:
        process_single_json_file(fwd.file_with_path, output_directory)
        # except Exception as e:
        #     print('Processing error for file ' + fwd.file_with_path)
        #     print(e)

