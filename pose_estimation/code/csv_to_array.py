import argparse
import os
import json
import numpy as np
import pandas as pd
import mediapipe as mp

TESTING_PHASE = True


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
FACE_CONNECTIONS = generate_connections_list(mp.solutions.holistic.FACEMESH_CONTOURS)


def get_landmarks_names_from_df(df: pd.DataFrame):
    landmarks_names = [cn[:(len(cn) - 2)] for cn in df.columns if cn[-2:] == '.x']
    return landmarks_names


def get_connection_name(landmark_name, connections_from, landmarks_enum=None):
    if landmarks_enum is None:
        landmark_index = int(landmark_name)
        if landmark_index >= len(connections_from):
            return None
        connected_from_index = connections_from[landmark_index]
        if connected_from_index is None:
            return None
        return str(connections_from[landmark_index])

    landmark_index = landmarks_enum[landmark_name].value
    if landmark_index >= len(connections_from):
        return None
    connected_from_index = connections_from[landmark_index]
    if connected_from_index is None:
        return None
    return landmarks_enum(connected_from_index).name


def remove_column_prefix(df: pd.DataFrame, prefix: str):
    df.rename(columns=lambda s: s[len(prefix):], inplace=True)
    return df


def add_polar_coordinates(landmarks_df: pd.DataFrame, connections_from, landmarks_enum=None):
    extended_landmarks_df = pd.DataFrame()
    landmarks_names = get_landmarks_names_from_df(landmarks_df)
    for landmark in landmarks_names:
        connected_landmark = get_connection_name(landmark, connections_from, landmarks_enum)
        if connected_landmark is not None:
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
            extended_landmarks_df[landmark + '.spherical'] = unavailability_from
            extended_landmarks_df[landmark + '.r'] = r
            extended_landmarks_df[landmark + '.theta'] = np.arccos(cos_theta)
            extended_landmarks_df[landmark + '.phi'] = np.arccos(cos_phi)

    return extended_landmarks_df


def impute_landmark_coordinates(landmarks_df: pd.DataFrame):
    landmarks_names = get_landmarks_names_from_df(landmarks_df)
    for landmark in landmarks_names:
        x = landmarks_df[landmark + '.x']
        y = landmarks_df[landmark + '.y']
        z = landmarks_df[landmark + '.z']

        landmarks_df[landmark + '.v'] = np.where(x.isna() | y.isna() | z.isna(), 1,  0)
        landmarks_df[landmark + '.x'].interpolate(method='linear',
                                                  axis="index",
                                                  limit_direction='both',
                                                  inplace=True)
        landmarks_df[landmark + '.y'].interpolate(method='linear',
                                                  axis="index",
                                                  limit_direction='both',
                                                  inplace=True)
        landmarks_df[landmark + '.z'].interpolate(method='linear',
                                                  axis="index",
                                                  limit_direction='both',
                                                  inplace=True)

    # landmarks_df.interpolate(method='linear',
    #                          axis="index",
    #                          limit_direction='both',
    #                          inplace=True)

    return landmarks_df


def back_to_pixels(landmarks_df, width, height):
    landmarks_names = get_landmarks_names_from_df(landmarks_df)
    for landmark in landmarks_names:
        landmarks_df[landmark + '.x'] = width * landmarks_df[landmark + '.x']
        landmarks_df[landmark + '.y'] = height * landmarks_df[landmark + '.y']
        landmarks_df[landmark + '.z'] = width * landmarks_df[landmark + '.z']

    return landmarks_df


def normalize(landmarks_df, norm_factor, suffixes=['.x', '.y', '.z', '.r']):
    landmarks_names = get_landmarks_names_from_df(landmarks_df)
    for landmark in landmarks_names:
        for suffix in suffixes:
            column_name = landmark + suffix
            landmarks_df[column_name] = landmarks_df[column_name] * norm_factor

    return landmarks_df


def process_single_json_file(json_file,
                             output_directory):

    with open(json_file, 'r') as f:
        properties = json.load(f)

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

    # In order to preserve angles and original proportions:
    face_df = back_to_pixels(face_df, properties['Width'], properties['Height'])
    pose_df = back_to_pixels(pose_df, properties['Width'], properties['Height'])
    left_hand_df = back_to_pixels(left_hand_df, properties['Width'], properties['Height'])
    right_hand_df = back_to_pixels(right_hand_df, properties['Width'], properties['Height'])

    face_df = impute_landmark_coordinates(face_df)
    pose_df = impute_landmark_coordinates(pose_df)
    left_hand_df = impute_landmark_coordinates(left_hand_df)
    right_hand_df = impute_landmark_coordinates(right_hand_df)

    face_extended = add_polar_coordinates(face_df, FACE_CONNECTIONS)
    pose_extended = add_polar_coordinates(pose_df, POSE_CONNECTIONS, mp.solutions.holistic.PoseLandmark)
    left_hand_extended = add_polar_coordinates(left_hand_df, HAND_CONNECTIONS, mp.solutions.holistic.HandLandmark)
    right_hand_extended = add_polar_coordinates(right_hand_df, HAND_CONNECTIONS, mp.solutions.holistic.HandLandmark)

    normalizing_factor = 1 / pose_extended[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER.name + ".r"]
    face_extended = normalize(face_extended, normalizing_factor)
    pose_extended = normalize(pose_extended, normalizing_factor)
    left_hand_extended = normalize(left_hand_extended, normalizing_factor)
    right_hand_extended = normalize(right_hand_extended, normalizing_factor)

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

        try:
            process_single_json_file(fwd.file_with_path, output_directory)
        except Exception as e:
            print('Processing error for file ' + fwd.file_with_path)
            print(e)

