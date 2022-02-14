import argparse
import json
import os
import numpy as np
import pandas as pd
import cv2 as cv
import mediapipe as mp
from tqdm import tqdm


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


def draw_face_landmarks(image, landmarks, draw_irises=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    if draw_irises:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_holistic.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    return image


def draw_pose_landmarks(image, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    return image


def draw_hand_landmarks(image, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

    return image


def video_to_landmarks(video, generate_new_video=True):
    face_columns_names = get_face_columns_names()
    pose_columns_names = get_pose_columns_names()
    left_hand_columns_names = get_left_hand_columns_names()
    right_hand_columns_names = get_right_hand_columns_names()

    face_nans_row = [np.nan for _ in range(len(face_columns_names))]
    pose_nans_row = [np.nan for _ in range(len(pose_columns_names))]
    left_hand_nans_row = [np.nan for _ in range(len(left_hand_columns_names))]
    right_hand_nans_row = [np.nan for _ in range(len(right_hand_columns_names))]

    mp_holistic = mp.solutions.holistic

    video_n_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    new_video = []
    face_frames_ls = []
    pose_frames_ls = []
    left_hand_frames_ls = []
    right_hand_frames_ls = []

    counter = 1

    with mp_holistic.Holistic(model_complexity=1,
                              smooth_landmarks=True,
                              refine_face_landmarks=False,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        success, frame = video.read()

        progress_bar = tqdm(total=video_n_frames,
                            unit="frames")
        while success:
            face_row = face_nans_row
            pose_row = pose_nans_row
            left_hand_row = left_hand_nans_row
            right_hand_row = right_hand_nans_row

            if generate_new_video:
                annotated_frame = frame.copy()
                annotated_frame.flags.writeable = True

            frame.flags.writeable = False
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = holistic.process(frame)

            # Extract face
            try:
                face_landmarks = results.face_landmarks
                if face_landmarks:
                    face_row = get_landmarks_coordinates_row(face_landmarks.landmark)
                    if generate_new_video:
                        annotated_frame = draw_face_landmarks(annotated_frame, face_landmarks)
            except Exception as err:
                print(f"\nFACE: Unexpected {err=}, {type(err)=}")
            face_frames_ls.append(face_row)

            # Extract pose
            try:
                pose_landmarks = results.pose_landmarks
                if pose_landmarks:
                    pose_row = get_landmarks_coordinates_row(pose_landmarks.landmark)
                    if generate_new_video:
                        annotated_frame = draw_pose_landmarks(annotated_frame, pose_landmarks)
            except Exception as err:
                print(f"\nPOSE: Unexpected {err=}, {type(err)=}")

            pose_frames_ls.append(pose_row)

            # Extract left hand
            try:
                left_hand_landmarks = results.left_hand_landmarks
                if left_hand_landmarks:
                    left_hand_row = get_landmarks_coordinates_row(left_hand_landmarks.landmark)
                    if generate_new_video:
                        annotated_frame = draw_hand_landmarks(annotated_frame, left_hand_landmarks)
            except Exception as err:
                print(f"\nLEFT HAND: Unexpected {err=}, {type(err)=}")

            left_hand_frames_ls.append(left_hand_row)

            # Extract right hand
            try:
                right_hand_landmarks = results.right_hand_landmarks
                if right_hand_landmarks:
                    right_hand_row = get_landmarks_coordinates_row(right_hand_landmarks.landmark)
                    if generate_new_video:
                        annotated_frame = draw_hand_landmarks(annotated_frame, right_hand_landmarks)
            except Exception as err:
                print(f"\nRIGHT HAND: Unexpected {err=}, {type(err)=}")

            right_hand_frames_ls.append(right_hand_row)

            if generate_new_video:
                new_video.append(annotated_frame)

            progress_bar.update()
            counter += 1
            success, frame = video.read()

        progress_bar.close()

    video_face = pd.DataFrame(face_frames_ls, columns=face_columns_names)
    video_pose = pd.DataFrame(pose_frames_ls, columns=pose_columns_names)
    video_left_hand = pd.DataFrame(left_hand_frames_ls, columns=left_hand_columns_names)
    video_right_hand = pd.DataFrame(right_hand_frames_ls, columns=right_hand_columns_names)

    return video_face, video_pose, video_left_hand, video_right_hand, new_video


def process_single_video_file(video_file, output_directory, save_annotated_video=True):
    output_video_codec = 'mp4v'
    resize_factor = 1
    output_video_suffix = '_annotated.avi'

    video_file_basename = os.path.basename(video_file)
    video_file_main_name = os.path.splitext(video_file_basename)[0]

    print('Processing file ' + video_file)

    video = cv.VideoCapture(video_file)

    h = int(video.get(cv.CAP_PROP_FOURCC))
    video_codec = chr(h & 0xff) + chr((h >> 8) & 0xff) + chr((h >> 16) & 0xff) + chr((h >> 24) & 0xff)
    video_n_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv.CAP_PROP_FPS)
    video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    if video_n_frames is None or video_n_frames == 0:
        print('WARNING: The file ' + video_file + ' does not look like video --> ignored')
        return

    print('\tCodec:    ' + str(video_codec))
    print('\tFPS:    ' + str(video_fps))
    print('\tFrames: ' + str(video_n_frames))
    print('\tSize:   ' + str(video_width) + 'x' + str(video_height))

    video_properties = {'File': os.path.abspath(video_file),
                        'Codec': video_codec,
                        'Frames': video_n_frames,
                        'FPS': video_fps,
                        'Width': video_width,
                        'Height': video_height}

    video_face_df, \
        video_pose_df, \
        video_left_hand_df, \
        video_right_hand_df, \
        new_video = video_to_landmarks(video, save_annotated_video)

    video.release()

    os.makedirs(output_directory, exist_ok=True)

    if save_annotated_video:
        output_video_path = os.path.join(output_directory,
                                         video_file_main_name + output_video_suffix)

        output_video_width = round(resize_factor * video_width)
        output_video_height = round(resize_factor * video_height)
        fourcc = cv.VideoWriter_fourcc(*output_video_codec)
        output_video = cv.VideoWriter(output_video_path,
                                      fourcc,
                                      video_fps,
                                      (output_video_width, output_video_height))
        for frame in new_video:
            if resize_factor != 1:
                frame = cv.resize(frame,
                                  (output_video_width, output_video_height),
                                  cv.INTER_AREA)
            output_video.write(frame)
        output_video.release()

    dfs = [video_face_df, video_pose_df, video_left_hand_df, video_right_hand_df]
    dfs_filenames = [video_file_main_name + "_face.csv",
                     video_file_main_name + "_pose.csv",
                     video_file_main_name + "_left_hand.csv",
                     video_file_main_name + "_right_hand.csv"]

    properties_file_name = os.path.join(output_directory, video_file_main_name + "_properties.json")
    with open(properties_file_name, 'w') as prop_file:
        prop_file.write(json.dumps(video_properties))

    for i in range(len(dfs)):
        df = dfs[i]
        filename = dfs_filenames[i]
        path = os.path.join(output_directory, filename)
        df.to_csv(path)

    print('Landmarks saved to files:')
    print(dfs_filenames)


class FileWithDirectory:
    def __init__(self, d, f_name):
        self.directory = d
        self.file_with_path = os.path.join(d, f_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to save outputs")
    parser.add_argument("save", help="true / false flag if to generate output file")
    args = parser.parse_args()
    print(args.input)

    files_with_directories = []

    root_directory = ''
    output_root_directory = args.output

    if os.path.isdir(args.input):
        root_directory = args.input
        for directory, _, files in os.walk(root_directory, topdown=True):
            for file in files:
                files_with_directories.append(FileWithDirectory(directory, file))
    elif os.path.isfile(args.input):
        files_with_directories = [FileWithDirectory('.', args.input)]
    else:
        raise Exception("Incorrect input file name or directory!")

    save_annotated_video = args.save and ((args.save == 'true') or (args.save == 't'))

    for fwd in files_with_directories:
        subdirectory = fwd.directory[len(root_directory):]
        output_directory = os.path.join(output_root_directory, subdirectory)

        try:
            process_single_video_file(fwd.file_with_path, output_directory, save_annotated_video)
        except Exception as e:
            print('Processing error for file ' + fwd.file_with_path)
            print(e)

