import argparse
import json
import os
import numpy as np
import pandas as pd
import cv2 as cv
import mediapipe as mp
from tqdm import tqdm

import csv_to_array as cta

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


def video_to_landmarks(video, generate_new_video=True, generate_segmentation_mask=False, smoothing=True):
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
    segmentation_frames_ls = []

    counter = 1

    with mp_holistic.Holistic(model_complexity=2,
                              smooth_landmarks=smoothing,
                              refine_face_landmarks=False,
                              enable_segmentation=generate_segmentation_mask,
                              min_detection_confidence=0.25,
                              min_tracking_confidence=0.25) as holistic:
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

            if generate_segmentation_mask:
                segmentation_frames_ls.append(results.segmentation_mask.tolist())

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

    return video_face, video_pose, video_left_hand, video_right_hand, new_video, segmentation_frames_ls


def save_segmentation_masks(segmentation_masks, name, output_dir):
    segmentation_file_name = os.path.join(output_dir, name + "_segmentations.json")
    with open(segmentation_file_name, 'w') as prop_file:
        prop_file.write(json.dumps(segmentation_masks))


def save_landmarks_csvs(dfs, name, output_dir):
    dfs_filenames = [name + "_face.csv",
                     name + "_pose.csv",
                     name + "_left_hand.csv",
                     name + "_right_hand.csv"]

    for i in range(len(dfs)):
        df = dfs[i]
        filename = dfs_filenames[i]
        path = os.path.join(output_dir, filename)
        df.to_csv(path)

    print('Landmarks saved to files:')
    print(dfs_filenames)


def process_single_video_file(video_file,
                              output_dir,
                              save_annotated=True,
                              generate_segmentation_mask=False):
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
        new_video,\
        segmentation_masks = video_to_landmarks(video, save_annotated, generate_segmentation_mask)

    video.release()

    os.makedirs(output_dir, exist_ok=True)

    if save_annotated:
        output_video_path = os.path.join(output_dir,
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

    if generate_segmentation_mask:
        save_segmentation_masks(segmentation_masks, video_file_main_name, output_dir)

    properties_file_name = os.path.join(output_dir, video_file_main_name + "_properties.json")
    with open(properties_file_name, 'w') as prop_file:
        prop_file.write(json.dumps(video_properties))

    save_landmarks_csvs([video_face_df, video_pose_df, video_left_hand_df, video_right_hand_df],
                        video_file_main_name,
                        output_dir)

    face_extended, \
        pose_extended, \
        left_hand_extended, \
        right_hand_extended = cta.extend_landmarks_dfs(video_face_df,
                                                       video_pose_df,
                                                       video_left_hand_df,
                                                       video_right_hand_df,
                                                       video_width,
                                                       video_height)

    np.savez_compressed(os.path.join(output_directory, video_file_main_name + ".npz"),
                        face=face_extended.to_numpy(),
                        pose=pose_extended.to_numpy(),
                        left_hand=left_hand_extended.to_numpy(),
                        right_hand=right_hand_extended.to_numpy())

    print('Arrays with landmarks saved')


def process_frames_in_directory(input_dir,
                                output_dir,
                                save_annotated=True,
                                generate_segmentation_mask=False):
    resize_factor = 1
    output_video_suffix = '_annotated.avi'

    frames_directory_name = os.path.basename(os.path.normpath(input_dir))

    print('Processing directory ' + frames_directory_name)

    frames = cv.VideoCapture(f'{input_dir}/{frames_directory_name}_%d.jpg', cv.CAP_IMAGES)
    frames_width = int(frames.get(cv.CAP_PROP_FRAME_WIDTH))
    frames_height = int(frames.get(cv.CAP_PROP_FRAME_HEIGHT))
    if frames_width == 0 or frames_height == 0:
        return

    face_df, \
        pose_df, \
        left_hand_df, \
        right_hand_df, \
        new_frames,\
        segmentation_masks = video_to_landmarks(frames, save_annotated, generate_segmentation_mask, smoothing=False)

    frames.release()

    os.makedirs(output_dir, exist_ok=True)

    if save_annotated:
        output_video_path = os.path.join(output_dir,
                                         frames_directory_name + output_video_suffix)

        #video_width, video_height = new_frames[0].size
        output_video_width = round(resize_factor * frames_width)
        output_video_height = round(resize_factor * frames_height)
        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        output_video = cv.VideoWriter(output_video_path,
                                      fourcc,
                                      1,
                                      (output_video_width, output_video_height))
        for frame in new_frames:
            if resize_factor != 1:
                frame = cv.resize(frame,
                                  (output_video_width, output_video_height),
                                  cv.INTER_AREA)
            output_video.write(frame)
        output_video.release()

    if generate_segmentation_mask:
        save_segmentation_masks(segmentation_masks, frames_directory_name, output_dir)

    save_landmarks_csvs([face_df, pose_df, left_hand_df, right_hand_df],
                        frames_directory_name,
                        output_dir)

    face_extended, \
        pose_extended, \
        left_hand_extended, \
        right_hand_extended = cta.extend_landmarks_dfs(face_df,
                                                       pose_df,
                                                       left_hand_df,
                                                       right_hand_df,
                                                       frames_width,
                                                       frames_height)

    np.savez_compressed(os.path.join(output_directory, frames_directory_name + ".npz"),
                        face=face_extended.to_numpy(),
                        pose=pose_extended.to_numpy(),
                        left_hand=left_hand_extended.to_numpy(),
                        right_hand=right_hand_extended.to_numpy())

    print('Arrays with landmarks saved')


class FileWithDirectory:
    def __init__(self, d, f_name):
        self.directory = d
        if f_name is not None:
            self.file_with_path = os.path.join(d, f_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--frames',
                       action='store_true')
    group.add_argument('--video',
                       action='store_true')
    parser.add_argument("--input",
                        help="Path to input directory")
    parser.add_argument("--output",
                        help="Path to save outputs")
    parser.add_argument("--annotate",
                        default=False,
                        action=argparse.BooleanOptionalAction,
                        help="Flag whether to generate output video")
    args = parser.parse_args()
    print(args.input)
    files_with_directories = []

    root_directory = ''
    output_root_directory = args.output
    save_annotated_video = args.annotate
    use_frames = args.frames
    use_video = args.video

    if os.path.isdir(args.input):
        root_directory = args.input
        if use_frames:
            for directory, _, _ in os.walk(root_directory, topdown=True):
                files_with_directories.append(FileWithDirectory(directory, None))
        elif use_video:
            for directory, _, files in os.walk(root_directory, topdown=True):
                for file in files:
                    files_with_directories.append(FileWithDirectory(directory, file))
    elif os.path.isfile(args.input):
        files_with_directories = [FileWithDirectory('.', args.input)]
    else:
        raise Exception("Incorrect input file name or directory!")

    for fwd in files_with_directories:
        subdirectory = fwd.directory[len(root_directory):]
        output_directory = os.path.join(output_root_directory, subdirectory)

        try:
            if use_frames:
                process_frames_in_directory(fwd.directory, output_directory, save_annotated_video)
            elif use_video:
                process_single_video_file(fwd.file_with_path, output_directory, save_annotated_video)
        except Exception as e:
            print('Processing error for file ' + fwd.file_with_path)
            print(e)

