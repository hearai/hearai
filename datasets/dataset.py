"""
This code was adopted from
https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
"""
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any
import warnings

# for non-latin encoding
import sys

if sys.version[0] == "2":
    reload(sys)
    sys.setdefaultencoding("utf-8")

ALL_HAMNOSYS_HEADS = {"symmetry_operator": 3,
            "hand_shape_base_form": 4,
            "hand_shape_thumb_position": 5,
            "hand_shape_bending": 6,
            "hand_position_finger_direction": 7,
            "hand_position_palm_orientation": 8,
            "hand_location_x": 9,
            "hand_location_y": 10,
            "distance": 11}

class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 
             1) The first element is the path to the video sample's
             frames excluding the root_datapath prefix,
             2) The  second element is the starting frame id of the video,
             3) the third element is the inclusive ending frame id of the video,
             4) the fourth element is the gloss,
             5) any following elements are labels in the case of multi label provided.
    """

    def __init__(self, row, root_datapath, classification_heads):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])
        self.classification_heads = classification_heads
        try:
            with open(
                os.path.join(self._path, row[0] + "_properties.json"), "r"
            ) as f:
                data = json.load(f)
            self.fps = int(data["FPS"])
        except:
            self.fps = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        # +1 because end frame is inclusive
        return self.end_frame - self.start_frame + 1

    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> Union[str, List[str]]:
        # just one label as gloss
        if "gloss" in self.classification_heads.keys():
            return int(self._data[3])
        # Sample associated with multiple labels - HamNoSys
        else:
            id_list = []
            for key in self.classification_heads.keys():
                id_list.append(self._data[ALL_HAMNOSYS_HEADS[key]])
            return [int(label) for label in id_list]


class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used. This dataset broadly corresponds to the frame
    sampling technique introduced in ``Temporal Segment Networks``
    at ECCV2016 https://arxiv.org/abs/1608.00859.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Args:
        root_path: The root path in which video folders lie.
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        is_pretraining: flag to launch pretrianing block
        classification_heads: dict with names of classification heads
                              and number of their classes
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders.
        use_landmarks: If True, additional landmarks are taken as annotations
        transform: Transform pipeline that receives a list of PIL images/frames.
        test_mode: If True, frames are taken from the center of each
                   segment, instead of a random location in each segment.
    """

    def __init__(
        self,
        root_path: str,
        annotationfile_path: str,
        is_pretraining: bool,
        classification_heads={},
        num_segments: int = -1,
        frames_per_segment: int = 1,
        time: Union[float, None] = None,
        imagefile_template: str = "{:s}_{:d}.jpg",
        use_frames: bool = True,
        use_landmarks: bool = False,
        use_face_landmarks: bool = False,
        transform=None,
        test_mode: bool = False,
    ):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.time = time
        self.imagefile_template = imagefile_template
        self.use_frames = use_frames
        self.use_landmarks = use_landmarks
        self.use_face_landmarks = use_face_landmarks
        self.transform = transform
        self.test_mode = test_mode
        self.is_pretraining = is_pretraining
        self.classification_heads = classification_heads

        self._parse_annotationfile()
        self._sanity_check_samples()
        

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(
            os.path.join(
                directory,
                self.imagefile_template.format(os.path.basename(directory), idx),
            )
        ).convert("RGB")

    def _parse_annotationfile(self):
        self.video_list = [
            VideoRecord(x.strip().split(), self.root_path,
                        self.classification_heads)
            for x in open(self.annotationfile_path)
        ]

    def _sanity_check_samples(self):
        if self.time is not None:
            self.frames_per_segment = 1
            warnings.warn(
                f"Dataset Warning: chosen time was set to {self.time}"
                f"frames_per_segment were set to 1!\n"
            )

        for id, record in enumerate(self.video_list):
            if any(not x.isdigit() for x in record._data[1:]):
                # Found datafile header. Removing it.
                del self.video_list[id]
                continue

            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                warnings.warn(
                    f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n"
                )

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                warnings.warn(
                    f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                    f"but the dataloader is set up to load "
                    f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                    f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                    f"error when trying to load this video.\n"
                )

    def _get_landmarks(self, video_name: str
                       ) -> Tuple["np.ndarray[float]", "np.ndarray[float]", "np.ndarray[float]", "np.ndarray[float]"]:
        landmarks = np.load(os.path.join(self.root_path, video_name, video_name + ".npz"))
        face = landmarks['face']
        left_hand = landmarks['left_hand']
        right_hand = landmarks['right_hand']
        pose = landmarks['pose']
        return face, right_hand, left_hand, pose

    def _get_start_indices(self, record: VideoRecord) -> "np.ndarray[int]":
        """
        For each segment, choose a start index from where frames
        are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        # choose start indices that are spread each time unit given by user
        if self.time is not None:
            if record.fps is None:
                sys.exit("Number of frames per second not provided")

            start_indices = np.array(
                [
                    int(x)
                    for x in range(record.num_frames)
                    if x % int(record.fps * self.time) == 0
                ]
            )
            if len(start_indices) > self.num_segments:
                warnings.warn(
                    f"Number of segments too small! "
                    f"{len(start_indices)} > {self.num_segments} "
                    f"Videos will be cut!"
                )
        # choose start indices that are perfectly evenly spread across the video frames.
        elif self.test_mode:
            distance_between_indices = (
                record.num_frames - self.frames_per_segment + 1
            ) / float(self.num_segments)

            start_indices = np.array(
                [
                    int(distance_between_indices / 2.0 + distance_between_indices * x)
                    for x in range(self.num_segments)
                ]
            )
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (
                record.num_frames - self.frames_per_segment + 1
            ) // self.num_segments

            start_indices = np.multiply(
                list(range(self.num_segments)), max_valid_start_index
            ) + np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indices

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple[
            "torch.Tensor[num_frames, channels, height, width]", Union[int, List[int]]
        ],
        Tuple[Any, Union[int, List[int]]],
    ]:
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.

        Args:
            idx: Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        record: VideoRecord = self.video_list[idx]

        frame_start_indices: "np.ndarray[int]" = self._get_start_indices(record)

        return self._get(record, frame_start_indices)

    def _get(
        self, record: VideoRecord, frame_start_indices: "np.ndarray[int]"
    ) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple[
            "torch.Tensor[num_frames, channels, height, width]", Union[int, List[int]]
        ],
        Tuple[Any, Union[str, List[str]]],
    ]:
        """
        Loads the frames of a video at the corresponding
        indices.

        Args:
            record: VideoRecord denoting a video sample.
            frame_start_indices: Indices from which to load consecutive frames from.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either
            1) a list of PIL images if no transform is used,
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH)
               in the range [0,1] if the transform "ImglistToTensor" is used,
            3) or anything else if a custom transform is used.
        """

        frame_start_indices = frame_start_indices + record.start_frame
        images = None
        if self.use_frames:
            images = []
        landmarks = None
        if self.use_landmarks:
            landmarks = {"right_hand": [], "left_hand": [], "pose": []}
            if self.use_face_landmarks:
                landmarks["face"] = []
            face, right_hand, left_hand, pose = self._get_landmarks(record._data[0])

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)
            if self.use_landmarks:
                landmarks["pose"].append(pose[frame_index, :])
                landmarks["right_hand"].append(right_hand[frame_index, :])
                landmarks["left_hand"].append(left_hand[frame_index, :])
                if self.use_face_landmarks:
                    landmarks["face"].append(face[frame_index, :])

            # load self.frames_per_segment consecutive frames
            if self.use_frames:
                for _ in range(self.frames_per_segment):
                    image = self._load_image(record.path, frame_index)
                    images.append(image)

                    if frame_index < record.end_frame:
                        frame_index += 1
        # create target in one-hot encoding form
        target = []
        labels = []
        if type(record.label) == int:
            labels.append(record.label)
        else:
            labels = record.label
        for value, class_label in zip(self.classification_heads.values(), labels):
            x = np.zeros(value["num_class"])
            x[class_label] = 1
            if self.is_pretraining:
                target.append(torch.tensor(x, dtype=torch.long))
            else:
                target.append(torch.tensor(x))

        if self.transform is not None and self.use_frames:
            img_list_to_tensor = ImglistToTensor()
            images = img_list_to_tensor(images)
            images = [self.transform(image=np.moveaxis(image.numpy(), 0, -1)) for image in images]
            images = torch.stack([torch.tensor(image['image']) for image in images])
            # images = self.transform(images)

        return images, {"target": target, "landmarks": landmarks}

    def __len__(self):
        return len(self.video_list)


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """

    @staticmethod
    def forward(
        img_list: List[Image.Image],
    ) -> "torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]":
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])


class PadCollate:
    """
    a variant of collate_fn that pads according to the 
    chosen total_length
    """

    def __init__(self, total_length: int):
        """
        args:
            total_length - total length to be padded
        """
        self.total_length = total_length

    def collate_fn_padd(self, batch: list):
        """
        Pads batch of variable length to have length total_length
        args:
            batch - list of [tensor, label, landmarks]

        note: if self.total_length is smaller than num of frames
        the sequence will be cut
        """
        batch_split = list(zip(*batch))
        seqs, targets_and_landmarks = batch_split[0], batch_split[1]

        targets = []
        landmarks = []
        for item in targets_and_landmarks:
            targets.append(item['target'])
            landmarks.append(item['landmarks'])

        new_batch = None
        if seqs[0] is not None:
            # get new dimensions
            dims = [len(seqs)]
            dims.extend(list(seqs[0].shape))
            dims[1] = self.total_length

            # pad all seqs to desired length with 0
            # or get rid of too many frames
            new_batch = torch.zeros(dims)
            for i, tensor in enumerate(seqs):
                length = tensor.size(0)
                if length <= self.total_length:
                    new_batch[i, :length, ...] = tensor
                else:
                    new_batch[i, ...] = tensor[: self.total_length, ...]

        # padding for landmarks
        stacked_landmarks = None
        if landmarks[0] is not None:
            stacked_landmarks = {}
            for landmark_name in landmarks[0].keys():
                padded_landmarks = []
                for single_video_landmarks in landmarks:
                    concatenated_landmarks = np.stack(single_video_landmarks[landmark_name], axis=0)
                    landmarks_frames_number = len(concatenated_landmarks)
                    if landmarks_frames_number < self.total_length:
                        padded_landmarks.append(
                            np.pad(concatenated_landmarks,
                                   pad_width=((0, self.total_length-landmarks_frames_number), (0, 0)),
                                   mode='edge')
                        )
                    else:
                        padded_landmarks.append(concatenated_landmarks[:self.total_length, ...])
                stacked_landmarks[landmark_name] = np.nan_to_num(np.stack(padded_landmarks, axis=0), nan=-1+10)

        return (
            new_batch,
            stacked_landmarks,
            [
                torch.stack([target[i] for target in targets], 0)
                for i in range(len(targets[0]))
            ]
        )

    def __call__(self, batch):
        return self.collate_fn_padd(batch)
