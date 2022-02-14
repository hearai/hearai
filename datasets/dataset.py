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
from utils.classification_mode import create_heads_dict

# for non-latin encoding
import sys

if sys.version[0] == "2":
    reload(sys)
    sys.setdefaultencoding("utf-8")


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

    def __init__(self, row, root_datapath, landmarks_path=None):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])
        try:
            with open(
                os.path.join(landmarks_path, row[0] + "_properties.json"), "r"
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
        if len(self._data) == 4:  # PLACEHOLDER
            return int(self._data[3])
        # TO DO - sample associated with multiple labels
        else:
            return [int(label) for label in self._data[3:]]


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
        classification_mode: mode for classification, choose from
                             classification_mode.py
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders.
        transform: Transform pipeline that receives a list of PIL images/frames.
        test_mode: If True, frames are taken from the center of each
                   segment, instead of a random location in each segment.
    """

    def __init__(
        self,
        root_path: str,
        annotationfile_path: str,
        classification_mode: str,
        num_segments: int = -1,
        frames_per_segment: int = 1,
        time: Union[float, None] = None,
        imagefile_template: str = "{:s}_{:d}.jpg",
        landmarks_path: Union[str, None] = None,
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
        self.landmarks_path = landmarks_path
        self.transform = transform
        self.test_mode = test_mode

        self._parse_annotationfile()
        self._sanity_check_samples()
        self.num_classes_dict = create_heads_dict(classification_mode)

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(
            os.path.join(
                directory,
                self.imagefile_template.format(os.path.basename(directory), idx),
            )
        ).convert("RGB")

    def _parse_annotationfile(self):
        self.video_list = [
            VideoRecord(x.strip().split(), self.root_path, self.landmarks_path)
            for x in open(self.annotationfile_path)
        ]

    def _sanity_check_samples(self):
        if self.time is not None:
            self.num_segments = -1
            self.frames_per_segment = 1
            print(
                f"\nDataset Warning: chosen time was set to {self.time} num_segments and "
                f"frames_per_segment were set to 1!\n"
            )

        for id, record in enumerate(self.video_list):
            if any(not x.isdigit() for x in record._data):
                # Found datafile header. Removing it.
                del self.video_list[id]
                continue

            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(
                    f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n"
                )

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                print(
                    f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                    f"but the dataloader is set up to load "
                    f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                    f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                    f"error when trying to load this video.\n"
                )

    def _get_landmarks(
        self, video_name: str, indices: "np.ndarray[int]", col_name: str = "Unnamed: 0"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        face = pd.read_csv(
            os.path.join(self.landmarks_path, video_name + "_face.csv"),
            index_col=col_name,
        ).loc[indices, :]
        left_hand = pd.read_csv(
            os.path.join(self.landmarks_path, video_name + "_left_hand.csv"),
            index_col=col_name,
        ).loc[indices, :]
        right_hand = pd.read_csv(
            os.path.join(self.landmarks_path, video_name + "_right_hand.csv"),
            index_col=col_name,
        ).loc[indices, :]
        pose = pd.read_csv(
            os.path.join(self.landmarks_path, video_name + "_pose.csv"),
            index_col=col_name,
        ).loc[indices, :]
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
                sys.exit("Number of rames per second not provided")

            start_indices = np.array(
                [
                    int(x)
                    for x in range(record.num_frames)
                    if x % int(record.fps * self.time) == 0
                ]
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
        images = []
        if self.landmarks_path is not None:
            landmarks = {"face": [], "right_hand": [], "left_hand": [], "pose": []}

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)
            if self.landmarks_path is not None:
                face, right_hand, left_hand, pose = self._get_landmarks(
                    record._data[0], frame_start_indices
                )
                landmarks["face"].append(face)
                landmarks["right_hand"].append(right_hand)
                landmarks["left_hand"].append(left_hand)
                landmarks["pose"].append(pose)

            # load self.frames_per_segment consecutive frames
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
        for value, class_label in zip(self.num_classes_dict.values(), labels):
            x = np.zeros(value)
            J = np.random.choice(class_label)
            x[J] = 1
            # x[class_label] = 1
            target.append(torch.tensor(x))

        if self.transform is not None:
            images = self.transform(images)

        return images, target  # , landmarks

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


def collate_fn_padd(batch):
    """
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """
    batch_split = list(zip(*batch))
    seqs, targs = batch_split[0], batch_split[1]
    ## padd
    batch = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return batch, [torch.stack([i[0] for i in targs], 0)]
