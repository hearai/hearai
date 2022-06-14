import os
from typing import Tuple, List

import torch
import torchvision.transforms as T

from datasets.dataset import VideoFrameDataset
from datasets.transforms_creator import TransformsCreator


class DatasetCreator:
    """
    DatasetCreator creates a video frames dataset and split it into train and val subsets.
    """

    def __init__(self,
                 data_paths: List[str],
                 classification_mode: dict,
                 classification_heads: int,
                 num_segments: int,
                 time: float,
                 use_frames: bool,
                 use_landmarks: bool,
                 ratio: float,
                 pre_training: bool,
                 transforms_creator: TransformsCreator):
        self.videos_root = data_paths
        self.classification_mode = classification_mode
        self.classification_heads = classification_heads
        self.num_segments = num_segments
        self.time = time
        self.use_frames = use_frames
        self.use_landmarks = use_landmarks
        self.ratio = ratio
        self.pre_training = pre_training
        self.transforms_creator = transforms_creator

    def get_train_and_val_subsets(self) -> Tuple[torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset]:
        train_transforms = self.transforms_creator.get_train_transforms()
        train_dataset = self._get_video_frame_datasets(train_transforms)

        val_transforms = self.transforms_creator.get_val_transforms()
        val_dataset = self._get_video_frame_datasets(val_transforms)

        train_len, val_len = self._get_split_lens(train_dataset)

        train_subset, _ = torch.utils.data.random_split(train_dataset, [train_len, val_len],
                                                        generator=torch.Generator().manual_seed(0))

        _, val_subset = torch.utils.data.random_split(val_dataset, [train_len, val_len],
                                                      generator=torch.Generator().manual_seed(0))

        return train_subset, val_subset

    def get_test_subset(self) -> torch.utils.data.dataset.Subset:
        test_transforms = self.transforms_creator.get_val_transforms()
        test_dataset = self._get_video_frame_datasets(test_transforms)
        return test_dataset

    def _get_video_frame_datasets(self, transform: T.Compose) -> torch.utils.data.ConcatDataset:
        datasets = []
        for video_root in self.videos_root:
            annotations_path = self._get_annotations_path(video_root)
            dataset = VideoFrameDataset(
                root_path=video_root,
                annotationfile_path=annotations_path,
                classification_heads=self.classification_heads,
                is_pretraining=self.pre_training,
                num_segments=self.num_segments,
                time=self.time,
                use_frames=self.use_frames,
                use_landmarks=self.use_landmarks,
                transform=transform,
                test_mode=True,
            )
            datasets.append(dataset)

        return torch.utils.data.ConcatDataset(datasets)

    def _get_split_lens(self, dataset: torch.utils.data.ConcatDataset) -> Tuple[int, int]:
        train_val_ratio = self.ratio
        train_len = round(len(dataset) * train_val_ratio)
        val_len = len(dataset) - train_len
        return train_len, val_len

    def _get_annotations_path(self, video_root: str) -> str:
        return os.path.join(video_root, f'test_{self.classification_mode}.txt')
