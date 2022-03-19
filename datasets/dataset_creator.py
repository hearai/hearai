import os

import pandas as pd
import sys
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader

from datasets.dataset import PadCollate, VideoFrameDataset
from datasets.transforms_creator import TransformsCreator


def iterative_train_test_split(X, y, train_size):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2, sample_distribution_per_fold=[1.0 - train_size, train_size]
    )
    train_indices, test_indices = next(stratifier.split(X, y))
    return train_indices, test_indices


class DatasetCreator:
    """
    DatasetCreator creates a video frames dataset and split it into train and val subsets.
    """

    def __init__(
        self,
        data_paths: list,
        classification_mode: dict,
        classification_heads,
        num_segments: int,
        time: float,
        landmarks: bool,
        ratio: float,
        pre_training: bool,
        transforms_creator: TransformsCreator,
    ):
        self.videos_root = data_paths
        self.classification_mode = classification_mode
        self.classification_heads = classification_heads
        self.num_segments = num_segments
        self.time = time
        self.landmarks = landmarks
        self.ratio = ratio
        self.pre_training = pre_training
        self.transforms_creator = transforms_creator

    def get_train_and_val_dataloaders(
        self, batch_size: int, workers: int, num_segments: int, mode: str = "random"
    ) -> (
        torch.utils.data.dataloader.DataLoader,
        torch.utils.data.dataloader.DataLoader,
    ):
        train_transforms = self.transforms_creator.get_train_transforms()
        train_dataset = self._get_video_frame_datasets(train_transforms)

        val_transforms = self.transforms_creator.get_val_transforms()
        val_dataset = self._get_video_frame_datasets(val_transforms)

        if mode == "stratify":
            train_idx, val_idx = self._get_stratify_indices()
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            # prepare dataloaders
            dataloader_train = DataLoader(
                train_dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=workers,
                collate_fn=PadCollate(total_length=num_segments),
                drop_last=False,
                sampler=train_sampler,
            )

            dataloader_val = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=workers,
                collate_fn=PadCollate(total_length=num_segments),
                drop_last=False,
                sampler=val_sampler,
            )
        elif mode == "random":
            train_len, val_len = self._get_split_lens(train_dataset)
            train_subset, _ = torch.utils.data.random_split(
                train_dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(0),
            )
            _, val_subset = torch.utils.data.random_split(
                val_dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(0),
            )
            # prepare dataloaders
            dataloader_train = DataLoader(
                train_subset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=workers,
                collate_fn=PadCollate(total_length=num_segments),
                drop_last=False,
            )

            dataloader_val = DataLoader(
                val_subset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=workers,
                collate_fn=PadCollate(total_length=num_segments),
                drop_last=False,
            )
        else:
            sys.exit("Wrong mode for loading datasets!")
        return dataloader_train, dataloader_val

    def _get_video_frame_datasets(
        self, transform: T.Compose
    ) -> torch.utils.data.ConcatDataset:
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
                landmarks=self.landmarks,
                transform=transform,
                test_mode=True,
            )
            datasets.append(dataset)

        return torch.utils.data.ConcatDataset(datasets)

    def _get_split_lens(self, dataset: torch.utils.data.ConcatDataset) -> (int, int):
        train_val_ratio = self.ratio
        train_len = round(len(dataset) * train_val_ratio)
        val_len = len(dataset) - train_len
        return train_len, val_len

    def _get_stratify_indices(self) -> (list, list):
        datasets = []
        for ann_root in self.videos_root:
            annotations_path = self._get_annotations_path(ann_root)
            data = pd.read_csv(annotations_path, sep=" ")
            datasets.append(data)
        all_data = pd.concat(datasets, ignore_index=True)
        all_data.dropna(axis=1, inplace=True)
        labels = all_data.drop(columns=["name", "start", "end"]).to_numpy().astype(int)
        name = all_data["name"].to_numpy()
        train, test = iterative_train_test_split(name, labels, train_size=self.ratio)
        return train, test

    def _get_annotations_path(self, video_root: str) -> str:
        return os.path.join(video_root, f"test_{self.classification_mode}.txt")
