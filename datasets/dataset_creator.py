import os

import torch
import torchvision.transforms as T

from datasets.dataset import ImglistToTensor, VideoFrameDataset


class DatasetCreator:
    """
    DatasetCreator creates a video frames dataset and split it into train and val subsets.
    """

    def __init__(self, data_paths: list, classification_mode: dict, classification_heads, num_segments: int,
                 time: float, landmarks: bool,
                 ratio: float, pre_training: bool):
        self.videos_root = data_paths
        self.classification_mode = classification_mode
        self.classification_heads = classification_heads
        self.num_segments = num_segments
        self.time = time
        self.landmarks = landmarks
        self.ratio = ratio
        self.pre_training = pre_training

    def get_train_and_val_subsets(self) -> (torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset):
        train_transforms = self._get_train_transforms()
        train_dataset = self._get_video_frame_datasets(train_transforms)

        val_transforms = self._get_val_transforms()
        val_dataset = self._get_video_frame_datasets(val_transforms)

        train_len, val_len = self._get_split_lens(train_dataset)

        _, train_subset = torch.utils.data.random_split(train_dataset, [train_len, val_len],
                                                        generator=torch.Generator().manual_seed(0))

        val_subset, _ = torch.utils.data.random_split(val_dataset, [train_len, val_len],
                                                      generator=torch.Generator().manual_seed(0))

        return train_subset, val_subset

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

    def _get_annotations_path(self, video_root: str) -> str:
        return os.path.join(video_root, f'test_{self.classification_mode}.txt')

    def _get_train_transforms(self) -> T.Compose:
        pil_augmentations = [

        ]
        tensor_augmentations = [
            T.RandomErasing(),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.1,
                          contrast=0.1,
                          saturation=0.1,
                          hue=0.05)
        ]
        return self._get_transforms(pil_augmentations, tensor_augmentations)

    def _get_val_transforms(self) -> T.Compose:
        pil_augmentations = []
        tensor_augmentations = []
        return self._get_transforms(pil_augmentations, tensor_augmentations)

    @staticmethod
    def _get_transforms(pil_augmentations: list, tensor_augmentations: list) -> T.Compose:
        return T.Compose(
            [
                *pil_augmentations,
                ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
                *tensor_augmentations,
                T.Resize(256),  # image batch, resize smaller edge to 256
                T.CenterCrop(256),  # image batch, center crop to square 256x256
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
