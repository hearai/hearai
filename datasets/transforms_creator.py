from typing import Dict

# import torchvision.transforms as T
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
from datasets.dataset import ImglistToTensor


class TransformsCreator:
    """
    Class for transforms creation and parametrization.
    """

    def __init__(self, augmentations_parameters: Dict = None):
        """
        Args:
            augmentations_parameters (Dict): Dict containing parameters regarding currently used augmentations.
                [Warning] Must contain fields:
                    - "apply_resize" (bool)
                    - "apply_center_crop" (bool)
                    - "apply_random_erasing" (bool)
                    - "apply_random_rotation" (bool)
                    - "apply_color_jitter" (bool)

                    - "resize_size" (int)
                    - "center_crop_size" (int)
                    - "random_erasing_probability" (float)
                    - "random_rotation_degree" (int)
                    - "color_jitter_brightness" (float)
                    - "color_jitter_contrast" (float)
                    - "color_jitter_saturation" (float)
                    - "color_jitter_brcolor_jitter_hueightness" (float)
        """

        self.apply_resize = augmentations_parameters["apply_resize"]
        self.apply_center_crop = augmentations_parameters["apply_center_crop"]
        self.apply_random_erasing = augmentations_parameters["apply_random_erasing"]
        self.apply_random_rotation = augmentations_parameters["apply_random_rotation"]
        self.apply_color_jitter = augmentations_parameters["apply_color_jitter"]

        self.resize_size = augmentations_parameters["resize_size"]

        self.center_crop_size = augmentations_parameters["center_crop_size"]

        self.random_erasing_probability = augmentations_parameters["random_erasing_probability"]

        self.random_rotation_degree = augmentations_parameters["random_rotation_degree"]

        self.color_jitter_brightness = augmentations_parameters["color_jitter_brightness"]
        self.color_jitter_contrast = augmentations_parameters["color_jitter_contrast"]
        self.color_jitter_saturation = augmentations_parameters["color_jitter_saturation"]
        self.color_jitter_hue = augmentations_parameters["color_jitter_hue"]

    def get_train_transforms(self) -> A.Compose:
        pil_augmentations = []
        tensor_augmentations = []

        # if self.apply_random_erasing:
        #     tensor_augmentations.append(T.RandomErasing(self.random_erasing_probability, scale=(0.1, 0.1)))
        #
        # if self.apply_random_rotation:
        #     tensor_augmentations.append(T.RandomRotation(degrees=self.random_rotation_degree))
        #
        # if self.apply_color_jitter:
        #     tensor_augmentations.append(T.ColorJitter(brightness=self.color_jitter_brightness,
        #                                               contrast=self.color_jitter_contrast,
        #                                               saturation=self.color_jitter_saturation,
        #                                               hue=self.color_jitter_hue))

        tensor_augmentations = [
            # A.Rotate(limit=5, p=0.9, border_mode=cv2.BORDER_CONSTANT),
            # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
            # A.RandomBrightnessContrast(p=0.2),
            # A.OneOf([
            #     A.Blur(blur_limit=3, p=0.5),
            #     A.ColorJitter(p=0.5),
            # ], p=1.0),
        ]

        return self._get_transforms(pil_augmentations, tensor_augmentations)

    def get_val_transforms(self) -> A.Compose:
        pil_augmentations = []
        tensor_augmentations = []
        return self._get_transforms(pil_augmentations, tensor_augmentations)

    def _get_transforms(self, pil_augmentations: list, tensor_augmentations: list) -> A.Compose:
        return A.Compose(
            [
                *pil_augmentations,
                # ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
                *tensor_augmentations,
                A.SmallestMaxSize(max_size=self.resize_size),  # image batch, resize smaller edge to 256
                A.CenterCrop(width=self.center_crop_size, height=self.center_crop_size),
                # # image batch, center crop to square 256x256
                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        )
